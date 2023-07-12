"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

Utilities that use the pruning spec to to actually prune the model in-place.

The main utility is `prune_model`, which will prune the provided model in
place, dropping and reordering weights so that the exported model is deployment
ready.

To read this file, start from `prune_model`.
"""

from typing import Any, Dict, List, Optional
from upscale.pruning.resolve import PruningSpec
from upscale.pruning.tracing import Metadata
import torch
import torch.nn as nn

from .utils import get_tensor_and_modules, get_tensor_to_module_metadata, group_indices_as_ranges


def generate_pruned_weight(
    weight: torch.Tensor,
    input_weight_indices: Optional[List[int]],
    output_weight_indices: Optional[List[int]],
    module: nn.Module,
) -> torch.Tensor:
    """Prune the provided tensor.

    Assumes the tensor is organized as (cout, cin, ...) or (cout,). The former
    includes convolutional and linear weights. The latter includes the
    respective layer's biases and broadcasted operands (e.g., scalar)

    Args:
        weight: Original tensor of parameters to be pruned
        input_weight_indices: Input weight indices for parameter. Assumed to be
            the second dimension.
        output_weight_indices: Output weight indices for parameter. Assumed to
            be first dimension.
        func_name: Name of the function this parameter was used in.
        module: The module this parameter belongs to.
        is_depthwise: Whether or not the provided module is a depthwise
            convolution.
        update_groups: Function to update the number of groups for a depthwise
            convolution.

    Returns:
        The resulting pruned and reordered tensor of parameters
    """
    module_name = module.__class__.__name__.lower()
    is_constant = len(weight.shape) == 1
    is_bias = is_constant and \
        ('conv' in module_name or 'linear' in module_name)
    is_depthwise = (
        'conv' in module.__class__.__name__.lower()
        and getattr(module, 'groups', 0) > 1
    )
    # or (input_weight_indices is not None and max(input_weight_indices) > 1
    # and len(weight.shape) > 1 and weight.shape[1] == 1):
    # TODO: do we need this
    if ('transpose' in module_name and not is_bias):
        output_weight_indices, input_weight_indices = \
            input_weight_indices, output_weight_indices
    if is_depthwise or (is_constant and not is_bias):
        output_weight_indices, input_weight_indices = \
            input_weight_indices, None
    if is_depthwise and output_weight_indices is not None:
        module.groups = len(output_weight_indices)
    if output_weight_indices:  # Prune output weights
        weight = weight[output_weight_indices]
    if input_weight_indices and len(weight.shape) > 1:  # Prune input weights
        weight = weight[:, input_weight_indices]
    assert (
        len(weight.shape) < 2
        or 'conv' not in module_name
        or (module.out_channels, module.in_channels == weight.shape[:2])
    )
    return weight


class Subselect(nn.Module):
    """PyTorch layer to subselect input channels.

    Args:
        indices: Channel indices to subselect from the input
        is_not_ane: Use all indices at once in a single index layer. Should
            be disabled for ANE/M1, where this is not supported. Should
            enable for GPU.

    >>> subselect = Subselect([0, 1])
    >>> x = torch.rand((1, 3, 28, 28))
    >>> subselect(x).shape
    torch.Size([1, 2, 28, 28])
    >>> subselect2 = Subselect(
    ...     [0, 1], is_not_ane=True, axis=2)
    >>> subselect2(x).shape
    torch.Size([1, 3, 2, 28])
    >>> subselect2 = Subselect([0, 1], axis=2)
    >>> subselect2(x).shape
    torch.Size([1, 3, 2, 28])
    """

    def __init__(
        self,
        indices: List[int],
        is_not_ane: bool = False,
        baseline: bool = False,
        axis: int = 1
    ):
        super().__init__()
        self.indices = indices
        self.is_not_ane = is_not_ane
        self.baseline = baseline
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use ranges as much as possible, to lessen number of tensor copies
        """
        if self.baseline:
            if self.is_not_ane:
                return x[self.slice()]
            else:
                return torch.cat([
                    x[self.slice(slice(i, i + 1))] for i in self.indices
                ], dim=1)
        if isinstance(self.indices, range):
            if self.is_not_ane:
                # NOTE: This means x[:, start:stop]
                return torch.narrow(
                    x, self.axis, self.indices.start,
                    self.indices.stop - self.indices.start)
            else:
                return x[self.slice(slice(
                    self.indices.start, self.indices.stop))]
        ranges = group_indices_as_ranges(self.indices)
        if self.is_not_ane:
            return x[self.slice()]
        else:
            return torch.cat([
                x[self.slice(slice(r.start, r.stop))] for r in ranges
            ], dim=1)

    def slice(self, indices=None):
        indices = indices or list(self.indices)
        slices = [slice(None) for _ in range(self.axis)]
        slices.append(indices)
        return tuple(slices)


def insert_subselection(
    metadata: Dict[str, Any],
    input_activation_indices: List,
    is_not_ane: bool = False,
    baseline: bool = False,
    channel_axis: int = 1,
):
    """
    Insert layer to subselect input tensor, in-place.

    Rather than computing subselection on the fly, insert an layer so that the
    resulting computation graph has this subselection mechanism built-in and
    exportable. This function does operate on the computation graph in place.

    Args:
        metadata: Metadata for the subselection mechanism. Needed to replace
            the original module with a subselected one.
        input_activation_indices: Indices into the old activation to obtain the
            new activation.
        Sequential: Sequential layer
        Subselect: Subselect layer
    """
    module = metadata['module']
    parent = metadata['parent']
    child_name = metadata['module_name']

    layer = nn.Sequential(Subselect(
        input_activation_indices,
        is_not_ane=is_not_ane,
        baseline=baseline,
        axis=channel_axis,
    ), module)

    if child_name.isdigit():  # if parent is sequential, re-assign item in list
        parent[int(child_name)] = layer
    else:
        setattr(parent, child_name, layer)


def prune_model(
    spec,
    is_not_ane: bool = False,
    baseline: bool = False,
):
    """Prune the provided model, by modifying weights in-place and inserting an
    activation subselection layer.

    Args:
        net: PyTorch model
        spec: a `PruningSpec` returned by generate_pruning_spec, which
            specifies activation and weight indices
    """
    tensor_to_module_metadata = get_tensor_to_module_metadata(spec.tensor_to_metadata)
    tensor_to_inputs_weight_indices = spec.tensor_to_inputs_weight_indices
    tensor_to_outputs_weight_indices = spec.tensor_to_outputs_weight_indices
    tensor_to_input_activation_indices = \
        spec.tensor_to_input_activation_indices

    for tensor_id, module_metadata in tensor_to_module_metadata.items():
        module_metadata = tensor_to_module_metadata[tensor_id]
        metadata = spec.tensor_to_metadata[tensor_id]
        for param in metadata.non_tracers:
            if len(param.shape) > 0:
                module = module_metadata.get('module', None)
                pruned = generate_pruned_weight(
                    weight=param.data,
                    # NOTE: [0] bc one set of indices per op
                    input_weight_indices=tensor_to_inputs_weight_indices.get(
                        tensor_id, [None])[0],
                    output_weight_indices=tensor_to_outputs_weight_indices.get(
                        tensor_id, [None])[0],
                    module=module,
                )
                param.data = pruned
        if tensor_id in tensor_to_input_activation_indices:
            insert_subselection(
                module_metadata,
                tensor_to_input_activation_indices[tensor_id],
                is_not_ane=is_not_ane,
                baseline=baseline,
                channel_axis=metadata.channel_axis,
            )
