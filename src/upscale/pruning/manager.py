"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

High-level API for using this pruning export implementation.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .mock import MockPrunedModel
from .pruner import prune_model
from .resolve import generate_pruning_spec
from .tracing import trace
from .utils import get_mask_indices


class PruningManager(nn.Module):
    """External API for using pruning utilities. Use manager to wrap and prune
    models.

    Instantiate any PyTorch model of your choice, along with its input.

    >>> import torch
    >>> from torchvision.models import resnet18
    >>> net = resnet18()
    >>> x = torch.rand((1, 3, 224, 224))

    Compute pruning parameters using a set of pruned input channel indices.

    >>> manager = PruningManager(net)
    >>> pruned_inputs = [('layer1.1.conv1.weight', 5)]
    >>> manager.compute([x], pruned_inputs=pruned_inputs)

    Then, during training, you can use the mock-pruned model. This mock-pruned
    model applies masks instead of modifying the model itself.

    >>> net_mock = manager.get_mock_model()
    >>> y_mock = net_mock(x)

    Finally, actually prune the model. Then run inference using the *original
    (now modified, in place) model.

    >>> manager.prune()
    >>> y_pruned = net(x)

    Check that both the mocked and pruned outputs match.

    >>> (y_pruned - y_mock).abs().max() < 1e-5
    tensor(True)

    Note that 1e-5 may seem generous. However, this is a strange artifact of
    consumer channel reordering. To obtain higher precision matches, using
    higher precision (fp64) input or disable consumer reordering
    (`reorder_consumer=False` for `PruningMangaer.compute`).
    """

    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.spec = None
        self.traced = None

    def forward(self, *args, **kwargs):
        return self.net.forward(*args, **kwargs)

    def trace(self, inputs: List[torch.Tensor], force: bool = False):
        if self.traced is None or force:
            self.traced = trace(self.net, inputs)
        return self.traced

    def compute(
        self,
        inputs: List[torch.Tensor],
        force: bool = False,
        pruned_inputs: Optional[List[Tuple]] = None,
        pruned_outputs: Optional[List[Tuple]] = None,
        **kwargs
    ):
        """
        Compute pruning specification for the current model.

        By default, if neither `pruned_inputs` nor `pruned_outputs` is
        provided, look for zero'ed out channels.

        This step is computed separately so that you can inspect the pruning
        specification before applying the pruning itself.

        Args:
            inputs: List of input tensors to provide to the network during a
                forward pass. Are passed as net(*inputs)
            force: Forcibly recompute the pruning spec, ignoring any
                pre-computed values
            pruned_inputs: Collection of input pruned channels, formatted as
                (tensor id OR param name, channel id)
            pruned_outputs: Collection of output pruned channels
            **kwargs: Any other keyword arguments are forwarded to
                `generate_pruning_spec`.
        """
        if self.spec is None or force:
            traced = self.trace(inputs, force)
            if pruned_inputs is None and pruned_outputs is None:
                pruned_inputs, pruned_outputs = get_mask_indices(traced.tensor_to_metadata)
            self.spec = generate_pruning_spec(
                traced, pruned_outputs, pruned_inputs, net=self.net, **kwargs)

    def get_mock_model(self):
        """
        Obtain a model you can run normally, just like any other PyTorch model.
        Emulates a pruned model.
        """
        assert self.spec is not None, (
            'Need to run PruningManager.compute before pruning or '
            'mocking pruning.'
        )
        return MockPrunedModel(self, spec=self.spec)

    def prune(self, is_not_ane: bool = False, baseline: bool = False):
        """
        Prune the wrapped model in-place. This is irreversible.

        Note that the wrapped model, after pruning, can't be reloaded after
        saving. The architecture will differ, so attempting to load a pruned
        checkpoint will result in errors. After this step, the only path
        forward is to fully export the model, e.g., using torchscript.
        """
        prune_model(self.spec, is_not_ane=is_not_ane, baseline=baseline)
        self.spec = None  # NOTE: clear pruning params, which are now invalid

