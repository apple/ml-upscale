"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.
"""

from collections import defaultdict
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


logger = logging.getLogger("pruning")


class SetLikeList(list):
    """Custom list that supports some syntactic sugar."""

    def __sub__(self, other):
        """
        >>> SetLikeList([1, 2, 3]) - {2}
        [1, 3]
        """
        return SetLikeList([item for item in self if item not in other])

    __isub__ = __sub__
    __rsub__ = __sub__

    def intersection(self, other: set):
        """
        >>> SetLikeList([1, 2, 3]).intersection({2, 1})  # respect og ordering
        [1, 2]
        """
        return SetLikeList([item for item in self if item in other])


def get_n_channels(tensor: torch.Tensor, axis: int = 1) -> int:
    """
    Get number of channels in tensor.

    >>> import torch
    >>> get_n_channels(torch.rand((1, 3, 224, 224)))
    3
    >>> get_n_channels(torch.rand((24,)))
    24
    >>> get_n_channels(torch.tensor(0.))
    1
    """
    if len(tensor.shape) > axis:
        return tensor.shape[axis]
    if len(tensor.shape) > 0:
        return tensor.shape[0]
    return 1


def channels_to_mapping(channels: List[Tuple]) -> Dict:
    """
    Convert iterable of (op idx, channel idx) to mapping from op idx to list
    of channel indices.

    If input is ordered, maintain order in mapping.
    """
    tensor_to_channels: Dict = defaultdict(
        set if isinstance(channels, set) else list)
    for tensor_id, channel in channels:
        container = tensor_to_channels[tensor_id]
        f = container.add if isinstance(container, set) else container.append
        f(channel)
    return {
        tensor_id: [channels]
        for tensor_id, channels in tensor_to_channels.items()
    }


def invert(indices: List[int], n: int) -> List[int]:
    """Invert a list of indices."""
    invalids = set(indices)
    return [i for i in range(n) if i not in invalids]


def invert_mapping_indices(
    tensor_to_channels: Dict[int, List],
    tensor_to_n_channels: Dict[int, List],
) -> Dict[int, List]:
    """Invert all indices in the provided mapping."""
    return {
        tensor_id: [
            invert(op_indices[0], n)
            for n in tensor_to_n_channels[tensor_id]
        ] for tensor_id, op_indices in tensor_to_channels.items()
    }


def group_indices_as_ranges(indices: List[int]) -> List[range]:
    """Group together sequential indices as ranges.

    The function currently only groups together increasing sequences into
    ranges.

    Args:
        indices: Collection of indices

    Returns:
        A list of range objects. Taken together, all the range objects will
            cover the same set of indices.

    >>> group_indices_as_ranges([7, 0, 1, 2, 8, 9, 4, 5])
    [range(7, 8), range(0, 3), range(8, 10), range(4, 6)]
    >>> group_indices_as_ranges([1, 0])
    [range(1, 2), range(0, 1)]
    """
    start, cur, ranges = None, None, []
    for i in indices:
        if start is None:
            start = i
        elif i != cur + 1:
            ranges.append(range(start, cur + 1))
            start = i
        cur = i
    if start is not None and cur is not None:  # capture last range
        ranges.append(range(start, cur + 1))
    return ranges


def get_tensor_and_modules(tensor_to_metadata):
    """
    Obtain a `tensor_to_metadata` dictionary by tracing a model. The output
    tensor will have a `y.tensor_to_metadata` property.
    """
    for tensor_id, metadata in tensor_to_metadata.items():
        for param in metadata.non_tracers:
            if not hasattr(param, '_metadata'):
                continue
            module = param._metadata['module']
            yield tensor_id, module
            break  # NOTE: Return just once per module


def get_tensor_to_module_metadata(tensor_to_metadata: Dict) -> Dict[str, Dict]:
    """
    Assemble metadata for the entire graph.

    Specifically, get mapping from tensor id to {module, parent module}
    pointers and names.

    Returns:
        Mapping from tensor id to extracted metadata.
    """
    tensor_to_module_metadata = {}
    for tensor_id, module in get_tensor_and_modules(tensor_to_metadata):
        module_data = module._metadata
        name = module_data['name'].split('.')[-1] \
            .replace('sequential_', '')  # a.b.c -> c
        tensor_to_module_metadata[tensor_id] = {
            'module_name': name,
            'module': module,
            'parent': module_data['parent'],
        }
    return tensor_to_module_metadata


def get_masked_channels(weight, dim=0):
    weight = weight.transpose(dim, 0)
    all_zeros = (weight.reshape(weight.shape[0], -1) == 0).all(dim=1)
    return list(map(int, all_zeros.nonzero()))


def get_mask_indices(tensor_to_metadata):
    tensor_to_module_metadata = get_tensor_to_module_metadata(tensor_to_metadata)
    pruned_inputs, pruned_outputs = [], []
    for tensor_id, metadata in tensor_to_module_metadata.items():
        module = metadata['module']
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            for lst, dim in (
                (pruned_inputs, 1),
                (pruned_outputs, 0)
            ):
                indices = get_masked_channels(module.weight, dim=dim)
                if indices:
                    lst.extend([(tensor_id, channel) for channel in indices])
    return pruned_inputs, pruned_outputs