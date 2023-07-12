"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

This file contains export algorithsm, converting a set of pruned channels to
lists of weight and activation indices. There are two primary 'modes' to
consider as a user:

    1. 'Constrained' mode. In this mode, you are guaranteed no memory copies
       during inference time. However, to accomplish this, the algorithm will
       grow the set of pruned channels so that all pruning masks in a segment
       are 'equal'. In other words, if two layers share an output, they must
       prune exactly the same set of channels. In short, guaranteed fast
       inference but inflexibility (may need more fine-tuning). This is found
       in `generate_constrained_indices`.

    2. 'Unconstrained' mode. In this mode, the original set of pruned channels
       is perfectly preserved. No additional channels are pruned. However,
       every layer may require its own set of pruned channels, distinct from
       other producers or consumers in the same segment. This means that memory
       copies may be incurred; to minimize this cost, we reorder both producer
       and consumer channels. In short, slower inference but greater
       flexibility (no retraining needed). This is found in
       `generate_unconstrained_indices`.

The utilities below compute indices to handle pruned channels. To reorder
channels to minimize memory copies, see the associated graph algorithm in
`reorder.py`.

To read this file, start from `generate_pruning_spec`.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .reorder import reorder_consumer_weights, reorder_producer_weights
from .utils import SetLikeList, invert_mapping_indices, logger, \
    channels_to_mapping
from .trace import Consumer, for_all_producers, for_all_sources, \
    producers_trace, prune_trace, reorder
from .tracing import Metadata, Tracer


@dataclass
class PruningSpec:
    """
    Pruning specification for export. Includes weight and activation reordering
    + pruning indices for a single op.
    """
    tensor_to_metadata: Dict[int, Dict] = field(default_factory=dict)
    tensor_to_inputs_masks: Dict[int, torch.Tensor] = \
        field(default_factory=dict)
    tensor_to_outputs_masks: Dict[int, torch.Tensor] = \
        field(default_factory=dict)
    tensor_to_inputs_weight_indices: Dict[int, SetLikeList] = \
        field(default_factory=dict)
    tensor_to_outputs_weight_indices: Dict[int, SetLikeList] = \
        field(default_factory=dict)
    tensor_to_input_activation_indices: Dict[int, SetLikeList] = \
        field(default_factory=dict)
    tensor_to_n_inputs_channels: Dict[int, SetLikeList] = \
        field(default_factory=dict)
    tensor_to_n_outputs_channels: Dict[int, SetLikeList] = \
        field(default_factory=dict)

    @property
    def tensor_to_inputs_weight_dropped(self) -> Dict[int, SetLikeList]:
        return invert_mapping_indices(
            self.tensor_to_inputs_weight_indices,
            self.tensor_to_n_inputs_channels)

    @property
    def tensor_to_outputs_weight_dropped(self) -> Dict[int, SetLikeList]:
        return invert_mapping_indices(
            self.tensor_to_outputs_weight_indices,
            self.tensor_to_n_outputs_channels)


def convert_names_to_ops(
    tensor_to_metadata: Dict[int, Metadata],
    pruned_inputs: List[Tuple],
    pruned_outputs: List[Tuple],
) -> Tuple[List, List]:
    """
    Convert parameter names to tensor ids.

    This allows users to specify pruned channels using more readable parameter
    names like 'conv1.weight' instead of unreadable and abstract tensor ids.
    """
    if (
        not (any(isinstance(name, str) for name, _ in pruned_inputs)
             or any(isinstance(name, str) for name, _ in pruned_outputs))
    ):
        return pruned_inputs, pruned_outputs

    param_to_tensor = {}
    for tensor_id, metadata in tensor_to_metadata.items():
        for param in metadata.non_tracers:
            if metadata := getattr(param, '_metadata', {}):
                param_to_tensor[metadata['name']] = tensor_id

    try:
        pruned_inputs = [
            (param_to_tensor[name] if isinstance(name, str) else name, channel)
            for name, channel in pruned_inputs]  # TODO: change back to set
        pruned_outputs = [
            (param_to_tensor[name] if isinstance(name, str) else name, channel)
            for name, channel in pruned_outputs]  # TODO: change back to set
    except KeyError as e:
        raise UserWarning(
            f"No such parameter named {e}. Must be one of: "
            f"{list(param_to_tensor.keys())[:30]}..."
        )
    return pruned_inputs, pruned_outputs


def generate_pruning_spec(
    y: torch.Tensor,
    pruned_outputs: Optional[List[Tuple]] = None,
    pruned_inputs: Optional[List[Tuple]] = None,
    constrained: bool = False,
    reorder_producer: bool = True,
    reorder_consumer: bool = True,
    net: nn.Module = None,
) -> PruningSpec:
    """
    Generate pruning masks from a pruning pattern by tracing channel
    modifications in the network.

    Args:
        net nn.Module: pytorch network to generate input masks for
        input_shape Tuple: shape of input tensor
        pruned_outputs List[Tuple]: list of (source tensor id, channel index)
            indicating pruned channels
        constrained bool: If true, grow set of pruned channels so that all
            masks are "equal". This avoids need for memory copies at inference
            time. If false, stay faithful to provided pruning specification,
            but use memory copies to do so.

    Returns:
        tensor_to_input_mask Dict[int, torch.Tensor]: Mapping from tensor id
            to a 0-1 mask tensor of shape (1, Cin, 1, 1)
        tensor_to_output_mask Dict[int, torch.Tensor]: Mapping from tensor id
            to a 0-1 mask tensor of shape (1, Cout, 1, 1)

    >>> from .tracing import trace
    >>> import torch.nn as nn
    >>> class Net(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv1 = nn.Conv2d(3, 10, 3)
    ...         self.conv2 = nn.Conv2d(10, 5, 3)
    ...     def forward(self, x):
    ...         x = torch.cat([x[:, -2:], x[:, :1]], dim=1)
    ...         x = self.conv1(x)
    ...         x = torch.cat([x[:, -5:], x[:, :5]], dim=1)
    ...         x = self.conv2(x)
    ...         return x
    ...
    >>> net = Net()
    >>> x = torch.rand((1, 3, 8, 8))
    >>> traced1 = trace(net, [x])
    >>> spec1 = generate_pruning_spec(
    ...     traced1, pruned_outputs=[('conv1.weight', 3)], constrained=True,
    ...     net=net)
    >>> traced2 = trace(net, [x])
    >>> spec2 = generate_pruning_spec(
    ...     traced2, pruned_inputs=[('conv2.weight', 8)], constrained=True,
    ...     net=net)
    >>> spec1.tensor_to_inputs_weight_indices == \
        spec2.tensor_to_inputs_weight_indices
    True
    >>> keys = list(sorted(spec1.tensor_to_inputs_masks.keys()))
    >>> spec1.tensor_to_inputs_masks[5]  # pruned conv out, as input to slice
    [tensor([1, 1, 1, 0, 1, 1, 1, 1, 1, 1])]
    >>> spec1.tensor_to_inputs_masks[7]  # dual inputs to cat
    [tensor([1, 1, 1, 0, 1]), tensor([1, 1, 1, 0, 1])]
    >>> spec1.tensor_to_inputs_masks[8]
    [tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 1])]
    """
    pruned_inputs = pruned_inputs or []
    pruned_outputs = pruned_outputs or []
    producers, consumers = get_producers_consumers(y)

    pruned_inputs, pruned_outputs = convert_names_to_ops(
        y.tensor_to_metadata, pruned_inputs, pruned_outputs)
    for tensor_id, _ in pruned_outputs:
        assert tensor_id in producers, (
            f"Invalid layer for output pruning: {tensor_id}. Must be one of: "
            f" {sorted(producers)}"
        )
    for tensor_id, _ in pruned_inputs:
        assert tensor_id in consumers, (
            f"Invalid layer for input pruning: {tensor_id}. Must be one of: "
            f"{sorted(consumers)}"
        )

    tensor_to_n_inputs_channels = {tensor_id: [
        len(channels) for channels in op_channels]
        for tensor_id, op_channels in y.tensor_to_traces.items()
    }
    tensor_to_n_outputs_channels = {
        tensor_id: [metadata.num_output_channels]
        for tensor_id, metadata in y.tensor_to_metadata.items()
    }

    tensor_to_input_activation_indices = {}
    if constrained:
        # Using pruned channels, find ignored channels. Repeat til convergence.
        tensor_to_inputs_weight_indices, tensor_to_outputs_weight_indices = \
            generate_constrained_indices(
                y.tensor_to_traces, pruned_inputs, pruned_outputs,
                tensor_to_n_inputs_channels, tensor_to_n_outputs_channels)
    else:
        # Or, keep pruned channels. After pruning, compute subselection.
        tensor_to_inputs_weight_indices, tensor_to_outputs_weight_indices, \
            tensor_to_input_activation_indices = \
            generate_unconstrained_indices(
                y.tensor_to_traces, producers, consumers, pruned_inputs,
                pruned_outputs, tensor_to_n_inputs_channels,
                tensor_to_n_outputs_channels, reorder_producer,
                reorder_consumer)

    tensor_to_inputs_masks = mapping_indices_to_masks(
        tensor_to_inputs_weight_indices, tensor_to_n_inputs_channels)
    tensor_to_outputs_masks = mapping_indices_to_masks(
        tensor_to_outputs_weight_indices, tensor_to_n_outputs_channels)

    return PruningSpec(
        tensor_to_metadata=y.tensor_to_metadata,
        tensor_to_inputs_masks=tensor_to_inputs_masks,
        tensor_to_outputs_masks=tensor_to_outputs_masks,
        tensor_to_inputs_weight_indices=tensor_to_inputs_weight_indices,
        tensor_to_outputs_weight_indices=tensor_to_outputs_weight_indices,
        tensor_to_input_activation_indices=tensor_to_input_activation_indices,
        tensor_to_n_inputs_channels=tensor_to_n_inputs_channels,
        tensor_to_n_outputs_channels=tensor_to_n_outputs_channels,
    )


def generate_unconstrained_indices(
    tensor_to_traces: Dict[int, List],
    producers: set,
    consumers: set,
    pruned_inputs: List[Tuple],
    pruned_outputs: List[Tuple],
    tensor_to_n_inputs_channels: Dict[int, List],
    tensor_to_n_outputs_channels: Dict[int, List],
    reorder_producer: bool = True,
    reorder_consumer: bool = True
) -> Tuple[Dict, Dict, Dict]:
    """
    Conservative variant, which remains fully faithful to the original set of
    pruned channels.

    NOTE: Does not currently support disagreeing output masks.

    Computes subselection indices needed for disagreeing input and output
    masks. Perform this subselection computation in 3 steps:

        1. Find all commonly-pruned channels. Only these are actually pruned.
        2. Restore all pruned producer channels mixed with a retained producer
           channel.
        3. Compute subselection indices, per consumer. Note these are indexed
           *after pruning.

    Step 2 above fixes the following scenario, where a commonly-pruned channel
    is mixed with a retained channel.

        y1 = x + x[:, ::-1]
        y2 = x

    In this example, we prune `y1[(0,2)]` and `y2[0]`, meaning only `x[0]` is
    commonly-pruned. However, this creates a problem for `y1[(0,2)]`, which is
    `x[0] (pruned) + x[2] (kept)`. Step 2 fixes this by retaining `x[0]`, only
    because it was mixed with the retained `x[2]` channel.

    For the variable names below:

        p = producer
        c = consumer
        u = user - any other layer in between a producer and consumer

    Args:
        tensor_to_traces: Mapping from tensor id to input tensor_to_traces
        pruned_inputs: SetLikeList of (tensor_id, channel id) tuples
        pruned_outputs: SetLikeList of (tensor_id, channel id) tuples

    >>> from .trace import trace_from_n_channels, union_traces
    >>> trace = lambda tensor_id: trace_from_n_channels(4, tensor_id)
    >>> tensor_to_traces = {
    ...     1: [trace(0)], 2: [trace(1)], 3: [trace(1)], 4: [trace(3)],
    ...     5: [trace(4)], 6: [trace(2)]
    ... }
    >>> outs = {i: [4] for i in range(6)}
    >>> ins = {i: [4] for i in range(1, 7)}
    >>> producers, consumers = {1, 2, 3, 4}, {2, 3, 4}
    >>> in_wgt_idx, out_wgt_idx, in_act_idx = generate_unconstrained_indices(
    ...     tensor_to_traces, producers, consumers, [(2, 0), (3, 1)], [], ins,
    ...     outs, reorder_producer=False, reorder_consumer=False
    ... )  # no prune + force no reorder
    >>> out_wgt_idx
    {}
    >>> in_act_idx
    {2: range(1, 4), 3: [0, 2, 3]}
    >>> in_wgt_idx, out_wgt_idx, in_act_idx = generate_unconstrained_indices(
    ...     tensor_to_traces, producers, consumers, [(2, 0), (3, 1)], [], ins,
    ...     outs)  # no prune + reorder
    >>> out_wgt_idx[1][0]
    [1, 2, 3, 0]
    >>> in_wgt_idx[3][0]
    [2, 3, 0]
    >>> in_act_idx
    {2: range(0, 3), 3: range(1, 4)}
    >>> in_wgt_idx, out_wgt_idx, in_act_idx = generate_unconstrained_indices(
    ...     tensor_to_traces, producers, consumers, [(2, 0), (3, 0), (3, 2)],
    ...     [], ins, outs)  # should prune (1, 0), subselect out (1, 2)
    >>> out_wgt_idx[1][0]
    [1, 3, 2]
    >>> in_act_idx
    {2: range(0, 3), 3: range(0, 2)}
    >>> in_wgt_idx, out_wgt_idx, in_act_idx = generate_unconstrained_indices(
    ...     tensor_to_traces, producers, consumers,
    ...     [(2, 3), (2, 2), (3, 3), (3, 2)], [], ins, outs
    ... )  # should prune (1, 3) AND (1, 2)
    >>> out_wgt_idx[1][0]
    [0, 1]
    >>> in_act_idx
    {2: range(0, 2), 3: range(0, 2)}
    >>> tensor_to_traces[3] = [trace(1)]
    >>> tensor_to_traces[2] = [union_traces([
    ...     tensor_to_traces[2][0],
    ...     tensor_to_traces[2][0][::-1]])]  # step 2 edge case above (see y1)
    >>> in_wgt_idx, out_wgt_idx, in_act_idx = generate_unconstrained_indices(
    ...     tensor_to_traces, producers, consumers, [(2, 0), (3, 0)], [], ins,
    ...     outs)  # no prune (see prev line)
    >>> out_wgt_idx
    {}
    >>> in_act_idx  # 3 = 0, 2 = 1, so [1, 1, 0] = [1, 2, 3]
    {2: [0, 1, 1], 3: range(1, 4)}
    >>> tensor_to_traces[2] = [
    ...     trace_from_n_channels(4, 1) + trace_from_n_channels(4, 4)]
    >>> ins[2] = [len(trace) for trace in tensor_to_traces[2]]
    >>> in_wgt_idx, out_wgt_idx, in_act_idx = generate_unconstrained_indices(
    ...     tensor_to_traces, producers, consumers, [(2, 0), (3, 0)], [], ins,
    ...     outs)  # should prune (1, 0)
    >>> out_wgt_idx[1][0]
    [1, 2, 3]
    >>> in_act_idx
    {2: range(0, 7), 3: range(0, 3)}
    >>> in_wgt_idx, out_wgt_idx, in_act_idx = generate_unconstrained_indices(
    ...     tensor_to_traces, producers, consumers,
    ...     [(2, 0), (2, 1), (3, 0), (3, 2)], [], ins, outs
    ... )  # prune (1, 0) + reorder
    >>> in_wgt_idx[3][0]
    [1, 3]
    >>> out_wgt_idx[1][0]
    [1, 3, 2]
    >>> in_act_idx
    {2: range(1, 7), 3: range(0, 2)}
    >>> tensor_to_traces[2] = [union_traces([trace(1), trace(4)])]
    >>> ins[2] = [len(trace) for trace in tensor_to_traces[2]]
    >>> in_wgt_idx, out_wgt_idx, in_act_idx = generate_unconstrained_indices(
    ...     tensor_to_traces, producers, consumers,
    ...     [(2, 0), (3, 0), (3, 1)], [], ins, outs)
    >>> out_wgt_idx
    {1: [[2, 3, 1]], 4: [[2, 3, 1]]}
    >>> in_act_idx
    {2: range(0, 3), 3: range(0, 2)}
    >>> tensor_to_traces = {
    ...     1: [trace(0)], 2: [trace(1)], 3: [trace(1)], 4: [trace(3)],
    ...     5: [trace(1)], 6: [trace(5)]}
    >>> n_output_channels = {i: [3] for i in range(4)}
    >>> n_input_channels = {i: [3] for i in range(1, 5)}
    >>> producers, consumers = {1, 3, 5}, {3, 5}
    >>> in_wgt_idx, out_wgt_idx, in_act_idx = generate_unconstrained_indices(
    ...     tensor_to_traces, producers, consumers, [(3, 1), (5, 0)], [], ins,
    ...     outs)  # prune + reorder - ensure users are reordered
    >>> out_wgt_idx
    {1: [[0, 2, 3, 1]]}
    >>> in_wgt_idx
    {3: [[0, 2, 3]], 5: [[2, 3, 1]], 2: [[0, 2, 3, 1]]}
    >>> in_act_idx
    {3: range(0, 3), 5: range(1, 4)}
    """
    tensor_to_c_unused = channels_to_mapping(pruned_inputs)
    c_to_data = {
        c_id: Consumer(
            c_id,
            inputs_trace[0],
            tensor_to_c_unused.get(c_id, [[]])[0]  # consumers have one input
        ) for c_id, inputs_trace in tensor_to_traces.items()
    }

    pruned_inputs = set(pruned_inputs)
    pruned_outputs = set(pruned_outputs)
    c_to_input_activation_indices = {}  # used to subselect inputs
    p_to_reordering = invert_mapping_indices(
        channels_to_mapping(pruned_outputs), tensor_to_n_outputs_channels
    )  # used to reorder output weights
    c_to_reordering = invert_mapping_indices(
        channels_to_mapping(pruned_inputs), tensor_to_n_inputs_channels
    )  # used to reorder input weights
    for i, (producer_ids, consumer_ids, user_ids) in enumerate(get_segments(
            tensor_to_traces, producers, global_consumers=consumers)):
        if not consumer_ids:
            continue

        segment_consumers = [c_to_data[c_id] for c_id in consumer_ids]
        if not any(
            consumer.unused_sources(tensor_to_n_outputs_channels)
            for consumer in segment_consumers
        ):
            continue

        # 1. Find channels that *all consumers do *not use
        p_pruned_sources = set.intersection(*[
            c.unused_sources(tensor_to_n_outputs_channels)
            for c in segment_consumers
        ])

        # 2. Restore channels mixed with retained channels.
        p_pruned_sources -= set.union(*[
            c.mixed_sources(p_pruned_sources) for c in segment_consumers])
        pruned_outputs = pruned_outputs | p_pruned_sources

        # 3. Compute output weight indices for producers
        if reorder_producer:
            p_to_reordering.update(reorder_producer_weights(segment_consumers))

        # 4. Compute input activation indices for consumers
        p_to_mapping = get_reordering_mapping(
            producer_ids, tensor_to_n_outputs_channels, p_pruned_sources,
            p_to_reordering)
        c_to_input_activation_indices.update({
            c.id: c.get_input_activation_indices(
                p_pruned_sources, p_to_mapping) for c in segment_consumers
            if producers_trace(c.trace).intersection(p_to_reordering)
            or c.unused_inputs
        })

        # 5. Compute input weight indices for consumers
        if reorder_consumer:  # reorder weights so act channels 'in order'
            reorder_consumer_weights(
                consumer_ids, c_to_input_activation_indices, c_to_reordering)

        # 6. Compute input weight indices for all users
        for u_id, input_traces in get_traces_for(
            user_ids, tensor_to_traces
        ).items():
            c_to_reordering[u_id] = compute_user_weight_indices(
                input_traces, p_pruned_sources, p_to_mapping)

        segment_in_act_idx = [
            (c_id, c_to_input_activation_indices[c_id])
            for c_id in consumer_ids if c_id in c_to_input_activation_indices
        ]
        segment_in_wgt_idx = [
            (c_id, c_to_reordering[c_id])
            for c_id in consumer_ids if c_id in c_to_reordering
        ]
        segment_out_wgt_idx = [
            (p_id, p_to_reordering[p_id])
            for p_id in producer_ids if p_id in p_to_reordering
        ]
        logger.debug(f"[{i}] Producers: {producer_ids} . Consumers: "
                     f"{consumer_ids} . Users: {user_ids}")
        logger.debug(f"[{i}] Input activation indices: {segment_in_act_idx}")
        logger.debug(f"[{i}] Input weight indices: {segment_in_wgt_idx}")
        logger.debug(f"[{i}] Output weight indices: {segment_out_wgt_idx}")

    tensor_to_inputs_weight_indices = c_to_reordering
    tensor_to_outputs_weight_indices = {
        **invert_mapping_indices(
            channels_to_mapping(pruned_outputs), tensor_to_n_outputs_channels),
        **p_to_reordering
    }
    return (
        tensor_to_inputs_weight_indices,
        tensor_to_outputs_weight_indices,
        c_to_input_activation_indices
    )


def get_segments(
    tensor_to_traces: Dict[int, List],
    global_producers: set,
    global_consumers: set
) -> List[Tuple]:
    """
    Get all segments (a.k.a., 'closed' set of producers and corresponding
    consumers)

    To do, iterate through all valid producers. Then, iteratively find all
    consumers, then find all producers, then find all consumers... etc. Repeat
    until the list of producers and consumers converges.

    >>> from .trace import trace_from_n_channels
    >>> trace = lambda tensor_id: trace_from_n_channels(3, tensor_id)
    >>> tensor_to_traces = {
    ...     1: [trace(0)], 2: [trace(1)], 3: [trace(1)], 4: [trace(3)],
    ...     5: [trace(4)]}
    >>> get_segments(tensor_to_traces, {1, 3}, {2, 3, 4})
    [({1}, {2, 3}, set()), ({3}, {4}, set())]
    >>> get_segments(tensor_to_traces, {1, 3}, {3, 4})
    [({1}, {3}, {2}), ({3}, {4}, set())]
    >>> tensor_to_traces = {
    ...     1: [trace(0)], 2: [trace(0)], 3: [trace(1) + trace(2)],
    ...     4: [trace(3)], 5: [trace(4)]}
    >>> get_segments(tensor_to_traces, {1, 2, 3}, {3, 4})
    [({1, 2}, {3}, set()), ({3}, {4}, set())]
    """
    producer_to_more_producers = defaultdict(set)
    producer_to_consumers = defaultdict(set)
    producer_to_users = defaultdict(set)
    for tensor_id, producer, producers in for_all_producers(tensor_to_traces):
        producer_to_more_producers[producer] |= producers
        mapping = producer_to_consumers \
            if tensor_id in global_consumers else producer_to_users
        mapping[producer].add(tensor_id)

    segments = []
    seen_producers = set()
    for producer in global_producers:
        if producer in seen_producers:  # producer belongs to one segment
            continue

        # grow set of producers until all affected producers are included
        producers, n_producers = {producer}, 0
        while n_producers < len(producers):
            n_producers = len(producers)
            producers |= set().union(*[
                producer_to_more_producers[producer]
                for producer in producers])
        seen_producers |= producers

        # convert producer set to consumer and user sets
        consumers = set().union(*[
            producer_to_consumers[producer] for producer in producers])
        users = set().union(*[
            producer_to_users[producer] for producer in producers])
        segments.append((producers, consumers, users))
    return segments


def get_traces_for(
    tensor_ids: set,
    tensor_id_to_traces: Dict[int, List]
) -> Dict[int, List]:
    """Get dictionary of traces for provided layers"""
    return {
        tensor_id: tensor_id_to_traces[tensor_id] for tensor_id in tensor_ids
    }


def get_reordering_mapping(
    producer_ids: set,
    tensor_to_n_output_channels: Dict[int, List],
    p_pruned_sources: Tuple[set],
    p_to_reordering: Dict[int, List]
) -> Dict[int, Dict]:
    """Get mapping from old channel indices to new, provided indices.

    Provided indices in current usage are pruned and reordered, so this mapping
    allows us to construct the new trace from an old trace, assuming the new
    pruning and reordering is applied.
    """
    p_to_ordering = {p_id: [
        [i for i in range(tensor_to_n_output_channels[p_id][0])
         if (p_id, i) not in p_pruned_sources]]
        for p_id in producer_ids
    }
    p_to_mapping = {  # producers only have one output
        p_id: dict(zip(p_to_ordering[p_id][0], p_to_reordering[p_id][0]))
        if p_id in p_to_reordering else {i: i for i in p_to_ordering[p_id][0]}
        for p_id in p_to_ordering
    }
    return p_to_mapping


def compute_user_weight_indices(
    input_traces: List,
    p_pruned_sources: set,
    p_to_mapping: Dict[int, Dict]
) -> List:
    """
    Compute input weight indices for all users, for a particular set of
    producers.
    """
    inputs_indices = []
    for input_trace in input_traces:
        new_input_trace = reorder(
            prune_trace(input_trace, p_pruned_sources),
            p_to_mapping
        )  # recompute trace after pruning
        inputs_indices.append([
            input_trace.index(channel_trace)
            for channel_trace in new_input_trace
        ])
    return inputs_indices


def generate_constrained_indices(
    tensor_to_traces: Dict[int, List],
    pruned_inputs: List[Tuple],
    pruned_outputs: List[Tuple],
    tensor_to_n_inputs_channels: Dict[int, List],
    tensor_to_n_outputs_channels: Dict[int, List]
) -> Tuple[Dict, Dict]:
    """
    Greedy variant of pruned-channel computation. Grows the set of pruned
    channels to effectively make masks "equal".

    Given initially pruned inputs and outputs, compute which other channels
    these pruned channels interact with. Grow the set of pruned channels until
    it is "closed", and no non-pruned channels interact with pruned channels.

    >>> from .trace import trace_from_n_channels
    >>> trace = lambda tensor_id: trace_from_n_channels(3, tensor_id)
    >>> tensor_to_traces = {
    ...     1: [trace(0)], 2: [trace(1)], 3: [trace(1)], 4: [trace(3)],
    ...     5: [trace(4)], 6: [trace(2)]}
    >>> n_outputs_channels = {i: [3] for i in range(6)}
    >>> n_inputs_channels = {i: [3] for i in range(1, 7)}
    >>> generate_constrained_indices(
    ...     tensor_to_traces, [(3, 1)], [], n_inputs_channels,
    ...     n_outputs_channels)
    ({3: [[0, 2]], 2: [[0, 2]]}, {1: [[0, 2]]})
    """
    pruned_output_channels = set(pruned_outputs)
    pruned_input_channels = set(pruned_inputs)
    n_pruned_channels = 0

    input_channels_to_sources = defaultdict(set)
    source_to_more_sources = defaultdict(set)
    for input_channel, source, channel_trace in for_all_sources(
            tensor_to_traces):
        source_to_more_sources[source] |= channel_trace
        if input_channel in pruned_input_channels:  # run only on pruned chs
            input_channels_to_sources[input_channel] |= channel_trace

    # grow list of pruned sources until all affected outputs are included
    pruned_output_channels |= set().union(*[
        input_channels_to_sources[pruned_input]
        for pruned_input in pruned_inputs])
    # if list of masked output channels has grown, keep running
    while len(pruned_output_channels) > n_pruned_channels:
        n_pruned_channels = len(pruned_output_channels)
        pruned_output_channels |= set().union(*[
            source_to_more_sources[source]
            for source in pruned_output_channels])

    source_to_input_channels = defaultdict(set)
    for input_channel, source, channel_trace in (
        for_all_sources(tensor_to_traces)
    ):
        if source in pruned_output_channels:  # run only on pruned sources
            source_to_input_channels[source].add(input_channel)

    # convert pruned sources into pruned consumer input channels
    pruned_input_channels = set().union(*[
        source_to_input_channels[source] for source in pruned_output_channels])

    tensor_to_inputs_weight_indices = invert_mapping_indices(
        channels_to_mapping(pruned_input_channels),
        tensor_to_n_inputs_channels)
    tensor_to_outputs_weight_indices = invert_mapping_indices(
        channels_to_mapping(pruned_output_channels),
        tensor_to_n_outputs_channels)
    return tensor_to_inputs_weight_indices, tensor_to_outputs_weight_indices


def mapping_indices_to_masks(
    mapping: Dict, tensor_to_n: Dict[int, int]
) -> Dict[int, List]:
    """Convert mapping of (op idx -> [channel indices]) to (op idx -> mask)"""
    return {
        tensor_id: [
            torch.tensor([int(i in indices) for i in range(n)])
            for indices, n in zip(op_indices, tensor_to_n[tensor_id])]
        for tensor_id, op_indices in mapping.items()
    }


def get_producers_consumers(y: Tracer) -> Tuple[set, set]:
    """Get all producers and consumers in the graph.

    Producers are any trace-reset layer that do not produce the final output.
    Consumers are any trace-reset layer that does not take the network input as
    input.
    """
    trace_reset_ops = {
        tensor_id for tensor_id, metadata in y.tensor_to_metadata.items()
        if metadata.is_trace_reset}
    producers = trace_reset_ops - producers_trace(y.tensor_trace)
    input_ops = {
        tensor_id for tensor_id, metadata in y.tensor_to_metadata.items()
        if metadata.is_input}
    non_consumers = {
        tensor_id for tensor_id, traces in y.tensor_to_traces.items()
        if any(input_ops.intersection(producers_trace(trace))
               for trace in traces)}  # consumers taking in input
    consumers = trace_reset_ops - non_consumers
    return producers, consumers
