"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

Data structures for keeping track of channel usage.

For consistency, we define several terms below to be used throughout the
pruning logic:

    Source: A tuple of (tensor id, channel id). If present, this 'source' value
        was used to compute the current value. True to the parent data
        structure, a Source is immutable.

    Trace:  A data structure that fully describes, for a tensor, which sources
        were used to compute each channel.

To read this file, start from `Trace`. Ideally, however, start from `Trace`
usage in `./tracing.py`.

Trace is a data structure that identifies, for each output channel, which input
channels contributed to it. This is itemized as a list of set of tuples, where
each tuple represents (source tensor id, source channel id). Each set
represents an output channel. At any point in the model, check this trace to
see which source channels were used for any output channel.

>>> t = trace_from_n_channels(3, 0)
>>> t
[{(0, 0)}, {(0, 1)}, {(0, 2)}]
>>> y = t[::-1] + t
>>> y[0]
{(0, 2)}
>>> (0, 2) in y[0]
True
>>> (0, 2) in y[2]
False
>>> sum([t, t], [])
[{(0, 0)}, {(0, 1)}, {(0, 2)}, {(0, 0)}, {(0, 1)}, {(0, 2)}]
>>> union_traces([t, t[::-1]])
[{(0, 2), (0, 0)}, {(0, 1)}, {(0, 2), (0, 0)}]
>>> flatten_trace(t)
[{(0, 1), (0, 2), (0, 0)}]
"""

import copy
import functools
import itertools
from dataclasses import dataclass
from typing import Dict, List, Union

from .utils import SetLikeList, invert


Source = tuple  # tuple of (tensor_id, channel_id)
Sources = set  # arbitrary collection of sources (e.g., set of pruned sources)
ChannelSources = list  # collection of sources that represent a channel
Trace = list


def channels_for_tensor_id(
    sources: Union[Sources, ChannelSources],
    tensor_id: int
) -> List:
    """Return all channels for the provided operation."""
    return [source[1] for source in sources if source[0] == tensor_id]


def no_pruned_sources(
    sources: Union[Sources, ChannelSources],
    pruned_sources: Union[Sources, ChannelSources]
) -> bool:
    """
    Check if any of the provided pruned sources are contained in this sequence
    of sources
    """
    return not any(source in sources for source in pruned_sources)


def union_sources(*sourcess: Union[Sources, ChannelSources]) -> Sources:
    """Union collections of sources together.

    >>> union_sources(Sources([Source((0, 1))]), Sources([Source((0, 2))]))
    {(0, 1), (0, 2)}
    """
    return Sources(itertools.chain(*sourcess))


def reorder(trace: Trace, tensor_id_to_reordering: Dict[int, List]) -> Trace:
    """
    Apply reordering mapping to this trace, dropping any missing ops or
    channels.

    Used to reorder user traces and consumer input traces. Use `reorder` when
    pruning and reordering simultaneously. In which case, missing channels are
    assumed to be pruned and are dropped from the trace.
    """
    reordered_trace = Trace()
    for channel in trace:
        reordered_channel = Sources()
        for source in channel:
            tensor_id, channel_id = source
            if channel_id in tensor_id_to_reordering.get(tensor_id, ()):
                reordered_channel.add(Source((
                    tensor_id,
                    tensor_id_to_reordering[tensor_id][channel_id])
                ))
        if reordered_channel:
            reordered_trace.append(reordered_channel)
    return reordered_trace


def reorder_safe(
    trace: Trace,
    source_to_source: Dict[Source, Source]
) -> Trace:  # TODO: combine with reorder
    """
    Apply reordering to this trace, preserving any sources not included.

    Used to reorder consumer input traces when reordering weights. Use this
    function when reordering without pruning simulanteously.
    """
    new_channels = Trace()
    for channel in trace:
        new_sources = Sources()
        for source in channel:
            new_sources.add(Source(source_to_source.get(source, source)))
        new_channels.append(new_sources)
    return new_channels


def memoize(f):
    # TODO: subclass list, or memoize this way? memoization may blow up memory
    cache = {}

    @functools.wraps(f)
    def decorator(trace):
        trace_id = hash(str(trace))  # NOTE: id -> too many collisions
        if trace_id not in cache:
            cache[trace_id] = f(trace)
        return cache[trace_id]
    return decorator


@memoize
def producers_trace(trace):
    producers = [producers_channel(channel_trace) for channel_trace in trace]
    return set().union(*producers)


def producers_channel(channel_sources):
    return set(source[0] for source in channel_sources)


def prune_trace(trace: Trace, pruned_sources: Sources):
    """
    Return a pruned version of the trace, where the provided sources are
    ommitted.
    """
    return Trace(channel for channel in trace
                 if no_pruned_sources(channel, pruned_sources))


def has_multiple_sources_per_channel(trace: Trace) -> bool:
    """Return if any channel has multiple sources"""
    return any(len(trace) > 1 for trace in trace)


def trace_from_n_channels(n_channels: int, tensor_id: int) -> Trace:
    """Create a trace for n different channels."""
    return Trace(Sources([Source((tensor_id, i))]) for i in range(n_channels))


def union_traces(traces: List[Trace]) -> Trace:
    """
    Take the union of multiple traces, preserving the location of each channel.
    """
    assert all(len(trace) == len(traces[0]) for trace in traces), \
        "Taking union of non-uniformly sized traces"
    return Trace(union_sources(*channel) for channel in zip(*traces))


def flatten_trace(trace: Trace) -> Trace:
    """
    Flatten the entire trace's input channels into just one output channel.

    >>> trace = Trace([Sources([Source((0, 1))]), Sources([Source((0, 2))])])
    >>> trace
    [{(0, 1)}, {(0, 2)}]
    >>> flatten_trace(trace)
    [{(0, 1), (0, 2)}]
    """
    return Trace([union_sources(*trace)])


@dataclass
class Consumer:
    """
    Data structure for a consumer.

    Includes operations that abstract away details of a Source object.
    """

    id: int
    trace: SetLikeList
    unused_inputs: SetLikeList

    @property
    def n_inputs(self) -> int:
        """Number of inputs for this consumer."""
        return len(self.trace)

    @property
    def used_inputs(self) -> List[int]:
        """Used input channels for this consumer."""
        return invert(self.unused_inputs, self.n_inputs)

    def used_sources(
        self,
        producer_ids: Union[set, None] = None
    ) -> ChannelSources:
        """Compute used producer channels. Retain ordering.

        >>> trace = trace_from_n_channels(3, 0)
        >>> c = Consumer(1, trace_from_n_channels(3, 0), [1])
        >>> c.used_sources()
        [(0, 0), (0, 2)]
        """
        producer_ids = producer_ids or set()
        used_sources: List = ChannelSources()
        for p_id in producer_ids or producers_trace(self.trace):
            p_used_ch = list()
            for c_ch_id in self.used_inputs:  # for ea consumer channel
                ch = channels_for_tensor_id(self.trace[c_ch_id], p_id)
                p_used_ch.extend(ch)  # skip if pruned
            used_sources.extend(
                (p_id, channel) for channel in p_used_ch)  # grab used sources
        return used_sources

    def unused_sources(
        self,
        tensor_to_n_outputs_channels: Dict[int, List],
        producer_ids: Union[set, None] = None
    ):
        """Compute unused producer channels.

        Note that we first accumulate all *used producer channels, then look at
        which producer channels are completely unused. This is important: We
        can't instead accumulate all pruned producer channels, because a
        producer channel may be used in *both pruned and not-pruned consumer
        channels.

        For example, say x is 3-dim, `y = x + x[:, ::-1]`, and `y[0]` is
        pruned. This would imply `x[(0,2)]` should be pruned. However, `y[2]`
        still uses `x[(0,2)]`, which means we should *not prune `x[(0,2)]`.

        >>> trace = trace_from_n_channels(3, 0)
        >>> c = Consumer(1, trace_from_n_channels(3, 0), [1, 2])
        >>> c.unused_sources({0: [3]})
        {(0, 1), (0, 2)}
        """
        producer_ids = producer_ids or set()
        unused_sources = Sources()
        for p_id in producer_ids or producers_trace(self.trace):
            # get all p channels used by this c channel
            sources = self.used_sources({p_id})
            p_used_channels = set(channels_for_tensor_id(sources, p_id))
            # compute unused channels
            n_output_channels = tensor_to_n_outputs_channels[p_id][0]
            p_unused_channels = set(range(n_output_channels)) - p_used_channels
            unused_sources |= {
                Source((p_id, channel)) for channel in p_unused_channels
            }  # assemble unused sources
        return unused_sources

    def mixed_sources(self, sources: set) -> set:
        """Get all sources that are mixed into a channel with one of the
        provided sources.

        >>> Consumer(1, [{(0, 0), (0, 1)}, {(0, 0), (0, 2)}, {(0, 3)}], []) \
                .mixed_sources({(0, 0)})
        {(0, 1), (0, 2), (0, 0)}
        """
        mixed_sources = set()
        for trace in self.trace:
            is_pruned = [source in sources for source in trace]
            if all(is_pruned):
                continue
            if not all([not p for p in is_pruned]):  # if any, retain all
                mixed_sources |= set(trace)
        return mixed_sources

    def get_input_activation_indices(
        self,
        p_pruned_sources: set,
        p_to_mapping: Dict[int, List]
    ) -> Union[List, range]:
        """
        Get indices for input activations, given the list of pruned sources and
        a mapping for channel reordering

        >>> Consumer(1, trace_from_n_channels(3, 0), []) \
                .get_input_activation_indices({(0, 0)}, {0: {1: 2, 2: 1}})
        [1, 0]
        """
        trace = copy.deepcopy(self.trace)
        input_trace_pruned = prune_trace(trace, p_pruned_sources)
        input_trace_reordered = reorder(input_trace_pruned, p_to_mapping)
        subselection = [
            input_trace_reordered.index(trace)
            for input_channel_id, trace in enumerate(self.trace)
            if input_channel_id in self.used_inputs
            and trace in input_trace_reordered
        ]

        assert len(subselection) > 0, 'Subselection mask includes no channels.'
        candidate = range(min(subselection), max(subselection) + 1)
        if subselection == list(candidate):
            return candidate
        return subselection


def for_all_sources(tensor_to_traces: Dict[int, List]):
    """
    Return generator over all sources in provided trace mapping.

    Every source is yielded with the corresponding input channel that uses it,
    and all other sources involved in that channel.
    """
    for tensor_id, input_traces in tensor_to_traces.items():
        for channels_trace in input_traces:
            for i, channel_trace in enumerate(channels_trace):
                for source in channel_trace:
                    yield (tensor_id, i), source, channel_trace


def for_all_producers(tensor_to_traces: Dict[int, List]):
    """
    Return generator over all producers in provided trace mapping.

    Every producer is yielded with the corresponding user's tensor id and as
    well as all the other producers used by that same operation.
    """
    for tensor_id, input_traces in tensor_to_traces.items():
        for input_trace in input_traces:
            producers = producers_trace(input_trace)
            for producer in producers:
                yield tensor_id, producer, producers
