"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

This file contains logic for reordering channels, to minimize amount of memory
copying needed during inference.

Our objective, informally, is to pick an ordering of channels, such that we
minimize inference-time memory copies. These copies occur if a consumer's
desired channels are found in disparate parts of the consumer's input. However,
we should instead think of this as ordering consumers instead, which will in
turn admit a valid channel ordering. This is effectively a graph problem:

  - Each node is a group of channels -- each group represents the channels that
    a consumer wants.
  - Node A has a directed edge to node B if A's last producer's channels are
    shared with B's first producer. Note that if all producer n's channels in
    A are shared, *then, producer n-1 can share channels, for the edge to
    exist. This is because reordering can only occur within a single producer's
    channels.

If we can find a path through this graph, without cycles, then we have a valid
ordering of channels that effects 0 memory copies. Odds are, such a path won't
always exist. As a result, we define an objective to optimize for, in the
absence of a 0-memory-copy solution.

Our objective, informally, is to maximize the number of channels our acyclic
path includes. This way, we minimize the number of channels that need to be
copied at inference time. To effect this:

  - Each node has a positive cost -- the number of channels in that node.
  - Each edge has a negative cost -- the number of channels shared by adjacent
    nodes.
  - The sum of a path's cost is the number of channels involved in that path.

Under this formulation, our objective is formally to find the maximum cost path
without cycles. This path then admits a channel ordering.

Notes:

  - We additionally look for all nodes that are subsets of other nodes. We dub
    these "sub-nodes". We simply rerun the algorithm on all sub-nodes for a
    single parent node. Knowing this, we can assume no nodes are subsets of one
    another (eliminates trivial cycles)
  - Note that the ordering is dependent on the consumer's ordering of input
    channels, *not just the producer's ordering of output channels. Our goal is
    to define a producer channel reordering such that each consumer's input is
    simply a view or subset of its original unpruned input. This is
    challenging, because each consumer may see a different ordering of producer
    output channels. *However, we simply ignore this and assume that producer
    channel orderings will match consumer channel orderings. This is true in
    most cases and simplifies the algorithm. Say we want to support this edge
    case in the future. To support this edge case, update the edges as you add
    nodes to the path. You would need to run this repeatedly, once per source
    node, so the algorithm runs in O(c^2).

To read this file, start from `reorder_{consumer,producer}_weights`. The latter
uses the aforementioned graph algorithm.
"""

from dataclasses import dataclass
from collections import defaultdict
from functools import total_ordering
from typing import Dict, List, Tuple, Union

import numpy as np

from .trace import Consumer, has_multiple_sources_per_channel, \
    producers_channel, reorder_safe
from .utils import SetLikeList, channels_to_mapping


def reorder_consumer_weights(
    consumer_ids: set,
    c_to_input_activation_indices: Dict[int, Union[List, range]],
    c_to_reordering: Dict[int, List]
):
    """
    Reorder consumer weights and activation indices, so that more activation
    indices are just 0-cost views, in-place

    >>> import torch
    >>> input_indices = [2, 0, 1]
    >>> c_to_inputs = {1: input_indices, 2: [0, 2]}
    >>> c_to_weights = {1: [[0, 1, 2]], 2: [[0, 1]]}
    >>> reorder_consumer_weights({1}, c_to_inputs, c_to_weights)
    >>> c_to_inputs
    {1: range(0, 3), 2: [0, 2]}
    >>> c_to_weights
    {1: [[1, 2, 0]], 2: [[0, 1]]}
    >>> _ = torch.manual_seed(0)
    >>> x, y = torch.rand((3)), torch.rand((3))
    >>> (x[input_indices] * y).sum() == (x * y[c_to_weights[1][0]]).sum()
    tensor(True)
    """
    for c_id in consumer_ids:
        subselection = c_to_input_activation_indices.get(c_id)
        if not subselection or isinstance(subselection, range):
            continue
        indices = c_to_reordering[c_id][0] if c_id in c_to_reordering else \
            list(range(len(subselection)))
        assert len(subselection) == len(indices), (
            f"# input channels {len(subselection)} should match # weight "
            f"channels {len(indices)}"
        )
        c_to_reordering[c_id] = [[
            indices[i] for i in list(np.argsort(subselection))]]
        c_to_input_activation_indices[c_id] = list(sorted(subselection))

        start, end = min(subselection), max(subselection)
        # if subselection is range(start, end) except jumbled (if in order
        # would prev been converted to range)
        if len(subselection) == end - start + 1:
            c_to_input_activation_indices[c_id] = range(start, end + 1)


def reorder_producer_weights(consumers: List[Consumer]) -> Dict[int, List]:
    """Computes weight index reordering for all producers.

    Args:
        consumers: List of `resolve.Consumer` objects representing consumers
                   and their used channels

    Returns:
        mapping from producer id to weight indices
    """
    consumer_ids = [c.id for c in consumers]
    p_to_reordering = {}

    # 3.a. If possible, reduce to single-producer case, per consumer.
    multi_producer_case = any(
        has_multiple_sources_per_channel(consumer.trace)
        for consumer in consumers)
    if multi_producer_case:
        equivalence = get_equivalent_sources(consumers)
        if not equivalence:
            # If un-reduce-able sources exist, no better reordering
            return {}

        new_to_olds, old_to_new = equivalence
        consumers = [
            Consumer(c.id, reorder_safe(c.trace, old_to_new), c.unused_inputs)
            for c in consumers]

    # 3.b. Compute reordering for all involved producers.
    id_to_p_used_sources = {c.id: c.used_sources() for c in consumers}
    sources_ordering = compute_reordering(consumer_ids, id_to_p_used_sources)
    p_to_reordering.update(channels_to_mapping(sources_ordering))

    # 3.c. Compute orderings for other producers (removed by reducing problem)
    if multi_producer_case:
        sources = sources_ordering[:]
        for source in sources_ordering:
            sources.extend(new_to_olds.get(source, []))
        p_to_reordering.update(channels_to_mapping(sources))
    return p_to_reordering


def get_equivalent_sources(consumers: List[Consumer]) -> Tuple[Dict, Dict]:
    """
    Find sets of equivalence classes, for sources.

    For example, say source (1, 1) is mixed with only (2, 1). Then, we can say
    (1, 1) is equivalent to (2, 1) because one is present in a trace iff the
    other is. The below computes these equivalence classes, so that
    multi-source channels can be reduced to single-source channels.

    Note if there are multiple channels from one producer, in one consumer
    channel, we abort the reordering. This is because multiple producer
    channels are unlikely to be improved.

    >>> from .trace import trace_from_n_channels, Consumer, union_traces
    >>> trace = lambda tensor_id: trace_from_n_channels(2, tensor_id)
    >>> consumer1 = Consumer(2, union_traces([trace(0), trace(1)]), {})
    >>> get_equivalent_sources([consumer1])[1]
    {(1, 0): (0, 0), (1, 1): (0, 1)}
    >>> consumer2 = Consumer(4, union_traces([trace(0), trace(2)]), {})
    >>> get_equivalent_sources([consumer1, consumer2])[1]
    {(1, 0): (0, 0), (2, 0): (0, 0), (1, 1): (0, 1), (2, 1): (0, 1)}
    """
    mapping = defaultdict(set)
    for consumer in consumers:
        for channel_trace in consumer.trace:
            tensor_id = min(producers_channel(channel_trace))

            new_source = None
            old_sources = set()
            for source in channel_trace:
                if source[0] == tensor_id:
                    if new_source is not None:
                        return None
                    new_source = source
                else:
                    old_sources.add(source)
                    if source in mapping:
                        old_sources |= mapping.pop(source)
            mapping[new_source] |= old_sources

    # reverse, from current to new source
    reverse_mapping = {}
    for new_source, old_sources in mapping.items():
        for old_source in old_sources:
            reverse_mapping[old_source] = new_source
    return dict(mapping), reverse_mapping


def compute_reordering(
    nodes: List[int],
    node_to_members: Dict[int, List]
) -> List[Tuple]:
    """
    Computes channel reordering to minimize amount of memory copying needed.
    Runs in O(c^2), where c is the number of consumers.

    Args:
        nodes List[int]: generic set of nodes
        node_to_members Dict[int, List[int]]: generic mapping from nodes to
            members

    This function is not aware of traces, networks etc. However, the variable
    names are named according to the original pruning application, to make this
    code readable.

    We run the algorithm in 4 steps:

        0. Use consumer ids as node ids. Each node and consumer corresponds to
           a set of channels.
        1. Find all sub-nodes. Subproblems we will run recursively on.
        2. Construct adjacency (and associated cost) matrix.
        3. From each node, find the maximum cost, acyclic path using dfs.
        4. Construct the channel ordering from the path.

    >>> compute_reordering(
    ...     {1, 2, 3}, {1: {0, 1}, 2: {1, 2}, 3: {2, 0, 4}}
    ... )  # pick biggest chunk (3)
    [1, 0, 2, 4]
    >>> compute_reordering(
    ...     {1, 2, 3, 4}, {1: {0, 1, 2}, 2: {0, 2}, 3: {2, 1}, 4: {3}}
    ... )  # force 1 order (using 2, 3)
    [0, 2, 1, 3]
    >>> compute_reordering(
    ...     {2, 3}, {2: {1, 2}, 3: {1}}
    ... )  # strict subset (does not cover all of subset)
    [1, 2]
    >>> compute_reordering(
    ...     {1, 2, 3, 4}, {1: {0,}, 2: {0, 1,}, 3: {0, 1, 2}, 4: {0, 1, 2}}
    ... )  # order should not change
    [0, 1, 2]
    >>> compute_reordering({2, 3}, {2: [(1, 2), (1, 3), (4, 0)], 3: [(1, 1)]})
    [(1, 2), (1, 3), (4, 0), (1, 1)]
    """
    # 1. Find all sub-nodes.  # TODO: figure out subset optimality
    parent_to_children = group_child_nodes(nodes, node_to_members)
    node_to_ordered_members = compute_child_node_reordering(
        parent_to_children, node_to_members)
    nodes = parent_to_children.keys()

    # 2. Construct adjacency and cost matrices.
    adj, cost = build_adjacency_cost(nodes, node_to_members)

    # 3. Iteratively find paths until entire graph is covered
    paths = max_cost_path(nodes, adj, cost)

    # 4. Build channel order from path
    channels = build_channel_order_from_paths(
        paths, node_to_members, node_to_ordered_members)

    return channels


def group_child_nodes(
    nodes: set,
    node_to_members: Dict[int, set]
) -> Dict[int, set]:
    """
    Cluster all child nodes by parent, in a mapping from parent to child nodes.

      1. For every node, find all other nodes that are subsets of it.
      2. Mark all such nodes as 'child nodes' for this 'parent node'.
      3. If any child node is itself a parent node, absorb all its child nodes.

    >>> group_child_nodes([1, 2, 3], {1: {1, 2}, 2: {1}, 3: {1, 2, 3}})
    {3: {1, 2}}
    >>> group_child_nodes({1, 2, 3}, {1: {0, 1}, 2: {1, 2}, 3: {2, 0, 4}})
    {1: set(), 2: set(), 3: set()}
    >>> group_child_nodes(
    ...     {1, 2, 3, 4},
    ...     {1: {0, 1}, 2: {0, 1, 2, 3}, 3: {0, 1, 2, 3, 4, 5},
    ...      4: {0, 1, 2, 3, 4, 5}})
    {3: {1, 2, 4}}
    """
    node_to_unordered = {
        node: set(members) for node, members in node_to_members.items()}

    parent_to_children = {parent: set() for parent in nodes}
    buffer = set(nodes)
    while buffer:
        parent = buffer.pop()
        for child in nodes:
            if (
                node_to_unordered[child].issubset(node_to_unordered[parent])
                and child != parent
            ):  # 1. is subset?
                buffer -= {child}  # safe version of .remove
                parent_to_children[parent].add(child)  # 2. mark as child node
                parent_to_children[parent] |= parent_to_children.pop(
                    child, set())  # 3. absorb child's children
    return dict(parent_to_children)


def compute_child_node_reordering(
    parent_to_children: Dict[int, set],
    node_to_members: Dict[int, set]
) -> Dict[int, List]:
    """
    Run `compute_reordering` on every set of child nodes.

    If a parent does not have children, simply sort its members. If a parent
    does, run `compute_reordering` on the child nodes, and use that to
    determine ordering for that node.
    """
    id_to_ordered_channels = {}
    for parent, children in parent_to_children.items():
        if not children:
            # can be in any order
            id_to_ordered_channels[parent] = sorted(node_to_members[parent])
            continue
        channels = compute_reordering(children, node_to_members)
        extend_list_as_set(channels, node_to_members[parent])
        id_to_ordered_channels[parent] = channels
    return id_to_ordered_channels


def extend_list_as_set(lst: List, lst_or_set: Union[List, set]):
    """Extend the list as though it was a set.

    >>> lst = [1, 2, 3]
    >>> extend_list_as_set(lst, {3, 4})
    >>> lst
    [1, 2, 3, 4]
    """
    membership = set(lst)
    for item in lst_or_set:
        if item not in membership:
            lst.append(item)


def build_adjacency_cost(
    nodes: set,
    node_to_members: Dict[int, set]
) -> Tuple[Dict, Dict]:
    """Build adjacency and cost matrices for from clusters of channels

    >>> adj, cost = build_adjacency_cost(
    ...     (1, 2, 3), {1: {2, 3}, 2: {1, 3}, 3: {1, 2}})
    >>> dict(adj)
    {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
    >>> dict(cost)  # doctest: +NORMALIZE_WHITESPACE
    {1: 2, (1, 2): -1, (1, 3): -1, 2: 2, (2, 1): -1, (2, 3): -1, 3: 2,
    (3, 1): -1, (3, 2): -1}
    >>> adj, cost = build_adjacency_cost(
    ...     (1, 2, 3, 4), {1: {0, 1, 2}, 2: {0, 2}, 3: {2, 1}, 4: {3}})
    >>> dict(adj)
    {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
    >>> dict(cost)  # doctest: +NORMALIZE_WHITESPACE
    {1: 3, (1, 2): -2, (1, 3): -2, 2: 2, (2, 1): -2, (2, 3): -1, 3: 2,
    (3, 1): -2, (3, 2): -1, 4: 1}
    >>> adj, cost = build_adjacency_cost(
    ...     (1, 2), {1: [(1, 0), (1, 1)], 2: [(1, 0), (1, 1), (4, 0), (4, 1)]})
    >>> dict(adj)  # only one directed edge from 1 -> 2
    {1: {2}}
    >>> dict(cost)
    {1: 2, (1, 2): -2, 2: 4}
    >>> adj, cost = build_adjacency_cost((1, 2, 3), {1: {2}, 2: {3}, 3: {1}})
    >>> dict(adj)  # no edges
    {}
    >>> dict(cost)  # make sure node costs still populated
    {1: 1, 2: 1, 3: 1}
    """
    node_to_unordered_members = {
        node: set(members) for node, members in node_to_members.items()}

    adj = defaultdict(set)
    cost = defaultdict(int)
    for parent in nodes:
        cost[parent] = len(node_to_members[parent])
        for child in nodes:
            if parent == child:
                continue
            channels_parent = node_to_members[parent]
            channels_child = node_to_members[child]
            channels_shared = node_to_unordered_members[parent].intersection(
                node_to_unordered_members[child])
            if channels_shared and should_edge_exist_between(
                channels_parent, channels_child, channels_shared
            ):
                adj[parent].add(child)
                cost[(parent, child)] = -len(channels_shared)
    return adj, cost


def should_edge_exist_between(
    channels_parent: set,
    channels_child: set,
    channels_shared: set
) -> bool:
    """Check if edge should exist between the provided parent and child.

    Node A has a directed edge to node B if A's last producer's channels are
    shared with B's first producer. Note that if all producer n's channels in A
    are shared, *then, producer n-1 can share channels, for the edge to exist.
    This is because reordering can only occur within a single producer's
    channels.
    """
    if (
        all(map(lambda item: isinstance(item, int), channels_parent))
        and all(map(lambda item: isinstance(item, int), channels_child))
    ):
        return True

    mapping_parent = channels_to_mapping(channels_parent)
    mapping_shared = channels_to_mapping(channels_shared)

    current_producer = None
    for producer, _ in channels_parent[::-1]:
        if current_producer == producer:
            continue
        current_producer = producer
        if (
            set(mapping_parent[producer][0]) - set(mapping_shared.pop(
                producer, [[]])[0])
            and mapping_shared
        ):
            return False
    return True


@total_ordering
@dataclass
class Path:
    """
    Data structure for encoding path with cost in a graph.

    >>> max([Path(cost=1), Path(cost=-1), Path(cost=9)])
    Path(nodes=(), cost=9)
    >>> Path(cost=10) > Path(cost=-1)
    True
    >>> Path(cost=2) + 2
    Path(nodes=(), cost=4)
    """
    nodes: Tuple[int] = ()
    cost: int = 0

    def is_valid(self, other):
        return hasattr(other, 'cost')

    def __eq__(self, other):
        return self.is_valid(other) and self.cost == other.cost

    def __lt__(self, other):
        return self.is_valid(other) and self.cost < other.cost

    def __add__(self, other):
        assert isinstance(other, (int, float))
        self.cost += other
        return self


def build_channel_order_from_paths(
    paths: List[Path],
    node_to_members: Dict[int, set],
    id_to_ordered_channels: Dict[int, List]
) -> List[int]:
    """Build channel order from paths in graph

    Args:
        paths List[Path]: list of linear paths in graph
        node_to_members Dict[int, set]: unordered channels per id
        id_to_ordered_channels Dict[int, list]: ordered channels per node, for
            some nodes only

    Returns:
        list of channel ids
    """
    channels = []
    for path in paths:
        for prv, cur, nxt in zip(
            (None,) + path.nodes[:-1],
            path.nodes,
            path.nodes[1:] + (None,)
        ):
            channels_prv = node_to_members.get(prv, set())
            channels_cur = SetLikeList(id_to_ordered_channels[cur]) or \
                SetLikeList(sorted(node_to_members[cur]))
            channels_nxt = node_to_members.get(nxt, set())
            extend_list_as_set(channels,
                               channels_cur.intersection(channels_prv))
            extend_list_as_set(channels,
                               channels_cur - channels_nxt - channels_prv)
            extend_list_as_set(channels,
                               channels_cur.intersection(channels_nxt))
    return channels


def max_cost_path(
    nodes: set,
    adj: Dict[int, set],
    cost: Dict[Union[Tuple, int], int]
) -> List[Path]:
    """Find the maximum path cost by naively computing the maximum path from
    all nodes.

    Note this currently naively, iteratively finds the max-cost path until all
    nodes are included in a path. This assumes that a partial ordering is
    better than none at all. However, if this is untrue (e.g., if any
    re-indexing causes a new tensor copy), then this is not needed. Simply
    randomly tack on all remaining channels. Another way to put this: Is memory
    copy or not a binary thing? Or does number of channels that need copying
    matter?

    >>> a, b, c, d, e = 'abcde'
    >>> max_cost_path((a, b, c),
    ...     {a: {b, c}, b: {a, c}, c: {a, b}}, {a: 1, b: 2, c: 2}
    ... )  # b -> c (2 + 2)
    [Path(nodes=('b', 'c'), cost=4), Path(nodes=('a',), cost=1)]
    >>> adj = {a: {b}, b: {c, d, e}, c: {b, d}, d: {b, c}, e: {b}}
    >>> cost = {a: 2, b: 3, c: 4, d: 2, e: 3}
    >>> max_cost_path(a, adj, cost)  # a -> b -> c (2 + 3 + 4)
    [Path(nodes=('a', 'b', 'c'), cost=9)]
    """
    invalids, paths = set(), []
    while len(invalids) < len(nodes):
        path = max(
            max_cost_path_from(
                source, adj, cost, invalids=invalids, path=(source,))
            for source in nodes if source not in invalids)
        paths.append(path)
        invalids |= set(path.nodes)
    return paths


def max_cost_path_from(
    source: int,
    adj: Dict[int, set],
    cost: Dict[Union[Tuple, int], int],
    invalids: set = set(),
    path: Tuple = ()
) -> Path:
    """
    Find the maximum cost, acyclic linear path in a graph, given the adjacency
    and cost matrices.

    >>> a, b, c, d, e = 'abcde'
    >>> max_cost_path_from(
    ...     a, {a: {b, c}, b: {a, c}, c: {a, b}}, {a: 1, b: 2, c: 1},
    ...     invalids={a}, path=(a,))  # a->b
    Path(nodes=('a', 'b'), cost=3)
    >>> adj = {a: {b}, b: {c, d, e}, c: {b, d}, d: {b, c}, e: {b}}
    >>> cost = {a: 2, b: 3, c: 4, d: 2, e: 3}
    >>> max_cost_path_from(
    ...     b, adj, cost, invalids={b}, path=(b,))  # b -> c (3 + 4)
    Path(nodes=('b', 'c'), cost=7)
    """
    child_invalids = invalids | {source} | adj[source]
    max_path = max([
        max_cost_path_from(child, adj, cost, child_invalids, path + (child,))
        + cost.get((source, child), 0)
        for child in adj[source] if child not in invalids
    ] + [Path(path)])
    max_path.cost += cost[source]
    return max_path
