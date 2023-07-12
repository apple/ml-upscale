"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.
"""

from upscale.pruning.tracing import trace
from upscale.pruning.resolve import get_producers_consumers, get_segments
from upscale.pruning.pruner import get_tensor_to_module_metadata
from upscale.masking.importance import Magnitude
from upscale.masking.utils import flatten_list_of_lists, unflatten_list_of_lists, DummyOp  # TODO: rm, only used for testing
import torch.nn as nn
import torch
from functools import partial


class MaskSegment:
    """Represents a group of layers that will be pruned in the same way.
    
    >>> weights = [torch.tensor([1, 2, 3, 4.]), torch.tensor([100, 99, 98, 97.])]
    >>> layers = [DummyOp(weights[0]), DummyOp(weights[1])]
    >>> _ = MaskManager([MaskSegment(layers)]).importance().mask()
    >>> layers[0].weight.tolist(), layers[1].weight.tolist()  # second layer dominates, prune last 2
    ([1.0, 2.0, 0.0, 0.0], [100.0, 99.0, 0.0, 0.0])
    >>> weights = [torch.tensor([1, 2, 3, 4.]), torch.tensor([100, 99, 98, 97.])]
    >>> layers = [DummyOp(weights[0]), DummyOp(weights[1])]
    >>> _ = MaskManager([MaskSegment(layers[:1], siblings=layers[1:])]).importance().mask()
    >>> layers[0].weight.tolist(), layers[1].weight.tolist()  # use first layer's weights
    ([0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 98.0, 97.0])
    """

    def __init__(self, layers, name='weight', dim=0, siblings=()):
        self.layers = layers
        self.siblings = siblings
        self.name = name
        self.dim = dim

    def get_weights(self, layers):
        return [getattr(layer, self.name).transpose(0, self.dim).detach() for layer in layers]

    def set_weights(self, layers, weights):
        for layer, weight in zip(layers, weights):
            param = nn.Parameter(weight.transpose(0, self.dim))
            old = getattr(layer, self.name)
            param.tensor_id = getattr(old, 'tensor_id', None)  # TODO: hack to preserve op ids in trace
            setattr(layer, self.name, param)

    def importance(self, heuristic=Magnitude()):
        if heuristic.name == 'weight':
            # TODO: broken in general (e.g., for densenet, need to use trace)
            unpruneds = [weight.reshape(weight.shape[0], -1) for weight in self.get_weights(self.layers)]
        else:
            unpruneds = [getattr(layer, heuristic.name) for layer in self.layers]

        # HACK: for densenet
        # TODO: 'better' combo for weird shapes like in densenet?
        largest = max(weight.shape[0] for weight in unpruneds)
        paddeds = [torch.cat([weight, torch.zeros(
            largest - weight.shape[0],
            weight.shape[-1]
        ).to(weight.device)]) for weight in unpruneds]

        unpruned = torch.cat(paddeds, dim=1)
        self._importance = heuristic(unpruned)
        return self

    def mask(self, mask):
        layers = list(self.layers) + list(self.siblings)

        weights = []
        for weight in self.get_weights(layers):
            _mask = mask.to(weight.device)
            while len(_mask.shape) < len(weight.shape):
                _mask = _mask[:, None]
            weights.append(_mask[:weight.shape[0]] * weight)  # HACK: slice is a hack for densenet

        self.set_weights(layers, weights)
        return self


class MaskManager:
    """Handles masking, globally, given a set of segments.
    
    >>> weights = [torch.tensor([1, 2, 3, 4.]), torch.tensor([5, 6, 7, 8.])]
    >>> layers = [DummyOp(weights[0]), DummyOp(weights[1])]
    >>> _ = MaskManager([MaskSegment(layers)]).importance().mask()
    >>> layers[0].weight.tolist(), layers[1].weight.tolist()  # prune 50% of each
    ([0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 7.0, 8.0])
    >>> layers = [DummyOp(weights[0]), DummyOp(weights[1])]
    >>> _ = MaskManager([MaskSegment(layers[:1]), MaskSegment(layers[1:])]).importance().mask()
    >>> layers[0].weight.tolist(), layers[1].weight.tolist()  # prune 50% globally smallest
    ([0.0, 0.0, 0.0, 0.0], [5.0, 6.0, 7.0, 8.0])
    """

    def __init__(self, segments):
        self.segments = segments
        assert self.segments, 'Need at least one segment for manager'

    def importance(self, heuristic=Magnitude()):
        self._importance = [segment.importance(heuristic=heuristic)._importance for segment in self.segments]
        return self
    
    def mask(self, amount=0.5):
        meta, importances = flatten_list_of_lists(self._importance)
        k = min(len(importances), int(amount * len(importances)) + 1)
        maximum, _ = torch.kthvalue(importances, k=k, dim=0)
        mask = torch.ones_like(importances).to(importances.device)
        mask[importances < maximum] = 0
        lol = unflatten_list_of_lists(meta, mask)
        for mask, segment in zip(lol, self.segments):
            segment.mask(mask)
        return self


class MaskingManager:
    """Handles masking for a model"""

    def __init__(self, model, side='input', method='unconstrained', is_global=False):
        self.model = model
        self.side = side
        self.method = method
        self.is_global = is_global
        self._segments = None

    def segments(self):
        if self._segments is None:
            segments = segment_model(
                model=self.model,
                side=self.side,
                method=self.method,
                is_global=self.is_global
            )
            if self.is_global:
                self._segments = [MaskManager(segments)]
            else:
                self._segments = [MaskManager([segment]) for segment in segments]
        return self._segments
    
    def importance(self, heuristic=Magnitude()):
        for segment in self.segments():
            segment.importance(heuristic=heuristic)
        return self
    
    def mask(self, amount=0.5):
        for segment in self.segments():
            segment.mask(amount=amount)
        return self


def link_bn_using_metadata(tensor_to_module_metadata):
    """
    Assumes tensor_id is assigned sequentially in the model
    """
    max_tensor_id = max(tensor_to_module_metadata)
    for tensor_id, module_metadata in tensor_to_module_metadata.items():
        module = module_metadata['module']
        if isinstance(module, nn.Conv2d):
            next_id = tensor_id + 1
            while next_id not in tensor_to_module_metadata and next_id < max_tensor_id:
                next_id += 1
            if tensor_id == max_tensor_id:  # this op is the last op
                module.bn = None
                continue
            next_module = tensor_to_module_metadata[next_id]['module']
            if isinstance(module, nn.BatchNorm2d):
                # NOTE: for densenet
                # if module.weight.shape[0] == next_module.weight.shape[0]:
                #     module.bn = next_module
                # else:
                #     module.bn = None
                assert module.weight.shape[0] == next_module.weight.shape[0]
                module.bn = next_module
            else:
                module.bn = None


def segment_model(model, side='input', method='unconstrained', is_global=False):
    # get segments
    y = trace(model, inputs=[torch.rand((1, 3, 224, 224)).cuda()])
    producers, consumers = get_producers_consumers(y)

    # get mapping from op ids to metadata (include layer itself)
    tensor_to_module_metadata = get_tensor_to_module_metadata(y.tensor_to_metadata)

    link_bn_using_metadata(tensor_to_module_metadata)

    segments = []
    skipped_segment = False
    for (producer_ids, consumer_ids, user_ids) in get_segments(y.tensor_to_traces, producers, consumers):
        # TODO: investigate why pruning first segment causes problems in r18. only global techniques can skip segments (because sparsity level applied globally)
        if is_global and model.__class__.__name__ == 'ResNet' and not skipped_segment:
            skipped_segment = True
            continue

        if side == 'input':
            # if input pruning, run on all consumers
            consumers = [tensor_to_module_metadata[c_id]['module'] for c_id in consumer_ids]
            consumers = list(filter(lambda c: c.weight.shape[1] != 3 and 'conv' in c.__class__.__name__.lower(), consumers))  # if input pruning, skip convs that take model input
            if not consumers:
                continue
            Segment = partial(MaskSegment, dim=1)
            if method == 'constrained':
                segments.append(Segment(consumers))
                segments[-1].metadatas = [tensor_to_module_metadata[c_id] for c_id in consumer_ids]  # HACK
            elif method == 'unconstrained':
                for c_id, consumer in zip(consumer_ids, consumers):
                    segments.append(Segment([consumer]))
                    segments[-1].metadatas = [tensor_to_module_metadata[c_id]] # HACK
            else:
                raise NotImplementedError(f"Unsupported method: {method}. Must be 'constrained' or 'unconstrained'")
        elif side == 'output':
            # if output pruning, run on all producers and their associated batch norms
            producers = [tensor_to_module_metadata[p_id]['module'] for p_id in producer_ids]
            producers = list(filter(lambda p: p.weight.shape[0] != 1000 and 'conv' in p.__class__.__name__.lower(), producers))  # if output pruning, skip convs that produce model output
            if not producers:
                continue
            Segment = partial(MaskSegment, dim=0)
            if method == 'constrained':
                segments.append(Segment(producers, siblings=[p.bn for p in producers if p.bn]))
            elif method == 'unconstrained':
                for producer in producers:
                    segments.append(Segment([producer], siblings=[producer.bn] if producer.bn else []))
        else:
            raise NotImplementedError(f"Unsupported side: {side}. Expected 'input' or 'input'.")

    # unlink bns
    for name, module in model.named_modules():
        if hasattr(module, 'bn'):
            del module.bn 
    return segments


def mask_model(model, side='input', method='unconstrained', amount=0.5, is_global=False, heuristic=Magnitude()):
    return MaskingManager(model, side=side, method=method, is_global=is_global) \
        .importance(heuristic=heuristic) \
        .mask(amount=amount)