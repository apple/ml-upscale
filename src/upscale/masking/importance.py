"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.
"""
import time
import logging

import torch
import torch.nn as nn
from pathlib import Path
from upscale.masking.utils import DummyOp  # TODO: rm, only used for testing


class Importance(nn.Module):
    needs_feature_map_statistics = False
    name = 'weight'

    def register(self, model):
        """Register hooks on model to collect feature map statistics."""
        raise NotImplementedError()

    def deregister(self, model):
        """Cleanup hooks on model."""
        raise NotImplementedError()

    def forward(self, x):
        """
        Compute importance using provided tensors associated with each segment's
        producers and consumers.
        """
        raise NotImplementedError()


class Magnitude(Importance):
    """Apply magnitude pruning.
    
    Pruning Filters for Efficient ConvNets (L1 norm)
    (ICLR 2017, https://arxiv.org/abs/1608.08710)

    Learning Structured Sparsity in Deep Learning (L2 norm)
    (NeurIPS 2016, https://arxiv.org/abs/1608.03665)

    Compute the L-p norm of every channel.
    """
    def __init__(self, p='fro'):
        super().__init__()
        self.p = p

    def forward(self, x):
        return torch.norm(x, dim=1, p=self.p)


class LAMP(Magnitude):
    """Apply a global, layer-adaptive magnitude pruning.

    Layer-Adaptive Sparsity for Magnitude-based Pruning
    (ICLR 2021, https://arxiv.org/abs/2010.07611)

    Original: https://github.com/jaeho-lee/layer-adaptive-sparsity/blob/817dad7abc1bfebcfbad7ae00af253e557c8749b/tools/pruners.py#L162
    
    Every channel norm is normalized by the cumulative sum of all 'surviving'
    channels' norms. Then, globally prune the least important channels.

    >>> from upscale.masking.mask import MaskManager, MaskSegment
    >>> net = nn.Sequential(
    ...     DummyOp(torch.tensor([1, 1, 1, 1.])),  # importances: [.25, .33, .5, 1]
    ...     DummyOp(torch.tensor([1, 2, 3, 4.])),  # importances: [.1, .22, .43, 1]
    ... )
    >>> _ = MaskManager([MaskSegment(net)]).importance(LAMP()).mask(0.5)
    >>> net[0].weight.flatten().tolist()
    [0.0, 0.0, 1.0, 1.0]
    >>> net[1].weight.flatten().tolist()
    [0.0, 0.0, 3.0, 4.0]
    """
    def forward(self, x):
        norms = super().forward(x)
        indices = torch.argsort(norms, dim=0, descending=True).flatten().tolist()
        numerator = norms[indices]
        denominator = torch.cumsum(numerator, dim=0)
        importance = numerator / denominator
        rindices = torch.arange(len(indices))[indices]
        return importance[rindices]


class FPGM(Magnitude):
    """Use geometeric median as importance.
    
    Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration
    (CVPR 2019, https://arxiv.org/abs/1811.00250)

    Original: https://github.com/he-y/filter-pruning-geometric-median/blob/76f0ff/pruning_imagenet.py#L499

    The original implementation prunes the 10% smallest weights, in some layers. We don't do this,
    as L-p norm importance degrades performance too quickly in post-training.
    - https://github.com/he-y/filter-pruning-geometric-median/blob/76f0ff/pruning_imagenet.py#L434
    - https://github.com/he-y/filter-pruning-geometric-median/blob/76f0ff/pruning_imagenet.py#L52

    The core of the method prunes filters closest to the geometric median, which can supposedly be
    represented by other filters already so is 'redundant'. This is the same as finding the filters
    that minimize distances to other filters. The smaller the total distance, the less important.
    https://github.com/he-y/filter-pruning-geometric-median/blob/76f0ff/pruning_imagenet.py#L462
    """
    def forward(self, x):
        return torch.cdist(x, x).sum(dim=0)


class HRank(Importance):
    """Use feature map ranks as importance.

    HRank: Filter Pruning using High-Rank Feature Map
    (CVPR 2020, https://arxiv.org/abs/2002.10179)

    Original: https://github.com/lmbxmu/HRank/blob/master/rank_generation.py#L205
    """
    needs_feature_map_statistics = True

    def __init__(self, side='input', out='./out'):
        super().__init__()
        assert side in ('input', 'output')
        self.side = side
        self.name = f"_hrank_{side}"
        self.out = Path(out) / 'hrank'
        self.out.mkdir(exist_ok=True, parents=True)

    def get_rank_path(self, model, module):
        path = self.out / f"hrank_{model._name}_{module._name}_{self.side}.pth"
        return path

    def get_rank(self, model, module):
        path = self.get_rank_path(model, module)
        if path.exists():
            return torch.load(path).cuda()
        return None

    def register(self, model):
        def hook(module, input, output):
            side = self.side
            x = input[0] if side == 'input' else output

            path = self.get_rank_path(model, module)
            if not path.exists():
                # compute and store ranks
                start = time.time()
                mean = torch.linalg.matrix_rank(x).float().mean(dim=0)
                end = time.time()
                setattr(module, self.name, mean)
                torch.save(mean, path)

                logging.info(f"Computed rank for {module._name} ({x.shape}) in {round(end - start, 2)}s")
            else:
                logging.debug(f"Loaded precomputed rank for {module._name})")

        model._handles = []
        is_hook_registered = False
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module._name = name
                path = self.get_rank_path(model, module)
                if not path.exists():
                    model._handles.append(module.register_forward_hook(hook))
                    is_hook_registered = True
        return is_hook_registered

    def deregister(self, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                rank = self.get_rank(model, module)
                assert rank is not None
                setattr(module, self.name, rank[:, None])
        while hasattr(model, '_handles') and model._handles:
            model._handles.pop().remove()

    def forward(self, x):
        return x[:, 0]