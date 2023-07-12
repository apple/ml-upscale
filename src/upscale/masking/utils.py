"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.
"""

import numpy as np
import torch
import torch.nn as nn
from itertools import chain

# TODO: move this + doctests to tests (only used for testing)
class DummyOp(nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)


def flatten_list_of_lists(lol):
    return [len(lst) for lst in lol], torch.tensor(list(chain(*lol)))


def unflatten_list_of_lists(meta, lst):
    starts = np.cumsum([0] + meta)
    lol = []
    for start, end in zip(starts, starts[1:]):
        lol.append(lst[start: end])
    return lol
