"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.
"""

import time

import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity
import torch
import torch.nn as nn
import numpy as np

class Latency:
    """Measure latency on device, using pytorch jit compilation"""

    # TODO: make a proper enum or just discard unused versions
    TORCH_BENCHMARK = 1
    TORCH_PROFILER = 2
    MANUAL = 3

    def __init__(
        self,
        n_iters: int=5,
        n_warmup: int=2,
        backend: int=TORCH_PROFILER
    ):
        self.n_iters = n_iters
        self.n_warmup = n_warmup
        self.backend = backend

        assert backend in (
            Latency.TORCH_BENCHMARK,
            Latency.TORCH_PROFILER,
            Latency.MANUAL
        )

    def forward(self, model: nn.Module, x: torch.Tensor):
        jit = torch.jit.trace(model, [x], strict=False).to(x.device)
        f = lambda: jit(x)
        if self.backend == Latency.TORCH_BENCHMARK:
            num_threads = 1
            timer = benchmark.Timer(stmt="f()",
                                    globals={
                                        "f": f,
                                    },
                                    num_threads=num_threads,
                                    label="Latency Measurement",
                                    sub_label="torch.utils.benchmark.")

            profile_result = timer.timeit(self.n_iters)
            return profile_result.mean, 0.0
        if self.backend == Latency.TORCH_PROFILER:
            assert torch.cuda.is_available() and next(model.parameters()).is_cuda
            stats = []
            for i in range(self.n_iters + self.n_warmup):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    with record_function("model_inference"):
                        f()
                if i > self.n_warmup:
                    stats.append(
                        prof
                        .key_averages()
                        .total_average()
                        .self_cuda_time_total / 1000.0
                    )
            mean, std = np.mean(stats), np.std(stats)
            return mean, std
        if self.backend == Latency.MANUAL:
            stats = []
            for i in range(self.n_iters + self.n_warmup):
                if torch.cuda.is_available():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    start = time.time()
                f()
                if torch.cuda.is_available():
                    end.record()
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end)
                else:
                    elapsed = time.time() - start
                if i > self.n_warmup:
                    stats.append(elapsed)
            mean, std = np.mean(stats), np.std(stats)
            return mean, std