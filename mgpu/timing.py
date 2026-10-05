"""Isolated operator timing; reference CPU timings are never GPU performance."""
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Callable
import math
import torch


@dataclass
class Timing:
    samples_ms: list[float]
    scope: str
    cache_policy: str

    @property
    def p50_ms(self):
        return median(self.samples_ms)

    def percentile(self, p: float):
        if not 0 <= p <= 1:
            raise ValueError("Percentile must be in [0,1]")
        values = sorted(self.samples_ms)
        index = (len(values) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)


def measure(fn: Callable[[], None], *, device: torch.device, warmup=10,
            repeats=30, evict_mb=0, graph=False) -> Timing:
    if warmup < 1 or repeats < 1 or evict_mb < 0:
        raise ValueError("warmup/repeats must be positive; evict_mb must be nonnegative")
    if device.type != "cuda" and (graph or evict_mb):
        raise ValueError("CUDA graphs and cache eviction require CUDA")
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            evict = torch.empty(evict_mb * 1024 * 1024, dtype=torch.uint8, device=device) if evict_mb else None
            invoke = fn
            if graph:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                invoke = g.replay
            # Initialize events before measuring. All single-GPU implementations
            # must honor PyTorch's CURRENT stream, not an untracked side stream.
            pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                     for _ in range(repeats)]
            samples = []
            for start, end in pairs:
                if evict is not None:
                    evict.fill_(37)  # heuristic only; not proof of a cold cache
                start.record()
                invoke()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
        scope = "cuda_graph_operator" if graph else "cuda_events_operator"
    else:
        samples = []
        for _ in range(repeats):
            start = perf_counter(); fn(); samples.append((perf_counter() - start) * 1000)
        scope = "cpu_reference_wall_time"
    return Timing(samples, scope, f"eviction_buffer_{evict_mb}MiB" if evict_mb else "warm_reused_inputs")
