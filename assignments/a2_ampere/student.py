from mgpu import contracts
from mgpu.config import KernelConfig
from mgpu.hardware import require
from mgpu.extension import load_extension


def gemm(a, b, out, *, variant="simt", config=None):
    contracts.gemm(a, b, out)
    names = ("simt", "mma", "async")
    if variant not in names:
        raise ValueError(f"Unknown GEMM variant: {variant}")
    cfg = KernelConfig.parse(config)
    require("ampere", a.device)
    load_extension("a2").gemm(a, b, out, names.index(variant),
        cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.stages, cfg.warps)
