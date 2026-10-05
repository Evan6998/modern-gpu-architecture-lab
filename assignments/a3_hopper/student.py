from mgpu import contracts
from mgpu.config import KernelConfig
from mgpu.hardware import require
from . import kernels


def gemm(a, b, out, *, variant="tma_wgmma", config=None):
    contracts.gemm(a, b, out, aligned=True)
    if variant != "tma_wgmma":
        raise ValueError("A3 GEMM variant must be tma_wgmma")
    cfg = KernelConfig.parse(config)
    require("hopper", a.device)
    kernels.launch_gemm(a, b, out, cfg)


def cluster_copy(x, out, *, variant="duplicate", config=None):
    contracts.cluster_copy(x, out)
    if variant not in ("duplicate", "multicast", "dsm"):
        raise ValueError("Unknown cluster-copy variant")
    cfg = KernelConfig.parse(config)
    require("hopper", x.device)
    kernels.launch_cluster_copy(x, out, variant, cfg)
