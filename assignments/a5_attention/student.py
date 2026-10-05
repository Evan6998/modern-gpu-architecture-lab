from mgpu import contracts
from mgpu.config import KernelConfig
from mgpu.hardware import require


def attention(q, k, v, out, *, variant="fused", config=None):
    contracts.attention(q, k, v, out)
    if variant != "fused":
        raise ValueError("Only the fused attention variant is supported")
    cfg = KernelConfig.parse(config)
    require("ampere", q.device)
    # Lazy import lets CPU reference tests run without Triton installed.
    from .kernels import launch_attention
    launch_attention(q, k, v, out, cfg)
