import torch
from mgpu import contracts
from mgpu.config import KernelConfig
from mgpu.hardware import require
from mgpu.quantization import PreparedQuantizedGemm
from . import kernels


def gemm(a, b, out, *, variant="tcgen05_1sm", config=None):
    contracts.gemm(a, b, out, aligned=True)
    if variant not in ("tcgen05_1sm", "tcgen05_2sm", "persistent"):
        raise ValueError("Unknown Blackwell GEMM variant")
    cfg = KernelConfig.parse(config)
    require("blackwell", a.device)
    kernels.launch_gemm(a, b, out, variant, cfg)


def prepare_quantized(a, bt):
    """Optional native repacking belongs here, outside steady-state GEMM timing.

    No mathematical GEMM occurs here. This identity preparation is scaffolding,
    not a substitute for the required low-precision hardware kernel.
    """
    prepared = PreparedQuantizedGemm(a, bt).validate()
    # TODO(A4): if required, populate prepared.backend with device-native layouts.
    return prepared


def quant_gemm(prepared, out, *, config=None):
    prepared.validate()
    a, bt = prepared.a, prepared.bt
    contracts.tensors(out, dtype=torch.float16)
    contracts.shape(out, (a.logical_shape[0], bt.logical_shape[0]))
    if out.device != a.data.device:
        raise ValueError("Output device mismatch")
    contracts.distinct_output(out, a.data, a.scales, bt.data, bt.scales)
    if a.logical_shape[0] % 128 or bt.logical_shape[0] % 128 or a.logical_shape[1] % 32:
        raise ValueError("A4 M,N must be multiples of 128; K a multiple of 32")
    if a.format == "mxfp4" and a.logical_shape[1] % 64:
        # One tcgen05 kind::mxf4 MMA consumes K=64, i.e. two 32-element scale
        # blocks. A zero-padded K%64==32 tail is outside this assignment.
        raise ValueError("A4 MXFP4 K must be a multiple of 64")
    cfg = KernelConfig.parse(config)
    require("blackwell", out.device)
    kernels.launch_quantized(prepared, out, cfg)
