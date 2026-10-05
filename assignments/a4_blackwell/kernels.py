"""Blackwell SM100 workspace. A Hopper WGMMA implementation is not sufficient."""


def launch_gemm(a, b, out, variant, cfg):
    # TODO(A4.1): allocate TMEM for FP32 accumulators; stage operands with TMA.
    # TODO(A4.2): tcgen05 MMA + completion barrier; drain before TMEM readback.
    # TODO(A4.3): TMEM -> registers -> global epilogue; release TMEM safely.
    # TODO(A4.4): select 1-CTA / 2-CTA MMA; then add persistent tile scheduling.
    # CuTe primitives are allowed; the scheduler/mainloop must be your own.
    raise NotImplementedError(f"A4: implement {variant} using tcgen05 + TMEM")


def launch_quantized(prepared, out, cfg):
    # FP8: E4M3FN operands; FP32 per-tensor scales can be applied in epilogue.
    # MXFP4: packed E2M1 + E8M0 scales (K-block=32); scale layout is part of task.
    # Both accumulate in FP32 and output FP16. Do not dequantize to FP16 and
    # call a dense GEMM: numerical tests alone cannot prove low-precision MMA.
    raise NotImplementedError(f"A4: implement native {prepared.a.format} GEMM")
