"""Hopper kernel workspace. CuTe DSL or an equivalent CUDA C++ backend is allowed.

Keep the public launcher signatures. Cache compilation/descriptors by complete
shape/dtype/stride/device/config keys, never by input values. Do not cache output.
"""


def launch_gemm(a, b, out, cfg):
    # TODO(A3.1): import cutlass.cute lazily and build layouts/TMA descriptors.
    # TODO(A3.2): allocate ring-buffer SMEM and initialize per-stage barriers.
    # TODO(A3.3): producer: wait-empty -> issue TMA -> publish-ready.
    # TODO(A3.4): consumer: wait-ready -> WGMMA commit/wait -> release stage.
    # TODO(A3.5): drain in-flight MMA before epilogue and SMEM reuse.
    # Use torch.cuda.current_stream(a.device); do not synchronize the device here.
    raise NotImplementedError("A3: implement TMA + WGMMA producer/consumer mainloop")


def launch_cluster_copy(x, out, variant, cfg):
    # One two-CTA cluster, tile [128,128] FP32. Both outputs must equal X.
    # duplicate: each CTA loads its own copy.
    # multicast: one TMA operation multicasts to both CTA SMEM allocations.
    # dsm: CTA0 loads, CTA1 reads remote SMEM after cluster synchronization.
    # Do not let CTA0 exit before CTA1 has completed its DSM reads.
    raise NotImplementedError(f"A3: implement two-CTA {variant} experiment")
