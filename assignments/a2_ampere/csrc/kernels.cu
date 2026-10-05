#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

// A[M,K], B[K,N], C[M,N], all row-major FP16. Accumulate in FP32.
// TODO(A2.1): SIMT shared-memory tiled GEMM.
// TODO(A2.2): warp-level mma.sync + ldmatrix; document lane/fragment layout.
// TODO(A2.3): replace synchronous global->shared copies with cp.async;
// retain prologue, steady state and drain. Handle K tails explicitly.
// You may use your SIMT kernel for edge tiles, but not a library GEMM.

__global__ void gemm_simt(const __half* a, const __half* b, __half* c,
                          int m, int n, int k) {
    // TODO: shared tiles, register accumulators, guarded epilogue.
}

void gemm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor out,
               int64_t variant, int64_t tile_m, int64_t tile_n,
               int64_t tile_k, int64_t stages, int64_t warps) {
    const c10::cuda::CUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    TORCH_CHECK(false, "TODO(A2): dispatch simt/mma/async GEMM");
    // Set dynamic-SMEM opt-in if needed; launch on stream.
    // C10_CUDA_KERNEL_LAUNCH_CHECK();
}
