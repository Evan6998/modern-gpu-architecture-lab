#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// TODO(A1.1): implement three variants, retaining all three for ablation.
// Input X[M,N] and output Y[N,M] are distinct contiguous FP32 tensors.
__global__ void transpose_naive(const float* x, float* y, int m, int n) {
    // TODO: derive a bounds-checked global index and write exactly one result.
}

// TODO(A1.2): shared-memory tiled transpose, then change its layout to remove
// bank conflicts. Every participating thread must reach block-wide barriers.

// TODO(A1.3): row-wise stable softmax, shared reduction and shuffle reduction.
// The last warp can be partial; never read a value from an inactive lane.
// Tests contain rows wider than a block and non-multiples of 32.

// `tile` is config["tile_m"] (default 32): the shared-memory tile edge for
// tiled/padded and the 2-D block edge for naive. `warps` is config["warps"]
// (default 4): warps per block for softmax. TORCH_CHECK the values you support;
// never silently substitute another value, because the CSV records the request.
void transpose_cuda(torch::Tensor x, torch::Tensor out, int64_t variant, int64_t tile) {
    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    TORCH_CHECK(false, "TODO(A1): implement and dispatch transpose kernels in csrc/kernels.cu");
    // Launch on stream, NOT the default stream; then check launch status.
    // transpose_naive<<<grid, block, 0, stream>>>(...);
    // C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_cuda(torch::Tensor x, torch::Tensor out, int64_t variant, int64_t warps) {
    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    TORCH_CHECK(false, "TODO(A1): implement shared/shuffle softmax");
}
