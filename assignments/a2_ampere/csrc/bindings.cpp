#include <torch/extension.h>
void gemm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor out,
              int64_t variant, int64_t tile_m, int64_t tile_n, int64_t tile_k,
              int64_t stages, int64_t warps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &gemm_cuda); }
