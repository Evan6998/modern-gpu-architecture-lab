#include <torch/extension.h>
void transpose_cuda(torch::Tensor x, torch::Tensor out, int64_t variant, int64_t tile);
void softmax_cuda(torch::Tensor x, torch::Tensor out, int64_t variant, int64_t warps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose", &transpose_cuda);
    m.def("softmax", &softmax_cuda);
}
