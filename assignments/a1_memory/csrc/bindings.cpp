#include <torch/extension.h>
void transpose_cuda(torch::Tensor x, torch::Tensor out, int64_t variant);
void softmax_cuda(torch::Tensor x, torch::Tensor out, int64_t variant);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose", &transpose_cuda);
    m.def("softmax", &softmax_cuda);
}
