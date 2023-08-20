#include <torch/extension.h>

torch::Tensor countCollinear(torch::Tensor points);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_collinear", &countCollinear, "Count Collinear Points");
}
