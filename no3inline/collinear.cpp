#include <torch/extension.h>
torch::Tensor reward_cuda(torch::Tensor m);

torch::Tensor calculate_reward_cuda(torch::Tensor m) {
    return reward_cuda(m);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_reward_cuda", &reward_cuda, "Reward (CUDA)");
}
