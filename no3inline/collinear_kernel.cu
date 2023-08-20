#include <torch/types.h>

typedef long long ll;

struct pt {
    ll x;
    ll y;
};

__device__ int turn(pt a, pt b, pt c) {
    if ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y) > 0) return 1;
    else if ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y) < 0) return -1;
    else return 0;
}

__global__ void rewardKernel(const pt* points, const int p_size, int* result) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p_size) {
        for (int j = i + 1; j < p_size; j++) {
            for (int l = j + 1; l < p_size; l++) {
                if (!turn(points[i], points[j], points[l])) {
                    atomicAdd(result, 1);
                }
            }
        }
    }
}

torch::Tensor reward_cuda(torch::Tensor m) {
    auto m_size = m.size(0);
    auto points = torch::empty({m_size * m_size, 2}, torch::kInt64).to(m.device());
    auto count = torch::zeros({1}, torch::kInt32).to(m.device());

    int index = 0;
    for (int i = 0; i < m_size; i++) {
        for (int j = 0; j < m_size; j++) {
            if (m[i][j].item<int>()) {
                points[index][0] = i;
                points[index][1] = j;
                index++;
            }
        }
    }

    const int threads = 256;
    const int blocks = (index + threads - 1) / threads;

    rewardKernel<<<blocks, threads>>>(reinterpret_cast<pt*>(points.data_ptr()), index, count.data_ptr<int>());

    return count;
}
