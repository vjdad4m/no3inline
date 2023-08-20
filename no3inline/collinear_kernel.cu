#include <torch/extension.h>

typedef long long ll;
typedef struct {
    ll x;
    ll y;
} pt;

__device__ int turn(pt a, pt b, pt c) {
    if ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y) > 0) return 1;
    else if ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y) < 0) return -1;
    else return 0;
}

__global__ void countCollinearKernel(pt *p, int p_size, int *result) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p_size) {
        for (int j = i + 1; j < p_size; j++) {
            for (int l = j + 1; l < p_size; l++) {
                if (!turn(p[i], p[j], p[l])) {
                    atomicAdd(result, 1);
                }
            }
        }
    }
}

torch::Tensor countCollinear(torch::Tensor points) {
    const int p_size = points.size(0);
    auto result = torch::zeros({1}, torch::kInt32).to(points.device());

    const int threads = 256;
    const int blocks = (p_size + threads - 1) / threads;
    
    countCollinearKernel<<<blocks, threads>>>(reinterpret_cast<pt*>(points.data_ptr()), p_size, result.data_ptr<int>());

    return result;
}
