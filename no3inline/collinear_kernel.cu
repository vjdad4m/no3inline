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
    const int i = blockIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < p_size && j < p_size && j > i) {
        for (int l = j + 1; l < p_size; l++) {
            if (!turn(p[i], p[j], p[l])) {
                atomicAdd(result, 1);
            }
        }
    }
}

torch::Tensor countCollinear(torch::Tensor points) {
    const int p_size = points.size(0);
    auto result = torch::zeros({1}, torch::kInt32).to(points.device());

    dim3 block_dim(1, 256);  // Here, 1 block in x and 256 threads in y.
    dim3 grid_dim(p_size, (p_size + block_dim.y - 1) / block_dim.y);
    
    countCollinearKernel<<<grid_dim, block_dim>>>(reinterpret_cast<pt*>(points.data_ptr()), p_size, result.data_ptr<int>());

    cudaDeviceSynchronize();
    return result;
}
