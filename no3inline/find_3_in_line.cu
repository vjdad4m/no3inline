#include <torch/extension.h>

__device__ bool are_points_on_same_line_kernel(int i1, int j1, int i2, int j2, int i3, int j3) {
    // bool ordered_points = (i1 < i2 && i2 < i3) || (j1 < j2 && j2 < j3);
    // if (!ordered_points) {
    //     return false;
    // }
    bool intersect = (i1 - i2) * (j1 - j3) == (i1 - i3) * (j1 - j2);
    return intersect;
}


__global__ void find_3_in_line_kernel(const bool* matrix, int N, int* count_result, int* idx_result) {
    int i1, j1, i2, j2, i3, j3;

    for (i1 = 0; i1 < N; i1++) {
        for (j1 = 0; j1 < N; j1++) {
            if (!matrix[i1 * N + j1]) {
                continue;
            }
            for (i2 = 0; i2 <= i1; i2++) {
                for (j2 = 0; j2 <= j1; j2++) {
                    if (!matrix[i2 * N + j2]) {
                        continue;
                    }
                    if (i1 == i2 && j1 == j2) {
                        continue;
                    }
                    for (i3 = 0; i3 <= i2; i3++) {
                        for (j3 = 0; j3 <= j2; j3++) {
                            if (!matrix[i3 * N + j3]) {
                                continue;
                            }
                            if ((i1 == i3 && j1 == j3) || (i2 == i3 && j2 == j3)) {
                                continue;
                            }
                            // Check if points lie on the same line
                            bool points_on_same_line = are_points_on_same_line_kernel(i1, j1, i2, j2, i3, j3);

                            // Accumulate the count if points are on the same line
                            if (points_on_same_line) {
                                int offset = atomicAdd(count_result, 1);

                                // Store the indices of the points
                                idx_result[offset * 6 + 0] = i1;
                                idx_result[offset * 6 + 1] = j1;

                                idx_result[offset * 6 + 2] = i2;
                                idx_result[offset * 6 + 3] = j2;

                                idx_result[offset * 6 + 4] = i3;
                                idx_result[offset * 6 + 5] = j3;

                                
                            }
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor find_3_in_line(torch::Tensor input_tensor) {
    const bool* matrix = input_tensor.data_ptr<bool>();
    int N = input_tensor.size(0);
    int count = 0;
    
    // Create a tensor to store the results (i1, j1, i2, j2, i3, j3) for each set of points
    std::vector<int> results_vector(N * N * 6, 0);
    int* dev_results;
    int* dev_count;


    cudaMalloc((void**)&dev_results, N * N * 6 * sizeof(int));
    cudaMalloc((void**)&dev_count, sizeof(int));
    cudaMemcpy(dev_count, &results_vector[0], sizeof(int), cudaMemcpyHostToDevice);

    find_3_in_line_kernel<<<1, 1>>>(matrix, N, dev_count, dev_results);
    cudaDeviceSynchronize();

    cudaMemcpy(&results_vector[0], dev_results, results_vector.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_results);
    cudaFree(dev_count);

    int num_sets = count;
    int num_results_per_set = 6;

    // Convert results_vector to a PyTorch tensor
    torch::Tensor results = torch::from_blob(
        &results_vector[0], 
        {num_sets, num_results_per_set}, 
        torch::kInt32
    ).clone();

    // reshape the results tensor to (num_sets, 3, 2)
    results = results.reshape({num_sets, 3, 2});

    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("find_3_in_line", &find_3_in_line, "Count 3 points in line");
}