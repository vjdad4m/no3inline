import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import no3inline


def generate_matrix(matrix_size, matrix_type):
    """Generate a matrix based on the specified type and size."""
    if matrix_type == 'torch':
        return torch.randint(0, 2, (matrix_size, matrix_size))
    elif matrix_type == 'numpy':
        return np.random.randint(0, 2, (matrix_size, matrix_size))
    else:
        return [[random.randint(0, 1)] * matrix_size for _ in range(matrix_size)]

def time_function(func, *args):
    """Time the execution of a function."""
    start = time.time()
    result = func(*args)
    end = time.time()
    return end - start, result

def benchmark(matrix_size, matrix_type):
    """Benchmark the no3inline.calculate_reward function."""
    arr = generate_matrix(matrix_size, matrix_type)
    elapsed_time, count = time_function(no3inline.calculate_reward, arr)
    return elapsed_time, count

def main():
    matrix_sizes = np.arange(1, 50, 1)
    matrix_types = ['numpy', 'torch', 'python']
    times = {matrix_type: [] for matrix_type in matrix_types}

    for matrix_size in tqdm.tqdm(matrix_sizes, desc='Benchmarking'):
        for matrix_type in matrix_types:
            _times = [benchmark(matrix_size, matrix_type)[0] for _ in range(5)]
            avg_time = np.mean(_times)
            times[matrix_type].append(avg_time)
    
    plot_results(matrix_sizes, times)

def plot_results(matrix_sizes, times):
    """Plot the benchmark results."""
    for matrix_type, timings in times.items():
        plt.plot(matrix_sizes, timings, label=matrix_type)
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
