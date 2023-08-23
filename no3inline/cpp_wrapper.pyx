# distutils: language = c++
# distutils: sources = no3inline/observe-same_loc.cpp

cimport cython
from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "observe-same_loc.h":
    long long sameLocSum(const vector[bool]& flattened_grid, int dim1, int dim2, int dim3)

def calculate_same_loc_sum(grid):
    cdef vector[bool] c_grid
    cdef int dim1 = len(grid)
    cdef int dim2 = len(grid[0])
    cdef int dim3 = len(grid[0][0])
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                c_grid.push_back(grid[i][j][k])
    
    result = sameLocSum(c_grid, dim1, dim2, dim3)
    return result