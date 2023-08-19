cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)

cdef extern from "reward.c":
    int reward(int **m, int m_size)

def calculate_reward(m):
    cdef int m_size = len(m)
    cdef int **c_m = <int **>malloc(m_size * sizeof(int *))

    for i in range(m_size):
        c_m[i] = <int *>malloc(m_size * sizeof(int))
        for j in range(m_size):
            c_m[i][j] = m[i][j]

    result = reward(c_m, m_size)

    for i in range(m_size):
        free(c_m[i])
    free(c_m)

    return result