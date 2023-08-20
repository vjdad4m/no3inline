import no3inline.wrapper

def calculate_reward(m):
    return no3inline.wrapper.calculate_reward(m)

try:
    from no3inline.collinear_cuda import calculate_reward_cuda
except ImportError:
    def calculate_reward_cuda(*args, **kwargs):
        raise ImportError("The CUDA module was not compiled. Please recompile with CUDA support.")