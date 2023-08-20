import no3inline.wrapper

def calculate_reward(m):
    return no3inline.wrapper.calculate_reward(m)

try:
    from .collinear_cuda import count_collinear
except ImportError:
    def count_collinear(*args, **kwargs):
        raise ImportError("The CUDA module was not compiled. Please recompile with CUDA support.")