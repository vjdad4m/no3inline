import argparse

import torch
from Cython.Build import cythonize
from setuptools import setup
from torch.utils import cpp_extension

parser = argparse.ArgumentParser(description="Build options")
parser.add_argument('--with-cuda', action='store_true', default=False, help="Explicitly request to compile with CUDA submodule")
parser.add_argument('--no-cuda', action='store_true', default=False, help="Explicitly request to not compile with CUDA submodule")
args, unknown = parser.parse_known_args()

ext_modules_list = []

# Include CUDA only if --with-cuda is given or CUDA is available, unless --no-cuda is explicitly provided
if (args.with_cuda or torch.cuda.is_available()) and not args.no_cuda:
    ext_modules_list.append(
        cpp_extension.CppExtension('no3inline._collinear_cuda', ['no3inline/collinear_kernel.cu', 'no3inline/collinear.cpp'],
                                   extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    )
    ext_modules_list.append(
        cpp_extension.CppExtension('no3inline._find_3_in_line', ['no3inline/find_3_in_line.cu'],
                                   extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    )

ext_modules_list.extend(cythonize("no3inline/wrapper.pyx"))

ext_modules_list.extend(cythonize("no3inline/cpp_wrapper.pyx"))

setup(
    name="no3inline",
    packages=["no3inline"],
    ext_modules=ext_modules_list,
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
