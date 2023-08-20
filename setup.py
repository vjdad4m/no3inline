import argparse

import torch
from Cython.Build import cythonize
from setuptools import setup
from torch.utils import cpp_extension

parser = argparse.ArgumentParser(description="Build options")
parser.add_argument('--with-cuda', action='store_true', default=False, help="Compile with CUDA submodule")
args, unknown = parser.parse_known_args()

ext_modules_list = []

if args.with_cuda or torch.cuda.is_available():
    ext_modules_list.append(
        cpp_extension.CppExtension('collinear_cuda', ['no3inline/collinear_kernel.cu', 'no3inline/collinear.cpp'],
                                   extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    )

ext_modules_list.extend(cythonize("no3inline/wrapper.pyx"))

setup(
    name="no3inline",
    packages=["no3inline"],
    ext_modules=ext_modules_list,
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
