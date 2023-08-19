from setuptools import setup
from Cython.Build import cythonize

setup(
    name = "no3inline",
    packages=["no3inline"],
    ext_modules = cythonize("no3inline/wrapper.pyx")
)