from setuptools import setup
from Cython.Build import cythonize

setup(
  name = 'randpy1',
  ext_modules = cythonize("randpy.pyx", language="c++"),
)
