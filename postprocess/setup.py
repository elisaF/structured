from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from glob import glob

extensions = [
  Extension(
    'mst',
    glob('mst.pyx')
    + glob('MaxSpanTree.cpp'),
    extra_compile_args=["-std=c++11", "-stdlib=libc++"])
]

setup(
  name='mst',
  ext_modules=cythonize(extensions))