# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/13 20:56
# @Author  : 成昭炜

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(ext_modules=cythonize("compute_overlap.pyx"), include_dirs=[np.get_include()])
