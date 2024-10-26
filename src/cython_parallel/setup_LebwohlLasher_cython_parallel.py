from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy


ext_modules = [
    Extension(
        "LebwohlLasher_cython_parallel",
        ["LebwohlLasher_cython_parallel.pyx"],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp'],
        include_dirs=[numpy.get_include()],
        language="c++"
    )
]

setup(
    name='LebwohlLasher_cython_parallel',
    ext_modules=cythonize(ext_modules),
)