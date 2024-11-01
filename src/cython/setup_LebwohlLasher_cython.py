from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy


ext_modules = [
    Extension(
        "LebwohlLasher_cython",
        ["LebwohlLasher_cython.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++"
    )
]

setup(
    name='LebwohlLasher_cython',
    ext_modules=cythonize(ext_modules),
)