from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
import numpy as np
import pybind11

class build_ext(build_ext_orig):
    def finalize_options(self):
        super().finalize_options()
        # Se establece la carpeta 'lib' como destino para la librería compilada
        self.build_lib = "lib"

module_secuential = [
    Extension(
        'utils',
        sources=['utils.cpp'],
        include_dirs=[np.get_include(), pybind11.get_include()],
        extra_compile_args=["-O3", "-fopenmp", "-march=native"],
        extra_link_args=["-O3", "-fopenmp", "-march=native"],
        language='c++'
    )
]

setup(
    name='utils',
    version='0.1',
    install_requires=['numpy', 'pybind11'],
    ext_modules=module_secuential,
)