from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
import numpy as np
import pybind11

class build_ext(build_ext_orig):
    def finalize_options(self):
        super().finalize_options()
        # Se establece la carpeta 'lib' como destino para la librería compilada
        self.build_lib = "lib"

module1 = [Extension(
        'utils',
        sources=['utils.cpp'],
        include_dirs=[np.get_include(), pybind11.get_include()],
        extra_compile_args=["-O3"],
        language='c++'
    ),]

module2 = [Extension(
        "utils2",
        sources=["utils2.cpp"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
            "/usr/include",
            "/usr/local/include",
            "/usr/include/x86_64-linux-gnu/mpich",
        ],
        libraries=["mpi"],  # Asegurar que se enlace con MPI
        library_dirs=["/usr/include", "/usr/local/include"],  # Agregar estas rutas
        extra_compile_args=["-lmpi"],
        language="c++",
    )]

module3 = [
    Extension(
        "utils3",
        sources=["utils3.cpp"],
        include_dirs=[pybind11.get_include(), np.get_include()],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp", "-O3"], 
        language="c++"
    )
]

setup(
    name='utils',
    version='0.1',
    install_requires=['numpy', 'pybind11'],
    ext_modules=module1
)

setup(
    name='utils2',
    version='0.1',
    install_requires=['numpy', 'pybind11', 'mpi4py'],
    ext_modules=module2
)

setup(
    name='utils3',
    version='0.1',
    install_requires=['numpy', 'pybind11'],
    ext_modules=module3
)