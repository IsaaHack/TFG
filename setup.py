from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
import pybind11
import subprocess
import sys
import glob

class CustomBuildExt(build_ext):
    def run(self):
        # Ejecutar make antes de compilar extensiones
        try:
            subprocess.check_call(['make'])
        except subprocess.CalledProcessError:
            sys.exit("Error: la compilación con `make` falló")
        
        # Continuar con la construcción normal
        build_ext.run(self)

        so_files = glob.glob('./*.so')
        for so_file in so_files:
            subprocess.check_call(['mv', so_file, 'problems/'])

module_secuential = [
    Extension(
        'utils',
        sources=['./cpp/utils.cpp'],
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
    cmdclass={'build_ext': CustomBuildExt},
)