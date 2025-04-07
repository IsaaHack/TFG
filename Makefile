#!/bin/bash

.PHONY: check-python

# Intenta detectar python o python3, pero fuerza python3 si la versión es Python 2
RAW_PYTHON := $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)
PYTHON_VERSION_RAW := $(shell $(RAW_PYTHON) --version 2>&1 | cut -d ' ' -f 2)
PYTHON_MAJOR := $(shell echo $(PYTHON_VERSION_RAW) | cut -d '.' -f 1)

ifeq ($(PYTHON_MAJOR),2)
    PYTHON_CMD := python3
else
    PYTHON_CMD := $(RAW_PYTHON)
endif

PYTHON_VERSION := $(shell $(PYTHON_CMD) --version 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1-2)
PYTHON_INCLUDES := $(shell $(PYTHON_CMD) -m pybind11 --includes)
PYTHON_SUFFIX := $(shell python$(PYTHON_VERSION)-config --extension-suffix)

all: build

check-python:
	@echo "Versión de Python detectada: $(PYTHON_VERSION)"

build: cpp cuda
	

cpp: utils.cpp utils_omp.cpp
	@echo "Building C++ file"
	$(PYTHON_CMD) setup.py build_ext --inplace

cuda: utils_gpu.cu
	@echo "Building CUDA file"
	nvcc -O3 -shared -std=c++14 --compiler-options -fPIC --extended-lambda \
	$(PYTHON_INCLUDES) \
	utils_gpu.cu -o "utils_gpu$(PYTHON_SUFFIX)" \
	-lcudart -L/usr/local/cuda/lib64 -lcublas

clean:
	$(PYTHON_CMD) setup.py clean
	rm -f utils_gpu$(PYTHON_SUFFIX)