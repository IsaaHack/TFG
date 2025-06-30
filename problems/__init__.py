"""
This module initializes the problems package and exposes key classes and utilities.

Imports:
    Problem, patch_problem (from .problem): Base class and patching utility for problem definitions.
    ClasProblem (from .clas_problem): Class for classification problems.
    TSPProblem (from .tsp_problem): Class for Traveling Salesman Problem instances.
    utils: General utility functions for problem handling.
    utils_gpu: GPU-specific utility functions for problem handling.
"""
from .problem import Problem, patch_problem
from .clas_problem import ClasProblem
from .tsp_problem import TSPProblem
from . import utils
from . import utils_gpu