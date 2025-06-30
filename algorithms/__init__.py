"""
This package initializes and exposes the main algorithm classes and constants.

Exports:
    Algorithm (class): Base class for all algorithms.
    MESSAGE_TAG (str): Tag used for message identification.
    FINISH_TAG (str): Tag used to indicate completion.
    ACO (class): Ant Colony Optimization algorithm implementation.
    PSO (class): Particle Swarm Optimization algorithm implementation.
    GA (class): Genetic Algorithm implementation.
"""

from .algorithm import Algorithm, MESSAGE_TAG, FINISH_TAG
from .aco import ACO
from .pso import PSO
from .ga import GA