'''
Module: problem

This module defines an abstract base class `Problem` for representing optimization or search problems,
providing a standardized interface for solution generation and fitness evaluation. It also includes
optional methods for leveraging different computational backends (OpenMP, GPU, Hybrid) for fitness
calculation. Additionally, the module provides a utility function `patch_problem` for dynamically
adding methods to instances of `Problem` or its subclasses.

Classes:
--------
    Problem (ABC): Abstract base class for defining optimization/search problems.
        - Methods:
            __init__(): Initializes environment variables for CUDA acceleration.
            generate_solution(num_samples=1): Abstract method to generate candidate solutions.
            fitness(solutions): Abstract method to evaluate fitness of solutions.
            fitness_omp(solutions): Optional, raises NotImplementedError by default.
            fitness_gpu(solutions): Optional, raises NotImplementedError by default.
            fitness_hybrid(solutions, speedup=1): Optional, raises NotImplementedError by default.

Functions:
    patch_problem(problem): Decorator factory to dynamically add methods to a Problem instance.
    - The `Problem` class is intended to be subclassed with concrete implementations for solution
      generation and fitness evaluation.
    - Environment variables for CUDA are set in the constructor to facilitate GPU acceleration.
    - The `patch_problem` function allows for runtime extension of Problem instances with new methods.
'''

from abc import ABC, abstractmethod
import os

class Problem(ABC):
    """
    Abstract base class for defining optimization or search problems.
    This class provides a template for problem definitions that require solution generation and evaluation (fitness).
    It also includes optional methods for evaluating solutions using different computational backends (OpenMP, GPU, Hybrid).

    Attributes:
        None

    Methods:
        generate_solution(num_samples=1):
            Abstract method to generate one or more candidate solutions for the problem.
        fitness(solutions):
            Abstract method to evaluate the fitness of one or more solutions.
        fitness_omp(solutions):
            Optional method to evaluate fitness using OpenMP parallelization.
            Raises NotImplementedError by default.
        fitness_gpu(solutions):
            Optional method to evaluate fitness using GPU acceleration.
            Raises NotImplementedError by default.
        fitness_hybrid(solutions, speedup=1):
            Optional method to evaluate fitness using a hybrid approach (e.g., CPU+GPU).
            Raises NotImplementedError by default.

    Notes:
        - Sets CUDA-related environment variables in the constructor to configure GPU acceleration.
        - Intended to be subclassed with concrete implementations for solution generation and fitness evaluation.
    """

    def __init__(self):
        """
        Initializes the object and sets environment variables to configure CUDA accelerators and enable TF32 support.

        Environment Variables Set:
            CUDA_ACCELERATOR: Specifies the CUDA accelerators to use ('cub', 'cutensor', 'cutensornet').
            CUDA_TF32: Enables TensorFloat-32 (TF32) computation on compatible hardware ('1').
        """

        os.environ['CUDA_ACCELERATOR'] = 'cub,cutensor,cutensornet'
        os.environ['CUDA_TF32'] = '1'

    @abstractmethod
    def generate_solution(self, num_samples=1):
        """
        Generates one or more solutions for the problem instance.

        Args:
            num_samples (int, optional): The number of solutions to generate. Defaults to 1.

        Returns:
            list or object: A list of generated solutions if num_samples > 1, otherwise a single solution object.

        Raises:
            NotImplementedError: If the method is not implemented.
        """

        pass

    @abstractmethod
    def fitness(self, solutions):
        """
        Calculates the fitness value(s) for the given solution(s).

        Args:
            solutions (Any): The solution or collection of solutions to evaluate. The type and structure depend on the problem context.

        Returns:
            Any: The computed fitness value(s) corresponding to the input solutions. The return type depends on the implementation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Note:
            This method should be overridden by subclasses to provide problem-specific fitness evaluation logic.
        """

        pass

    def fitness_omp(self, solutions):
        """
        Calculates the fitness of the given solutions using an OpenMP-based approach.

        Args:
            solutions (iterable): A collection of candidate solutions to be evaluated.

        Returns:
            float or list: The computed fitness value(s) for the provided solutions.

        Raises:
            NotImplementedError: If the method is not implemented.

        Note:
            This method is intended to leverage OpenMP for parallel computation of fitness values.
        """

        raise NotImplementedError("OpenMP fitness not implemented")

    def fitness_gpu(self, solutions):
        """
        Calculates the fitness of the given solutions using GPU acceleration.

        Args:
            solutions: An iterable or array-like object containing the solutions to evaluate.

        Returns:
            The fitness values corresponding to each solution.

        Raises:
            NotImplementedError: If the method is not implemented for GPU computation.
        """

        raise NotImplementedError("GPU fitness not implemented")
    
    def fitness_hybrid(self, solutions, speedup=1):
        """
        Calculates the fitness of a set of solutions using a hybrid evaluation method.

        Args:
            solutions (iterable): A collection of candidate solutions to be evaluated.
            speedup (float, optional): A factor to accelerate the fitness computation. Defaults to 1.

        Returns:
            float or list: The computed fitness value(s) for the provided solutions.

        Raises:
            NotImplementedError: If the method is not implemented.

        Note:
            This method should be implemented in subclasses to provide a hybrid fitness evaluation strategy.
        """

        raise NotImplementedError("Hybrid fitness not implemented")

def patch_problem(problem):
    """
    Adds a method to an instance of the Problem class or its subclasses.
    This function acts as a decorator factory that checks if the provided `problem` argument
    is an instance of the `Problem` class (or its subclasses). It returns a decorator that,
    when applied to a function, attaches that function as a method to the given `problem` instance.

    Args:
        problem: An instance of the Problem class or its subclasses to which the method will be added.

    Raises:
        TypeError: If `problem` is not an instance of Problem or its subclasses.
        TypeError: If the decorated object is not callable.
        
    Returns:
        A decorator that adds the decorated function as a method to the `problem` instance.
    """

    #Comprobar si el argumento forma parte de la clase Problem o de la clase hija
    if not isinstance(problem, Problem):
        raise TypeError("The argument must be an instance of the Problem class or its subclasses.")
    def add_method(func):
        if not callable(func):
            raise TypeError("The argument must be a callable function.")
        setattr(problem, func.__name__, func)
        return func
        