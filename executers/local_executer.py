'''
This module defines a set of executer classes for evaluating the fitness of populations in optimization problems using different computational resources.

Classes
-------
- LocalExecuter: Abstract base class for executers that operate on a given problem instance locally.
- SingleCoreExecuter: Evaluates population fitness using a single CPU core.
- MultiCoreExecuter: Evaluates population fitness in parallel using multiple CPU cores (OpenMP).
- GpuExecuter: Evaluates population fitness using GPU acceleration.
- HybridExecuter: Evaluates and benchmarks population fitness using a hybrid method, dynamically adjusting a speedup factor.

Constants
---------

- NUM_SAMPLES: Number of samples for hybrid fitness evaluation benchmarking.

Each executer class implements an `execute(population)` method, which delegates the fitness evaluation to the corresponding method of the associated problem instance.
'''

from abc import abstractmethod
from . import Executer

# Number of samples for hybrid fitness evaluation for benchmarking
NUM_SAMPLES = 5

class LocalExecuter(Executer):
    """
    LocalExecuter is an abstract base class that inherits from Executer and is designed to execute operations on a given problem instance locally.
    
    Attributes:
        problem: The problem instance to be solved or manipulated.

    Methods:
        execute(population):
            Abstract method to execute operations on the provided population. Must be implemented by subclasses.

    Args:
        problem: An instance representing the problem to be executed.

    Raises:
        NotImplementedError: If the execute method is not implemented in a subclass.
    """

    def __init__(self, problem):
        """
        Initializes the instance with the given problem.

        Args:
            problem: The problem instance or object to be associated with this executor.
        """
        self.problem = problem

    @abstractmethod
    def execute(self, population):
        """
        Executes an operation on the given population.

        Args:
            population: The population to be processed. The type and structure of the population
                should be specified in the implementing subclass or usage context.

        Returns:
            None

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """

        pass

class SingleCoreExecuter(LocalExecuter):
    """
    SingleCoreExecuter is a subclass of LocalExecuter that evaluates the fitness of a population using a single core.

    Methods
    -------
    execute(population):
        Evaluates the fitness of the given population by calling the problem's fitness function.

    Parameters
    ----------
    population : iterable
        The collection of individuals whose fitness will be evaluated.

    Returns
    -------
    fitness_values : Any
        The result of the problem's fitness function applied to the population.
    """

    def execute(self, population):
        """
        Evaluates the fitness of a given population using the associated problem's fitness function.

        Args:
            population: The population to be evaluated. The expected type and structure depend on the problem definition.

        Returns:
            The fitness value(s) of the provided population as computed by the problem's fitness function.
        """

        return self.problem.fitness(population)

class MultiCoreExecuter(LocalExecuter):
    """
    MultiCoreExecuter is a subclass of LocalExecuter that executes the evaluation of a population's fitness using multiple CPU cores.

    Methods
    -------
    execute(population):
        Evaluates the fitness of the given population in parallel using OpenMP-enabled fitness calculation.

    Parameters
    ----------
    population : iterable
        The collection of individuals whose fitness will be evaluated.

    Returns
    -------
    result
        The result of the parallel fitness evaluation as returned by `self.problem.fitness_omp(population)`.
    """

    def execute(self, population):
        """
        Evaluates the fitness of a given population using the problem's fitness_omp method.

        Args:
            population: The population to be evaluated. The expected type and structure depend on the problem definition.

        Returns:
            The result of the fitness_omp evaluation, typically a list or array of fitness values corresponding to the individuals in the population.
        """

        return self.problem.fitness_omp(population)

class GpuExecuter(LocalExecuter):
    """
    GpuExecuter is a subclass of LocalExecuter designed to evaluate the fitness of a population using GPU acceleration.

    Methods
    -------
    execute(population):
        Evaluates the fitness of the given population using the GPU-accelerated fitness function defined in self.problem.

        Parameters:
            population: The population of individuals to be evaluated.

        Returns:
            The fitness values computed by the GPU.
    """

    def execute(self, population):
        """
        Executes the fitness evaluation for a given population using GPU acceleration.

        Args:
            population: The population of individuals to be evaluated.

        Returns:
            The fitness values of the population as computed by the problem's GPU-based fitness function.
        """

        return self.problem.fitness_gpu(population)

class HybridExecuter(LocalExecuter):
    """
    HybridExecuter is a subclass of LocalExecuter designed to execute and benchmark a population using a hybrid fitness evaluation method.

    Attributes:
        problem: An object representing the optimization problem, expected to provide a `fitness_hybrid` method.
        s_gpu_omp (float): A parameter used to track and update the speedup factor for the hybrid fitness evaluation.

    Methods:
        __init__(problem):
            Initializes the HybridExecuter with the given problem and sets the initial speedup factor.
        benchmark(population):
            Benchmarks the hybrid fitness evaluation on the provided population for a fixed number of samples,
            updating the speedup factor accordingly.
        execute(population):
            Evaluates the fitness of the given population using the hybrid method and updates the speedup factor.
            Returns the computed fitness values.
    """

    def __init__(self, problem):
        """
        Initializes the instance with the given problem.

        Args:
            problem: The problem instance or configuration to be associated with this executor.

        Attributes:
            problem: Stores the provided problem instance.
            s_gpu_omp (int): Speedup factor for hybrid fitness evaluation, initialized to 1.
        """

        self.problem = problem
        self.s_gpu_omp = 1

    def benchmark(self, population):
        """
        Evaluates the fitness of a given population multiple times using a hybrid fitness function.
        For each sample in the predefined number of samples (NUM_SAMPLES), this method calls
        the `fitness_hybrid` function of the associated problem, passing the population and the
        current state of the speedup factor (`self.s_gpu_omp`).

        Args:
            population: The population to be evaluated for fitness.

        Side Effects:
            Updates the internal state variable `self.s_gpu_omp` after each fitness evaluation.
        """

        for _ in range(NUM_SAMPLES):
            _, self.s_gpu_omp = self.problem.fitness_hybrid(population, self.s_gpu_omp)

    def execute(self, population):
        """
        Evaluates the fitness of a given population using a hybrid fitness function.
        This method calls the `fitness_hybrid` function of the associated problem, passing the current
        population and the current speedup value (`self.s_gpu_omp`). It updates `self.s_gpu_omp` with the
        new value returned by the fitness function and returns the computed fitness values.

        Args:
            population (iterable): The population of individuals to evaluate.

        Returns:
            fitness_values (iterable): The fitness values computed for the given population.
        """

        fitness_values, self.s_gpu_omp = self.problem.fitness_hybrid(population, self.s_gpu_omp)

        #self.s_gpu_omp = (self.s_gpu_omp + new_speedup) / 2
        return fitness_values