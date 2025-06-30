from abc import ABC, abstractmethod
from executers import SingleCoreExecuter, MultiCoreExecuter, GpuExecuter, HybridExecuter
from time import time
import tqdm
import numpy as np
import cupy as cp

MESSAGE_TAG = 0
FINISH_TAG = 1

class Algorithm(ABC):
    '''Base class for optimization algorithms that operate on a given problem instance and support multiple execution strategies.
    This abstract class provides a framework for implementing optimization algorithms, handling the initialization of execution backends (single-core, multi-core, GPU, or hybrid), enforcing required problem methods, managing random seeds, and offering progress bar utilities for iterative or time-limited algorithms.

    Attributes:
        required_methods: List or set of method names that must be implemented by the problem.
        executer: The execution backend used to run the algorithm (single-core, multi-core, GPU, or hybrid).
        progress_bar: Progress bar object for tracking algorithm progress (initialized as None).
        print_mode: Mode for progress bar updates ('iterations' or 'timelimit').
        max_iter: Maximum number of iterations (if applicable).
        timelimit: Time limit in seconds (if applicable).
        start_time: Start time for time-limited progress tracking.
        seed: Random seed for reproducibility.
        
    Methods:
        __init__(problem, required_methods, executer_type='hybrid', executer=None):
            Raises ValueError if an unknown or invalid executer type is provided.
        init_seed(seed):
            Sets the random seed for both NumPy and CuPy random number generators for reproducibility.
        check_required_methods():
            Ensures that the problem instance implements all required methods.
            Raises ValueError if any required method is missing.
        print_init(time_start, iterations, timelimit):
            Initializes and configures the progress bar based on the stopping criterion (iterations or time limit).
            Raises ValueError for invalid stopping criteria.
        print_update(best_fit, n_iters=1):
        print_end():
        fit():
            Abstract method to be implemented by subclasses, defining the training or optimization logic.
    '''
    def __init__(self, problem, required_methods, executer_type='hybrid', executer=None):
        """
        Initializes the algorithm with the specified problem, required methods, and execution strategy.

        Args:
            problem: The problem instance to be solved.
            required_methods: A list or set of method names that must be implemented.
            executer_type (str, optional): The type of executer to use. Options are 'single', 'multi', 'gpu', or 'hybrid'. Defaults to 'hybrid'.
            executer (optional): An instance of an executer. If None, an executer is created based on `executer_type`.

        Raises:
            ValueError: If an unknown executer type is provided or if the provided executer is not a valid type.

        Notes:
            - If `executer` is None, an executer is instantiated based on `executer_type`.
            - For 'hybrid' executer type, a benchmark is performed using a generated solution.
            - Ensures all required methods are implemented.
        """
        self.problem = problem

        if executer is None:
            if executer_type == 'single':
                self.executer = SingleCoreExecuter(problem)
            elif executer_type == 'multi':
                self.executer = MultiCoreExecuter(problem)
            elif executer_type == 'gpu':
                self.executer = GpuExecuter(problem)
            elif executer_type == 'hybrid':
                self.executer = HybridExecuter(problem)
                self.executer.benchmark(problem.generate_solution(16))
            else:
                raise ValueError(f"Unknown executer type: {executer_type} available: single, multi, gpu, hybrid")
        else:
            if not isinstance(executer, (SingleCoreExecuter, MultiCoreExecuter, GpuExecuter, HybridExecuter)):
                raise ValueError("Invalid executer type provided.")
            self.executer = executer

        self.required_methods = required_methods
        self.check_required_methods()
        self.progress_bar = None

    def init_seed(self, seed):
        """
        Initializes the random seed for both NumPy and CuPy random number generators.

        Args:
            seed (int): The seed value to set for reproducibility.

        Sets:
            self.seed (int): Stores the provided seed value.
            np.random.seed: Sets the seed for NumPy's random number generator.
            cp.random.seed: Sets the seed for CuPy's random number generator.
        """
        self.seed = seed
        np.random.seed(self.seed)
        cp.random.seed(self.seed)

    def check_required_methods(self):
        """
        Checks whether the problem instance implements all required methods.

        Iterates over the list of required method names specified in `self.required_methods`
        and verifies that each method exists in the `self.problem` object. If any required
        method is missing, raises a ValueError indicating which method is not implemented.

        Raises:
            ValueError: If the problem class does not implement a required method.
        """
        for method in self.required_methods:
            if not hasattr(self.problem, method):
                raise ValueError(f"Problem class must implement the {method} method.")
            
    def print_init(self, time_start, iterations, timelimit):
        """
        Initializes and configures the progress bar for the algorithm based on the stopping criterion.

        Parameters:
            time_start (float): The starting time of the algorithm, used for time-based progress tracking.
            iterations (int or float): The maximum number of iterations to run. Use np.inf for unlimited iterations.
            timelimit (float): The time limit in seconds for the algorithm to run. Use np.inf for unlimited time.

        Raises:
            ValueError: If both iterations and timelimit are set to np.inf (unlimited).
            ValueError: If iterations is less than or equal to 0.
            ValueError: If timelimit is less than or equal to 0.

        Side Effects:
            Sets the progress bar mode ('timelimit' or 'iterations'), initializes the progress bar,
            and stores relevant parameters as instance attributes.
        """
        if iterations == np.inf and timelimit == np.inf:
            raise ValueError("Either iterations or timelimit must be set to a finite value.")
        if iterations <= 0:
            raise ValueError("Iterations must be greater than 0.")
        if timelimit is None:
            timelimit = np.inf
        if timelimit <= 0:
            raise ValueError("Timelimit must be greater than 0.")
        
        if iterations == np.inf:
            self.print_mode = 'timelimit'
        else:
            self.print_mode = 'iterations'
        
        self.max_iter = iterations
        self.timelimit = timelimit

        if self.print_mode == 'timelimit':
            self.start_time = time_start
            self.progress_bar = tqdm.tqdm(
                total=self.timelimit,
                desc="⏱ Time Progress",
                unit="s",
                bar_format="{l_bar}▕{bar}▏ Elapsed: {elapsed}, ETA: {remaining} {postfix}"
            )
        else:
            self.progress_bar = tqdm.tqdm(
                total=self.max_iter,
                desc="Iterations",
                unit="iter",
                leave=True,
            )

    def print_update(self, best_fit, n_iters=1):
        """
        Updates the progress bar and displays the current best fitness value.

        Depending on the print mode, either updates the progress bar based on elapsed time
        ('timelimit' mode) or by a specified number of iterations. Also sets a postfix string
        to display the best fitness value.

        Args:
            best_fit (float): The current best fitness value to display.
            n_iters (int, optional): Number of iterations to update the progress bar by (default is 1).
        """
        if self.progress_bar is not None:
            if self.print_mode == 'timelimit':
                elapsed_time = round(time() - self.start_time, 2)
                elapsed_time = min(elapsed_time, self.timelimit)
                self.progress_bar.n = elapsed_time
            else:
                self.progress_bar.update(n_iters)

            self.progress_bar.set_postfix_str(f"Best fitness: {best_fit:.4f}")

    def print_end(self):
        """
        Closes and resets the progress bar if it exists.

        This method checks if the `progress_bar` attribute is not `None`. If so, it closes the progress bar and sets the attribute to `None` to clean up resources.
        """
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    @abstractmethod
    def fit(self):
        """
        Trains or fits the model to the provided data.

        This method should be implemented by subclasses to define the training logic
        for the algorithm. It typically processes input data and adjusts internal
        parameters to optimize performance.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass