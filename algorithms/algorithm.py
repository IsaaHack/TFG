from abc import ABC, abstractmethod
from executers import SingleCoreExecuter, MultiCoreExecuter, GpuExecuter, HybridExecuter
from time import time
import tqdm
import numpy as np
import cupy as cp

class Algorithm(ABC):
    def __init__(self, problem, required_methods, executer_type='hybrid', executer=None):
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
        self.seed = seed
        np.random.seed(self.seed)
        cp.random.seed(self.seed)

    def check_required_methods(self):
        for method in self.required_methods:
            if not hasattr(self.problem, method):
                raise ValueError(f"Problem class must implement the {method} method.")
            
    def print_init(self, time_start, iterations, timelimit):
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
        if self.progress_bar is not None:
            if self.print_mode == 'timelimit':
                elapsed_time = round(time() - self.start_time, 2)
                elapsed_time = min(elapsed_time, self.timelimit)
                self.progress_bar.n = elapsed_time
            else:
                self.progress_bar.update(n_iters)

            self.progress_bar.set_postfix_str(f"Best fitness: {best_fit:.4f}")

    def print_end(self):
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    @abstractmethod
    def fit(self):
        pass