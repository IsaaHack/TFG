from abc import ABC, abstractmethod
from executers.executer import SingleCoreExecuter, MultiCoreExecuter, GpuExecuter, HybridExecuter

class Algoritm(ABC):
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
                self.executer.benchmark(problem.generate_solution(10))
            else:
                raise ValueError(f"Unknown executer type: {executer_type} available: single, multi, gpu, hybrid")
        else:
            if not isinstance(executer, (SingleCoreExecuter, MultiCoreExecuter, GpuExecuter, HybridExecuter)):
                raise ValueError("Invalid executer type provided.")
            self.executer = executer

        self.required_methods = required_methods
        self.check_required_methods()

    def check_required_methods(self):
        for method in self.required_methods:
            if not hasattr(self.problem, method):
                raise ValueError(f"Problem class must implement the {method} method.")

    @abstractmethod
    def fit(self):
        pass