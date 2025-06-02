from abc import ABC, abstractmethod
from . import Executer

NUM_SAMPLES = 5

class LocalExecuter(Executer):
    def __init__(self, problem):
        self.problem = problem

    @abstractmethod
    def execute(self, population):
        pass

class SingleCoreExecuter(LocalExecuter):
    def execute(self, population):
        return self.problem.fitness(population)

class MultiCoreExecuter(LocalExecuter):
    def execute(self, population):
        return self.problem.fitness_omp(population)

class GpuExecuter(LocalExecuter):
    def execute(self, population):
        return self.problem.fitness_gpu(population)

class HybridExecuter(LocalExecuter):
    def __init__(self, problem):
        self.problem = problem
        self.s_gpu_omp = 1

    def benchmark(self, population):
        for _ in range(NUM_SAMPLES):
            _, self.s_gpu_omp = self.problem.fitness_hybrid(population, self.s_gpu_omp)

    def execute(self, population):
        fitness_values, self.s_gpu_omp = self.problem.fitness_hybrid(population, self.s_gpu_omp)

        #self.s_gpu_omp = (self.s_gpu_omp + new_speedup) / 2
        return fitness_values