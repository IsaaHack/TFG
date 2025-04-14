from abc import ABC, abstractmethod
import time

NUM_SAMPLES = 10

class Executer(ABC):
    def __init__(self, problem):
        self.problem = problem

    @abstractmethod
    def execute(self, population):
        pass

class SingleCoreExecuter(Executer):
    def execute(self, population):
        self.problem.fitness(population)
        pass

class MultiCoreExecuter(Executer):
    def execute(self, population):
        return self.problem.fitness_omp(population)

class GpuExecuter(Executer):
    def execute(self, population):
        return self.problem.fitness_gpu(population)

class HybridExecuter(Executer):
    def __init__(self, problem):
        self.problem = problem
        self.s_gpu_omp = 1

    def benchmark(self, population):
        t = time.time()
        for i in range(NUM_SAMPLES):
            self.problem.fitness_omp(population)
        t = time.time() - t 
        time_omp = t / NUM_SAMPLES

        t = time.time()
        for i in range(NUM_SAMPLES):
            self.problem.fitness_gpu(population)
        t = time.time() - t
        time_gpu = t / NUM_SAMPLES
        self.s_gpu_omp = time_omp / time_gpu

    def execute(self, population):
        fitness_values, self.s_gpu_omp = self.problem.fitness_hybrid(population, self.s_gpu_omp)

        #self.s_gpu_omp = (self.s_gpu_omp + new_speedup) / 2
        return fitness_values