import problems.problem as problem
import utils, utils_omp, utils_gpu
import cupy as cp
import numpy as np

class ClasProblem(problem.Problem):
    def __init__(self, X, Y, blocking=True):
        self.X = X
        self.Y = Y
        self.X_gpu = cp.asarray(X, dtype=cp.float32, blocking=blocking, order='C')
        self.Y_gpu = cp.asarray(Y, dtype=cp.int32, blocking=blocking, order='C')
        self.n_samples = len(X)
        self.n_features = len(X[0])

    def fitness(self, solution):
        return utils.fitness(solution, self.X, self.Y)
    
    def fitness_omp(self, solution):
        return utils_omp.fitness_omp(solution, self.X, self.Y)
    
    def fitness_gpu(self, solution):
        solution_gpu = cp.asarray(solution, dtype=cp.float32, blocking=True, order='C')

        utils_gpu.warmup()
        return utils_gpu.fitness_cuda(
                utils_gpu.create_capsule(solution_gpu.data.ptr),
                utils_gpu.create_capsule(self.X_gpu.data.ptr),
                utils_gpu.create_capsule(self.Y_gpu.data.ptr),
                self.n_samples,
                self.n_features
        )