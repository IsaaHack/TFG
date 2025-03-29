import problems.problem as problem
import utils, utils2, utils3, utils_gpu
import os
import cupy as cp

class ClasProblem(problem.Problem):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_gpu = cp.asarray(X, dtype=cp.float32, blocking=True, order='C')
        self.Y_gpu = cp.asarray(Y, dtype=cp.int32, blocking=True, order='C')
        self.n_samples = len(X)
        self.n_features = len(X[0])

    def fitness(self, solution):
        return utils.fitness(solution, self.X, self.Y)
    
    def fitness_mpi(self, solution):
        return utils2.fitness_mpi(solution, self.X, self.Y)
    
    def fitness_omp(self, solution):
        os.environ["OMP_NUM_THREADS"] = "8"
        return utils3.fitness_omp(solution, self.X, self.Y)
    
    def fitness_gpu(self, solution):
        os.environ["OMP_NUM_THREADS"] = "8"
        utils_gpu.warmup()
        solution_gpu = cp.asarray(solution, dtype=cp.float32, blocking=True, order='C')
        solution_ptr = solution_gpu.data.ptr

        return utils_gpu.fitness_cuda(
                utils_gpu.create_capsule(solution_ptr),
                utils_gpu.create_capsule(self.X_gpu.data.ptr),
                utils_gpu.create_capsule(self.Y_gpu.data.ptr),
                self.n_samples,
                self.n_features
        )