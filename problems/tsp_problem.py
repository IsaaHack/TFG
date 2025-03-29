import problems.problem as problem
import utils, utils2, utils3, utils_gpu
import os
import cupy as cp
import numpy as np

class TSPProblem(problem.Problem):
    def __init__(self, distances):
        self.distances = distances
        self.distances_np = np.asarray(distances, dtype=np.float32)
        self.n_cities = len(distances)
        self.distances_gpu = cp.asarray(distances, dtype=cp.float32, blocking=True, order='C')

    def fitness(self, solution):
        return utils.fitness_tsp(self.distances_np, np.asarray(solution, dtype=np.int32))
    
    def fitness_mpi(self, solution):
        return utils2.fitness_tsp_mpi(self.distances, solution)
    
    def fitness_omp(self, solution):
        os.environ["OMP_NUM_THREADS"] = "8"
        return utils3.fitness_tsp_omp(self.distances_np, np.asarray(solution, dtype=np.int32))
    
    def fitness_gpu(self, solution):
        utils_gpu.warmup()
        distances_ptr = self.distances_gpu.data.ptr
        solution_gpu = cp.asarray(solution, dtype=cp.int32, blocking=True, order='C')
        solution_ptr = solution_gpu.data.ptr

        utils_gpu.warmup()
        return utils_gpu.fitness_tsp_cuda(
                utils_gpu.create_capsule(distances_ptr),
                utils_gpu.create_capsule(solution_ptr),
                self.n_cities)
