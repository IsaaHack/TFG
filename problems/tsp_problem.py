import problems.problem as problem
import utils, utils_omp, utils_gpu
import cupy as cp
import numpy as np

class TSPProblem(problem.Problem):
    def __init__(self, distances, blocking=True):
        self.distances = distances
        self.n_cities = len(distances)
        self.distances_gpu = cp.asarray(distances, dtype=cp.float32, blocking=blocking, order='C')

    def generate_solution(self, num_samples=1):
        if num_samples == 1:
            return np.arange(self.n_cities).astype(np.int32)
        else:
            base_perm = np.tile(np.arange(self.n_cities), (num_samples, 1)).astype(np.int32)
            np.apply_along_axis(np.random.shuffle, 1, base_perm)
            return base_perm

    def fitness(self, solution):
        return utils.fitness_tsp(self.distances, solution)
    
    def fitness_omp(self, solution):
        return utils_omp.fitness_tsp_omp(self.distances, solution)
    
    def fitness_gpu(self, solution):
        solution_gpu = cp.asarray(solution, dtype=cp.int32, blocking=True, order='C')

        utils_gpu.warmup()
        return utils_gpu.fitness_tsp_cuda(
                utils_gpu.create_capsule(self.distances_gpu.data.ptr),
                utils_gpu.create_capsule(solution_gpu.data.ptr),
                self.n_cities
                )
