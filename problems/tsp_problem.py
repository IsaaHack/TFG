import problems.problem as problem
import utils, utils_omp, utils_gpu
import cupy as cp

class TSPProblem(problem.Problem):
    def __init__(self, distances, blocking=True):
        self.distances = distances
        self.n_cities = len(distances)
        self.distances_gpu = cp.asarray(distances, dtype=cp.float32, blocking=blocking, order='C')

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
