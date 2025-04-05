import problems.problem as problem
import utils, utils_omp, utils_gpu
import cupy as cp
import numpy as np

SQRT_03 = np.sqrt(0.3)

class ClasProblem(problem.Problem):
    def __init__(self, X, Y, blocking=True):
        self.X = X
        self.Y = Y
        self.X_gpu = cp.asarray(X, dtype=cp.float32, blocking=blocking, order='C')
        self.Y_gpu = cp.asarray(Y, dtype=cp.int32, blocking=blocking, order='C')
        self.n_samples = len(X)
        self.n_features = len(X[0])

    def generate_solution(self, num_samples=1):
        if num_samples == 1:
            return np.random.uniform(0, 1, size=(self.n_features)).astype(np.float32)
        else:
            return np.random.uniform(0, 1, size=(num_samples, self.n_features)).astype(np.float32)

    def fitness(self, solution):
        return utils.fitness(solution, self.X, self.Y)
    
    def fitness_omp(self, solution):
        return utils_omp.fitness_omp(solution, self.X, self.Y)
    
    def fitness_gpu(self, solution):
        solution_gpu = cp.asarray(solution, dtype=cp.float32, blocking=True, order='C')

        return utils_gpu.fitness_cuda(
                utils_gpu.create_capsule(solution_gpu.data.ptr),
                utils_gpu.create_capsule(self.X_gpu.data.ptr),
                utils_gpu.create_capsule(self.Y_gpu.data.ptr),
                self.n_samples,
                self.n_features
        )
    
    def clas_rate(self, solution):
        return utils.clas_rate(solution, self.X, self.Y)
    
    def red_rate(self, solution):
        return utils.red_rate(solution)
    
    def crossover(self, population, crossover_rate, alpha : float = 0.3):
        estimated_crossovers : int = np.floor(crossover_rate * len(population) / 2).astype(int)

        for i in range(estimated_crossovers):
            parent1 : int = 2*i
            parent2 : int = parent1 + 1

            c_max : np.ndarray[float] = np.maximum(population[parent1], population[parent2])
            c_min : np.ndarray[float] = np.minimum(population[parent1], population[parent2])

            I : np.ndarray[float] = c_max - c_min

            population[parent1] = np.random.uniform(c_min - alpha * I, c_max + alpha * I)
            population[parent2] = np.random.uniform(c_min - alpha * I, c_max + alpha * I)

        population[:2*estimated_crossovers] = np.clip(population[:2*estimated_crossovers], 0, 1)


    def mutation(self, population, mutation_rate):
        estimated_mutations = int(mutation_rate * population.shape[0])

        mutation = np.random.normal(0, SQRT_03, estimated_mutations)
        genes_to_mutate = np.random.randint(0, population.shape[1], estimated_mutations)
        people_to_mutate = np.random.randint(0, population.shape[0], estimated_mutations)

        population[people_to_mutate, genes_to_mutate] = np.clip(population[people_to_mutate, genes_to_mutate] + mutation, 0, 1)