import problems.problem as problem
import utils, utils_omp, utils_gpu
import cupy as cp
import numpy as np

SQRT_03 = np.sqrt(0.3)

class ClasProblem(problem.Problem):
    def __init__(self, X, Y, threshold=0.1):
        self.X_gpu = cp.asarray(X, dtype=cp.float32, order='C')
        self.Y_gpu = cp.asarray(Y, dtype=cp.int32, order='C')
        self.X = X
        self.Y = Y
        self.n_samples = len(X)
        self.n_features = len(X[0])
        self.threshold = threshold

        cp.cuda.Device().synchronize()

    def generate_solution(self, num_samples=1):
        if num_samples == 1:
            return np.random.uniform(0, 1, size=(self.n_features)).astype(np.float32)
        else:
            return np.random.uniform(0, 1, size=(num_samples, self.n_features)).astype(np.float32)

    def fitness(self, solutions):
        if len(solutions.shape) == 1:
            return utils.fitness(solutions, self.X, self.Y)
        else:
            return np.array([utils.fitness(solution, self.X, self.Y) for solution in solutions])
    
    def fitness_omp(self, solutions):
        if len(solutions.shape) == 1:
            return utils_omp.fitness_omp(solutions, self.X, self.Y)
        else:
            return np.array([utils_omp.fitness_omp(solution, self.X, self.Y) for solution in solutions])
    
    def fitness_gpu(self, solutions):
        solutions_gpu = cp.asarray(solutions, dtype=cp.float32, order='C')
        cp.cuda.Device().synchronize()  # Sincroniza el dispositivo antes de llamar a la función CUDA

        if len(solutions.shape) == 1:
            return utils_gpu.fitness_cuda(
                    utils_gpu.create_capsule(solutions_gpu.data.ptr),
                    utils_gpu.create_capsule(self.X_gpu.data.ptr),
                    utils_gpu.create_capsule(self.Y_gpu.data.ptr),
                    self.n_samples,
                    self.n_features
            )
        else:
            return np.array([utils_gpu.fitness_cuda(
                    utils_gpu.create_capsule(solutions_gpu[i].data.ptr),
                    utils_gpu.create_capsule(self.X_gpu.data.ptr),
                    utils_gpu.create_capsule(self.Y_gpu.data.ptr),
                    self.n_samples,
                    self.n_features
            ) for i in range(solutions.shape[0])])
    
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