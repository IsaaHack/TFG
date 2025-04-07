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

        X_capsule = utils_gpu.create_capsule(self.X_gpu.data.ptr)
        Y_capsule = utils_gpu.create_capsule(self.Y_gpu.data.ptr)
        
        cp.cuda.Device().synchronize()  # Sincroniza el dispositivo antes de llamar a la función CUDA

        if len(solutions.shape) == 1:
            return utils_gpu.fitness_cuda(
                    utils_gpu.create_capsule(solutions_gpu.data.ptr),
                    X_capsule,
                    Y_capsule,
                    self.n_samples,
                    self.n_features
            )
        else:
            return np.array([utils_gpu.fitness_cuda(
                    utils_gpu.create_capsule(solutions_gpu[i].data.ptr),
                    X_capsule,
                    Y_capsule,
                    self.n_samples,
                    self.n_features
            ) for i in range(solutions.shape[0])])
    
    def clas_rate(self, solution):
        return utils.clas_rate(solution, self.X, self.Y)
    
    def red_rate(self, solution):
        return utils.red_rate(solution)
    
    def crossover(self, population, crossover_rate, alpha: float = 0.3):
        n_pairs = int(np.floor(crossover_rate * len(population) / 2))
        if n_pairs == 0:
            return

        # Índices de los padres en parejas consecutivas
        idx_even = np.arange(0, 2 * n_pairs, 2)
        idx_odd = np.arange(1, 2 * n_pairs, 2)

        # Extrae los padres en forma vectorizada
        parents1 = population[idx_even]
        parents2 = population[idx_odd]

        # Calcula c_max, c_min e I para cada par de padres
        c_max = np.maximum(parents1, parents2)
        c_min = np.minimum(parents1, parents2)
        I = c_max - c_min

        # Genera dos conjuntos de descendientes utilizando la distribución uniforme
        offspring1 = np.random.uniform(c_min - alpha * I, c_max + alpha * I)
        offspring2 = np.random.uniform(c_min - alpha * I, c_max + alpha * I)

        # Asigna los descendientes de vuelta a la población
        population[idx_even] = offspring1
        population[idx_odd] = offspring2

        # Asegura que todos los valores estén en el rango [0,1]
        population[:2 * n_pairs] = np.clip(population[:2 * n_pairs], 0, 1)



    def mutation(self, population, mutation_rate):
        estimated_mutations = int(mutation_rate * population.shape[0])

        mutation = np.random.normal(0, SQRT_03, estimated_mutations)
        genes_to_mutate = np.random.randint(0, population.shape[1], estimated_mutations)
        people_to_mutate = np.random.randint(0, population.shape[0], estimated_mutations)

        population[people_to_mutate, genes_to_mutate] = np.clip(population[people_to_mutate, genes_to_mutate] + mutation, 0, 1)