import problems.problem as problem
from . import utils, utils_gpu
import cupy as cp
import numpy as np

SQRT_03 = np.sqrt(0.3)
MIN_STD = 1e-3
MAX_STD = 0.25

class ClasProblem(problem.Problem):
    def __init__(self, X, Y, threshold=0.1, alpha=0.25):
        self.X_gpu = cp.asarray(X, dtype=cp.float32, order='C')
        self.Y_gpu = cp.asarray(Y, dtype=cp.int32, order='C')
        self.X = X
        self.Y = Y
        self.n_samples = len(X)
        self.n_features = len(X[0])
        self.threshold = threshold
        self.alpha = alpha

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
            fitness_values = np.empty(1, dtype=np.float32)
        else:
            fitness_values = np.empty(solutions.shape[0], dtype=np.float32)

        utils.fitness_omp(solutions, self.X, self.Y, fitness_values)

        return fitness_values[0] if len(solutions.shape) == 1 else fitness_values
    
    def fitness_gpu(self, solutions):
        solutions_gpu = cp.asarray(solutions, dtype=cp.float32, order='C')
        fitness_values = cp.empty(solutions_gpu.shape[0], dtype=cp.float32)

        cp.cuda.Device().synchronize()  # Sincroniza el dispositivo antes de llamar a la función CUDA

        # Vectorización de todas las operaciones
        mask = solutions_gpu >= self.threshold  # (n_solutions, n_features)
        sqrt_weights = cp.multiply(cp.sqrt(solutions_gpu), mask)  # (n_solutions, n_features)

        # Crear matriz 3D de características ponderadas (n_solutions, n_samples, n_features)
        X_all = cp.ascontiguousarray(self.X_gpu[None, :, :] * sqrt_weights[:, None, :])

        # Cálculo eficiente de todas las matrices de distancias
        G = cp.einsum('ijk,ilk->ijl', X_all, X_all)
        norms = cp.sum(cp.square(X_all), axis=2)  # (n_solutions, n_samples)
        D = norms[:, :, None] + norms[:, None, :] - 2 * G

        # Configurar diagonales a infinito
        n_samples = self.X_gpu.shape[0]
        D[:, cp.arange(n_samples), cp.arange(n_samples)] = cp.inf

        # Cálculo vectorizado de predicciones y métricas
        index_pred = cp.argmin(D, axis=2)  # Índices de vecinos más cercanos
        prediction_labels = self.Y_gpu[index_pred]  # Etiquetas predichas

        clas_rate = 100 * cp.mean(prediction_labels == self.Y_gpu, axis=1)  # Tasa de acierto
        red_rate = 100 * cp.sum(~mask, axis=1) / self.n_features  # Tasa de reducción

        # Cálculo final del fitness
        fitness_values = clas_rate * 0.75 + red_rate * 0.25
        return fitness_values.get()  # Convertir de cupy a numpy
    
    def fitness_gpu2(self, solutions):
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
        
    def fitness_hybrid(self, solutions, speedup=1):
        if len(solutions.shape) == 1:
            fitness_values = np.empty(1, dtype=np.float32)
        else:
            fitness_values = np.empty(solutions.shape[0], dtype=np.float32)

        # Llamar a la función de GPU
        new_speedup = utils_gpu.fitness_hybrid(
                solutions,
                self.X,
                self.Y,
                utils_gpu.create_capsule(self.X_gpu.data.ptr),
                utils_gpu.create_capsule(self.Y_gpu.data.ptr),
                fitness_values,
                self.n_samples,
                self.n_features,
                speedup
        )

        return fitness_values[0] if len(solutions.shape) == 1 else fitness_values, new_speedup
    
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

    def crossover2(self, population, crossover_rate, alpha: float = 0.3):
        # Calcular número de pares
        n_pairs = int(np.floor(crossover_rate * len(population) / 2))
        idx_even = np.arange(0, 2 * n_pairs, 2, dtype=np.int32)
        idx_odd = np.arange(1, 2 * n_pairs, 2, dtype=np.int32)

        # Generar valores aleatorios uniformes para cada descendiente
        rand_uniform1 = np.random.rand(n_pairs, population.shape[1]).astype(np.float32)
        rand_uniform2 = np.random.rand(n_pairs, population.shape[1]).astype(np.float32)

        return utils.crossover_blx(population, idx_even, idx_odd, rand_uniform1, rand_uniform2, alpha=alpha)

    def mutation(self, population, mutation_rate):
        estimated_mutations = int(mutation_rate * population.shape[0])

        mutation = np.random.normal(0, SQRT_03, estimated_mutations)
        genes_to_mutate = np.random.randint(0, population.shape[1], estimated_mutations)
        people_to_mutate = np.random.randint(0, population.shape[0], estimated_mutations)

        population[people_to_mutate, genes_to_mutate] = np.clip(population[people_to_mutate, genes_to_mutate] + mutation, 0, 1)

    def initialize_pheromones(self):
        means = np.random.uniform(0, 1, size=self.n_features).astype(np.float32)
        stds = np.full(self.n_features, 0.2).astype(np.float32)
        return means, stds
    
    def construct_solutions(self, colony_size, pheromones, alpha, beta, out=None):
        means, stds = pheromones

        # Ajustar medias usando alpha (control de explotación)
        adjusted_means = np.clip(means * (1.0 / (alpha + 1e-5)), 0.0, 1.0)
        
        # Ajustar desviaciones usando beta (control de exploración)
        adjusted_stds = np.clip(stds * (1.0 / (beta + 1e-5)), MIN_STD, MAX_STD)
        
        # Muestreo vectorizado con influencia de alpha en las medias
        solutions = np.random.normal(
            adjusted_means,
            adjusted_stds,
            size=(colony_size, self.n_features)
        ).astype(np.float32)
        
        solutions = np.clip(solutions, 0.0, 1.0)

        if out is not None:
            out[:] = solutions

        return solutions
    
    def update_pheromones(self, pheromones, colony, fitness_values, evaporation_rate):
        means, stds = pheromones
        fitness = np.array(fitness_values, dtype=np.float32)

        # Normalizar fitness para usar como pesos
        fitness = fitness - np.min(fitness)
        fitness = fitness / (np.max(fitness) + 1e-10)
        weights = 1.0 - fitness  # Mejor fitness → peso más alto
        weights /= np.sum(weights)

        # Calcular nueva media ponderada
        new_means = (1 - 0.9) * means + 0.1 * np.average(colony, axis=0, weights=weights)

        # Estimar desviación estándar con varianza ponderada (o usar la de la élite)
        elite_count = max(1, len(colony) // 5)
        elite = colony[np.argsort(fitness_values)[:elite_count]]
        elite_std = np.std(elite, axis=0)
        
        new_stds = (1 - evaporation_rate * 0.5) * stds + 0.5 * elite_std
        new_stds = np.clip(new_stds, MIN_STD, MAX_STD)

        return new_means, new_stds

    
    def reset_pheromones(self, pheromones):
        pheromones = self.initialize_pheromones()
        return pheromones
    

