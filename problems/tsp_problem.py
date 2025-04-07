import problems.problem as problem
import utils, utils_omp, utils_gpu
import cupy as cp
import numpy as np

class TSPProblem(problem.Problem):
    def __init__(self, distances):
        self.distances_gpu = cp.asarray(distances, dtype=cp.float32, order='C')
        self.distances = distances
        self.n_cities = len(distances)

        cp.cuda.Device().synchronize()

    def generate_solution(self, num_samples=1):
        if num_samples == 1:
            return np.arange(self.n_cities).astype(np.int32)
        else:
            base_perm = np.tile(np.arange(self.n_cities), (num_samples, 1)).astype(np.int32)
            np.apply_along_axis(np.random.shuffle, 1, base_perm)
            return base_perm

    def fitness(self, solutions):
        if len(solutions.shape) == 1:
            return utils.fitness_tsp(self.distances, solutions)
        else:
            return np.array([utils.fitness_tsp(self.distances, solution) for solution in solutions])
    
    def fitness_omp(self, solutions):
        if len(solutions.shape) == 1:
            return utils_omp.fitness_tsp_omp(self.distances, solutions)
        else:
            return np.array([self.fitness_omp(solution) for solution in solutions])
    
    def fitness_gpu(self, solutions):
        unique = False
        if len(solutions.shape) == 1:
            solutions = np.expand_dims(solutions, axis=0)
            unique = True

        solutions_gpu = cp.asarray(solutions, dtype=cp.int32, order='C')
        fitness_gpu = cp.empty(solutions.shape[0], dtype=cp.float32, order='C')
        cp.cuda.Device().synchronize()

        utils_gpu.fitness_tsp_cuda(
                utils_gpu.create_capsule(self.distances_gpu.data.ptr),
                utils_gpu.create_capsule(solutions_gpu.data.ptr),
                utils_gpu.create_capsule(fitness_gpu.data.ptr),
                self.n_cities,
                fitness_gpu.shape[0]
        )

        if unique:
            return cp.asnumpy(fitness_gpu)[0]
        else:
            return cp.asnumpy(fitness_gpu)
    
    def crossover(self, population, crossover_rate):
        # Calcula el número de cruces a realizar (por parejas)
        estimated_crossovers = int(np.floor(crossover_rate * len(population) / 2))

        for k in range(estimated_crossovers):
            parent1 = 2 * k
            parent2 = parent1 + 1

            # Order Crossover (OX)
            start, end = np.sort(np.random.choice(self.n_cities, size=2, replace=False))
            child1 = np.full(self.n_cities, -1, dtype=np.int32)  # Inicializa el hijo con -1
            child2 = np.full(self.n_cities, -1, dtype=np.int32)  # Inicializa el hijo con -1

            # Copia el segmento seleccionado de cada padre
            child1[start:end+1] = population[parent1, start:end+1]
            child2[start:end+1] = population[parent2, start:end+1]

            p1_index = (end + 1) % self.n_cities
            p2_index = (end + 1) % self.n_cities

            # Rellena los hijos con las ciudades faltantes en el orden de aparición
            for j in range(self.n_cities):
                if population[parent2, j] not in child1:
                    child1[p1_index] = population[parent2, j]
                    p1_index = (p1_index + 1) % self.n_cities

                if population[parent1, j] not in child2:
                    child2[p2_index] = population[parent1, j]
                    p2_index = (p2_index + 1) % self.n_cities

            # Reemplaza los padres en la población por los nuevos hijos
            population[parent1] = child1
            population[parent2] = child2


    def mutation(self, population, mutation_rate):
        n_individuals = population.shape[0]
        # Calcula la cantidad de individuos a mutar
        estimated_mutations = int(mutation_rate * n_individuals)
        # Selecciona aleatoriamente los índices de individuos a mutar sin repetición
        individual_indices = np.random.choice(n_individuals, size=estimated_mutations, replace=False)
        
        for i in individual_indices:
            # Selecciona dos índices de ciudades aleatorios para swap mutation
            index1, index2 = np.random.choice(self.n_cities, size=2, replace=False)
            population[i][index1], population[i][index2] = population[i][index2], population[i][index1]
        
            # Swap mutation
            # population[i][index1] = np.random.randint(self.n_cities)
            # population[i][index2] = np.random.randint(self.n_cities)
            # Inversion mutation
            # start, end = sorted(np.random.choice(self.n_cities, size=2, replace=False))
            # population[i][start:end+1] = population[i][start:end+1][::-1]
            # Scramble mutation
            # start, end = sorted(np.random.choice(self.n_cities, size=2, replace=False))
            # temp = population[i][start:end+1].copy()
            # np.random.shuffle(temp)
            # population[i][start:end+1] = temp
