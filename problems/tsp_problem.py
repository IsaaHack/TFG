import problems.problem as problem
import utils, utils_gpu
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
            fitness_value = np.empty(1, dtype=np.float32)
        else:
            fitness_value = np.empty(solutions.shape[0], dtype=np.float32)
        
        utils.fitness_tsp_omp(self.distances, solutions, fitness_value)

        return fitness_value[0] if len(solutions.shape) == 1 else fitness_value
    
    def fitness_gpu(self, solutions):
        unique = False
        if len(solutions.shape) == 1:
            solutions = np.expand_dims(solutions, axis=0)
            unique = True

        solutions_gpu = cp.asarray(solutions, dtype=cp.int32, order='C')
        fitness_gpu = cp.empty(solutions.shape[0], dtype=cp.float32, order='C')
        solutions_capsule = utils_gpu.create_capsule(solutions_gpu.data.ptr)
        distances_capsule = utils_gpu.create_capsule(self.distances_gpu.data.ptr)
        fitness_capsule = utils_gpu.create_capsule(fitness_gpu.data.ptr)

        cp.cuda.Device().synchronize()

        utils_gpu.fitness_tsp_cuda(
                distances_capsule,
                solutions_capsule,
                fitness_capsule,
                self.n_cities,
                fitness_gpu.shape[0]
        )

        if unique:
            return cp.asnumpy(fitness_gpu)[0]
        else:
            return cp.asnumpy(fitness_gpu)
        
    def fitness_hybrid(self, solutions, speedup=1):
        if len(solutions.shape) == 1:
            fitness_values = np.empty(1, dtype=np.float32)
        else:
            fitness_values = np.empty(solutions.shape[0], dtype=np.float32)

        new_speedup = utils_gpu.fitness_tsp_hybrid(
                self. distances,
                utils_gpu.create_capsule(self.distances_gpu.data.ptr),
                solutions,
                fitness_values,
                self.n_cities,
                fitness_values.shape[0],
                speedup
        )

        return fitness_values[0] if len(solutions.shape) == 1 else fitness_values, new_speedup
    
    def crossover(self, population, crossover_rate):
        num_crossovers = int(np.floor(crossover_rate * len(population) / 2))

        random_starts = np.empty(num_crossovers, dtype=np.int32)
        random_ends = np.empty(num_crossovers, dtype=np.int32)
        for k in range(num_crossovers):
            # Seleccionar dos índices aleatorios sin reemplazo y ordenarlos (ya se requiere que estén sorted)
            random_starts[k], random_ends[k] = np.sort(np.random.choice(self.n_cities, size=2, replace=False))

        return utils.crossover_tsp(population, random_starts, random_ends)
    
    def crossover2(self, population, crossover_rate):
        num_crossovers = int(np.floor(crossover_rate * len(population) / 2))
        n_cities = self.n_cities
        in_segment = np.zeros(n_cities, dtype=bool)  # Preallocate boolean array for reuse

        for k in range(num_crossovers):
            parent1 = 2 * k
            parent2 = parent1 + 1

            # Randomly select and sort segment indices
            start, end = np.sort(np.random.choice(n_cities, size=2, replace=False))
            
            # Initialize children with empty arrays
            child1 = np.empty(n_cities, dtype=np.int32)
            child2 = np.empty(n_cities, dtype=np.int32)
            
            # Copy selected segment from parents
            segment1 = population[parent1, start:end+1]
            segment2 = population[parent2, start:end+1]
            child1[start:end+1] = segment1
            child2[start:end+1] = segment2
            
            # Calculate remaining indices (wrapping around)
            remaining_idx = np.concatenate((np.arange(end+1, n_cities), np.arange(0, start)))
            
            # Fill child1 with remaining genes from parent2
            in_segment.fill(False)
            in_segment[segment1] = True
            child1[remaining_idx] = population[parent2][~in_segment[population[parent2]]]
            
            # Fill child2 with remaining genes from parent1
            in_segment.fill(False)
            in_segment[segment2] = True
            child2[remaining_idx] = population[parent1][~in_segment[population[parent1]]]
            
            # Replace parents with children
            population[parent1] = child1
            population[parent2] = child2

        return population
    
    def mutation2(self, population, mutation_rate):
        n_individuals = population.shape[0]
        estimated_mutations = int(mutation_rate * n_individuals)
        if estimated_mutations == 0:
            return
        
        # Seleccionar aleatoriamente los índices de individuos a mutar sin repetición
        individual_indices = np.random.choice(n_individuals, size=estimated_mutations, replace=False)

        # Generar índices para swap mutation:
        # Primer índice aleatorio
        indices1 = np.random.randint(0, self.n_cities, size=estimated_mutations)
        # Segundo índice: primero se genera un entero en [0, n_cities-1) y luego se ajusta:
        indices2 = np.random.randint(0, self.n_cities - 1, size=estimated_mutations)
        indices2 = np.where(indices2 >= indices1, indices2 + 1, indices2)

        return utils.mutation_tsp(population, individual_indices, indices1, indices2)

    def mutation(self, population, mutation_rate):
        n_individuals = population.shape[0]
        estimated_mutations = int(mutation_rate * n_individuals)
        if estimated_mutations == 0:
            return

        # Selecciona aleatoriamente los índices de individuos a mutar sin repetición
        individual_indices = np.random.choice(n_individuals, size=estimated_mutations, replace=False)
        
        # Genera índices aleatorios para swap mutation:
        # Se utiliza np.random.randint para generar el primer índice.
        indices1 = np.random.randint(0, self.n_cities, size=estimated_mutations)
        # Para el segundo índice, se generan números entre 0 y n_cities-2. 
        # Luego, si el índice generado es mayor o igual que el índice 1 correspondiente, se le suma 1 para evitar la repetición.
        indices2 = np.random.randint(0, self.n_cities - 1, size=estimated_mutations)
        indices2 = np.where(indices2 >= indices1, indices2 + 1, indices2)

        # Realiza el intercambio vectorizado
        temp = population[individual_indices, indices1].copy()
        population[individual_indices, indices1] = population[individual_indices, indices2]
        population[individual_indices, indices2] = temp
        
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
