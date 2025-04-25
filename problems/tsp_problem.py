import problems.problem as problem
import utils, utils_gpu
import cupy as cp
import numpy as np

# Raw kernel source
_raw_kernel_code = r"""
extern "C" __global__ void construct_kernels(
    const int nCities,
    const float* distances,
    const float* pheromonesAlpha,
    const float* heuristicMatrix,
    const float* randArr,
    int* solutions)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= gridDim.x * blockDim.x) return;

    extern __shared__ bool visited[];
    bool* v = visited + threadIdx.x * nCities;
    for (int i = 0; i < nCities; ++i) v[i] = false;

    int first = (int)(randArr[idx * nCities] * nCities);
    solutions[idx * nCities] = first;
    v[first] = true;

    for (int step = 1; step < nCities; ++step) {
        int prev = solutions[idx * nCities + step - 1];
        float sumProb = 0.0;
        for (int j = 0; j < nCities; ++j) {
            if (!v[j]) {
                sumProb += pheromonesAlpha[prev * nCities + j] *
                           heuristicMatrix[prev * nCities + j];
            }
        }
        float threshold = randArr[idx * nCities + step] * sumProb;
        float cumulative = 0.0;
        int nextCity = 0;
        for (int j = 0; j < nCities; ++j) {
            if (!v[j]) {
                cumulative += pheromonesAlpha[prev * nCities + j] *
                              heuristicMatrix[prev * nCities + j];
                if (cumulative >= threshold) {
                    nextCity = j;
                    break;
                }
            }
        }
        solutions[idx * nCities + step] = nextCity;
        v[nextCity] = true;
    }
}
"""

# Compile kernel globally
constructor_kernel_tsp = cp.RawKernel(_raw_kernel_code, 'construct_kernels')

class TSPProblem(problem.Problem):
    def __init__(self, distances):
        self.distances_gpu = cp.asarray(distances, dtype=cp.float32, order='C')
        self.distances_gpu = cp.ascontiguousarray(self.distances_gpu)
        self.distances = distances
        self.n_cities = len(distances)

        cp.cuda.Device().synchronize()

    def generate_solution(self, num_samples=1):
        if num_samples == 1:
            return np.random.permutation(self.n_cities).astype(np.int32)
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
    
    def fitness_gpu2(self, solutions):
        # Convertir a array de CuPy una sola vez
        solutions_gpu = cp.asarray(solutions, dtype=cp.int32)
        
        # Crear versión desplazada de las rutas (para calcular siguiente ciudad)
        shifted = cp.roll(solutions_gpu, shift=-1, axis=1)
        
        # Calcular todas las distancias entre ciudades consecutivas simultáneamente
        distances = self.distances_gpu[solutions_gpu, shifted]
        
        # Sumar las distancias por ruta (axis=1)
        total_distances = cp.sum(distances, axis=1)
        
        # Fitness = negativo de la distancia total
        return -total_distances.get()
    
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
        rng = np.random.default_rng(seed=np.random.randint(0, 2**32 - 1))

        # Generación vectorizada de puntos de inicio y fin (sin duplicados ni sorting)
        start = rng.integers(0, self.n_cities - 1, size=num_crossovers)
        remaining = self.n_cities - start - 1
        end = start + 1 + rng.integers(0, remaining, size=num_crossovers)

        return utils.crossover_tsp(population, start, end)
    
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

    def initialize_pheromones(self):
        pheromones = cp.ones((self.n_cities, self.n_cities), dtype=cp.float32)
        pheromones /= self.n_cities
        cp.fill_diagonal(pheromones, 0.0)  # Evitar feromonas en la diagonal
        return pheromones
    
    def construct_solutions2(self, colony_size, pheromones, alpha, beta):
        # Inicializar el arreglo de soluciones y el seguimiento de ciudades visitadas.
        solutions = np.empty((colony_size, self.n_cities), dtype=np.int32)
        # Seleccionar aleatoriamente la ciudad inicial para cada hormiga.
        # Se asume colony_size <= self.n_cities. Si no fuera el caso, se puede modificar la selección.
        solutions[:, 0] = np.random.choice(self.n_cities, colony_size, replace=False)
        visited = np.zeros((colony_size, self.n_cities), dtype=bool)
        visited[np.arange(colony_size), solutions[:, 0]] = True

        for step in range(1, self.n_cities):
            # La ciudad actual de cada hormiga
            current_cities = solutions[:, step - 1]
            
            # Extraer de forma vectorizada la feromona y la distancia para cada solución
            pheromone = np.power(pheromones[current_cities, :], alpha)  # (colony_size, n_cities)
            distance_segment = self.distances[current_cities, :].copy()  # (colony_size, n_cities)
            # Evitar división por cero en la diagonal (misma ciudad)
            distance_segment[distance_segment == 0] = 1e-10
            heuristic = np.power(1.0 / distance_segment, beta)
            
            # Calcular la probabilidad sin normalizar
            probabilities = pheromone * heuristic
            
            # Enmascarar las ciudades ya visitadas (asigna 0 a las posiciones ya visitadas)
            probabilities[visited] = 0.0
            
            # Manejo de filas con suma cero
            row_sums = probabilities.sum(axis=1)
            zero_rows = row_sums == 0
            if np.any(zero_rows):
                for idx in np.where(zero_rows)[0]:
                    unvisited = ~visited[idx]
                    probabilities[idx, unvisited] = 1.0 / np.count_nonzero(unvisited)
                row_sums = probabilities.sum(axis=1)
            
            # Normalizar las probabilidades
            probabilities = probabilities / row_sums[:, np.newaxis]
            
            # Selección vectorizada de la siguiente ciudad usando cumsum.
            cumulative_probs = np.cumsum(probabilities, axis=1)
            random_nums = np.random.rand(colony_size, 1)  # Una muestra por cada hormiga
            # Para cada hormiga se elige el índice (ciudad) donde el valor acumulado supera el número aleatorio.
            next_cities = (cumulative_probs >= random_nums).argmax(axis=1)
            
            solutions[:, step] = next_cities
            visited[np.arange(colony_size), next_cities] = True

        return solutions
    
    def construct_solutions3(self, colony_size, pheromones, alpha, beta):
        n_cities = self.n_cities
        solutions = np.empty((colony_size, n_cities), dtype=np.int32)
        solutions[:, 0] = np.random.choice(n_cities, colony_size, replace=False)
        
        # Precomputación de matrices (solo una vez)
        if not hasattr(self, 'heuristic_matrix'):
            inv_dist = np.divide(1.0, self.distances, 
                            out=np.full_like(self.distances, 1e10, dtype=np.float64),
                            where=self.distances != 0)
            self.heuristic_matrix = np.power(inv_dist, beta, dtype=np.float64)
        
        # Buffers preasignados
        visited = np.zeros((colony_size, n_cities), dtype=np.uint8)
        visited[np.arange(colony_size), solutions[:, 0]] = 1
        prob_buffer = np.zeros((colony_size, n_cities), dtype=np.float64)
        cum_buffer = np.zeros_like(prob_buffer)
        
        # Máscaras y constantes
        #ones = np.ones((colony_size, 1), dtype=np.float32)
        epsilon = np.finfo(np.float64).eps
        
        for step in range(1, n_cities):
            current = solutions[:, step-1]
            
            # Cálculo vectorizado sin loops
            np.power(pheromones[current], alpha, out=prob_buffer)
            prob_buffer *= self.heuristic_matrix[current]
            np.multiply(prob_buffer, 1 - visited, out=prob_buffer)
            
            # Manejo de ceros con operaciones matriciales
            row_sums = prob_buffer.sum(axis=1, keepdims=True)
            np.divide(prob_buffer, row_sums + epsilon, out=prob_buffer, where=row_sums > 0)
            np.multiply(prob_buffer, (row_sums > 0).astype(np.float64), out=prob_buffer)
            
            # Selección por búsqueda binaria vectorizada
            np.cumsum(prob_buffer, axis=1, out=cum_buffer)
            rand = np.random.rand(colony_size, 1).astype(np.float64)
            next_cities = (cum_buffer >= rand).argmax(axis=1)
            
            # Actualización sin loops usando índices mágicos
            idx = np.arange(colony_size)
            solutions[:, step] = next_cities
            visited[idx, next_cities] = 1

        return solutions
    
    def construct_solutions4(self, colony_size, pheromones, alpha, beta, out=None):
        n_cities = self.n_cities
        if out is not None:
            solutions = out
        else:
            solutions = np.empty((colony_size, n_cities), dtype=np.int32)
        solutions[:, 0] = np.random.choice(n_cities, colony_size, replace=False)
        
        # Precompute heuristic_matrix with current beta, checking for changes
        if not hasattr(self, 'heuristic_beta') or self.heuristic_beta != beta:
            inv_dist = np.divide(1.0, self.distances, 
                            out=np.full_like(self.distances, 1e10, dtype=np.float64),
                            where=self.distances != 0)
            if beta == 1:
                self.heuristic_matrix = inv_dist
            else:
                self.heuristic_matrix = np.power(inv_dist, beta, dtype=np.float64)
            self.heuristic_beta = beta  # Track the beta used
        
        # Buffers preallocados
        visited = np.zeros((colony_size, n_cities), dtype=np.uint8)
        visited[np.arange(colony_size), solutions[:, 0]] = 1
        prob_buffer = np.zeros((colony_size, n_cities), dtype=np.float64)
        cum_buffer = np.zeros_like(prob_buffer)
        mask_buffer = np.zeros_like(prob_buffer)  # Para (1 - visited) como float64
        epsilon = np.finfo(np.float64).eps
        
        # Precalcular todos los números aleatorios
        rand_matrix = np.random.rand(colony_size, n_cities - 1)
        
        for step in range(1, n_cities):
            current = solutions[:, step-1]
            rand = rand_matrix[:, step-1].reshape(-1, 1)  # Formato (colony_size, 1)
            
            # Calcular prob_buffer
            current_pheromones = pheromones[current]
            np.power(current_pheromones, alpha, out=prob_buffer)
            prob_buffer *= self.heuristic_matrix[current]
            
            # Aplicar máscara usando buffer preasignado
            np.subtract(1, visited, out=mask_buffer)
            prob_buffer *= mask_buffer
            
            # Normalizar
            row_sums = prob_buffer.sum(axis=1, keepdims=True)
            np.divide(prob_buffer, row_sums + epsilon, out=prob_buffer, where=row_sums > 0)
            np.multiply(prob_buffer, (row_sums > 0).astype(np.float64), out=prob_buffer)
            
            # Seleccionar siguiente ciudad
            np.cumsum(prob_buffer, axis=1, out=cum_buffer)
            next_cities = (cum_buffer >= rand).argmax(axis=1)
            
            # Actualizar soluciones y visitados
            solutions[:, step] = next_cities
            visited[np.arange(colony_size), next_cities] = 1

        return solutions
    
    def construct_solutions(self, colony_size, pheromones, alpha, beta, out=None):
        n = self.n_cities

        # allocate GPU buffer for solutions
        gpu_solutions = cp.empty((colony_size, n), dtype=cp.int32)

        # Compute pheromones^alpha each call to reflect updates
        pheromones_alpha = cp.asarray(pheromones, dtype=cp.float32)
        pheromones_alpha = cp.power(pheromones_alpha, alpha)

        # precompute heuristicMatrix
        if not hasattr(self, 'heuristic_beta') or self.heuristic_beta != beta:
            inv_dist = cp.where(self.distances_gpu != 0,
                                1.0 / self.distances_gpu,
                                cp.finfo(cp.float32).max)
            self.heuristic_matrix = cp.power(inv_dist, beta, dtype=cp.float32)
            self.heuristic_beta = beta

        # random thresholds
        rand = cp.random.random((colony_size, n), dtype=cp.float32)

        threads = 32
        blocks = (colony_size + threads - 1) // threads
        shared_mem = threads * n * cp.dtype(cp.bool_).itemsize

        # Launch global kernel
        constructor_kernel_tsp(
            (blocks,), (threads,),
            (
                n,
                self.distances_gpu.ravel(),
                pheromones_alpha.ravel(),
                self.heuristic_matrix.ravel(),
                rand.ravel(),
                gpu_solutions.ravel(),
            ),
            shared_mem=shared_mem
        )

        # transfer results back to CPU numpy array
        result = cp.asnumpy(gpu_solutions)
        if out is not None:
            out[:] = result
            return out
        return result
    
    def construct_solutions777(self, colony_size, pheromones, alpha, beta, out=None):
        rng = np.random.default_rng(seed=np.random.randint(0, 2**32 - 1))

        if out is not None:
            solutions = out
        else:
            solutions = np.empty((colony_size, self.n_cities), dtype=np.int32)

        solutions[:, 0] = rng.choice(self.n_cities, colony_size)
        
        # Precompute heuristic_matrix with current beta, checking for changes
        if not hasattr(self, 'heuristic_beta') or self.heuristic_beta != beta:
            inv_dist = np.divide(1.0, self.distances, 
                            out=np.full_like(self.distances, 1e10, dtype=np.float64),
                            where=self.distances != 0)
            if beta == 1:
                self.heuristic_matrix = inv_dist
            else:
                self.heuristic_matrix = np.power(inv_dist, beta, dtype=np.float64)
            self.heuristic_beta = beta  # Track the beta used
        
        visited = np.zeros((colony_size, self.n_cities), dtype=np.uint8)
        visited[np.arange(colony_size), solutions[:, 0]] = 1
        
        # Precalcular todos los números aleatorios
        rand_matrix = rng.random((colony_size, self.n_cities - 1))

        pheromones_np = pheromones.get()

        utils.construct_solutions_tsp_inner(
            solutions,
            visited,
            pheromones_np,
            self.heuristic_matrix,
            rand_matrix,
            colony_size,
            self.n_cities,
            alpha,
            np.finfo(np.float64).eps
        )
    
    def update_pheromones(self, pheromones, colony, fitness_values, evaporation_rate):
        # Evaporación global: se reduce la feromona
        pheromones *= (1 - evaporation_rate)

        colony_gpu = cp.asarray(colony, dtype=cp.int32)
        
        # Extraer los índices de ciudades consecutivas:
        # city_from tendrá la primera parte de cada tour y city_to la segunda.
        city_from = colony_gpu[:, :-1]  # forma: (n_soluciones, n_cities-1)
        city_to = colony_gpu[:, 1:]     # forma: (n_soluciones, n_cities-1)
        
        # Calcular la contribución para cada solución 
        # (se usa broadcasting para dividir cada fila por el fitness correspondiente)
        deposit = -1.0 / fitness_values[:, None]
        
        # Actualizamos la feromona en ambas direcciones utilizando np.add.at para manejar índices repetidos
        cp.add.at(pheromones, (city_from, city_to), deposit)
        cp.add.at(pheromones, (city_to, city_from), deposit)
        
        return pheromones
    
    def update_pheromones2(self, pheromones, colony, fitness_values, evaporation_rate):
        return utils.update_pheromones_tsp(
            pheromones,
            colony,
            fitness_values,
            evaporation_rate
        )
    
    def reset_pheromones(self, pheromones):
        pheromones *= 0
        pheromones += 1 / self.n_cities
        cp.fill_diagonal(pheromones, 0.0)
        return pheromones
