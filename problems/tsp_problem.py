from . import Problem
from . import utils, utils_gpu
import cupy as cp
import numpy as np
import numba

# Raw kernel source
_raw_kernel_code = r"""
extern "C" __global__ void construct_kernels(
    const int nCities,
    const int colony_size,
    const float* distances,
    const float* probabilities,
    const float* randArr,
    int* solutions,
    bool* visited_global)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= colony_size) return;

    bool* visited = visited_global + idx * nCities;
    
    // Inicializar visited a falso
    for(int i = 0; i < nCities; ++i) visited[i] = false;
    
    // Seleccionar primera ciudad
    //int first = (int)(randArr[idx * nCities] * nCities);
    int first = 0;
    solutions[idx * nCities] = first;
    visited[first] = true;

    for(int step = 1; step < nCities; ++step) {
        int prev = solutions[idx * nCities + step - 1];
        float sumProb = 0.0f;
        
        // Calcular suma de probabilidades
        for(int j = 0; j < nCities; ++j) {
            if(!visited[j]) {
                sumProb += probabilities[prev * nCities + j];
            }
        }
        
        // Seleccionar ciudad
        float threshold = randArr[idx * nCities + step] * sumProb;
        float cumulative = 0.0f;
        int nextCity = -1;
        
        for(int j = 0; j < nCities; ++j) {
            if(!visited[j]) {
                cumulative += probabilities[prev * nCities + j];
                if(cumulative >= threshold) {
                    nextCity = j;
                    break;
                }
            }
        }
        
        // Actualizar solucion y visited
        solutions[idx * nCities + step] = nextCity;
        visited[nextCity] = true;
    }
}
"""

two_opt_kernel_src = r'''
extern "C" __global__
void two_opt_kernel(int* tours, float* distances,
                    int num_tours, int n) {
    int tour_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (tour_id >= num_tours) return;

    int* tour = tours + tour_id * n;
    bool improved = false;

    for(int i = 1; i < n - 2; ++i) {
        int a = tour[i - 1];
        int b = tour[i];

        float dist_ab = distances[a * n + b];
        for(int k = i + 1; k < n - 1; ++k) {
            int c = tour[k];
            int d = tour[k + 1];
            float delta = (dist_ab + distances[c * n + d])
                        - (distances[a * n + c] + distances[b * n + d]);
            if (delta > 1e-6f) {
                int left = i, right = k;
                while (left < right) {
                    int tmp       = tour[left];
                    tour[left++]  = tour[right];
                    tour[right--] = tmp;
                }
                improved = true;
                break;
            }
        }
        if (improved) break;
    }
}
'''

two_opt_kernel = cp.RawKernel(two_opt_kernel_src, 'two_opt_kernel')
constructor_kernel_tsp = cp.RawKernel(_raw_kernel_code, 'construct_kernels')

class TSPProblem(Problem):
    def __init__(self, distances, executer='gpu'):
        self.distances_gpu = cp.asarray(distances, dtype=cp.float32, order='C')
        self.distances_gpu = cp.ascontiguousarray(self.distances_gpu)
        self.distances = distances
        self.n_cities = len(distances)
        self.execution = executer

        cp.cuda.Device().synchronize()

        if executer not in ['single', 'multi', 'gpu']:
            raise ValueError(f"Unsupported execution type: {executer}. Supported types are 'single', 'multi', 'gpu'.")

        if executer == 'single':
            self.crossover = self.crossover_cpu
            self.construct_solutions = self.construct_solutions_cpu
            self.initialize_pheromones = self.initialize_pheromones_cpu
            self.update_pheromones = self.update_pheromones_cpu
            self.reset_pheromones = self.reset_pheromones_cpu
            self.update_velocity = self.update_velocity_cpu
            self.update_position = self.update_position_cpu
        elif executer == 'multi':
            self.construct_solutions = self.construct_solutions_cpu_multi
            self.initialize_pheromones = self.initialize_pheromones_cpu
            self.update_pheromones = self.update_pheromones_cpu
            self.reset_pheromones = self.reset_pheromones_cpu
            self.update_position = self.update_position_cpu_multi

    def generate_solution(self, num_samples=1):
        if num_samples == 1:
            # Fijar la ciudad 0 como punto de inicio
            rest = np.random.permutation(np.arange(1, self.n_cities)).astype(np.int32)
            return np.concatenate(([0], rest))
        else:
            # Generar múltiples permutaciones con ciudad 0 fija al inicio
            solutions = np.zeros((num_samples, self.n_cities), dtype=np.int32)
            for i in range(num_samples):
                rest = np.random.permutation(np.arange(1, self.n_cities)).astype(np.int32)
                solutions[i] = np.concatenate(([0], rest))
            return solutions

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
        rng = np.random.default_rng(seed=np.random.randint(0, 2**32 - 1))

        # Asegurar que start >= 1 para no modificar la ciudad 0
        start = rng.integers(1, self.n_cities - 1, size=num_crossovers)
        remaining = self.n_cities - start - 1
        end = start + 1 + rng.integers(0, remaining, size=num_crossovers)

        return utils.crossover_tsp(population, start, end)
    
    def crossover_cpu(self, population, crossover_rate):
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

    def mutation(self, population, mutation_rate):
        n_individuals = population.shape[0]
        estimated_mutations = int(mutation_rate * n_individuals)
        if estimated_mutations == 0:
            return

        # Selecciona aleatoriamente los índices de individuos a mutar sin repetición
        individual_indices = np.random.choice(n_individuals, size=estimated_mutations, replace=False)
        
        # Genera índices aleatorios para swap mutation:
        # Se utiliza np.random.randint para generar el primer índice.
        indices1 = np.random.randint(1, self.n_cities, size=estimated_mutations)
        # Para el segundo índice, se generan números entre 0 y n_cities-2. 
        # Luego, si el índice generado es mayor o igual que el índice 1 correspondiente, se le suma 1 para evitar la repetición.
        indices2 = np.random.randint(1, self.n_cities - 1, size=estimated_mutations)
        indices2 = np.where(indices2 >= indices1, indices2 + 1, indices2)

        # Realiza el intercambio vectorizado
        temp = population[individual_indices, indices1].copy()
        population[individual_indices, indices1] = population[individual_indices, indices2]
        population[individual_indices, indices2] = temp

    def initialize_pheromones(self):
        pheromones = cp.random.uniform(1/(self.n_cities*self.n_cities), 1/self.n_cities, (self.n_cities, self.n_cities), dtype=cp.float32)
        cp.fill_diagonal(pheromones, 0.0)  # Evitar feromonas en la diagonal
        return pheromones
    
    def initialize_pheromones_cpu(self):
        pheromones = np.random.uniform(1/(self.n_cities*self.n_cities), 1/self.n_cities, (self.n_cities, self.n_cities))
        np.fill_diagonal(pheromones, 0.0)  # Evitar feromonas en la diagonal
        return pheromones
    
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
            self.heuristic_matrix = cp.ascontiguousarray(self.heuristic_matrix)
            self.heuristic_beta = beta

        probabilites = cp.multiply(pheromones_alpha, self.heuristic_matrix)

        # random thresholds
        rand = cp.random.random((colony_size, n), dtype=cp.float32)

        # Crear buffer global para visited
        visited_global = cp.zeros((colony_size, n), dtype=cp.bool_)

        threads = 32
        blocks = (colony_size + threads - 1) // threads
        #shared_mem = threads * n * cp.dtype(cp.bool_).itemsize

        # Launch global kernel
        constructor_kernel_tsp(
            (blocks,), (threads,),
            (
                n,
                colony_size,
                self.distances_gpu.ravel(),
                probabilites.ravel(),
                rand.ravel(),
                gpu_solutions.ravel(),
                visited_global.ravel()
            ),
        )

        # transfer results back to CPU numpy array
        if out is not None:
            cp.asnumpy(gpu_solutions, out=out)
            # for i in range(colony_size):
            #     out[i] = self.two_opt(out[i],self.distances)
            return out
        return cp.asnumpy(gpu_solutions)
    
    def construct_solutions_cpu(self, colony_size, pheromones, alpha, beta, out=None):
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

        utils.construct_solutions_tsp_inner_cpu(
            solutions,
            visited,
            pheromones,
            self.heuristic_matrix,
            rand_matrix,
            colony_size,
            self.n_cities,
            alpha,
            np.finfo(np.float64).eps
        )
    
    def construct_solutions_cpu_multi(self, colony_size, pheromones, alpha, beta, out=None):
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

        utils.construct_solutions_tsp_inner(
            solutions,
            visited,
            pheromones,
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
    
    def update_pheromones_cpu(self, pheromones, colony, fitness_values, evaporation_rate):
        return utils.update_pheromones_tsp(
            pheromones,
            colony,
            fitness_values,
            evaporation_rate
        )
    
    def reset_pheromones(self, pheromones):
        pheromones = cp.random.uniform(1/(self.n_cities*self.n_cities), 1/self.n_cities, (self.n_cities, self.n_cities), dtype=cp.float32)
        cp.fill_diagonal(pheromones, 0.0)
        return pheromones
    
    def reset_pheromones_cpu(self, pheromones):
        pheromones = np.random.uniform(1/(self.n_cities*self.n_cities), 1/self.n_cities, (self.n_cities, self.n_cities))
        np.fill_diagonal(pheromones, 0.0)
        return pheromones
    
    def generate_velocity(self, num_samples=1):
        # Genera intercambios aleatorios iniciales basados en la longitud del tour
        n = self.n_cities
        velocities = []
        max_swaps = max(2, int(n**0.5))  # Límite basado en sqrt(n)
        for _ in range(num_samples):
            num_swaps = np.random.randint(1, max_swaps)
            swaps = []
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                swaps.append((i, j))
            velocities.append(swaps)
        return velocities
    
    @staticmethod
    @numba.jit(nopython=True)
    def _get_swap_sequence(from_tour, to_tour):
        seq = []
        ft = from_tour.copy()
        position_map = {city: idx for idx, city in enumerate(ft)}
        for idx in range(len(ft)):
            target_city = to_tour[idx]
            if ft[idx] == target_city:
                continue
            swap_idx = position_map[target_city]
            seq.append((idx, swap_idx))
            # Actualiza el mapa de posiciones
            position_map[ft[swap_idx]] = idx
            position_map[ft[idx]] = swap_idx
            # Realiza el intercambio
            ft[idx], ft[swap_idx] = ft[swap_idx], ft[idx]
        return seq

    def update_velocity(self, swarm, velocity, p_best, g_best, inertia_weight, cognitive_weight, social_weight):
        new_velocity = []

        max_swaps = np.maximum(2, int(self.n_cities**0.5))  # Límite basado en sqrt(n)
        for i, tour in enumerate(swarm):
            # Componente de inercia: parte de la velocidad anterior
            inertia_count = int(inertia_weight * len(velocity[i]))
            vel_inertia = velocity[i][:inertia_count]  # Tomar primeros 'inertia_count' intercambios
            
            # Componente cognitivo: intercambios hacia p_best[i]
            swaps_cog = utils.get_swap_sequence(tour, p_best[i])
            k1 = int(cognitive_weight * len(swaps_cog))
            vel_cognitive = swaps_cog[:k1]
            
            # Componente social: intercambios hacia g_best
            swaps_soc = utils.get_swap_sequence(tour, g_best)
            k2 = int(social_weight * len(swaps_soc))
            vel_social = swaps_soc[:k2]
            
            # Combinar componentes
            combined = vel_inertia + vel_cognitive + vel_social
            combined = list(set(combined))  # Eliminar duplicados
            np.random.shuffle(combined)
            new_velocity.append(combined[:np.minimum(len(combined), max_swaps)])
            #new_velocity.append(combined)

        return new_velocity
    
    def update_velocity_cpu(self, swarm, velocity, p_best, g_best, inertia_weight, cognitive_weight, social_weight):
        new_velocity = []

        max_swaps = np.maximum(2, int(self.n_cities**0.5))  # Límite basado en sqrt(n)
        for i, tour in enumerate(swarm):
            # Componente de inercia: parte de la velocidad anterior
            inertia_count = int(inertia_weight * len(velocity[i]))
            vel_inertia = velocity[i][:inertia_count]  # Tomar primeros 'inertia_count' intercambios
            
            # Componente cognitivo: intercambios hacia p_best[i]
            swaps_cog = TSPProblem._get_swap_sequence(tour, p_best[i])
            k1 = int(cognitive_weight * len(swaps_cog))
            vel_cognitive = swaps_cog[:k1]
            
            # Componente social: intercambios hacia g_best
            swaps_soc = TSPProblem._get_swap_sequence(tour, g_best)
            k2 = int(social_weight * len(swaps_soc))
            vel_social = swaps_soc[:k2]
            
            # Combinar componentes
            combined = vel_inertia + vel_cognitive + vel_social
            combined = list(set(combined))  # Eliminar duplicados
            np.random.shuffle(combined)
            new_velocity.append(combined[:np.minimum(len(combined), max_swaps)])
            #new_velocity.append(combined)

        return new_velocity
    
    @staticmethod
    @numba.jit(nopython=True)
    def two_opt_cpu(tours, distances):
        for tour in tours:
            n = tour.size
            improved = False
            for i in range(1, n - 2):
                a = tour[i - 1]
                b = tour[i]
                for k in range(i + 1, n - 1):
                    c = tour[k]
                    d = tour[k + 1]
                    # calculate cost difference
                    delta = (distances[a, b] + distances[c, d]) - (distances[a, c] + distances[b, d])
                    if delta > 1e-6:
                        # perform swap and exit
                        #tour[i:k + 1] = tour[i:k + 1][::-1]
                        #tour[i:k + 1] = np.flip(tour[i:k + 1])
                        start = i
                        end = k
                        while start < end:
                            tour[start], tour[end] = tour[end], tour[start]
                            start += 1
                            end -= 1
                        improved = True
                        break
                if improved:
                    break
        return tours
    
    @staticmethod
    def two_opt(tours, distances):
        tours_gpu = cp.asarray(tours, dtype=cp.int32,   order='C')

        num_tours, n = tours_gpu.shape

        # Un bloque por tour, 1 hilo por bloque
        threads = 256
        blocks = (num_tours + threads - 1) // threads

        # Itera el kernel hasta max_iters (o podrías parar temprano con _flags)
        two_opt_kernel(
            (blocks,), (threads,),
            (tours_gpu, distances, num_tours, n)
        )

        # Trae el resultado de vuelta a CPU
        return cp.asnumpy(tours_gpu, out=tours)

    def update_position(self, swarm, velocity):
        # Aplica cada swap de la velocidad sobre el tour
        for i, swaps in enumerate(velocity):
            tour = swarm[i]
            for a, b in swaps:
                tour[a], tour[b] = tour[b], tour[a]
        
        swarm = TSPProblem.two_opt(swarm, self.distances_gpu)

        return swarm
    
    def update_position_cpu(self, swarm, velocity):
        # Aplica cada swap de la velocidad sobre el tour
        for i, swaps in enumerate(velocity):
            tour = swarm[i]
            for a, b in swaps:
                tour[a], tour[b] = tour[b], tour[a]
        
        swarm = TSPProblem.two_opt_cpu(swarm, self.distances)

        return swarm
    
    def update_position_cpu_multi(self, swarm, velocity):
        # Aplica cada swap de la velocidad sobre el tour
        for i, swaps in enumerate(velocity):
            tour = swarm[i]
            for a, b in swaps:
                tour[a], tour[b] = tour[b], tour[a]
        
        utils.two_opt(swarm, self.distances)

        return swarm