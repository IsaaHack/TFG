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
    """
    TSPProblem implements the Traveling Salesman Problem (TSP) for use in metaheuristic algorithms such as Ant Colony Optimization (ACO) and Particle Swarm Optimization (PSO), supporting both CPU and GPU execution.

    Parameters
    ----------
    distances : np.ndarray
        A 2D array representing the distance matrix between cities.
    executer : str, optional
        Execution mode: 'single' (CPU), 'multi' (multi-core CPU), or 'gpu' (GPU, default).

    Attributes
    ----------
    distances : np.ndarray
        The distance matrix on CPU.
    distances_gpu : cp.ndarray
        The distance matrix on GPU (if applicable).
    n_cities : int
        Number of cities in the problem.
    execution : str
        Selected execution mode.

    Methods
    -------
    generate_solution(num_samples=1)
        Generates one or more random TSP solutions (permutations), always starting from city 0.
    fitness(solutions)
        Computes the total distance (fitness) of one or more TSP solutions using CPU.
    fitness_omp(solutions)
        Computes the fitness using an OpenMP-accelerated function.
    fitness_gpu(solutions)
        Computes the fitness using a GPU-accelerated function.
    fitness_hybrid(solutions, speedup=1)
        Computes the fitness using a hybrid CPU/GPU approach.
    crossover(population, crossover_rate)
        Applies crossover to the population using a GPU/CPU-agnostic method.
    crossover_cpu(population, crossover_rate)
        Applies crossover to the population using a CPU implementation.
    mutation(population, mutation_rate)
        Applies swap mutation to the population.
    initialize_pheromones()
        Initializes the pheromone matrix on GPU.
    initialize_pheromones_cpu()
        Initializes the pheromone matrix on CPU.
    construct_solutions(colony_size, pheromones, alpha, beta, out=None)
        Constructs solutions using ACO on GPU.
    construct_solutions_cpu(colony_size, pheromones, alpha, beta, out=None)
        Constructs solutions using ACO on CPU.
    construct_solutions_cpu_multi(colony_size, pheromones, alpha, beta, out=None)
        Constructs solutions using ACO on multi-core CPU.
    update_pheromones(pheromones, colony, fitness_values, evaporation_rate)
        Updates pheromones on GPU.
    update_pheromones_cpu(pheromones, colony, fitness_values, evaporation_rate)
        Updates pheromones on CPU.
    reset_pheromones(pheromones)
        Resets pheromones on GPU.
    reset_pheromones_cpu(pheromones)
        Resets pheromones on CPU.
    generate_velocity(num_samples=1)
        Generates initial random swap sequences (velocities) for PSO.
    update_velocity(swarm, velocity, p_best, g_best, inertia_weight, cognitive_weight, social_weight)
        Updates particle velocities for PSO on GPU.
    update_velocity_cpu(swarm, velocity, p_best, g_best, inertia_weight, cognitive_weight, social_weight)
        Updates particle velocities for PSO on CPU.
    two_opt_cpu(tours, distances)
        Applies the 2-opt local search to tours on CPU.
    two_opt(tours, distances)
        Applies the 2-opt local search to tours on GPU.
    update_position(swarm, velocity)
        Applies velocity swaps and 2-opt to update swarm positions on GPU.
    update_position_cpu(swarm, velocity)
        Applies velocity swaps and 2-opt to update swarm positions on CPU.
    update_position_cpu_multi(swarm, velocity)
        Applies velocity swaps and 2-opt to update swarm positions on multi-core CPU.
    """
    def __init__(self, distances, executer='gpu'):
        """
        Initializes the TSP problem instance with the given distance matrix and execution mode.

        Args:
            distances (array-like): A 2D array or matrix representing the distances between cities.
            executer (str, optional): The execution mode to use. Supported values are:
                - 'single': Use single-threaded CPU execution.
                - 'multi': Use multi-threaded CPU execution.
                - 'gpu': Use GPU execution (default).

        Raises:
            ValueError: If an unsupported execution type is provided.

        Side Effects:
            - Converts the distance matrix to a contiguous CuPy array for GPU execution.
            - Sets up method bindings according to the selected execution mode.
            - Synchronizes the GPU device if using GPU execution.
        """

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
        """
        Generates one or more random solutions (tours) for the Traveling Salesman Problem (TSP),
        always starting from city 0.

        Args:
            num_samples (int, optional): Number of solutions to generate. Defaults to 1.

        Returns:
            np.ndarray: 
                - If num_samples == 1: A 1D array representing a single tour, starting at city 0.
                - If num_samples > 1: A 2D array of shape (num_samples, n_cities), where each row is a tour starting at city 0.
        """

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
        """
        Calculates the fitness value(s) for one or more TSP solutions.

        Args:
            solutions (np.ndarray): A single solution (1D array) or multiple solutions (2D array)
                representing the order of cities in the TSP route(s).

        Returns:
            float or np.ndarray: The fitness value for the solution if a single solution is provided,
                or an array of fitness values if multiple solutions are provided.

        Notes:
            - Uses the `utils.fitness_tsp` function to compute the fitness based on the provided distances.
            - If `solutions` is a 1D array, returns a single fitness value.
            - If `solutions` is a 2D array, returns an array of fitness values, one for each solution.
        """

        if len(solutions.shape) == 1:
            return utils.fitness_tsp(self.distances, solutions)
        else:
            return np.array([utils.fitness_tsp(self.distances, solution) for solution in solutions])
    
    def fitness_omp(self, solutions):
        """
        Calculates the fitness value(s) for one or more TSP solutions using an optimized parallel method.

        Parameters:
            solutions (np.ndarray): A single solution (1D array) or a batch of solutions (2D array) representing TSP tours.

        Returns:
            float or np.ndarray: The fitness value for the provided solution(s). Returns a single float if a single solution is provided, or a NumPy array of floats for multiple solutions.

        Notes:
            This method uses a parallelized implementation (`utils.fitness_tsp_omp`) to efficiently compute fitness values.
        """

        if len(solutions.shape) == 1:
            fitness_value = np.empty(1, dtype=np.float32)
        else:
            fitness_value = np.empty(solutions.shape[0], dtype=np.float32)
        
        utils.fitness_tsp_omp(self.distances, solutions, fitness_value)

        return fitness_value[0] if len(solutions.shape) == 1 else fitness_value
    
    def fitness_gpu(self, solutions):
        """
        Calculates the fitness values for a batch of TSP solutions using GPU acceleration.

        Parameters
        ----------
        solutions : np.ndarray
            An array of TSP solutions (routes), where each solution is a sequence of city indices.
            Can be a 1D array (single solution) or 2D array (batch of solutions).

        Returns
        -------
        np.ndarray or float
            The fitness values (total route distances) for each solution. Returns a float if a single solution
            was provided, or a 1D numpy array of floats for a batch of solutions.

        Notes
        -----
        This method leverages GPU computation via CuPy and custom CUDA kernels for efficient evaluation.
        """

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
        """
        Calculates the fitness values for a set of TSP solutions using a hybrid CPU/GPU approach.

        Args:
            solutions (np.ndarray): An array of TSP solutions. Each solution is typically a permutation of city indices.
            speedup (int, optional): A parameter to control the speedup factor for the GPU computation. Defaults to 1.

        Returns:
            tuple:
                - np.float32 or np.ndarray: The fitness value(s) for the provided solution(s). Returns a single float if a single solution is provided, or an array of floats for multiple solutions.
                - int: The potentially updated speedup value after computation.

        Notes:
            - Utilizes GPU acceleration for fitness computation via the `utils_gpu.fitness_tsp_hybrid` function.
            - If a single solution is provided (1D array), returns a single fitness value; otherwise, returns an array of fitness values.
        """

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
        """
        Performs crossover operation on a population of TSP solutions.
        This method selects pairs of individuals from the population and applies a crossover
        operation to generate new offspring. The crossover points are randomly chosen for each
        pair, ensuring that the starting city (city 0) is not altered.

        Args:
            population (np.ndarray): The current population of TSP solutions, where each individual
                is represented as a sequence of city indices.
            crossover_rate (float): The proportion of the population to undergo crossover, in the
                range [0, 1].

        Returns:
            np.ndarray: The new population after applying the crossover operation.
        """

        num_crossovers = int(np.floor(crossover_rate * len(population) / 2))
        rng = np.random.default_rng(seed=np.random.randint(0, 2**32 - 1))

        # Asegurar que start >= 1 para no modificar la ciudad 0
        start = rng.integers(1, self.n_cities - 1, size=num_crossovers)
        remaining = self.n_cities - start - 1
        end = start + 1 + rng.integers(0, remaining, size=num_crossovers)

        return utils.crossover_tsp(population, start, end)
    
    def crossover_cpu(self, population, crossover_rate):
        """
        Performs ordered crossover (OX) on a population of individuals for the Traveling Salesman Problem (TSP).
        This method modifies the given population in-place by applying the ordered crossover operator to pairs of parents,
        producing new offspring that inherit segments from both parents while preserving the order and uniqueness of cities.

        Args:
            population (np.ndarray): A 2D NumPy array of shape (population_size, n_cities), where each row represents a tour (permutation of cities).
            crossover_rate (float): The proportion of the population to undergo crossover, in the range [0, 1].

        Returns:
            np.ndarray: The modified population array after crossover.

        Notes:
            - The number of crossovers performed is determined by `crossover_rate * len(population) / 2`.
            - Each crossover operates on consecutive pairs of individuals in the population.
            - The crossover is performed in-place, replacing parents with their offspring.
        """

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
        """
        Applies swap mutation to a subset of individuals in the population.
        For each selected individual, two random positions (excluding the first city)
        are chosen and their values are swapped, introducing variation in the population.

        Args:
            population (np.ndarray): 2D array of shape (n_individuals, n_cities) representing the current population,
                                     where each row is a permutation of city indices.
            mutation_rate (float): Probability (between 0 and 1) that determines the fraction of individuals to mutate.

        Returns:
            None: The population is modified in place.
        """

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
        """
        Initializes the pheromone matrix for the TSP problem.

        The pheromone values are randomly initialized between 1/(n_cities^2) and 1/n_cities
        for each pair of cities, except for the diagonal (i.e., pheromones[i, i]), which is set to 0
        to prevent self-loops.

        Returns:
            cp.ndarray: A (n_cities, n_cities) matrix of pheromone values with zeros on the diagonal.
        """
        pheromones = cp.random.uniform(1/(self.n_cities*self.n_cities), 1/self.n_cities, (self.n_cities, self.n_cities), dtype=cp.float32)
        cp.fill_diagonal(pheromones, 0.0)  # Evitar feromonas en la diagonal
        return pheromones
    
    def initialize_pheromones_cpu(self):
        """
        Initializes the pheromone matrix for the TSP problem on the CPU.

        The pheromone values are randomly initialized between 1/(n_cities^2) and 1/n_cities
        for each pair of cities, except for the diagonal, which is set to 0 to prevent
        pheromone trails on self-loops.

        Returns:
            np.ndarray: A (n_cities, n_cities) matrix representing the initial pheromone levels.
        """
        pheromones = np.random.uniform(1/(self.n_cities*self.n_cities), 1/self.n_cities, (self.n_cities, self.n_cities))
        np.fill_diagonal(pheromones, 0.0)  # Evitar feromonas en la diagonal
        return pheromones
    
    def construct_solutions(self, colony_size, pheromones, alpha, beta, out=None):
        """
        Constructs a batch of TSP solutions using an Ant Colony Optimization (ACO) approach on the GPU.

        This method generates `colony_size` candidate solutions for the Traveling Salesman Problem (TSP)
        by simulating the probabilistic construction of tours based on pheromone and heuristic information.
        The computation is accelerated using GPU arrays and a custom CUDA kernel.

        Args:
            colony_size (int): Number of solutions (ants) to construct in parallel.
            pheromones (array-like): 2D array of pheromone values for each edge.
            alpha (float): Exponent for the influence of pheromone values.
            beta (float): Exponent for the influence of heuristic information (inverse distance).
            out (np.ndarray, optional): Optional output array to store the resulting solutions.

        Returns:
            np.ndarray: Array of shape (colony_size, n_cities) containing the constructed tours,
                        where each row is a permutation of city indices representing a tour.

        Notes:
            - Uses GPU acceleration via CuPy for efficient computation.
            - The method updates internal heuristic matrices if `beta` changes.
            - If `out` is provided, results are written in-place and also returned.
        """
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
        """
        Constructs a set of solutions for the TSP problem using Ant Colony Optimization (ACO) on the CPU.
        This method generates `colony_size` solutions (tours) by simulating the behavior of ants, 
        where each ant incrementally builds a tour based on pheromone trails and heuristic information.
        The function supports reusing an output array and efficiently updates the heuristic matrix 
        only when the `beta` parameter changes.

        Parameters
        ----------
        colony_size : int
            The number of solutions (ants) to construct.
        pheromones : np.ndarray
            A 2D array of shape (n_cities, n_cities) representing the pheromone levels on each edge.
        alpha : float
            The exponent applied to pheromone values in the transition probability calculation.
        beta : float
            The exponent applied to heuristic values (inverse distance) in the transition probability calculation.
        out : np.ndarray, optional
            An optional preallocated array of shape (colony_size, n_cities) to store the constructed solutions.
            If None, a new array is allocated.

        Returns
        -------
        solutions : np.ndarray
            An array of shape (colony_size, n_cities) containing the constructed tours, where each row 
            represents a sequence of city indices visited by an ant.

        Notes
        -----
        - The method precomputes a heuristic matrix (inverse distance raised to `beta`) and only updates it 
          if `beta` changes from the previous call.
        - Random numbers for solution construction are precomputed for efficiency.
        - The actual construction of solutions is delegated to the `utils.construct_solutions_tsp_inner_cpu` function.
        """
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
        """
        Constructs multiple TSP solutions in parallel on the CPU using an Ant Colony Optimization (ACO) approach.

        Parameters
        ----------
        colony_size : int
            The number of solutions (ants) to construct in parallel.
        pheromones : np.ndarray
            A 2D array representing the pheromone levels between cities.
        alpha : float
            The exponent for the influence of pheromone trails.
        beta : float
            The exponent for the influence of heuristic information (inverse distance).
        out : np.ndarray, optional
            An optional preallocated output array to store the constructed solutions.
            If None, a new array is created.

        Returns
        -------
        solutions : np.ndarray
            An array of shape (colony_size, n_cities) containing the constructed TSP solutions,
            where each row represents a tour (sequence of city indices).

        Notes
        -----
        - This method precomputes the heuristic matrix for the given beta value and caches it for efficiency.
        - The construction process is randomized and uses a different random seed on each call.
        - The actual solution construction is performed by the `utils.construct_solutions_tsp_inner` function.
        """
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
        """
        Updates the pheromone matrix based on the solutions found by the colony and their fitness values.
        This method performs global pheromone evaporation and deposits new pheromones along the paths taken by each solution in the colony.
        The amount of pheromone deposited is inversely proportional to the fitness value of each solution.

        Args:
            pheromones (cp.ndarray): The current pheromone matrix (2D array) representing pheromone levels between cities.
            colony (np.ndarray or cp.ndarray): Array of shape (n_solutions, n_cities) where each row is a tour (sequence of city indices).
            fitness_values (np.ndarray or cp.ndarray): Array of shape (n_solutions,) containing the fitness (e.g., tour length) for each solution.
            evaporation_rate (float): The rate at which pheromones evaporate globally (between 0 and 1).

        Returns:
            cp.ndarray: The updated pheromone matrix after evaporation and deposition.
        """
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
        """
        Updates the pheromone matrix for the TSP problem using the provided colony and fitness values.

        This method applies pheromone evaporation and deposits new pheromones based on the solutions
        found by the colony of ants. The update is performed on the CPU.

        Args:
            pheromones (np.ndarray): The current pheromone matrix representing the desirability of each edge.
            colony (np.ndarray): The collection of solutions (paths) generated by the ants.
            fitness_values (np.ndarray): The fitness (quality) of each solution in the colony.
            evaporation_rate (float): The rate at which pheromones evaporate from the matrix.

        Returns:
            np.ndarray: The updated pheromone matrix after evaporation and deposition.
        """
        return utils.update_pheromones_tsp(
            pheromones,
            colony,
            fitness_values,
            evaporation_rate
        )
    
    def reset_pheromones(self, pheromones):
        """
        Resets the pheromone matrix with random values for an Ant Colony Optimization algorithm.

        The pheromone values are initialized uniformly at random between 1/(n_cities^2) and 1/n_cities,
        where n_cities is the number of cities in the problem. The diagonal elements, representing
        self-loops, are set to 0.

        Args:
            pheromones (cp.ndarray): The current pheromone matrix (not used in this implementation).

        Returns:
            cp.ndarray: The reset pheromone matrix with new random values and zero diagonal.
        """
        pheromones = cp.random.uniform(1/(self.n_cities*self.n_cities), 1/self.n_cities, (self.n_cities, self.n_cities), dtype=cp.float32)
        cp.fill_diagonal(pheromones, 0.0)
        return pheromones
    
    def reset_pheromones_cpu(self, pheromones):
        """
        Resets the pheromone matrix for the TSP problem on the CPU by assigning random values within a specified range.

        The pheromone values are initialized uniformly at random between 1/(n_cities^2) and 1/n_cities for each pair of cities,
        except for the diagonal elements, which are set to 0 (no self-loops).

        Args:
            pheromones (np.ndarray): The current pheromone matrix (not used in this implementation).

        Returns:
            np.ndarray: The reset pheromone matrix with updated values.
        """
        pheromones = np.random.uniform(1/(self.n_cities*self.n_cities), 1/self.n_cities, (self.n_cities, self.n_cities))
        np.fill_diagonal(pheromones, 0.0)
        return pheromones
    
    def generate_velocity(self, num_samples=1):
        """
        Generates a list of random swap operations (velocities) for use in permutation-based optimization algorithms.

        Each velocity is represented as a list of tuple pairs, where each tuple (i, j) indicates a swap between the i-th and j-th cities in the tour.

        Args:
            num_samples (int, optional): Number of velocity samples to generate. Defaults to 1.

        Returns:
            list[list[tuple[int, int]]]: A list containing 'num_samples' velocity samples. Each sample is a list of swap operations (tuples of city indices).
        """
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
        """
        Generates a sequence of swaps to transform one tour into another.

        Given two permutations (tours) of the same set of cities, this function computes
        the minimal sequence of swap operations required to convert `from_tour` into `to_tour`.
        Each swap is represented as a tuple of indices (i, j), indicating that the elements
        at positions i and j in the current tour should be swapped.

        Args:
            from_tour (list): The starting tour, represented as a list of city identifiers.
            to_tour (list): The target tour, represented as a list of city identifiers.

        Returns:
            list of tuple: A list of (i, j) tuples, where each tuple represents a swap operation
            to be applied sequentially to transform `from_tour` into `to_tour`.
        """
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
        """
        Updates the velocity of each particle in the swarm for the TSP Particle Swarm Optimization algorithm.
        The velocity is represented as a sequence of swap operations that, when applied to a tour, move it towards better solutions.
        The new velocity is computed as a combination of three components:
            - Inertia: retains part of the previous velocity.
            - Cognitive: moves the particle towards its personal best solution.
            - Social: moves the particle towards the global best solution.
        Duplicate swaps are removed, and the total number of swaps is limited based on the number of cities.

        Args:
            swarm (list of list of int): Current population of tours (particles), each represented as a list of city indices.
            velocity (list of list of tuple): Current velocities for each particle, where each velocity is a list of swap operations.
            p_best (list of list of int): Personal best tours found by each particle.
            g_best (list of int): Global best tour found by the swarm.
            inertia_weight (float): Weight for the inertia component.
            cognitive_weight (float): Weight for the cognitive component.
            social_weight (float): Weight for the social component.

        Returns:
            list of list of tuple: Updated velocities for each particle, each as a list of swap operations.
        """
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
        """
        Updates the velocity of each particle in the swarm for the TSP using the Particle Swarm Optimization (PSO) algorithm.
        This method computes the new velocity for each particle by combining three components:
        - Inertia: Retains part of the previous velocity.
        - Cognitive: Moves the particle towards its personal best solution.
        - Social: Moves the particle towards the global best solution.
        The velocity is represented as a sequence of swap operations that transform the current tour towards the target tour.

        Args:
            swarm (list of list of int): Current population of tours (particles), each represented as a list of city indices.
            velocity (list of list of tuple): Current velocities for each particle, where each velocity is a list of swap operations (tuples).
            p_best (list of list of int): Personal best tours found by each particle.
            g_best (list of int): Global best tour found by the swarm.
            inertia_weight (float): Weight for the inertia component (previous velocity).
            cognitive_weight (float): Weight for the cognitive component (personal best).
            social_weight (float): Weight for the social component (global best).

        Returns:
            list of list of tuple: Updated velocities for each particle, each as a list of swap operations.
        """
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
        """
        Performs a single iteration of the 2-opt optimization algorithm on a batch of TSP tours.

        This function attempts to improve each tour in the input by reversing a single segment
        if it results in a shorter total tour length, according to the provided distance matrix.
        Only the first improving move found for each tour is applied.

        Args:
            tours (np.ndarray): A 2D numpy array of shape (batch_size, n), where each row represents
                a tour as a sequence of city indices.
            distances (np.ndarray): A 2D numpy array of shape (n, n) representing the distance matrix
                between cities.

        Returns:
            np.ndarray: The potentially improved batch of tours, with the same shape as the input.
        """
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
        """
        Optimizes a batch of tours using the 2-opt algorithm on the GPU.

        Args:
            tours (numpy.ndarray): Array of shape (num_tours, n) representing the initial tours.
            distances (cupy.ndarray): 2D array of shape (n, n) containing the distance matrix, stored on the GPU.

        Returns:
            numpy.ndarray: Array of optimized tours with the same shape as the input `tours`.

        Notes:
            - This function uses a custom CUDA kernel (`two_opt_kernel`) to perform the optimization in parallel.
            - The input `tours` is transferred to the GPU, processed, and the result is copied back to the CPU.
            - The function assumes that `distances` is already a CuPy array on the GPU.
        """
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
        """
        Updates the positions of a swarm of tours by applying a sequence of swap operations (velocity) to each tour.
        Each element in the velocity list corresponds to a list of swap operations for a tour in the swarm. 
        After applying all swaps, the method performs a 2-opt optimization on the entire swarm to further improve the tours.
        
        Args:
            swarm (list of list of int): The current population of tours, where each tour is a list of city indices.
            velocity (list of list of tuple): The velocity for each tour, represented as a list of swap operations (tuples of indices).

        Returns:
            list of list of int: The updated swarm after applying the velocity and 2-opt optimization.
        """
        # Aplica cada swap de la velocidad sobre el tour
        for i, swaps in enumerate(velocity):
            tour = swarm[i]
            for a, b in swaps:
                tour[a], tour[b] = tour[b], tour[a]
        
        swarm = TSPProblem.two_opt(swarm, self.distances_gpu)

        return swarm
    
    def update_position_cpu(self, swarm, velocity):
        """
        Updates the positions of a swarm of tours by applying a sequence of swap operations (velocity) to each tour,
        and then refines the tours using the 2-opt local search algorithm.

        Args:
            swarm (list of list of int): The current population of tours, where each tour is a list of city indices.
            velocity (list of list of tuple of int): The velocity for each tour, represented as a list of swap operations.
                Each swap operation is a tuple (a, b) indicating the indices in the tour to be swapped.

        Returns:
            list of list of int: The updated swarm of tours after applying the velocity swaps and 2-opt optimization.
        """
        # Aplica cada swap de la velocidad sobre el tour
        for i, swaps in enumerate(velocity):
            tour = swarm[i]
            for a, b in swaps:
                tour[a], tour[b] = tour[b], tour[a]
        
        swarm = TSPProblem.two_opt_cpu(swarm, self.distances)

        return swarm
    
    def update_position_cpu_multi(self, swarm, velocity):
        """
        Updates the positions of a swarm of tours by applying a series of swap operations (velocity) to each tour,
        then performs a 2-opt optimization on the entire swarm.

        Args:
            swarm (list of list of int): The current population of tours, where each tour is a list of city indices.
            velocity (list of list of tuple): The velocity for each tour, represented as a list of swap operations (tuples of indices).

        Returns:
            list of list of int: The updated swarm after applying the swaps and 2-opt optimization.
        """
        # Aplica cada swap de la velocidad sobre el tour
        for i, swaps in enumerate(velocity):
            tour = swarm[i]
            for a, b in swaps:
                tour[a], tour[b] = tour[b], tour[a]
        
        utils.two_opt(swarm, self.distances)

        return swarm