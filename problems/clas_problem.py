from . import Problem
from . import utils, utils_gpu
import cupy as cp
import numpy as np

# Constants for the problem
SQRT_03 = np.sqrt(0.3)
MIN_STD = 1e-3
MAX_STD = 0.25
MAX_V = np.sqrt(0.5)

class ClasProblem(Problem):
    """
    ClasProblem is a classification problem class designed for feature selection and optimization using evolutionary and swarm-based algorithms. It supports CPU and GPU (CuPy) acceleration for fitness evaluation and provides methods for solution generation, evaluation, and population management.

    Attributes:
        X (np.ndarray): Feature matrix (CPU).
        Y (np.ndarray): Label vector (CPU).
        X_gpu (cp.ndarray): Feature matrix (GPU).
        Y_gpu (cp.ndarray): Label vector (GPU).
        n_samples (int): Number of samples in the dataset.
        n_features (int): Number of features in the dataset.
        threshold (float): Threshold for feature selection.
        alpha (float): Weighting factor for classification vs. reduction rate.

    Methods:
        __init__(X, Y, threshold=0.1, alpha=0.75):
            Initializes the ClasProblem instance with data and parameters.
        generate_solution(num_samples=1):
            Generates random solutions for the feature selection problem.
        fitness(solutions):
            Evaluates the fitness of solutions using a CPU-based function.
        fitness_omp(solutions):
            Evaluates the fitness of solutions using an OpenMP-accelerated function.
        fitness_gpu(solutions):
            Evaluates the fitness of solutions using GPU acceleration (CuPy).
        fitness_gpu2(solutions):
            Alternative GPU-based fitness evaluation using custom CUDA kernels.
        fitness_hybrid(solutions, speedup=1):
            Hybrid fitness evaluation combining CPU and GPU methods.
        clas_rate(solution):
            Computes the classification accuracy rate for a given solution.
        red_rate(solution):
            Computes the feature reduction rate for a given solution.
        predict(X_test, solution):
            Predicts labels for test data using the selected features.
        crossover(population, crossover_rate, alpha=0.3):
            Performs crossover operation on the population for evolutionary algorithms.
        mutation(population, mutation_rate):
            Applies mutation to the population.
        initialize_pheromones():
            Initializes pheromone means and standard deviations for ACO algorithms.
        construct_solutions(colony_size, pheromones, alpha, beta, out=None):
            Constructs new solutions based on pheromone information.
        update_pheromones(pheromones, colony, fitness_values, evaporation_rate):
            Updates pheromone values based on colony performance.
        reset_pheromones(pheromones):
            Resets pheromones to initial values.
        generate_velocity(num_samples=1):
            Generates random velocities for PSO algorithms.
        update_velocity(population, velocity, p_best, g_best, inertia_weight, cognitive_weight, social_weight):
            Updates particle velocities in PSO.
        update_position(population, velocity):
            Updates particle positions in PSO, applying boundary constraints.
    """

    def __init__(self, X, Y, threshold=0.1, alpha=0.75):
        """
        Initializes the classification problem with input data, labels, and optional parameters.

        Parameters:
            X (array-like): Input feature data.
            Y (array-like): Target labels.
            threshold (float, optional): Threshold value for classification or other purposes. Default is 0.1.
            alpha (float, optional): Alpha parameter for algorithm-specific tuning. Default is 0.75.

        Attributes:
            X_gpu (cp.ndarray): Input features transferred to GPU as a CuPy array (float32).
            Y_gpu (cp.ndarray): Target labels transferred to GPU as a CuPy array (int32).
            X (array-like): Original input feature data.
            Y (array-like): Original target labels.
            n_samples (int): Number of samples in the dataset.
            n_features (int): Number of features per sample.
            threshold (float): Stored threshold value.
            alpha (float): Stored alpha value.

        Synchronizes the current CUDA device after data transfer.
        """

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
        """
        Generates random solution(s) for the classification problem.

        Parameters:
            num_samples (int, optional): Number of solutions to generate. 
                If 1 (default), returns a single solution vector. 
                If greater than 1, returns an array of solution vectors.

        Returns:
            np.ndarray: A NumPy array of shape (n_features,) if num_samples == 1,
                or shape (num_samples, n_features) if num_samples > 1, containing
                random float32 values uniformly sampled from [0, 1).
        """

        if num_samples == 1:
            return np.random.uniform(0, 1, size=(self.n_features)).astype(np.float32)
        else:
            return np.random.uniform(0, 1, size=(num_samples, self.n_features)).astype(np.float32)

    def fitness(self, solutions):
        """
        Calculates the fitness value(s) for one or more solutions.
        If a single solution is provided (1D array), computes its fitness directly.
        If multiple solutions are provided (2D array), computes the fitness for each solution.

        Args:
            solutions (np.ndarray): A single solution (1D array) or multiple solutions (2D array).

        Returns:
            float or np.ndarray: The fitness value for the single solution, or an array of fitness values for multiple solutions.
        """

        if len(solutions.shape) == 1:
            return utils.fitness(solutions, self.X, self.Y, self.alpha, self.threshold)
        else:
            return np.array([utils.fitness(solution, self.X, self.Y, self.alpha, self.threshold) for solution in solutions])
    
    def fitness_omp(self, solutions):
        """
        Calculates the fitness values for the given solutions using the OMP (Orthogonal Matching Pursuit) method.

        Parameters:
            solutions (np.ndarray): Array of candidate solutions. Can be a 1D array (single solution) or 2D array (multiple solutions).

        Returns:
            np.ndarray or float: Fitness values for the provided solutions. Returns a single float if a single solution is provided, 
            otherwise returns a 1D numpy array of fitness values.

        Notes:
            - Uses the `utils.fitness_omp` function to perform the actual fitness computation.
            - The fitness is computed based on the input data `self.X`, target values `self.Y`, regularization parameter `self.alpha`, 
              and threshold `self.threshold`.
        """

        if len(solutions.shape) == 1:
            fitness_values = np.empty(1, dtype=np.float32)
        else:
            fitness_values = np.empty(solutions.shape[0], dtype=np.float32)

        utils.fitness_omp(solutions, self.X, self.Y, fitness_values, self.alpha, self.threshold)

        return fitness_values[0] if len(solutions.shape) == 1 else fitness_values
    
    def fitness_gpu(self, solutions):
        """
        Computes the fitness values for a batch of solutions using GPU acceleration with CuPy.
        This method evaluates each solution by applying feature selection and transformation,
        then computes classification and reduction rates based on nearest neighbor predictions.
        If GPU memory is insufficient, it falls back to an alternative implementation.

        Parameters
        ----------
        solutions : array-like or cupy.ndarray, shape (n_solutions, n_features)
            The set of candidate solutions to evaluate. Each row represents a solution
            with feature weights or selection values.

        Returns
        -------
        numpy.ndarray, shape (n_solutions,)
            The computed fitness values for each solution.

        Notes
        -----
        - The fitness is a weighted sum of classification accuracy and feature reduction rate,
          controlled by the `self.alpha` parameter.
        - Uses GPU arrays (`cupy`) for efficient computation. Falls back to `fitness_gpu2`
          if a GPU memory error occurs.
        - Assumes `self.X_gpu` (features), `self.Y_gpu` (labels), `self.threshold`,
          `self.n_features`, and `self.alpha` are defined and reside on the GPU.
        """

        try:
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
        except cp.cuda.memory.OutOfMemoryError as e:
            # Si hay un error de memoria, se usa la función alternativa
            return self.fitness_gpu2(solutions)

        # Cálculo vectorizado de predicciones y métricas
        index_pred = cp.argmin(D, axis=2)  # Índices de vecinos más cercanos
        prediction_labels = self.Y_gpu[index_pred]  # Etiquetas predichas

        clas_rate = 100 * cp.mean(prediction_labels == self.Y_gpu, axis=1)  # Tasa de acierto
        red_rate = 100 * cp.sum(~mask, axis=1) / self.n_features  # Tasa de reducción

        # Cálculo final del fitness
        fitness_values = clas_rate * self.alpha + red_rate * (1-self.alpha)
        return fitness_values.get()  # Convertir de cupy a numpy
    
    def fitness_gpu2(self, solutions):
        """
        Computes the fitness value(s) for the given solution(s) using GPU acceleration.
        This method transfers the input solutions to the GPU, prepares the necessary data capsules,
        and calls a CUDA-accelerated fitness function. It supports both single-solution (1D array)
        and multiple-solution (2D array) inputs.

        Parameters
        ----------
        solutions : numpy.ndarray or cupy.ndarray
            An array of solutions to evaluate. Can be a 1D array (single solution) or a 2D array
            (multiple solutions, shape: [n_solutions, n_features]).

        Returns
        -------
        float or numpy.ndarray
            The computed fitness value(s) for the input solution(s). Returns a single float if a
            single solution is provided, or a numpy array of floats if multiple solutions are provided.

        Notes
        -----
        - Requires GPU support and appropriate CUDA kernels via `utils_gpu`.
        - Synchronizes the GPU device before computation to ensure data consistency.
        """

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
                    self.n_features,
                    self.alpha, 
                    self.threshold
            )
        else:
            return np.array([utils_gpu.fitness_cuda(
                    utils_gpu.create_capsule(solutions_gpu[i].data.ptr),
                    X_capsule,
                    Y_capsule,
                    self.n_samples,
                    self.n_features,
                    self.alpha,
                    self.threshold
            ) for i in range(solutions.shape[0])])
        
    def fitness_hybrid(self, solutions, speedup=1):
        """
        Computes the fitness values for a set of solutions using a hybrid CPU/GPU approach.
        This method evaluates the fitness of one or more candidate solutions by leveraging a GPU-accelerated
        function. It supports both single-solution and batch evaluation. The function also returns an updated
        speedup value, which may be used to tune the GPU computation.

        Parameters
        ----------
        solutions : np.ndarray
            An array of candidate solutions to be evaluated. Can be a 1D array (single solution) or a 2D array
            (multiple solutions).
        speedup : int or float, optional
            A parameter to control or record the speedup factor for GPU computation (default is 1).

        Returns
        -------
        fitness_values : float or np.ndarray
            The computed fitness value(s) for the provided solution(s). Returns a single float if a single
            solution is provided, or a 1D numpy array of floats for multiple solutions.
        new_speedup : int or float
            The updated speedup value as returned by the GPU function.
        """

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
                speedup,
                self.alpha,
                self.threshold
        )

        return fitness_values[0] if len(solutions.shape) == 1 else fitness_values, new_speedup
    
    def clas_rate(self, solution):
        """
        Calculates the classification rate for a given solution.
        This method delegates the computation to the `utils.clas_rate` function,
        passing the provided solution along with the instance's feature matrix (self.X),
        target labels (self.Y), and classification threshold (self.threshold).

        Args:
            solution: The solution or model parameters to evaluate.

        Returns:
            float: The classification rate (accuracy) of the solution.
        """

        return utils.clas_rate(solution, self.X, self.Y, self.threshold)
    
    def red_rate(self, solution):
        """
        Calculates the reduction rate for a given solution using the specified threshold.

        Args:
            solution: The solution object or data structure to evaluate.

        Returns:
            float: The reduction rate computed by the utils.red_rate function.
        """

        return utils.red_rate(solution, self.threshold)
    
    def predict(self, X_test, solution):
        """
        Predicts the output for the given test data using the provided solution.

        Args:
            X_test (array-like): Test data to predict.
            solution (object): Solution or model parameters to use for prediction.

        Returns:
            array-like: Predicted values for the test data.
        """

        return utils.predict(X_test, solution, self.X, self.Y, self.threshold)
    
    def crossover(self, population, crossover_rate, alpha: float = 0.3):
        """
        Performs blend crossover (BLX-alpha) on a population of individuals.
        This method selects pairs of parents from the population and generates offspring by sampling uniformly within an extended range defined by the parents and the alpha parameter. The offspring replace the parents in the population. All resulting values are clipped to the [0, 1] range.

        Args:
            population (np.ndarray): The population of individuals, where each individual is represented as a vector of real values.
            crossover_rate (float): The proportion of the population to undergo crossover (between 0 and 1).
            alpha (float, optional): The BLX-alpha parameter that controls the extent of the sampling range beyond the parents' values. Default is 0.3.

        Returns:
            None: The population is modified in place.
        """

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
        """
        Applies mutation to a population matrix by randomly altering gene values.
        Each mutation selects a random individual and gene, then adds Gaussian noise to the gene value,
        clipping the result to the range [0, 1]. The number of mutations is determined by the mutation rate.

        Args:
            population (np.ndarray): The population matrix of shape (num_individuals, num_genes).
            mutation_rate (float): The probability of mutation per individual (between 0 and 1).

        Returns:
            None: The population is modified in place.
        """

        estimated_mutations = int(mutation_rate * population.shape[0])

        mutation = np.random.normal(0, SQRT_03, estimated_mutations)
        genes_to_mutate = np.random.randint(0, population.shape[1], estimated_mutations)
        people_to_mutate = np.random.randint(0, population.shape[0], estimated_mutations)

        population[people_to_mutate, genes_to_mutate] = np.clip(population[people_to_mutate, genes_to_mutate] + mutation, 0, 1)

    def initialize_pheromones(self):
        """
        Initializes the pheromone parameters for each feature.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - means (np.ndarray): Randomly initialized means for each feature, sampled uniformly between 0 and 1.
                - stds (np.ndarray): Standard deviations for each feature, initialized to 0.2.
        """

        means = np.random.uniform(0, 1, size=self.n_features).astype(np.float32)
        stds = np.full(self.n_features, 0.2).astype(np.float32)
        return means, stds
    
    def construct_solutions(self, colony_size, pheromones, alpha, beta, out=None):
        """
        Constructs a set of candidate solutions for the colony using vectorized sampling 
        from normal distributions influenced by pheromone information and exploration/exploitation parameters.

        Args:
            colony_size (int): Number of solutions (ants) to generate.
            pheromones (tuple of np.ndarray): Tuple containing means and standard deviations 
                (means, stds) for each feature, representing the pheromone trails.
            alpha (float): Parameter controlling exploitation; higher values increase exploitation 
                by reducing the influence of the mean.
            beta (float): Parameter controlling exploration; higher values increase exploration 
                by reducing the influence of the standard deviation.
            out (np.ndarray, optional): Optional output array to store the generated solutions.

        Returns:
            np.ndarray: Array of shape (colony_size, n_features) containing the generated solutions, 
            with values clipped to the [0.0, 1.0] range.
        """

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
        """
        Updates the pheromone parameters (means and standard deviations) for an ant colony optimization algorithm.

        Args:
            pheromones (tuple): A tuple containing the current means and standard deviations (means, stds) of the pheromone model.
            colony (np.ndarray): The current population of solutions (ants), shape (n_ants, n_features).
            fitness_values (list or np.ndarray): Fitness values for each solution in the colony. Lower values indicate better solutions.
            evaporation_rate (float): The rate at which pheromone information evaporates (decays) in each iteration.

        Returns:
            tuple: Updated means and standard deviations (new_means, new_stds) for the pheromone model.

        Notes:
            - Fitness values are normalized and inverted to compute weights, so better solutions have higher influence.
            - Means are updated using a weighted average of the colony, favoring better solutions.
            - Standard deviations are updated using the elite subset of the colony to maintain exploration.
            - Standard deviations are clipped to remain within predefined bounds (MIN_STD, MAX_STD).
        """

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
        """
        Resets the pheromone matrix to its initial state.

        Parameters:
            pheromones: The current pheromone matrix (not used in this implementation).

        Returns:
            The newly initialized pheromone matrix.
        """

        pheromones = self.initialize_pheromones()
        return pheromones
    
    def generate_velocity(self, num_samples=1):
        """
        Generates random velocity vectors for particles.

        Parameters:
            num_samples (int, optional): Number of velocity vectors to generate. 
                If 1 (default), returns a single vector of shape (n_features,).
                If greater than 1, returns an array of shape (num_samples, n_features).

        Returns:
            np.ndarray: Random velocity vector(s) with values uniformly sampled from [-1, 1].
        """

        if num_samples == 1:
            return np.random.uniform(-1, 1, size=(self.n_features)).astype(np.float32)
        else:
            return np.random.uniform(-1, 1, size=(num_samples, self.n_features)).astype(np.float32)
        
    def update_velocity(self, population, velocity, p_best, g_best, inertia_weight, cognitive_weight, social_weight):
        """
        Updates the velocity of particles in a Particle Swarm Optimization (PSO) algorithm.

        Parameters:
            population (np.ndarray): Current positions of the particles.
            velocity (np.ndarray): Current velocities of the particles.
            p_best (np.ndarray): Personal best positions of the particles.
            g_best (np.ndarray): Global best position found by the swarm.
            inertia_weight (float): Inertia weight factor controlling exploration and exploitation.
            cognitive_weight (float): Cognitive coefficient (particle's own experience).
            social_weight (float): Social coefficient (swarm's experience).

        Returns:
            np.ndarray: Updated velocities for the particles, clipped to the range [-MAX_V, MAX_V].
        """

        # Calcular la velocidad usando la fórmula de PSO
        r1 = np.random.rand(*population.shape).astype(np.float32)
        r2 = np.random.rand(*population.shape).astype(np.float32)

        cognitive_component = cognitive_weight * r1 * (p_best - population)
        social_component = social_weight * r2 * (g_best - population)

        new_velocity = inertia_weight * velocity + cognitive_component + social_component

        new_velocity = np.clip(new_velocity, -MAX_V, MAX_V)

        #print("Mean velocity:", np.mean(np.abs(new_velocity)))

        return new_velocity
    
    def update_position(self, population, velocity):
        """
        Updates the positions of individuals in the population based on their velocities,
        applying boundary conditions and ensuring all positions remain within [0, 1].

        Parameters:
            population (np.ndarray): The current positions of the population.
            velocity (np.ndarray): The velocities to apply to the population.

        Returns:
            np.ndarray: The updated positions of the population, clipped to the [0, 1] range.
            
        Notes:
            - If an updated position goes out of bounds ([0, 1]), the velocity is reversed and halved (simulating a bounce),
              and the position is updated again.
            - All positions are clipped to ensure they remain within the valid range.
        """

        # Actualizar la posición de la población
        population += velocity

        # Aplicar rebote si la posición está fuera de los límites
        mask = (population < 0) | (population > 1)
        velocity[mask] *= -0.5

        population[mask] += velocity[mask]

        # Asegurarse de que los valores estén dentro del rango [0, 1]
        population = np.clip(population, 0, 1)
        return population
    

