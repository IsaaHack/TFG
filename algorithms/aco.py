from algorithms.algorithm import Algorithm
import numpy as np
from time import time
import cupy as cp

class ACO(Algorithm):
    def __init__(self, problem, colony_size=50, evaporation_rate=0.1, iterations=100, alpha=1.0, beta=2.0, seed=None,reset_threshold=100, executer_type='single', executer=None, timelimit=np.inf):
        # Se definen los métodos requeridos que el problema debe implementar.
        required_methods = ['fitness', 'initialize_pheromones', 'construct_solutions', 'update_pheromones', 'reset_pheromones']
        super().__init__(problem, iterations, required_methods, executer_type, executer, timelimit)
        
        self.colony_size = colony_size
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.reset_threshold = reset_threshold
        self.timelimit = timelimit

        if iterations == np.inf and timelimit == np.inf:
            raise ValueError("Either iterations or timelimit must be set to a finite value.")
        if colony_size <= 0:
            raise ValueError("Colony size must be greater than 0.")
        if evaporation_rate <= 0 or evaporation_rate >= 1:
            raise ValueError("Evaporation rate must be between 0 and 1.")
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")
        if beta <= 0:
            raise ValueError("Beta must be greater than 0.")
        if reset_threshold <= 0:
            raise ValueError("Reset threshold must be greater than 0.")
        if iterations <= 0:
            raise ValueError("Iterations must be greater than 0.")

    def fit(self):
        time_start = time()

        if self.seed is not None:
            np.random.seed(self.seed)
            cp.random.seed(self.seed)

        # Inicializa las feromonas del problema.
        pheromones = self.problem.initialize_pheromones()
        best = None
        best_fit = -np.inf

        iteration = 0
        no_improvement = 0
        
        self.print_init(time_start)

        fitness_values = np.empty(self.colony_size, dtype=np.float32)
        colony = self.problem.generate_solution(self.colony_size)

        while iteration < self.iterations and time() - time_start < self.timelimit:
            # Se generan soluciones a partir de las feromonas.
            self.problem.construct_solutions(self.colony_size, pheromones, self.alpha, self.beta, out=colony)
            if best is not None:
                # Se reinicializan las soluciones de la colonia con la mejor solución encontrada.
                colony[0] = best
                fitness_values[0] = best_fit
                # Se evalúan las soluciones generadas.
                fitness_values[1:] = self.executer.execute(colony[1:])
            else:
                # Se evalúan las soluciones generadas.
                fitness_values = self.executer.execute(colony)
            
            # Se selecciona la mejor solución de la colonia.
            best_iteration_idx = np.argmax(fitness_values)
            best_iteration_fit = fitness_values[best_iteration_idx]

            # Se actualiza la mejor solución global si es necesario.
            if best_iteration_fit > best_fit:
                best = np.copy(colony[best_iteration_idx])
                best_fit = best_iteration_fit
                no_improvement = 0
            else:
                no_improvement += 1

            # Actualización de feromonas a partir de las soluciones actuales.
            # Se delega la actualización en el problema.
            pheromones = self.problem.update_pheromones(pheromones, colony, fitness_values, self.evaporation_rate)

            # Si no hay mejora en varias iteraciones se reinicia la colonia.
            if no_improvement >= self.reset_threshold:
                # Se reinicializa las feromonas
                pheromones = self.problem.reset_pheromones(pheromones)
                # Se reinicializa el contador de no mejora
                no_improvement = 0

            iteration += 1
            self.print_update(best_fit, 1)

        self.print_end()

        return np.copy(best)
