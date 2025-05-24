from . import Algorithm
import numpy as np
from time import time
import cupy as cp

class PSO(Algorithm):
    def __init__(self, problem, swarm_size=100, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.0, iterations=100, seed=None, executer_type='single', executer=None, timelimit=np.inf):
        # Se definen los métodos requeridos que el problema debe implementar.
        required_methods = ['fitness', 'generate_solution', 'update_velocity', 'update_position']
        super().__init__(problem, iterations, required_methods, executer_type, executer, timelimit)

        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.iterations = iterations
        self.seed = seed

        if iterations == np.inf and timelimit == np.inf:
            raise ValueError("Either iterations or timelimit must be set to a finite value.")
        if swarm_size <= 0:
            raise ValueError("Swarm size must be greater than 0.")
        if inertia_weight < 0 or inertia_weight > 1:
            raise ValueError("Inertia weight must be between 0 and 1.")
        if cognitive_weight < 0 or cognitive_weight > 1:
            raise ValueError("Cognitive weight must be between 0 and 1.")
        if social_weight < 0 or social_weight > 1:
            raise ValueError("Social weight must be between 0 and 1.")
        if iterations <= 0:
            raise ValueError("Iterations must be greater than 0.")
        
    def fit(self):
        time_start = time()

        self.init_seed(self.seed)

        self.print_init(time_start)

        # Inicializar el enjambre
        swarm = self.problem.generate_solution(self.swarm_size)
        velocity = self.problem.generate_velocity(self.swarm_size)
        fitness = self.executer.execute(swarm)

        p_best = np.copy(swarm)
        p_best_fitness = np.copy(fitness)
        g_best_idx = np.argmax(p_best_fitness)
        g_best = np.copy(p_best[g_best_idx])
        g_best_fitness = p_best_fitness[g_best_idx]

        inertia = self.inertia_weight

        iteration = 1
        
        self.print_update(g_best_fitness)

        while iteration < self.iterations and time() - time_start < self.timelimit:
            # Actualizar inercia
            inertia = max(0.1, self.inertia_weight * (1 - iteration / self.iterations))

            # Actualizar velocidad y posición usando métodos del problema
            velocity = self.problem.update_velocity(swarm, velocity, p_best, g_best,
                                                    inertia, self.cognitive_weight, self.social_weight)
            self.problem.update_position(swarm, velocity)

            # Evaluar fitness
            fitness = self.executer.execute(swarm)

            # Actualizar el mejor personal
            improved = fitness > p_best_fitness
            p_best[improved] = swarm[improved]
            p_best_fitness[improved] = fitness[improved]

            # Actualizar el mejor global
            current_best_idx = np.argmax(p_best_fitness)
            current_best_fitness = p_best_fitness[current_best_idx]
            if current_best_fitness > g_best_fitness:
                g_best = np.copy(p_best[current_best_idx])
                g_best_fitness = current_best_fitness

            iteration += 1
            self.print_update(g_best_fitness)

        self.print_end()
        return np.copy(g_best)