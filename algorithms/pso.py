from . import Algorithm
import numpy as np
from time import time
from . import MESSAGE_TAG, FINISH_TAG
import pickle, zlib

class PSO(Algorithm):
    def __init__(self, problem, swarm_size=100, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.0, seed=None, executer='single', reset_threshold=100):
        # Se definen los métodos requeridos que el problema debe implementar.
        required_methods = ['fitness', 'generate_solution', 'update_velocity', 'update_position']
        super().__init__(problem, required_methods, executer)

        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.seed = seed

        if swarm_size <= 0:
            raise ValueError("Swarm size must be greater than 0.")
        if inertia_weight < 0 or inertia_weight > 1:
            raise ValueError("Inertia weight must be between 0 and 1.")
        if cognitive_weight < 0 or cognitive_weight > 1:
            raise ValueError("Cognitive weight must be between 0 and 1.")
        if social_weight < 0 or social_weight > 1:
            raise ValueError("Social weight must be between 0 and 1.")
        
    def fit(self, iterations, timelimit=None, verbose=True):
        if timelimit is None:
            timelimit = np.inf
        if timelimit < 0:
                raise ValueError("Timelimit must be a non-negative value.")
        if iterations <= 0:
            raise ValueError("Iterations must be greater than 0.")
        time_start = time()

        self.init_seed(self.seed)

        if verbose:
            self.print_init(time_start, iterations, timelimit)

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
        no_improvement = 0
        
        self.print_update(g_best_fitness)

        while iteration < iterations and time() - time_start < timelimit:
            # Actualizar inercia
            inertia = max(0.1, self.inertia_weight * (1 - iteration / iterations))

            # Actualizar velocidad y posición usando métodos del problema
            velocity = self.problem.update_velocity(swarm, velocity, p_best, g_best,
                                                    inertia, self.cognitive_weight, self.social_weight)
            swarm = self.problem.update_position(swarm, velocity)

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
                no_improvement = 0  # Reiniciar contador de no mejoras
            else:
                no_improvement += 1

            if no_improvement >= 100:  # Si no hay mejoras en 100 iteraciones, reiniciar el enjambre
                swarm = self.problem.generate_solution(self.swarm_size)
                velocity = self.problem.generate_velocity(self.swarm_size)
                
                swarm[0] = np.copy(g_best)  # Mantener el mejor global en la primera posición
                fitness[0] = g_best_fitness  # Mantener su fitness
                fitness[1:] = self.executer.execute(swarm[1:])  # Evaluar el resto del enjambre

                p_best = np.copy(swarm)
                p_best_fitness = np.copy(fitness)

                g_best_idx = np.argmax(p_best_fitness)
                g_best = np.copy(p_best[g_best_idx])
                g_best_fitness = p_best_fitness[g_best_idx]

                no_improvement = 0  # Reiniciar contador de no mejoras

            iteration += 1
            self.print_update(g_best_fitness)



        self.print_end()
        return np.copy(g_best)
    
    def fit_mpi(self, comm, rank, timelimit, sendto, receivefrom, verbose=True):
        if timelimit < 0:
            raise ValueError("Timelimit must be a non-negative value.")

        iterations = np.inf

        time_start = time()

        self.init_seed(self.seed)

        if verbose and rank == 0:
            self.print_init(time_start, iterations, timelimit)

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

        while iteration < iterations and time() - time_start < timelimit:
            # Actualizar inercia
            inertia = max(0.1, self.inertia_weight * (1 - iteration / iterations))

            # Actualizar velocidad y posición usando métodos del problema
            velocity = self.problem.update_velocity(swarm, velocity, p_best, g_best,
                                                    inertia, self.cognitive_weight, self.social_weight)
            swarm = self.problem.update_position(swarm, velocity)

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
                # Enviar el mejor resultado al siguiente proceso
                data = (rank, g_best, g_best_fitness)
                data_serialized = zlib.compress(pickle.dumps(data), level=9)
                comm.send(data_serialized, dest=sendto, tag=MESSAGE_TAG)

            # Recibir el mejor resultado del proceso recievefrom
            hay_mensaje = comm.Iprobe(source=receivefrom, tag=MESSAGE_TAG)
            while hay_mensaje:
                rank_received, best_received, best_fit_received = pickle.loads(zlib.decompress(comm.recv(source=receivefrom, tag=MESSAGE_TAG)))
                if best_fit_received > g_best_fitness:
                    g_best = np.copy(best_received)
                    g_best_fitness = best_fit_received
                if rank_received != rank:
                    data = (rank_received, g_best, g_best_fitness)
                    data_serialized = zlib.compress(pickle.dumps(data), level=9)
                hay_mensaje = comm.Iprobe(source=receivefrom, tag=MESSAGE_TAG)

            iteration += 1
            
            self.print_update(g_best_fitness)

        self.print_end()

        data = (rank, g_best, g_best_fitness)
        data_serialized = zlib.compress(pickle.dumps(data), level=9)
        comm.send(data_serialized, dest=sendto, tag=FINISH_TAG)
        
        return np.copy(g_best)