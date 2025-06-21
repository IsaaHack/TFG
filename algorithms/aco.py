from . import Algorithm
import numpy as np
from time import time
from . import MESSAGE_TAG, FINISH_TAG
import pickle, zlib

class ACO(Algorithm):
    def __init__(self, problem, colony_size=50, evaporation_rate=0.1, alpha=1.0, beta=2.0, seed=None, reset_threshold=100, executer='single'):
        # Se definen los métodos requeridos que el problema debe implementar.
        required_methods = ['fitness', 'initialize_pheromones', 'construct_solutions', 'update_pheromones', 'reset_pheromones']
        super().__init__(problem, required_methods, executer)
        
        self.colony_size = colony_size
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.reset_threshold = reset_threshold

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

        # Inicializa las feromonas del problema.
        pheromones = self.problem.initialize_pheromones()
        best = None
        best_fit = -np.inf

        iteration = 0
        no_improvement = 0

        fitness_values = np.empty(self.colony_size, dtype=np.float32)
        colony = self.problem.generate_solution(self.colony_size)

        while iteration < iterations and time() - time_start < timelimit:
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
    
    def fit_mpi(self, comm, rank, timelimit, sendto, receivefrom, verbose=True):
        if timelimit < 0:
            raise ValueError("Timelimit must be a non-negative value.")

        iterations = np.inf

        time_start = time()

        self.init_seed(self.seed)

        # Inicializa las feromonas del problema.
        pheromones = self.problem.initialize_pheromones()
        best = None
        best_fit = -np.inf

        iteration = 0
        no_improvement = 0
        
        if verbose and rank == 0:
            self.print_init(time_start, iterations, timelimit)

        fitness_values = np.empty(self.colony_size, dtype=np.float32)
        colony = self.problem.generate_solution(self.colony_size)

        while iteration < iterations and time() - time_start < timelimit:
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
                # Enviar el mejor resultado al siguiente proceso
                data = (rank, best, best_fit)
                data_serialized = zlib.compress(pickle.dumps(data), level=9)
                comm.send(data_serialized, dest=sendto, tag=MESSAGE_TAG)
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

            # Recibir el mejor resultado del proceso receivefrom
            hay_mensaje = comm.Iprobe(source=receivefrom, tag=MESSAGE_TAG)
            while hay_mensaje:
                rank_received, best_received, best_fit_received = pickle.loads(zlib.decompress(comm.recv(source=receivefrom, tag=MESSAGE_TAG)))
                if best_fit_received > best_fit:
                    best = np.copy(best_received)
                    best_fit = best_fit_received
                if rank_received != rank:
                    data = (rank, best, best_fit)
                    data_serialized = zlib.compress(pickle.dumps(data), level=9)
                    comm.send(data_serialized, dest=sendto, tag=MESSAGE_TAG)
                hay_mensaje = comm.Iprobe(source=receivefrom, tag=MESSAGE_TAG)

            iteration += 1
            self.print_update(best_fit, iteration)

        self.print_end()

        data = (rank, best, best_fit)
        data_serialized = zlib.compress(pickle.dumps(data), level=9)
        comm.send(data_serialized, dest=sendto, tag=FINISH_TAG)

        return np.copy(best)
