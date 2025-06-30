from . import Algorithm
import numpy as np
from time import time
from . import MESSAGE_TAG, FINISH_TAG
import pickle, zlib

class PSO(Algorithm):
    '''
    Particle Swarm Optimization (PSO) algorithm implementation.
    This class provides an implementation of the Particle Swarm Optimization (PSO) algorithm for solving optimization problems.
    PSO is a population-based stochastic optimization technique inspired by the social behavior of birds flocking or fish schooling.
    The algorithm maintains a swarm of particles, where each particle represents a candidate solution. Particles move through the
    solution space by updating their velocities and positions based on their own experience and the experience of their neighbors.

    Attributes:
        swarm_size (int): Number of particles in the swarm.
        inertia_weight (float): Inertia weight for velocity update.
        cognitive_weight (float): Cognitive weight for velocity update.
        social_weight (float): Social weight for velocity update.
        seed (int or None): Random seed for reproducibility.
        reset_threshold (int): Number of iterations without improvement before resetting a particle.

    Methods:
        __init__(self, problem, swarm_size=100, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.0, seed=None, executer='single', reset_threshold=100):
            Initializes the PSO algorithm with the specified parameters and validates their ranges.
        fit(self, iterations, timelimit=None, verbose=True):
            Runs the PSO algorithm for a specified number of iterations or until a time limit is reached.
            Initializes the swarm, updates velocities and positions, evaluates fitness, and tracks the best solutions.
            Supports partial swarm reset if no improvement is observed for a specified threshold.
        fit_mpi(self, comm, rank, timelimit, sendto, receivefrom, verbose=True):
            Executes the PSO algorithm in a distributed fashion using MPI for communication.
            Each MPI process maintains its own swarm and communicates the best found solutions with neighboring processes.
            Supports swarm reset and inter-process communication for best solution sharing.
    '''
    def __init__(self, problem, swarm_size=100, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.0, seed=None, executer='single', reset_threshold=100):
        """
        Initializes the Particle Swarm Optimization (PSO) algorithm with the specified parameters.

        Args:
            problem: An object representing the optimization problem. Must implement 'fitness', 'generate_solution', 'update_velocity', and 'update_position' methods.
            swarm_size (int, optional): Number of particles in the swarm. Must be greater than 0. Default is 100.
            inertia_weight (float, optional): Inertia weight for velocity update. Must be between 0 and 1. Default is 0.5.
            cognitive_weight (float, optional): Cognitive weight for velocity update. Must be between 0 and 1. Default is 1.0.
            social_weight (float, optional): Social weight for velocity update. Must be between 0 and 1. Default is 1.0.
            seed (int, optional): Random seed for reproducibility. Default is None.
            executer (str, optional): Execution mode, e.g., 'single' or other supported modes. Default is 'single'.
            reset_threshold (int, optional): Number of iterations without improvement before resetting a particle. Default is 100.

        Raises:
            ValueError: If any of the provided parameters are outside their valid ranges.
        """
        # Se definen los métodos requeridos que el problema debe implementar.
        required_methods = ['fitness', 'generate_solution', 'update_velocity', 'update_position']
        super().__init__(problem, required_methods, executer)

        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.seed = seed
        self.reset_threshold = reset_threshold

        if swarm_size <= 0:
            raise ValueError("Swarm size must be greater than 0.")
        if inertia_weight < 0 or inertia_weight > 1:
            raise ValueError("Inertia weight must be between 0 and 1.")
        if cognitive_weight < 0 or cognitive_weight > 1:
            raise ValueError("Cognitive weight must be between 0 and 1.")
        if social_weight < 0 or social_weight > 1:
            raise ValueError("Social weight must be between 0 and 1.")
        
    def fit(self, iterations, timelimit=None, verbose=True):
        """
        Optimize the problem using Particle Swarm Optimization (PSO).
        This method runs the PSO algorithm for a specified number of iterations or until a time limit is reached.
        It initializes the swarm, updates velocities and positions, evaluates fitness, and tracks the best solutions.
        If no improvement is observed for a specified number of iterations, the swarm is partially reset.

        Args:
            iterations (int): Maximum number of iterations to run the algorithm. Must be greater than 0.
            timelimit (float, optional): Maximum allowed runtime in seconds. If None, runs without a time limit.
            verbose (bool, optional): If True, prints progress information during execution.

        Returns:
            np.ndarray: The best solution found by the swarm.

        Raises:
            ValueError: If `timelimit` is negative or `iterations` is not greater than 0.
        """
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

            if no_improvement >= self.reset_threshold:
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
        """
        Runs the Particle Swarm Optimization (PSO) algorithm in a distributed fashion using MPI for communication.
        This method executes the PSO algorithm, where each MPI process maintains its own swarm and communicates
        the best found solutions with neighboring processes. The algorithm continues until the specified time limit
        is reached or the maximum number of iterations is exceeded.

        Parameters:
            comm: MPI communicator object used for inter-process communication.
            rank (int): The rank (ID) of the current MPI process.
            timelimit (float): Maximum allowed runtime in seconds. Must be non-negative.
            sendto (int): Rank of the process to which the best solution should be sent.
            receivefrom (int): Rank of the process from which to receive the best solution.
            verbose (bool, optional): If True, prints progress information. Default is True.

        Returns:
            np.ndarray: The best solution found by this process.

        Raises:
            ValueError: If `timelimit` is negative.
            
        Notes:
            - The method assumes that the problem instance and executer are properly initialized.
            - Communication between processes is performed using compressed and pickled data.
            - The algorithm supports swarm reset if no improvement is observed for a specified threshold.
        """
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

        no_improvement = 0 

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
                no_improvement = 0  # Reiniciar contador de no mejoras
            else:
                no_improvement += 1

            if no_improvement >= self.reset_threshold:
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