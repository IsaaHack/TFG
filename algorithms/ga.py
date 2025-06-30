from . import Algorithm
import numpy as np
from time import time
from . import MESSAGE_TAG, FINISH_TAG
import pickle, zlib

class GA(Algorithm):
    '''Genetic Algorithm (GA) implementation for solving optimization problems.
    This class provides a flexible and extensible framework for applying genetic algorithms to a wide range of problems. 
    It supports both single-process and MPI-based parallel execution, with features such as tournament selection, 
    elitism, population reset, and customizable mutation and crossover rates.

    Attributes:
        population_size (int): Number of individuals in the population.
        mutation_rate (float): Probability of mutation for each individual.
        crossover_rate (float): Probability of crossover between individuals.
        seed (int or None): Random seed for reproducibility.
        tournament_size (int): Number of individuals in each tournament selection.
        reset_threshold (int): Number of generations without improvement before resetting the population.
        executer (str): Execution mode (e.g., 'single').
        
    Methods:
        __init__(self, problem, population_size=100, mutation_rate=0.08, crossover_rate=0.7, seed=None, tournament_size=3, reset_threshold=100, executer='single'):
            Initializes the genetic algorithm with the specified parameters and validates input ranges.
        initialize_population(self):
            Initializes the population using the problem's generate_solution method.
        selection(self, population, fitnesess):
            Performs tournament selection to create a new population from the current one.
        reset_population(self, best, best_fit):
            Resets the population, preserving the best individual and its fitness.
        fit(self, iterations, timelimit=None, verbose=True):
            Runs the genetic algorithm for a specified number of generations or until a time limit is reached.
            Applies selection, crossover, mutation, and elitism, with optional population reset.
            Returns the best solution found.
        fit_mpi(self, comm, rank, timelimit, sendto, receivefrom, verbose=True):
            Runs the genetic algorithm in parallel using MPI, exchanging best solutions between processes.
            Returns the best solution found by this process.

        ValueError: 
        
        If any parameter is out of its valid range or if required arguments are missing.
        - The problem instance must implement 'fitness', 'generate_solution', 'mutation', and 'crossover' methods.
        - Elitism ensures the best solution is retained across generations.
        - Population reset helps avoid stagnation if no improvement is observed for a specified threshold.
        - MPI-based execution allows for distributed optimization and solution sharing between processes.
    '''
    def __init__(self, problem, population_size=100, mutation_rate=0.08, crossover_rate=0.7, seed=None, tournament_size=3, reset_threshold=100, executer='single'):
        """
        Initializes the genetic algorithm with the specified parameters.

        Args:
            problem: The problem instance to be solved. Must implement 'fitness', 'generate_solution', 'mutation', and 'crossover' methods.
            population_size (int, optional): Number of individuals in the population. Must be greater than 0. Default is 100.
            mutation_rate (float, optional): Probability of mutation for each individual. Must be between 0 and 1. Default is 0.08.
            crossover_rate (float, optional): Probability of crossover between individuals. Must be between 0 and 1. Default is 0.7.
            seed (int, optional): Random seed for reproducibility. Default is None.
            tournament_size (int, optional): Number of individuals participating in tournament selection. Must be greater than 0. Default is 3.
            reset_threshold (int, optional): Number of generations without improvement before resetting the population. Default is 100.
            executer (str, optional): Execution mode, e.g., 'single'. Default is 'single'.

        Raises:
            ValueError: If any of the parameters are out of their valid ranges.
        """
        required_methods = ['fitness', 'generate_solution', 'mutation', 'crossover']

        super().__init__(problem, required_methods, executer)

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.tournament_size = tournament_size
        self.reset_threshold = reset_threshold

        if population_size <= 0:
            raise ValueError("Population size must be greater than 0.")
        if mutation_rate < 0 or mutation_rate > 1:
            raise ValueError("Mutation rate must be between 0 and 1.")
        if crossover_rate < 0 or crossover_rate > 1:
            raise ValueError("Crossover rate must be between 0 and 1.")
        if tournament_size <= 0:
            raise ValueError("Print frequency must be greater than 0.")

    def initialize_population(self):
        """
        Initializes the population for the genetic algorithm.

        Returns:
            list: A list of solutions generated by the problem's generate_solution method,
                  with the number of solutions equal to the specified population size.
        """
        return self.problem.generate_solution(self.population_size)
            
    def selection(self, population, fitnesess):
        """
        Selects a new population from the current population using tournament selection.

        Args:
            population (np.ndarray): The current population of individuals, typically a 2D array where each row represents an individual.
            fitnesess (np.ndarray): Array of fitness values corresponding to each individual in the population.

        Returns:
            np.ndarray: The new population selected via tournament selection, with the same shape as the input population.

        Notes:
            - Tournament selection is performed by randomly selecting groups of individuals (of size `self.tournament_size`) and choosing the individual with the highest fitness from each group to form the new population.
            - Assumes that `self.tournament_size` is defined and is an integer greater than 0.
        """
        new_population = np.empty_like(population)

        random_indexes = np.random.randint(0, population.shape[0], size=(population.shape[0], self.tournament_size))
        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)
        new_population = population[random_indexes[np.arange(population.shape[0]), best_in_each_group]]

        return new_population
    
    def reset_population(self, best, best_fit):
        """
        Resets the population for the genetic algorithm, preserving the best individual.
        Parameters:
            best (np.ndarray): The best individual from the previous generation.
            best_fit (float): The fitness value of the best individual.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the new population array and the corresponding fitness values array, 
            with the best individual and its fitness at the first position.
        """
        new_population = self.initialize_population()
        new_population[0] = best
        
        fitness_values = np.empty(self.population_size)
        fitness_values[0] = best_fit
        fitness_values[1:] = self.executer.execute(new_population[1:])

        return new_population, fitness_values
    
    def fit(self, iterations, timelimit=None, verbose=True):
        """
        Trains the genetic algorithm for a specified number of iterations or until a time limit is reached.

        Args:
            iterations (int): The maximum number of generations to run the algorithm.
            timelimit (float, optional): The maximum time (in seconds) to run the algorithm. If None, runs without a time limit. Defaults to None.
            verbose (bool, optional): If True, prints progress updates during execution. Defaults to True.

        Returns:
            np.ndarray: The best solution found by the genetic algorithm.

        Raises:
            ValueError: If 'timelimit' is negative or 'iterations' is not greater than 0.

        Notes:
            - The method initializes the population and iteratively applies selection, crossover, and mutation.
            - Elitism is used to retain the best solution found so far.
            - If no improvement is observed for a specified number of generations ('reset_threshold'), the population is reset.
            - Progress and results are printed if 'verbose' is True.
        """
        if timelimit is None:
            timelimit = np.inf
        if timelimit < 0:
                raise ValueError("Timelimit must be a non-negative value.")
        if iterations <= 0:
            raise ValueError("Iterations must be greater than 0.")

        time_start = time()

        if verbose:
            self.print_init(time_start, iterations, timelimit)

        self.init_seed(self.seed)

        population = self.initialize_population()
        fitness_values = self.executer.execute(population)
        actual_generation = 1

        best = np.copy(population[np.argmax(fitness_values)])
        best_fit = fitness_values[np.argmax(fitness_values)]

        no_improvement = 0

        self.print_update(best_fit)

        while actual_generation < iterations and time() - time_start < timelimit:
            # Selection
            new_population = self.selection(population, fitness_values)
            # Crossover
            self.problem.crossover(new_population, self.crossover_rate)
            # Mutation
            self.problem.mutation(new_population, self.mutation_rate)

            # Evaluate fitness
            fitness_values = self.executer.execute(new_population)

            # Elitism
            best_new_index = np.argmax(fitness_values)
            worst_new_index = np.argmin(fitness_values)

            best_new = new_population[best_new_index]
            best_new_fit = fitness_values[best_new_index]

            if best_new_fit > best_fit:
                best = best_new
                best_fit = best_new_fit
                no_improvement = 0
            else:
                new_population[worst_new_index] = best
                fitness_values[worst_new_index] = best_fit
                no_improvement += 1

            # Reset if there is not improvement
            if no_improvement >= self.reset_threshold:
                population, fitness_values = self.reset_population(best, best_fit)
                best_fit = np.argmax(fitness_values)
                best = np.copy(population[best_fit])
                best_fit = fitness_values[best_fit]
                no_improvement = 0
            else:
                population = new_population

            actual_generation += 1
            self.print_update(best_fit)

        self.print_end()

        return np.copy(best)

    def fit_mpi(self, comm, rank, timelimit, sendto, receivefrom, verbose=True):
        """
        Runs the genetic algorithm using MPI for parallel execution and communication between processes.
        This method evolves a population of candidate solutions using selection, crossover, mutation, and elitism.
        It exchanges the best solutions between MPI processes to improve convergence and diversity. The algorithm
        runs until the specified time limit is reached.
        Parameters:
            comm: MPI communicator object used for inter-process communication.
            rank (int): The rank (ID) of the current MPI process.
            timelimit (float): Maximum allowed runtime in seconds. Must be non-negative.
            sendto (int): Rank of the process to which the best solution should be sent.
            receivefrom (int): Rank of the process from which to receive the best solution.
            verbose (bool, optional): If True, prints progress information. Defaults to True.
        Returns:
            np.ndarray: The best solution found by this process.
        Raises:
            ValueError: If `timelimit` is negative.
        Notes:
            - The method uses compressed and pickled data for MPI communication.
            - Implements elitism and population reset if no improvement is observed for a threshold number of generations.
            - The best solution is sent to the next process and received from the previous process in a ring topology.
        """
        if timelimit < 0:
            raise ValueError("Timelimit must be a non-negative value.")
            
        iterations = np.inf
            
        time_start = time()

        self.init_seed(self.seed)

        if verbose and rank == 0:
            self.print_init(time_start, iterations, timelimit)

        population = self.initialize_population()
        fitness_values = self.executer.execute(population)
        actual_generation = 1

        best = np.copy(population[np.argmax(fitness_values)])
        best_fit = fitness_values[np.argmax(fitness_values)]

        no_improvement = 0

        self.print_update(best_fit, rank)

        while actual_generation < iterations and time() - time_start < timelimit:
            # Selection
            new_population = self.selection(population, fitness_values)
            # Crossover
            self.problem.crossover(new_population, self.crossover_rate)
            # Mutation
            self.problem.mutation(new_population, self.mutation_rate)

            # Evaluate fitness
            fitness_values = self.executer.execute(new_population)

            # Elitism
            best_new_index = np.argmax(fitness_values)
            worst_new_index = np.argmin(fitness_values)

            best_new = new_population[best_new_index]
            best_new_fit = fitness_values[best_new_index]

            if best_new_fit > best_fit:
                best = best_new
                best_fit = best_new_fit
                no_improvement = 0
                #Enviar la mejor solución a el siguiente proceso
                data = (rank, best, best_fit)
                data_serialized = zlib.compress(pickle.dumps(data), level=9)
                comm.send(data_serialized, dest=sendto, tag=MESSAGE_TAG)
            else:
                new_population[worst_new_index] = best
                fitness_values[worst_new_index] = best_fit
                no_improvement += 1

            # Reset if there is not improvement
            if no_improvement >= self.reset_threshold:
                population, fitness_values = self.reset_population(best, best_fit)
                best_fit = np.argmax(fitness_values)
                best = np.copy(population[best_fit])
                best_fit = fitness_values[best_fit]
                no_improvement = 0
            else:
                population = new_population

            # Recibir el mejor resultado del proceso receivefrom
            hay_mensaje = comm.Iprobe(source=receivefrom, tag=MESSAGE_TAG)
            while hay_mensaje:
                rank_received, best_received, best_fit_received = pickle.loads(zlib.decompress(comm.recv(source=receivefrom, tag=MESSAGE_TAG)))
                if best_fit_received > best_fit:
                    best = np.copy(best_received)
                    best_fit = best_fit_received
                    # Sustituir la peor solución por la mejor recibida
                    worst_new_index = np.argmin(fitness_values)
                    new_population[worst_new_index] = best
                    fitness_values[worst_new_index] = best_fit
                if rank_received != rank:
                    data = (rank, best, best_fit)
                    data_serialized = zlib.compress(pickle.dumps(data), level=9)
                    comm.send(data_serialized, dest=sendto, tag=MESSAGE_TAG)
                hay_mensaje = comm.Iprobe(source=receivefrom, tag=MESSAGE_TAG)

            actual_generation += 1
            self.print_update(best_fit, rank)

        self.print_end()

        data = (rank, best, best_fit)
        data_serialized = zlib.compress(pickle.dumps(data), level=9)
        comm.send(data_serialized, dest=sendto, tag=FINISH_TAG)

        return np.copy(best)
    