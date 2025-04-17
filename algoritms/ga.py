from algoritms.algoritm import Algoritm
import numpy as np
from time import time

class GA(Algoritm):
    def __init__(self, problem, population_size=100, mutation_rate=0.08, crossover_rate=0.7, generations=100, seed=None, tournament_size=3, executer_type='hybrid', executer=None, timelimit=np.inf, print_freq=None):
        required_methods = ['fitness', 'generate_solution', 'mutation', 'crossover']
        super().__init__(problem, required_methods, executer_type, executer)

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.seed = seed
        self.tournament_size = tournament_size
        self.timelimit = timelimit
        self.print_freq = print_freq

        if generations == np.inf and timelimit == np.inf:
            raise ValueError("Either generations or timelimit must be set to a finite value.")
        if population_size <= 0:
            raise ValueError("Population size must be greater than 0.")
        if mutation_rate < 0 or mutation_rate > 1:
            raise ValueError("Mutation rate must be between 0 and 1.")
        if crossover_rate < 0 or crossover_rate > 1:
            raise ValueError("Crossover rate must be between 0 and 1.")
        if tournament_size <= 0:
            raise ValueError("Tournament size must be greater than 0.")
        if print_freq is not None and print_freq <= 0:
            raise ValueError("Print frequency must be greater than 0.")
        if generations <= 0:
            raise ValueError("Generations must be greater than 0.")

    def initialize_population(self):
        return self.problem.generate_solution(self.population_size)
            
    def selection(self, population, fitnesess):
        new_population = np.empty_like(population)

        random_indexes = np.random.randint(0, population.shape[0], size=(population.shape[0], self.tournament_size))
        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)
        new_population = population[random_indexes[np.arange(population.shape[0]), best_in_each_group]]

        return new_population
    
    def reset_population(self, best, best_fit):
        new_population = self.initialize_population()
        new_population[0] = best
        
        fitness_values = np.empty(self.population_size)
        fitness_values[0] = best_fit
        fitness_values[1:] = self.executer.execute(new_population[1:])

        return new_population, fitness_values
    
    def fit(self):
        time_start = time()

        if self.seed is not None:
            np.random.seed(self.seed)

        population = self.initialize_population()
        fitness_values = self.executer.execute(population)
        actual_generation = 1
        best = np.copy(population[np.argmax(fitness_values)])
        best_fit = fitness_values[np.argmax(fitness_values)]
        no_improvement = 0
        gap = 0.1

        if self.generations != np.inf and self.print_freq is None:
            frequency = self.generations * gap
        elif self.print_freq is None:
            frequency = 1000
        else:
            frequency = self.print_freq

        while actual_generation < self.generations and time() - time_start < self.timelimit:
            # Selection
            new_population = self.selection(population, fitness_values)
            # Crossover
            self.problem.crossover(new_population, self.crossover_rate)
            # Mutation
            self.problem.mutation(new_population, self.mutation_rate)

            # Evaluate fitness
            fitness_values = self.executer.execute(new_population)

            # Replace the worst individuals with the best from the previous generation(Elitism)
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

            if no_improvement >= 100:
                #print("Resetting population due to no improvement.")
                population, fitness_values = self.reset_population(best, best_fit)
                best_fit = np.argmax(fitness_values)
                best = np.copy(population[best_fit])
                best_fit = fitness_values[best_fit]
                no_improvement = 0
            else:
                population = new_population

            actual_generation += 1

            if self.generations != np.inf and self.timelimit == np.inf:
                self.print_iter(actual_generation, self.generations, best_fit, frequency=frequency)
            else:
                self.print_time(actual_generation, self.generations, time_start, self.timelimit, best_fit, frequency=frequency)

        return np.copy(best)


    