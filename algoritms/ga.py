from algoritms.algoritm import Algoritm
import numpy as np

class GA(Algoritm):
    def __init__(self, problem, population_size=100, mutation_rate=0.08, crossover_rate=0.7, generations=100, seed=None):
        self.problem = problem
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.seed = seed

        self.required_methods = ['fitness', 'generate_solution', 'mutation', 'crossover']
        self.check_required_methods()

    def initialize_population(self):
        return self.problem.generate_solution(self.population_size)
    
    def check_required_methods(self):
        for method in self.required_methods:
            if not hasattr(self.problem, method):
                raise ValueError(f"Problem class must implement the {method} method.")
            
    def selection(self, population, fitnesess):
        new_population = np.empty_like(population)

        random_indexes = np.random.randint(0, population.shape[0], size=(population.shape[0], 3))
        best_in_each_group = np.argmax(fitnesess[random_indexes], axis=1)
        new_population = population[random_indexes[np.arange(population.shape[0]), best_in_each_group]]

        return new_population
    
    def fit(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        population = self.initialize_population()
        fitness_values = np.array([self.problem.fitness_gpu(individual) for individual in population])
        actual_generation = 0
        best = np.copy(population[np.argmax(fitness_values)])
        best_fit = fitness_values[np.argmax(fitness_values)]

        while actual_generation < self.generations:
            # Selection
            new_population = self.selection(population, fitness_values)
            # Crossover
            self.problem.crossover(new_population, self.crossover_rate)
            # Mutation
            self.problem.mutation(new_population, self.mutation_rate)

            # Evaluate fitness of the new population
            fitness_values = np.array([self.problem.fitness_gpu(individual) for individual in new_population])

            # Replace the worst individuals with the best from the previous generation(Elitism)
            best_new_index = np.argmax(fitness_values)
            worst_new_index = np.argmin(fitness_values)

            best_new = new_population[best_new_index]
            best_new_fit = fitness_values[best_new_index]

            if best_new_fit > best_fit:
                best = best_new
                best_fit = best_new_fit
            else:
                new_population[worst_new_index] = best
                fitness_values[worst_new_index] = best_fit

            population = new_population

            actual_generation += 1

        return np.copy(best)


    