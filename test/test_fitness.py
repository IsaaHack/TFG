import numpy as np
from problems import ClasProblem, TSPProblem
import time
from sklearn.datasets import load_digits
from .common import generate_distance_matrix, tsp_optimal_solution, experiment_fitness, get_digits_data

def main():
    np.random.seed(42)  # Para reproducibilidad

    X_train, y_train = get_digits_data(preprocess=True)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Definimos pesos para cada una de las características
    problem = ClasProblem(X_train, y_train)
    weights = problem.generate_solution(num_samples=64)

    print('----------------Fitness de clasificación-------------------')

    # Definimos el número de repeticiones
    N = 5

    experiment_fitness(problem, weights, N)

    print("-----------------------Fitness de TSP------------------------")


    np.random.seed(np.random.randint(0, 1000))  # Para reproducibilidad

    # Definir número de ciudades
    num_cities = 20000
    dist_matrix = generate_distance_matrix(num_cities)
    problem = TSPProblem(dist_matrix)

    print("Number of cities:", num_cities)
    print("Distance matrix shape:", dist_matrix.shape)
    if num_cities <= 11:
        time_start = time.time()
        path, cost = tsp_optimal_solution(dist_matrix)
        time_end = time.time()

        print("Optimal path:", path)
        print("Optimal cost:", cost)
        print("Time to find optimal path:", time_end - time_start)

    # Definimos 1024 soluciones aleatorias
    solutions = problem.generate_solution(num_samples=1024)
    N = 20

    experiment_fitness(problem, solutions, N)

if __name__ == "__main__":
    main()

