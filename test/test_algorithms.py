from .common import generate_distance_matrix, tsp_optimal_solution, experiment_fitness, get_digits_data
import numpy as np
from problems import ClasProblem, TSPProblem
from algorithms import GA, ACO, PSO
import time

clas_iters = 300
tsp_iters = 100

ga_params_clas = {
    'population_size': 50,
    'seed': 42,
}
ga_params_tsp = {
    'population_size': 1024,
    'seed': 42,
    'mutation_rate': 0.2,
}
aco_params_clas = {
    'colony_size': 50,
    'seed': 42,
    'evaporation_rate': 0.03,
    'alpha': 1.0,
    'beta': 1.5
}
aco_params_tsp = {
    'colony_size': 1024,
    'seed': 42,
    'alpha': 1.7,
    'beta': 1.2
}
pso_params_clas = {
    'swarm_size': 50,
    'seed': 42,
    'inertia_weight': 0.9,
    'cognitive_weight': 0.4,
    'social_weight': 0.7
}
pso_params_tsp = {
    'swarm_size': 1024,
    'seed': 42,
    'inertia_weight': 0.9,
    'cognitive_weight': 0.4,
    'social_weight': 0.7
}

def main(algorithm_name='ga', executer_type='gpu', seed=42, n_cities=100):
    np.random.seed(seed)
    print("Using seed:", seed)

    X_train, y_train = get_digits_data(preprocess=True)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    dist_matrix = generate_distance_matrix(n_cities)
    print("Number of cities:", n_cities)

    # Definimos pesos para cada una de las características
    problem = ClasProblem(X_train, y_train)
    problem2 = TSPProblem(dist_matrix)

    if algorithm_name == 'ga':
        algorithm = GA(problem, **ga_params_clas, executer=executer_type)
        algorithm2 = GA(problem2, **ga_params_tsp, executer=executer_type)
    elif algorithm_name == 'aco':
        algorithm = ACO(problem, **aco_params_clas, executer=executer_type)
        algorithm2 = ACO(problem2, **aco_params_tsp, executer=executer_type)
    elif algorithm_name == 'pso':
        algorithm = PSO(problem, **pso_params_clas, executer=executer_type)
        algorithm2 = PSO(problem2, **pso_params_tsp, executer=executer_type)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
    print(f"Starting {algorithm_name.upper()} Clas...")

    start = time.time()
    weights = algorithm.fit(iterations=clas_iters)
    end = time.time()

    print("Time:", end - start)
    fit = problem.fitness(weights)
    class_rate = problem.clas_rate(weights)
    red_rate = problem.red_rate(weights)

    print("Classification rate:", class_rate)
    print("Reduction rate:", red_rate)
    print("Fitness from", algorithm_name.upper() + ":", fit)

    print("-----------------------------------------------------------")
    print(f"Starting {algorithm_name.upper()} TSP...")

    start = time.time()
    path = algorithm2.fit(iterations=tsp_iters)
    end = time.time()

    print("Time:", end - start)
    fit = problem2.fitness(path)
    print("Fitness from", algorithm_name.upper() + ":", -fit)

    # Verify path
    print("Verifying path...")
    if len(path) == len(set(path)) and np.all(np.isin(path, np.arange(n_cities))):
        print("Path is valid")
    else:
        print("Path is invalid")

if __name__ == "__main__":
    # Obtener el nombre del algoritmo desde la línea de comandos
    import argparse
    parser = argparse.ArgumentParser(description='Run optimization algorithms.')
    parser.add_argument('-a', '--algorithm', type=str, default='ga', choices=['ga', 'aco', 'pso'], help='Algorithm to run: ga, aco, or pso')
    parser.add_argument('-e', '--executer_type', type=str, default='gpu', choices=['single', 'multi', 'gpu', 'hybrid'], help='Execution type: single, multi, gpu, or hybrid')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('-n', '--n_cities', type=int, default=100, help='Number of cities for TSP problem')
    args = parser.parse_args()
    main(algorithm_name=args.algorithm, executer_type=args.executer_type, seed=args.seed, n_cities=args.n_cities)