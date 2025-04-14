from executers.executer import ClusterExecuter

def main():
    import numpy as np
    from scipy.spatial.distance import cdist

    def generate_distance_matrix(n_cities):
        # Genera coordenadas aleatorias (x, y) para cada ciudad en un rango de 0 a 100
        cities = np.random.rand(n_cities, 2) * 100

        # Calcula la matriz de distancias usando cdist (vectorizado y rápido)
        distance_matrix = cdist(cities, cities, "minkowski", p=2).astype(np.float32)

        return distance_matrix
    
    n_cities = 40
    dist_matrix = generate_distance_matrix(n_cities)

    executer = ClusterExecuter(
        filename="worker.py",
        nodes=["localhost", "localhost"],
        problem_import="problems.tsp_problem.TSPProblem",
        algorithm_import="algoritms.ga.GA",
        problem_args={
            "distances": dist_matrix.tolist()
        },
        algorithm_args={
            0: {
                "population_size": 1024,
                "mutation_rate": 0.08,
                "crossover_rate": 0.7,
                "generations": 100,
                "seed": 42,
                "executer_type": 'hybrid',
            },
            1: {
                "population_size": 1024,
                "mutation_rate": 0.08,
                "crossover_rate": 0.7,
                "generations": 100,
                "seed": 61,
                "executer_type": 'gpu',
            }
        }
    )

    executer.execute()

if __name__ == "__main__":
    main()