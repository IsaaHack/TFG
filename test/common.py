import numpy as np
import itertools
from scipy.spatial.distance import cdist
import timeit
from problems import utils_gpu
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

def get_digits_data(preprocess=True):
    digits = load_digits()
    X_train = np.array(digits.data, dtype=np.float32)
    y_train = np.array(digits.target, dtype=np.int32)

    if preprocess:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_train = np.array(X_train, dtype=np.float32)

    return X_train, y_train

def experiment_fitness(problem, weights, N):
    # Fitness en CPU
    def cpu_function():
        return problem.fitness(weights)

    cpu_times = timeit.repeat(cpu_function, number=1, repeat=N)
    print("Fitness CPU:", problem.fitness(weights)[0])
    print(f"Time (CPU): {np.mean(cpu_times):.6f}s ± {np.std(cpu_times):.6f}s")

    print("-----------------------------------------------------------")

    # Fitness con OpenMP
    def omp_function():
        return problem.fitness_omp(weights)

    omp_times = timeit.repeat(omp_function, number=1, repeat=N)
    print("Fitness OpenMP:", problem.fitness_omp(weights)[0])
    print(f"Time (OpenMP): {np.mean(omp_times):.6f}s ± {np.std(omp_times):.6f}s")

    print("-----------------------------------------------------------")

    # Fitness en GPU
    def gpu_function():
        return problem.fitness_gpu(weights)

    utils_gpu.warmup()
    gpu_times = timeit.repeat(gpu_function, number=1, repeat=N)
    print("Fitness GPU:", problem.fitness_gpu(weights)[0])
    print(f"Time (GPU): {np.mean(gpu_times):.6f}s ± {np.std(gpu_times):.6f}s")

    print("-----------------------------------------------------------")

    # # Fitness Hybrid
    # def hybrid_function():
    #     return problem.fitness_hybrid(weights)
    # hybrid_times = timeit.repeat(hybrid_function, number=1, repeat=N)
    # print("Fitness Hybrid:", problem.fitness_hybrid(weights)[0][0])    
    # print(f"Time (Hybrid): {np.mean(hybrid_times):.6f}s ± {np.std(hybrid_times):.6f}s")
    # print("-----------------------------------------------------------")

    # Speedups
    cpu_speedup = np.mean(cpu_times) / np.mean(omp_times)
    gpu_speedup = np.mean(cpu_times) / np.mean(gpu_times)
    #hybrid_speedup = np.mean(cpu_times) / np.mean(hybrid_times)
    print(f"Speedup CPU vs OpenMP: {cpu_speedup:.2f}x")
    print(f"Speedup CPU vs GPU: {gpu_speedup:.2f}x")
    #print(f"Speedup CPU vs Hybrid: {hybrid_speedup:.2f}x")
    print("-----------------------------------------------------------")

def generate_distance_matrix(n_cities):
    # Genera coordenadas aleatorias (x, y) para cada ciudad en un rango de 0 a 100
    cities = np.random.rand(n_cities, 2) * 100

    # Calcula la matriz de distancias usando cdist (vectorizado y rápido)
    distance_matrix = cdist(cities, cities, "minkowski", p=2).astype(np.float32)

    return distance_matrix

def tsp_optimal_solution(dist_matrix):
    n = dist_matrix.shape[0]
    # Generar todas las permutaciones de ciudades excluyendo la 0
    perms = np.array(list(itertools.permutations(np.arange(1, n))))
    
    # Agregar la ciudad 0 al inicio y al final de cada ruta
    n_routes = perms.shape[0]
    paths = np.hstack([np.zeros((n_routes, 1), dtype=int), perms, np.zeros((n_routes, 1), dtype=int)])
    
    # Calcular los costos de todas las rutas de forma vectorizada:
    # paths[:, :-1] y paths[:, 1:] tienen dimensiones (n_routes, n)
    # Para cada ruta se suma la distancia entre cada par consecutivo de ciudades.
    costs = np.sum(dist_matrix[paths[:, :-1], paths[:, 1:]], axis=1)
    
    # Seleccionar la ruta con el costo mínimo
    best_idx = np.argmin(costs)
    best_cost = costs[best_idx]
    best_path = paths[best_idx]
    
    return best_path, best_cost