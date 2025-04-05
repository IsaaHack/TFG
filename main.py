import numpy as np
from problems.clas_problem import ClasProblem
import time
from sklearn.datasets import load_digits
import timeit

digits = load_digits()
X_train = np.array(digits.data, dtype=np.float32)
y_train = np.array(digits.target, dtype=np.int32)

# Normalizamos los datos con un MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train = np.array(X_train, dtype=np.float32)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Definimos pesos para cada una de las características
problem = ClasProblem(X_train, y_train)
weights = problem.generate_solution()

print('----------------Fitness de clasificación-------------------')

# Definimos el número de repeticiones
N = 50

# Fitness en CPU
def cpu_function():
    return problem.fitness(weights)

cpu_times = timeit.repeat(cpu_function, number=1, repeat=N)
print("Fitness CPU:", problem.fitness(weights))
print(f"Time (CPU): {np.mean(cpu_times):.6f}s ± {np.std(cpu_times):.6f}s")

print("-----------------------------------------------------------")

# Fitness con OpenMP
def omp_function():
    return problem.fitness_omp(weights)

omp_times = timeit.repeat(omp_function, number=1, repeat=N)
print("Fitness OpenMP:", problem.fitness_omp(weights))
print(f"Time (OpenMP): {np.mean(omp_times):.6f}s ± {np.std(omp_times):.6f}s")

print("-----------------------------------------------------------")

# Fitness en GPU
def gpu_function():
    return problem.fitness_gpu(weights)

gpu_times = timeit.repeat(gpu_function, number=1, repeat=N)
print("Fitness GPU:", problem.fitness_gpu(weights))
print(f"Time (GPU): {np.mean(gpu_times):.6f}s ± {np.std(gpu_times):.6f}s")

print("-----------------------------------------------------------")

# Speedups
cpu_speedup = np.mean(cpu_times) / np.mean(omp_times)
gpu_speedup = np.mean(cpu_times) / np.mean(gpu_times)
print(f"Speedup CPU vs OpenMP: {cpu_speedup:.2f}x")
print(f"Speedup CPU vs GPU: {gpu_speedup:.2f}x")
print(f"Speedup OpenMP vs GPU: {np.mean(omp_times) / np.mean(gpu_times):.2f}x")
print("-----------------------------------------------------------")



print("-----------------------Fitness de TSP------------------------")
from problems.tsp_problem import TSPProblem
from scipy.spatial.distance import cdist

def generate_distance_matrix(n_cities):
    # Genera coordenadas aleatorias (x, y) para cada ciudad en un rango de 0 a 100
    cities = np.random.rand(n_cities, 2) * 100

    # Calcula la matriz de distancias usando cdist (vectorizado y rápido)
    distance_matrix = cdist(cities, cities, "minkowski", p=2).astype(np.float32)

    return distance_matrix

# Definir número de ciudades
num_cities = 20000  # Cambia el número de ciudades aquí
dist_matrix = generate_distance_matrix(num_cities)
problem = TSPProblem(dist_matrix)

print("Number of cities:", num_cities)
print("Distance matrix shape:", dist_matrix.shape)

# Definimos una solución aleatoria (ruta)
solution = problem.generate_solution()

print("---------------------------------------------------------")
# Fitness en CPU
def cpu_function():
    return problem.fitness(solution)

cpu_times = timeit.repeat(cpu_function, number=1, repeat=N)
print("Fitness CPU:", problem.fitness(solution))
print(f"Time (CPU): {np.mean(cpu_times):.6f}s ± {np.std(cpu_times):.6f}s")

print("---------------------------------------------------------")

# Fitness con OpenMP
def omp_function():
    return problem.fitness_omp(solution)

omp_times = timeit.repeat(omp_function, number=1, repeat=N)
print("Fitness OpenMP:", problem.fitness_omp(solution))
print(f"Time (OpenMP): {np.mean(omp_times):.6f}s ± {np.std(omp_times):.6f}s")

print("---------------------------------------------------------")

# Fitness en GPU
def gpu_function():
    return problem.fitness_gpu(solution)

gpu_times = timeit.repeat(gpu_function, number=1, repeat=N)
print("Fitness GPU:", problem.fitness_gpu(solution))
print(f"Time (GPU): {np.mean(gpu_times):.6f}s ± {np.std(gpu_times):.6f}s")

print("---------------------------------------------------------")

# Speedups
cpu_speedup = np.mean(cpu_times) / np.mean(omp_times)
gpu_speedup = np.mean(cpu_times) / np.mean(gpu_times)
print(f"Speedup CPU vs OpenMP: {cpu_speedup:.2f}x")
print(f"Speedup CPU vs GPU: {gpu_speedup:.2f}x")
print(f"Speedup OpenMP vs GPU: {np.mean(omp_times) / np.mean(gpu_times):.2f}x")
print("---------------------------------------------------------")

print("..........................ALGORITMS..........................")

print("---------------------------GA-------------------------------")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

from algoritms.ga import GA

problem = ClasProblem(X_train, y_train)
ga = GA(problem, population_size=50, generations=300, seed=42)

print("Starting GA...")
start = time.time()
weights = ga.fit()
end = time.time()
print("Time:", end - start)

fit = problem.fitness_gpu(weights)
class_rate = problem.clas_rate(weights)
red_rate = problem.red_rate(weights)

print("Classification rate:", class_rate)
print("Reduction rate:", red_rate)
print("Fitness from GA:", fit)