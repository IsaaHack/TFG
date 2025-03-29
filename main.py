import numpy as np
from problems.clas_problem import ClasProblem
import time
from sklearn.datasets import load_digits

digits = load_digits()
X_train = np.array(digits.data, dtype=np.float32)
y_train = np.array(digits.target, dtype=np.int32)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Definimos pesos para cada una de las características
weights = np.random.rand(X_train.shape[1]).astype(np.float32)
problem = ClasProblem(X_train, y_train)

print('----------------Fitness de clasificación-------------------')

start = time.time()
fit = problem.fitness(weights)
end = time.time()


print("Fitness CPU:", fit)
print("Time:", end - start)
print("---------------------------------------------------------")

start = time.time()
fit3 = problem.fitness_omp(weights)
end = time.time()
print("Fitness OpenMP:", fit3)
print("Time:", end - start)
print("---------------------------------------------------------")

start = time.time()
fit4 = problem.fitness_gpu(weights)
end = time.time()
print("Fitness GPU:", fit4)
print("Time:", end - start)


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
solution = np.random.permutation(num_cities).astype(np.int32)

print("---------------------------------------------------------")
start = time.time()
fit = problem.fitness(solution)
end = time.time()
print("Fitness CPU:", fit)
print("Time:", end - start)
print("---------------------------------------------------------")

start = time.time()
fit3 = problem.fitness_omp(solution)
end = time.time()
print("Fitness OpenMP:", fit3)
print("Time:", end - start)
print("---------------------------------------------------------")

start = time.time()
fit4 = problem.fitness_gpu(solution)
end = time.time()
print("Fitness GPU:", fit4)
print("Time:", end - start)
print("---------------------------------------------------------")