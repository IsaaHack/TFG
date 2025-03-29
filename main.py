import numpy as np
from problems.clas_problem import ClasProblem
import random
import time
from sklearn.datasets import load_digits

digits = load_digits()
X_train = digits.data.tolist()
y_train = digits.target.tolist()

print("X_train shape:", np.array(X_train).shape)
print("y_train shape:", np.array(y_train).shape)

# Definimos pesos para cada una de las características
weights = [random.uniform(0, 1) for _ in range(len(X_train[0]))]
problem = ClasProblem(X_train, y_train)

print('----------------Fitness de clasificación-------------------')

start = time.time()
fit = problem.fitness(weights)
end = time.time()


print("Fitness CPU:", fit)
print("Time:", end - start)
print("---------------------------------------------------------")

start = time.time()
fit2 = problem.fitness_mpi(weights)
end = time.time()
print("Fitness MPI:", fit2)
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
import random
import numpy as np
from scipy.spatial.distance import cdist

def generate_distance_matrix(n_cities):
    # Genera coordenadas aleatorias (x, y) para cada ciudad en un rango de 0 a 100
    cities = np.random.rand(n_cities, 2) * 100  

    # Calcula la matriz de distancias usando cdist (vectorizado y rápido)
    distance_matrix = cdist(cities, cities, metric='euclidean')

    return distance_matrix

# Definir número de ciudades
num_cities = 6000  # Cambia el número de ciudades aquí
dist_matrix = generate_distance_matrix(num_cities).tolist()
problem = TSPProblem(dist_matrix)

print("Number of cities:", num_cities)
print("Distance matrix shape:", np.array(dist_matrix).shape)

# Definimos una solución aleatoria (ruta)
solution = list(range(num_cities))
random.shuffle(solution)

print("---------------------------------------------------------")
start = time.time()
fit = problem.fitness(solution)
end = time.time()
print("Fitness CPU:", fit)
print("Time:", end - start)
print("---------------------------------------------------------")

start = time.time()
fit2 = problem.fitness_mpi(solution)
end = time.time()
print("Fitness MPI:", fit2)
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