import time
from problems.clas_problem import ClasProblem
from problems.tsp_problem import TSPProblem
from sklearn.datasets import load_digits
import numpy as np
from scipy.spatial.distance import cdist
import itertools
from algoritms.ga import GA

np.random.seed(42)  # Para reproducibilidad

digits = load_digits()
X_train = np.array(digits.data, dtype=np.float32)
y_train = np.array(digits.target, dtype=np.int32)

# Normalizamos los datos con un MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train = np.array(X_train, dtype=np.float32)

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

# Definir número de ciudades
num_cities = 100  # Cambia el número de ciudades aquí
dist_matrix = generate_distance_matrix(num_cities)

print("Number of cities:", num_cities)
print("Distance matrix shape:", dist_matrix.shape)
if num_cities <= 11:
    time_start = time.time()
    path, cost = tsp_optimal_solution(dist_matrix)
    time_end = time.time()

    print("Optimal path:", path)
    print("Optimal cost:", cost)
    print("Time to find optimal path:", time_end - time_start)

print("---------------------------GA-------------------------------")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

problem = ClasProblem(X_train, y_train)
problem2 = TSPProblem(dist_matrix)
ga = GA(problem, population_size=50, generations=300, seed=42, executer_type='gpu')

print("Starting GA Clas...")
start = time.time()
weights = ga.fit()
end = time.time()
print("Time:", end - start)

fit = problem.fitness(weights)
class_rate = problem.clas_rate(weights)
red_rate = problem.red_rate(weights)

print("Classification rate:", class_rate)
print("Reduction rate:", red_rate)
print("Fitness from GA:", fit)

print("-----------------------------------------------------------")

ga = GA(problem2, population_size=1024, generations=100, seed=42, mutation_rate=0.2, executer_type='gpu')
print("Starting GA TSP...")

start = time.time()
path = ga.fit()
end = time.time()
print("Time:", end - start)
fit = problem2.fitness(path)
print("Fitness from GA:", -fit)

#Verify path
print("Verifying path...")
if len(path) == len(set(path)) and np.all(np.isin(path, np.arange(num_cities))):
    print("Path is valid")