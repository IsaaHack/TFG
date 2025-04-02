import numpy as np
from problems.clas_problem import ClasProblem
import time
from sklearn.datasets import load_digits

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
num_cities = 1000  # Cambia el número de ciudades aquí
dist_matrix = generate_distance_matrix(num_cities)
problem = TSPProblem(dist_matrix)

print("Number of cities:", num_cities)
print("Distance matrix shape:", dist_matrix.shape)

# Definimos una solución aleatoria (ruta)
solution = problem.generate_solution()

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

print("---------------------------GA-------------------------------")
#Coger el dataset de breast cancer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#Load the breast cancer dataset
breast_cancer = load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Normalizamos los datos con un MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = np.array(X, dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

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