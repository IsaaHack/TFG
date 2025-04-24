from abc import ABC, abstractmethod
import time
import os
import json
import argparse
import subprocess
import numpy as np
import importlib

NUM_SAMPLES = 5

class Executer(ABC):
    def __init__(self):
        super().__init__()

class LocalExecuter(Executer):
    def __init__(self, problem):
        self.problem = problem

    @abstractmethod
    def execute(self, population):
        pass

class SingleCoreExecuter(LocalExecuter):
    def execute(self, population):
        return self.problem.fitness(population)

class MultiCoreExecuter(LocalExecuter):
    def execute(self, population):
        return self.problem.fitness_omp(population)

class GpuExecuter(LocalExecuter):
    def execute(self, population):
        return self.problem.fitness_gpu(population)

class HybridExecuter(LocalExecuter):
    def __init__(self, problem):
        self.problem = problem
        self.s_gpu_omp = 1

    def benchmark(self, population):
        for _ in range(NUM_SAMPLES):
            _, self.s_gpu_omp = self.problem.fitness_hybrid(population, self.s_gpu_omp)

    def execute(self, population):
        fitness_values, self.s_gpu_omp = self.problem.fitness_hybrid(population, self.s_gpu_omp)

        #self.s_gpu_omp = (self.s_gpu_omp + new_speedup) / 2
        return fitness_values
    
class ClusterExecuter(Executer):
    def __init__(self, filename: str, nodes: list,
                 problem_import: str, problem_args,
                 algorithm_import: str, algorithm_args):
        self.filename = filename
        self.nodes = nodes
        self.problem_import = problem_import
        self.problem_args = problem_args
        self.algorithm_import = algorithm_import
        self.algorithm_args = algorithm_args

    def execute(self):
        self.create_hosts()

        args = self.prepare_arguments()
        
        # Ejecutar el comando en tiempo real
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Leer y mostrar la salida en tiempo real
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                print(line, end='')  # end='' para no agregar doble salto de línea

        process.wait()  # Esperar a que termine el proceso

        self.delete_hosts()

    def create_hosts(self):
        # Crear el archivo hosts.txt
        with open("hosts.txt", "w") as f:
            for node in self.nodes:
                f.write(f"{node}\n")

    def prepare_arguments(self):
        num_nodes = len(self.nodes)
        base_cmd = [
            "mpirun", "-n", str(num_nodes), "--hostfile", "hosts.txt",
            "python3", self.filename,
            "--problem_import", self.problem_import
        ]

        # Procesar problem_args: se construye un diccionario para cada nodo
        if isinstance(self.problem_args, dict):
            # Verificar si es específico por nodo (claves 0,1,...,n-1)
            expected_keys = set(range(num_nodes))
            actual_keys = set(self.problem_args.keys())
            if expected_keys == actual_keys:
                config = self.problem_args  # Configuración diferenciada por nodo
            else:
                # Si no cumple, se asume que es una configuración común para todos
                config = {i: self.problem_args for i in range(num_nodes)}
        elif isinstance(self.problem_args, (list, str)):
            # Convertir a tokens (si es cadena se separa por espacios)
            tokens = self.problem_args if isinstance(self.problem_args, list) else self.problem_args.split()
            if len(tokens) % 2 != 0:
                raise ValueError("La configuración común debe tener pares de argumento-valor")
            common_args = {}
            for i in range(0, len(tokens), 2):
                key = tokens[i].lstrip("-")  # Quitar prefijo '-' si existe
                value = tokens[i+1]
                common_args[key] = value
            config = {i: common_args for i in range(num_nodes)}
        else:
            raise ValueError("Tipo de 'problem_args' no soportado.")

        # Serializar la configuración en JSON
        json_config = json.dumps(config)
        base_cmd += ["--problem_args", json_config]

        # Agregar el algoritmo y sus argumentos
        base_cmd += ["--algorithm_import", self.algorithm_import]

        # Procesar algoritmo_args de forma similar
        if isinstance(self.algorithm_args, dict):
            expected_keys = set(range(num_nodes))
            actual_keys = set(self.algorithm_args.keys())
            if expected_keys == actual_keys:
                algo_config = self.algorithm_args
            else:
                algo_config = {i: self.algorithm_args for i in range(num_nodes)}
        elif isinstance(self.algorithm_args, (list, str)):
            tokens = self.algorithm_args if isinstance(self.algorithm_args, list) else self.algorithm_args.split()
            if len(tokens) % 2 != 0:
                raise ValueError("La configuración común del algoritmo debe tener pares de argumento-valor")
            common_args = {}
            for i in range(0, len(tokens), 2):
                key = tokens[i].lstrip("-")
                value = tokens[i+1]
                common_args[key] = value
            algo_config = {i: common_args for i in range(num_nodes)}
        else:
            raise ValueError("Tipo de 'algorithm_args' no soportado.")

        # Serializar la configuración del algoritmo en JSON y agregarla a la línea de comando
        json_algo_config = json.dumps(algo_config)
        base_cmd += ["--algorithm_args", json_algo_config]

        return base_cmd

    def delete_hosts(self):
        # Borrar el archivo hosts.txt
        try:
            os.remove("hosts.txt")
        except OSError:
            print("Error: hosts.txt file not found.")


def cluster_executer_arg_parser(rank):
    parser = argparse.ArgumentParser()
    # Nota: Se esperan las opciones de importación en lugar de un nombre de problema
    parser.add_argument("--problem_import", type=str, required=True)
    parser.add_argument("--problem_args", type=str, required=True)
    parser.add_argument("--algorithm_import", type=str, required=True)
    parser.add_argument("--algorithm_args", type=str, required=True)
    args = parser.parse_args()

    # Decodificar la configuración JSON del problema
    problem_config = json.loads(args.problem_args)
    # Extraer la configuración específica del nodo, permitiendo claves numéricas como string o int
    node_problem_args = problem_config.get(str(rank), problem_config.get(rank))

    # Decodificar la configuración JSON del algoritmo
    algorithm_config = json.loads(args.algorithm_args)
    node_algorithm_args = algorithm_config.get(str(rank), algorithm_config.get(rank))
    
    return args.problem_import, node_problem_args, args.algorithm_import, node_algorithm_args


def cluster_executer_get_problem(problem_import, problem_args):
    """
    Carga dinámicamente el módulo y la clase especificados en problem_import.
    Se espera que problem_import sea una cadena con la ruta completa, por ejemplo:
      "problems.tsp_problem.TSPProblem"
    """
    # Dividir la cadena para separar el módulo y la clase
    try:
        module_path, class_name = problem_import.rsplit(".", 1)
    except ValueError as e:
        raise ValueError(f"El valor de problem_import ('{problem_import}') no es válido. "
                         f"Debe tener la forma 'ruta_del_módulo.NombreDeLaClase': {e}")

    # Importar el módulo
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"No se pudo importar el módulo '{module_path}': {e}")

    # Obtener la clase desde el módulo importado
    try:
        problem_class = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"El módulo '{module_path}' no define la clase '{class_name}': {e}")

    # Procesar argumentos: convertir listas a arrays de NumPy cuando sea necesario
    for key, value in problem_args.items():
        if isinstance(value, list):
            if all(isinstance(i, int) for i in value):
                problem_args[key] = np.array(value, dtype=np.int32)
            elif all(isinstance(i, float) for i in value):
                problem_args[key] = np.array(value, dtype=np.float64)
            else:
                problem_args[key] = np.array(value, dtype=np.float32)
    return problem_class(**problem_args)


def cluster_executer_get_algorithm(problem, algorithm_import, algorithm_args):
    """
    Carga dinámicamente el módulo y la clase especificados en algorithm_import.
    Se espera que algorithm_import tenga la forma:
      "algorithms.some_algorithm.SomeAlgorithm"
    """
    try:
        module_path, class_name = algorithm_import.rsplit(".", 1)
    except ValueError as e:
        raise ValueError(f"El valor de algorithm_import ('{algorithm_import}') no es válido. "
                         f"Debe tener la forma 'ruta_del_módulo.NombreDeLaClase': {e}")

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"No se pudo importar el módulo '{module_path}': {e}")

    try:
        algorithm_class = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"El módulo '{module_path}' no define la clase '{class_name}': {e}")

    # Procesar argumentos: convertir listas a arrays de NumPy cuando corresponda
    for key, value in algorithm_args.items():
        if isinstance(value, list):
            if all(isinstance(i, int) for i in value):
                algorithm_args[key] = np.array(value, dtype=np.int32)
            elif all(isinstance(i, float) for i in value):
                algorithm_args[key] = np.array(value, dtype=np.float64)
            else:
                algorithm_args[key] = np.array(value, dtype=np.float32)
    return algorithm_class(problem, **algorithm_args)

def cluster_executer_main(rank):
    problem_import, problem_args, algorithm_import, algorithm_args = cluster_executer_arg_parser(rank)
    problem = cluster_executer_get_problem(problem_import, problem_args)
    algorithm = cluster_executer_get_algorithm(problem, algorithm_import, algorithm_args)

    solution = algorithm.fit()
    fitness = problem.fitness(solution)

    print(f"[Rank {rank}] Solución: {solution}")
    print(f"[Rank {rank}] Fitness: {fitness}")