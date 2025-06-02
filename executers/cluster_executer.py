from . import Executer
import os
import subprocess

class ClusterExecuter(Executer):
    def __init__(self, algorithms):
        super().__init__()
        self.algorithms = algorithms

        if isinstance(algorithms, list):
            self.n_algorithms = len(algorithms)
        else:
            self.n_algorithms = 1

        if self.n_algorithms <= 0:
            raise ValueError("Number of algorithms must be greater than 0.")

    def execute(self, rank, size, timelimit=None, verbose=False):
        if self.n_algorithms == 1:
            # Ejecutar un único problema con un único algoritmo
            return self.algorithms.fit_mpi(rank, size, timelimit, verbose=verbose)
        else:
            if size != self.n_algorithms:
                raise ValueError("Number of algorithms must match the number of processes.")
            # Ejecutar un único problema con múltiples algoritmos
            return self.algorithms[rank].fit_mpi(rank, size, timelimit, verbose=verbose)
        

def create_hosts(nodes):
    # Crear el archivo hosts.txt
    with open("hosts.txt", "w") as f:
        for node in nodes:
            f.write(f"{node}\n")

def delete_hosts():
    # Borrar el archivo hosts.txt
    try:
        os.remove("hosts.txt")
    except OSError:
        print("Error: hosts.txt file not found.")

def cluster_execute_run(filename, nodes, program_args=None):
    """
    Ejecuta un script en paralelo en los nodos especificados usando MPI.
    """
    create_hosts(nodes)


    if program_args is None:
        program_args = []

    # Ejecutar el comando en tiempo real
    args = ["mpirun", "-n", str(len(nodes)), "--hostfile", "hosts.txt", "python3", filename] + program_args
    
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Leer y mostrar la salida en tiempo real
    while True:
        line = process.stdout.readline()
        if line == '' and process.poll() is not None:
            break
        if line:
            print(line, end='')  # end='' para no agregar doble salto de línea

    process.wait()  # Esperar a que termine el proceso

    delete_hosts()