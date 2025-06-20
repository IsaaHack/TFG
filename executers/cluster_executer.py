from . import Executer
import os
import subprocess
import numpy as np
from mpi4py import MPI

MESSAGE_TAG = 0
FINISH_TAG = 1

class ClusterExecuter(Executer):
    def __init__(self, algorithms, type='master-slave'):
        super().__init__()
        self.algorithms = algorithms

        if isinstance(algorithms, list):
            self.n_algorithms = len(algorithms)
        else:
            self.n_algorithms = 1

        if self.n_algorithms <= 0:
            raise ValueError("Number of algorithms must be greater than 0.")
        
        if type not in ['master-slave', 'distributed']:
            raise ValueError("Type must be either 'master-slave' or 'distributed'.")
        
        self.type = type

    def execute(self, comm, rank, size, timelimit=None, verbose=False):
        if self.type == 'master-slave':
            return self.execute_master_slave(comm, rank, size, timelimit, verbose)
        elif self.type == 'distributed':
            return self.execute_distributed(comm, rank, size, timelimit, verbose)
        else:
            raise ValueError(f"Unknown execution type: {self.type}. Available types: 'master-slave', 'distributed'.")
            
        
    def execute_master_slave(self, comm, rank, size, timelimit=None, verbose=False):
        sendto = 0
        receivefrom = 0

        if rank != 0 and size > 1:
            if self.n_algorithms == 1:
                # Ejecutar un único problema con un único algoritmo
                return self.algorithms.fit_mpi(comm, rank, timelimit, sendto, receivefrom, verbose=verbose)
            else:
                if size != self.n_algorithms:
                    raise ValueError("Number of algorithms must match the number of processes.")
                # Ejecutar un único problema con múltiples algoritmos
                return self.algorithms[rank].fit_mpi(comm, rank, timelimit, sendto, receivefrom, verbose=verbose)
        else:
            best = None
            best_fit = -np.inf
            n_improvements_rank = np.zeros(size-1, dtype=int)

            end = False
            while not end:
                status = MPI.Status()
                comm.probe(MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

                source = status.source
                message = comm.recv(source=source, tag=MESSAGE_TAG, status=status)
                _, best_recieved, best_fit_recieved = message

                if best_fit_recieved > best_fit:
                    best = np.copy(best_recieved)
                    best_fit = best_fit_recieved
                    n_improvements_rank[source-1] += 1

                if status.tag == FINISH_TAG:
                    end = True
                else:
                    # Enviar el mejor resultado a todos los procesos
                    for i in range(1, size):
                        comm.send((rank, best, best_fit), dest=i, tag=MESSAGE_TAG)

            return np.copy(best)


    def execute_distributed(self, comm, rank, size, timelimit=None, verbose=False):
        sendto = (rank + 1) % size
        receivefrom = (rank - 1) % size

        if self.n_algorithms == 1:
            # Ejecutar un único problema con un único algoritmo
            return self.algorithms.fit_mpi(comm, rank, timelimit, sendto, receivefrom, verbose=verbose)
        else:
            if size != self.n_algorithms:
                raise ValueError("Number of algorithms must match the number of processes.")
            # Ejecutar un único problema con múltiples algoritmos
            return self.algorithms[rank].fit_mpi(comm, rank, timelimit, sendto, receivefrom, verbose=verbose)
        

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