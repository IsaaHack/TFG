'''
cluster_executer.py
This module provides the ClusterExecuter class and utility functions for executing algorithms in parallel or distributed computing environments using MPI (Message Passing Interface). It supports both master-slave and distributed execution modes, allowing for flexible parallelization strategies across multiple nodes or processes.

Classes
-------
ClusterExecuter
    A class for managing and executing one or more algorithms in a cluster environment using MPI. Supports both 'master-slave' and 'distributed' execution types.

Functions
---------
    Creates a 'hosts.txt' file listing the provided nodes, to be used as a hostfile for MPI execution.
    Deletes the 'hosts.txt' file from the current directory, handling errors if the file does not exist.
cluster_execute_run(filename, nodes, program_args=None)
    Executes a Python script across a cluster of nodes using MPI (mpirun), streaming output in real time and cleaning up the hostfile after execution.

Constants
---------
MESSAGE_TAG : int
    Tag used for standard MPI messages.
FINISH_TAG : int
    Tag used to indicate process completion in MPI communication.

Dependencies
------------
- mpi4py
- numpy
- pickle
- zlib
- subprocess
- os
- time

Usage
-----
This module is intended to be used as part of a distributed or parallel computing workflow, where algorithms need to be executed across multiple processes or nodes using MPI.
'''

from . import Executer
import os
import subprocess
import numpy as np
import pickle, zlib
from time import sleep

MESSAGE_TAG = 0
FINISH_TAG = 1

class ClusterExecuter(Executer):
    """
    ClusterExecuter is a class for executing algorithms in a parallel or distributed computing environment using MPI.
    Parameters
    ----------
    algorithms : object or list of objects
        A single algorithm instance or a list of algorithm instances to be executed in parallel.
    type : str, optional (default='master-slave')
        The execution mode. Must be either 'master-slave' or 'distributed'.
    Attributes
    ----------
    algorithms : object or list of objects
        The algorithm(s) to be executed.
    n_algorithms : int
        The number of algorithms to be executed.
    type : str
        The execution mode.
    Methods
    -------
    execute(comm, rank, size, timelimit=None, verbose=False)
        Executes the algorithms according to the specified execution type ('master-slave' or 'distributed').
    execute_master_slave(comm, rank, size, timelimit=None, verbose=False)
        Executes the algorithms in a master-slave parallelization scheme.
    execute_distributed(comm, rank, size, timelimit=None, verbose=False)
        Executes the algorithms in a distributed parallelization scheme.
    Raises
    ------
    ValueError
        If the number of algorithms is less than or equal to zero.
        If the execution type is not 'master-slave' or 'distributed'.
        If the number of algorithms does not match the number of processes (when required).
    """
    def __init__(self, algorithms, type='master-slave'):
        """
        Initializes the cluster executor with the specified algorithms and execution type.
        Args:
            algorithms (list or object): A list of algorithm instances or a single algorithm instance to be managed by the executor.
            type (str, optional): The execution type, either 'master-slave' or 'distributed'. Defaults to 'master-slave'.
        Raises:
            ValueError: If the number of algorithms is less than or equal to 0.
            ValueError: If the provided type is not 'master-slave' or 'distributed'.
        """
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
        """
        Executes the cluster task based on the specified execution type.

        Parameters:
            comm: The MPI communicator object used for inter-process communication.
            rank (int): The rank of the current process within the communicator.
            size (int): The total number of processes in the communicator.
            timelimit (Optional[float]): Optional time limit for the execution (in seconds). Defaults to None.
            verbose (bool): If True, enables verbose output. Defaults to False.

        Returns:
            The result of the execution method corresponding to the selected type.

        Raises:
            ValueError: If the execution type is unknown. Supported types are 'master-slave' and 'distributed'.
        """
        if self.type == 'master-slave':
            return self.execute_master_slave(comm, rank, size, timelimit, verbose)
        elif self.type == 'distributed':
            return self.execute_distributed(comm, rank, size, timelimit, verbose)
        else:
            raise ValueError(f"Unknown execution type: {self.type}. Available types: 'master-slave', 'distributed'.")
            
        
    def execute_master_slave(self, comm, rank, size, timelimit=None, verbose=False):
        """
        Executes a master-slave parallel optimization using MPI.

        This method coordinates the execution of one or multiple algorithms across multiple MPI processes.
        The master process (rank 0) collects results from worker processes, tracks the best solution found,
        and broadcasts improvements to all workers. Worker processes execute their assigned algorithm(s)
        and communicate results to the master.

        Parameters:
            comm (mpi4py.MPI.Comm): The MPI communicator.
            rank (int): The rank of the current MPI process.
            size (int): The total number of MPI processes.
            timelimit (float, optional): Time limit for the optimization (per process). Default is None.
            verbose (bool, optional): If True, prints progress and debug information. Default is False.

        Returns:
            np.ndarray: The best solution found (on the master process).
        
        Raises:
            ValueError: If the number of algorithms does not match the number of worker processes.
        """
        sendto = 0
        receivefrom = 0

        if rank != 0 and size > 1:
            if self.n_algorithms == 1:
                # Ejecutar un único problema con un único algoritmo
                return self.algorithms.fit_mpi(comm, rank, timelimit, sendto, receivefrom, verbose=False)
            else:
                if size-1 != self.n_algorithms:
                    raise ValueError("Number of algorithms must match the number of processes.")
                # Ejecutar un único problema con múltiples algoritmos
                return self.algorithms[rank-1].fit_mpi(comm, rank, timelimit, sendto, receivefrom, verbose=False)
        else:
            from mpi4py import MPI
            best = None
            best_fit = -np.inf
            n_improvements_rank = np.zeros(size-1, dtype=int)

            end = False
            end_number = 0

            while not end:
                status = MPI.Status()
                comm.probe(MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

                source = status.source
                message = comm.recv(source=source, tag=MPI.ANY_TAG, status=status)
                _, best_recieved, best_fit_recieved = pickle.loads(zlib.decompress(message))

                if best_fit_recieved > best_fit:
                    best = np.copy(best_recieved)
                    best_fit = best_fit_recieved
                    n_improvements_rank[source-1] += 1
                    if verbose:
                        print(f"New best from {source}: fitness = {best_fit}")
                    # Enviar el mejor resultado a todos los procesos
                    data = (rank, best, best_fit)
                    data_serialized = zlib.compress(pickle.dumps(data), level=9)
                    for i in range(1, size):
                        # Serializar el mensaje para evitar problemas con tipos de datos complejos
                        if i != source:
                            comm.send(data_serialized, dest=i, tag=MESSAGE_TAG)

                if status.tag == FINISH_TAG:
                    end_number += 1
                    if verbose:
                        print(f"Process {source} finished. Total finished: {end_number}/{size-1}")
                    if end_number == size - 1:
                        end = True
                        if verbose:
                            print("All processes finished.")

            return np.copy(best)


    def execute_distributed(self, comm, rank, size, timelimit=None, verbose=False):
        """
        Executes distributed training or evaluation of algorithms using MPI communication.

        Parameters:
            comm: MPI communicator object used for inter-process communication.
            rank (int): The rank (ID) of the current process.
            size (int): Total number of processes participating in the computation.
            timelimit (float, optional): Maximum allowed time for execution (in seconds). Defaults to None.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Returns:
            The result of the fit_mpi method from the algorithm(s), which may vary depending on implementation.

        Raises:
            ValueError: If the number of algorithms does not match the number of processes when multiple algorithms are used.

        Notes:
            - If only one algorithm is present, it is executed across all processes.
            - If multiple algorithms are present, each process executes a different algorithm.
            - The sendto and receivefrom variables determine the neighboring processes for communication in a ring topology.
        """
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
    """
    Creates a file named 'hosts.txt' and writes each node from the given list to a new line in the file.

    Args:
        nodes (list): A list of node addresses or hostnames to be written to the file.

    Returns:
        None
    """
    # Crear el archivo hosts.txt
    with open("hosts.txt", "w") as f:
        for node in nodes:
            f.write(f"{node}\n")

def delete_hosts():
    """
    Deletes the 'hosts.txt' file from the current directory.

    Attempts to remove the 'hosts.txt' file. If the file does not exist,
    an error message is printed to inform the user.

    Raises:
        OSError: If an error occurs other than the file not being found.
    """
    # Borrar el archivo hosts.txt
    try:
        os.remove("hosts.txt")
    except OSError:
        print("Error: hosts.txt file not found.")

def cluster_execute_run(filename, nodes, program_args=None):
    """
    Executes a Python script across a cluster of nodes using MPI (mpirun), streaming output in real time.
    Args:
        filename (str): The path to the Python script to execute.
        nodes (list): A list of node addresses or hostnames to include in the cluster.
        program_args (list, optional): Additional command-line arguments to pass to the Python script. Defaults to None.
    Behavior:
        - Creates a hostfile from the provided nodes.
        - Runs the script using mpirun with the specified nodes and arguments.
        - Streams the combined stdout and stderr output in real time to the console.
        - Cleans up the hostfile after execution.
    Raises:
        Any exceptions raised by subprocess.Popen or related I/O operations will propagate.
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