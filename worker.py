from mpi4py import MPI
from executers.executer import cluster_executer_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Ejecutar la función principal del ejecutor de clúster
cluster_executer_main(rank)