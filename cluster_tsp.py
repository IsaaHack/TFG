from executers import ClusterExecuter
from mpi4py import MPI
import argparse
import numpy as np
from problems import TSPProblem
from algorithms import ACO, GA, PSO
import os

from tsp import read_tsp_file, extract_coords, build_distance_matrix

def main(tsp_file, timelimit=60, executer='gpu'):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    meta, node_lines, edge_lines = read_tsp_file(tsp_file)
    coords_arr = extract_coords(node_lines) if node_lines else None
    dist_matrix = build_distance_matrix(meta, coords_arr, edge_lines)

    dist_matrix_np = np.array(dist_matrix, dtype=np.float32)

    if rank == 0:
        print("Number of cities:", dist_matrix_np.shape[0])
        print("Distance matrix shape:", dist_matrix_np.shape)

    problem = TSPProblem(dist_matrix_np)

    algorithm0 = GA(problem, population_size=1024, seed=42, executer=executer, mutation_rate=0.12, crossover_rate=0.85, tournament_size=6)
    algorithm1 = ACO(problem, colony_size=1024, seed=42, executer=executer, alpha=1.2, beta=4.0, evaporation_rate=0.01)
    algorithm2 = PSO(problem, swarm_size=1024, seed=42, executer=executer, inertia_weight=0.4, cognitive_weight=0.6, social_weight=0.7)

    algorithms = [algorithm0, algorithm1, algorithm2]
    executer = ClusterExecuter(algorithms, type='master-slave')
    path = executer.execute(comm, rank, size, timelimit=timelimit, verbose=True)
    fit = problem.fitness(path)

    #Reunir resultados de todos los procesos
    results = comm.gather((rank, path, fit), root=0)

    if rank == 0:
        # Procesar resultados
        results = sorted(results, key=lambda x: -x[2])  # Ordenar por fitness
        for r in results:
            print(f"Rank {r[0]}: Path fitness = {-r[2]}")

        path = results[0][1]  # Mejor solución encontrada
        fit = results[0][2]

        print("Best path found:", path)
        print("Best fitness:", -fit)

        # Intentar cargar solución óptima si existe
        opt_path = args.tsp_file.replace(".tsp", ".opt.tour")
        if os.path.exists(opt_path):
            print(f"\nArchivo de óptimo encontrado: {opt_path}")
            with open(opt_path, 'r') as f:
                lines = f.readlines()

            opt_tour = []
            reading = False
            for line in lines:
                line = line.strip()
                if line == 'EOF':
                    break
                if reading:
                    opt_tour.extend(int(x) - 1 for x in line.split())  # TSP indexado desde 1
                if line == 'TOUR_SECTION':
                    reading = True

            # Eliminar -1 final si lo tiene
            if opt_tour and opt_tour[-1] == -2:
                opt_tour = opt_tour[:-1]

            opt_tour = np.array(opt_tour, dtype=np.int32)

            #print("Óptimo encontrado:", opt_tour)

            # Validar longitud
            if len(opt_tour) != dist_matrix_np.shape[0]:
                print("⚠️ Longitud del tour óptimo no coincide con la dimensión.")
            else:
                fitness_opt = problem.fitness(np.array(opt_tour))
                print("Fitness del óptimo:", -fitness_opt)

                # Comparar fitness
                gap = np.round(-(fit - fitness_opt) / abs(fitness_opt), 4)*100
                print("La solución encontrada es", gap, "% peor que la solución óptima")

                if gap < 0:
                    print("✅ La solución encontrada es mejor que el óptimo registrado (posible error en el óptimo)")
                elif gap > 0:
                    print("📉 La solución encontrada es peor que el óptimo registrado.")
                else:
                    print("🎯 La solución encontrada es igual al óptimo.")

        else:
            gap = np.nan
            print("\nNo se encontró archivo con la solución óptima.")

        # Si el archivo results/cluster_tsp_results.csv no existe, lo creamos
        results_file = 'results/cluster_tsp_results.csv'
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                f.write("Name,Cities,Algorithm,Executer,Timelimit,Fitness,Gap\n")

        # Guardar el nombre del archivo y los resultados
        with open(results_file, 'a') as f:
            f.write(f"{os.path.basename(tsp_file)},{dist_matrix_np.shape[0]},{algorithm0.name},{executer},{timelimit},{-fit:.4f},{gap:.4f}\n")

    MPI.Finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run TSP algorithms on a TSPlib .tsp file using MPI.'
    )
    parser.add_argument('tsp_file', help='TSPlib .tsp file path')
    parser.add_argument('-t', '--timelimit', type=int, default=60, help='Time limit for the algorithm in seconds (default: 60)')
    parser.add_argument('-e', '--executer', type=str, default='gpu', choices=['single', 'multi', 'gpu', 'hybrid'], help='Execution type: single, multi, gpu, or hybrid (default: gpu)')
    args = parser.parse_args()

    main(args.tsp_file, timelimit=args.timelimit)