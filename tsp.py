import os
import argparse
import numpy as np
from algorithms import ACO, GA, PSO
from time import time
from problems.tsp_problem import TSPProblem

def parse_header(lines):
    """
    Parse header lines of a TSPlib file into a metadata dictionary.
    """
    meta = {}
    for line in lines:
        if ':' in line:
            key, val = line.split(':', 1)
            meta[key.strip()] = val.strip()
    return meta


def read_tsp_file(path):
    """
    Read a .tsp file, separating header, node coordinates, and explicit edge weights.
    """
    with open(path, 'r') as f:
        raw = [l.strip() for l in f if l.strip() and not l.startswith('COMMENT')]

    header_lines, node_lines, edge_lines = [], [], []
    mode = 'header'

    for line in raw:
        if line.startswith('NODE_COORD_SECTION'):
            mode = 'nodes'; continue
        if line.startswith('EDGE_WEIGHT_SECTION'):
            mode = 'edges'; continue
        if line.startswith('DISPLAY_DATA_SECTION') or line == 'EOF':
            break
        if mode == 'header':    header_lines.append(line)
        elif mode == 'nodes':   node_lines.append(line)
        else:                   edge_lines.append(line)

    return parse_header(header_lines), node_lines, edge_lines


def extract_coords(node_lines):
    """
    Convert node coordinate lines into an (n,2) NumPy array.
    """
    data = [line.split()[1:3] for line in node_lines]
    return np.array(data, dtype=float)


def build_distance_matrix(meta, coords_arr, edge_lines):
    """
    Build the distance matrix using NumPy vectorization.
    """
    n = int(meta['DIMENSION'])
    etype = meta.get('EDGE_WEIGHT_TYPE', 'EUC_2D')

    if etype == 'EXPLICIT':
        weights = np.fromiter(
            (int(w) for line in edge_lines for w in line.split()),
            dtype=int
        )
        fmt = meta.get('EDGE_WEIGHT_FORMAT', 'FULL_MATRIX')
        if fmt == 'FULL_MATRIX':
            matrix = weights.reshape((n, n))
        elif fmt in ('UPPER_ROW', 'UPPER_DIAG_ROW'):
            matrix = np.zeros((n, n), dtype=int)
            tri_k = 0 if fmt == 'UPPER_ROW' else 1
            i_idx, j_idx = np.triu_indices(n, k=tri_k)
            if len(weights) != len(i_idx):
                raise ValueError(f"Mismatch between weights length ({len(weights)}) and expected elements ({len(i_idx)}) for {fmt}.")
            matrix[i_idx, j_idx] = weights
            matrix[j_idx, i_idx] = weights
        else:
            raise NotImplementedError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")
    else:
        if coords_arr is None:
            raise ValueError("Coordinates required for metric distances.")
        # Compute pairwise Euclidean distances
        diff = coords_arr[:, None, :] - coords_arr[None, :, :]
        dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        matrix = dist

    return matrix


def main(tsp_file, algorithm, executer='gpu', iterations=None, timelimit=None, verbose=True):
    meta, node_lines, edge_lines = read_tsp_file(tsp_file)
    coords_arr = extract_coords(node_lines) if node_lines else None
    dist_matrix = build_distance_matrix(meta, coords_arr, edge_lines)

    dist_matrix_np = np.array(dist_matrix, dtype=np.float32)

    if verbose:
        print("Number of cities:", dist_matrix_np.shape[0])
        print("Distance matrix shape:", dist_matrix_np.shape)

    if executer == 'hybrid':
        problem_executer = 'gpu'
    else:
        problem_executer = executer

    problem = TSPProblem(dist_matrix_np, executer=problem_executer)
    if algorithm == 'ga':
        if verbose:
            print("Using Genetic Algorithm...")
        algoritm = GA(problem, population_size=1024, seed=42, executer=executer, mutation_rate=0.12, crossover_rate=0.85, tournament_size=6)
        if iterations is None:
            iterations = 1000
    elif algorithm == 'aco':
        if verbose:
            print("Using Ant Colony Optimization...")
        algoritm = ACO(problem, colony_size=1024, seed=42, executer=executer, alpha=1.2, beta=4.0, evaporation_rate=0.01)
        if iterations is None:
            iterations = 1000
    elif algorithm == 'pso':
        if verbose:
            print("Using Particle Swarm Optimization...")
        algoritm = PSO(problem, swarm_size=1024, seed=42, executer=executer, inertia_weight=0.4, cognitive_weight=0.6, social_weight=0.7)
        if iterations is None:
            iterations = 1000

    if verbose:
        print("Starting Algorithm...")
    start = time()
    path = algoritm.fit(iterations, timelimit=timelimit, verbose=verbose)
    end = time()
    fit = problem.fitness(path)

    if verbose:
        print("Time:", end - start)
        print("Fitness:", -fit)
        print("Path:", path)

        #Verify path
        print("Verifying path...")
        if len(path) == len(set(path)) and np.all(np.isin(path, np.arange(len(dist_matrix)))):
            print("Path is valid")
        else:
            print("Path is invalid")
            print("Repeated cities or out of range")
    
    # Intentar cargar solución óptima si existe
    opt_path = tsp_file.replace(".tsp", ".opt.tour")
    if os.path.exists(opt_path):
        if verbose:
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
            if verbose:
                raise ValueError("Length of optimal tour does not match the number of cities in the distance matrix.")
        else:
            fitness_opt = problem.fitness(np.array(opt_tour))
            gap = np.round(-(fit - fitness_opt) / abs(fitness_opt), 4)*100

            if verbose:
                print("Fitness del óptimo:", -fitness_opt)
                # Comparar fitness
                print("La solución encontrada es", gap, "% peor que la solución óptima")
                if gap < 0:
                    print("✅ La solución encontrada es mejor que el óptimo registrado (posible error en el óptimo)")
                elif gap > 0:
                    print("📉 La solución encontrada es peor que el óptimo registrado.")
                else:
                    print("🎯 La solución encontrada es igual al óptimo.")

    else:
        gap = np.nan
        if verbose:
            print("\nNo se encontró archivo con la solución óptima.")

    # Si el archivo results/tsp_results.csv no existe, lo creamos
    results_file = 'results/tsp_results.csv'
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("Name,Cities,Algorithm,Executer,Iterations,Timelimit,Fitness,Gap,Time\n")
    with open(results_file, 'a') as f:
        f.write(f"{os.path.splitext(os.path.basename(tsp_file))[0]},{dist_matrix_np.shape[0]},{algorithm},{executer},{iterations},{timelimit},{-fit},{gap},{end - start:.2f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract distance matrix from a TSPlib .tsp file using NumPy.")
    parser.add_argument('tsp_file', help='TSPlib .tsp file path')
    parser.add_argument('-a','--algorithm', choices=['aco', 'ga', 'pso'], default='aco',
                        help='Algorithm to use (default: aco)')
    parser.add_argument('-e', '--executer', choices=['single', 'multi', 'gpu', 'hybrid'], default='gpu',
                        help='Execution type: single, multi, gpu, or hybrid (default: gpu)')
    parser.add_argument('-i', '--iterations', type=int, default=None,
                        help='Number of iterations for the algorithm (default: None, will use default for each algorithm)')
    parser.add_argument('-t', '--timelimit', type=int, default=None,
                        help='Time limit for the algorithm in seconds (default: None, will use default for each algorithm)')
    args = parser.parse_args()
    main(args.tsp_file, args.algorithm, args.executer, args.iterations, args.timelimit)