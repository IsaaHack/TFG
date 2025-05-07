import os
import argparse
import numpy as np
from algoritms.aco import ACO
from algoritms.ga import GA
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract distance matrix from a TSPlib .tsp file using NumPy.")
    parser.add_argument('tsp_file', help='TSPlib .tsp file path')
    parser.add_argument('--algorithm', choices=['aco', 'ga'], default='aco',
                        help='Algorithm to use (default: aco)')
    args = parser.parse_args()

    meta, node_lines, edge_lines = read_tsp_file(args.tsp_file)
    coords_arr = extract_coords(node_lines) if node_lines else None
    dist_matrix = build_distance_matrix(meta, coords_arr, edge_lines)

    dist_matrix_np = np.array(dist_matrix, dtype=np.float32)

    print("Number of cities:", dist_matrix_np.shape[0])
    print("Distance matrix shape:", dist_matrix_np.shape)

    problem = TSPProblem(dist_matrix_np)
    if args.algorithm == 'ga':
        print("Using Genetic Algorithm...")
        algoritm = GA(problem, population_size=1024, generations=100, seed=42, executer_type='gpu', mutation_rate=0.2)
    else:
        algoritm = ACO(problem, colony_size=1024, iterations=100000, seed=42, executer_type='multi', alpha=1.5, beta=3.0, evaporation_rate=0.01)

    print("Starting Algorithm...")
    start = time()
    path = algoritm.fit()
    end = time()
    print("Time:", end - start)
    fit = problem.fitness(path)
    print("Fitness:", -fit)

    #print("Path:", path)

    #Verify path
    print("Verifying path...")
    if len(path) == len(set(path)) and np.all(np.isin(path, np.arange(len(dist_matrix)))):
        print("Path is valid")
    else:
        print("Path is invalid")
        print("Repeated cities or out of range")
    
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
            ratio = np.round(-(fit - fitness_opt) / abs(fitness_opt), 4)*100
            print("La solución encontrada es", ratio, "% peor que la solución óptima")

            if ratio < 0:
                print("✅ La solución encontrada es mejor que el óptimo registrado (posible error en el óptimo)")
            elif ratio > 0:
                print("📉 La solución encontrada es peor que el óptimo registrado.")
            else:
                print("🎯 La solución encontrada es igual al óptimo.")

    else:
        print("\nNo se encontró archivo con la solución óptima.")

if __name__ == '__main__':
    main()