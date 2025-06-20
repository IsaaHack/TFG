import argparse
from executers import cluster_execute_run

def main():
    parser = argparse.ArgumentParser(
        description='Run a cluster script'
    )
    parser.add_argument('-p', '--problem', type=str, default='tsp', help='Problem to solve (default: tsp)')
    parser.add_argument('-pf', '--problem_file', type=str, default='datasets/TSP/berlin52.tsp', help='Problem file path (default: data/berlin52.tsp)')

    args = parser.parse_args()
    if args.problem == 'tsp':
        filename = 'cluster_tsp.py'
    elif args.problem == 'clas':
        filename = 'cluster_clas.py'
    else:
        raise ValueError(f"Unsupported problem type: {args.problem}. Supported types are 'tsp' and 'clas'.")

    program_args = args.problem_file.split() if args.problem_file else []
    nodes = ['compute2', 'compute3', 'compute4']

    print(f"Running {filename} on nodes: {nodes} with problem file: {args.problem_file}")

    cluster_execute_run(filename, nodes, program_args)

if __name__ == "__main__":
    main()