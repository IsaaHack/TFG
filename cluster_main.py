import argparse
from executers import cluster_execute_run

def main(problem='tsp', problem_file='', nodes=None, executer='gpu', timelimit=60, verbose=True):
    if problem == 'tsp':
        filename = 'cluster_tsp.py'
    elif problem == 'clas':
        filename = 'cluster_clas.py'
    else:
        raise ValueError(f"Unsupported problem type: {problem}. Supported types are 'tsp' and 'clas'.")
    
    if executer not in ['single', 'multi', 'gpu', 'hybrid']:
        raise ValueError(f"Unsupported executer type: {executer}. Supported types are 'single', 'multi', 'gpu', and 'hybrid'.")
    
    if timelimit <= 0:
        raise ValueError(f"Invalid timelimit: {timelimit}. It must be a positive integer.")

    program_args = problem_file.split() if problem_file else []
    program_args += ['-e', executer, '-t', str(timelimit)]
    if verbose:
        program_args.append('-v')

    if nodes is None:
        nodes = ['compute5', 'compute2', 'compute3', 'compute4']

    if verbose:
        print(f"Running {filename} on nodes: {nodes} with problem file: {problem_file}")

    cluster_execute_run(filename, nodes, program_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run a cluster script'
    )
    parser.add_argument('-p', '--problem', type=str, default='tsp', help='Problem to solve (default: tsp)')
    parser.add_argument('-pf', '--problem_file', type=str, help='Problem file path')
    parser.add_argument('-n', '--nodes', type=str, nargs='+', default=['compute5', 'compute2', 'compute3', 'compute4'], help='Nodes to run the script on (default: compute5 compute2 compute3 compute4)')
    parser.add_argument('-e', '--executer', type=str, default='gpu', choices=['single', 'multi', 'gpu', 'hybrid'], help='Execution type: single, multi, gpu, or hybrid (default: gpu)')
    parser.add_argument('-t', '--timelimit', type=int, default=60, help='Time limit for the algorithm in seconds (default: 60)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output (default: True)', default=True)
    args = parser.parse_args()
    main(problem=args.problem, 
         problem_file=args.problem_file, 
         nodes=args.nodes, 
         executer=args.executer, 
         timelimit=args.timelimit, 
         verbose=args.verbose)