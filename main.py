from test.test_algorithms import main as test_algorithms_main
from test.test_fitness import main as test_fitness_main
from clas import main as clas_main
from tsp import main as tsp_main
from cluster_main import main as cluster_main
import argparse
import numpy as np
import os
import itertools

def main(test_type=None, algorithm=None, executer=None, tsp_file=None,
         problem=None):
    
    if test_type is None:  # Modo interactivo
        print("What would you like to test?")
        print("1. Test algorithms")
        print("2. Test fitness functions")
        print("3. Clas experiments")
        print("4. TSP experiments")
        print("5. Cluster experiments")
        print("6. Exit")
        choice = input("Enter 1, 2, 3, 4, 5, or 6: ")
    else:
        choice = test_type

    if choice == '1' or choice == 1:
        if algorithm is None:
            print('Welcome to the Algorithms test suite!')
            print("1. Genetic Algorithm (GA)")
            print("2. Ant Colony Optimization (ACO)")
            print("3. Particle Swarm Optimization (PSO)")
            algorithm_choice = input("Enter 1, 2, or 3: ")
            algorithm = {'1': 'ga', '2': 'aco', '3': 'pso'}.get(algorithm_choice)
            if algorithm is None:
                print("Invalid algorithm choice.")
                return
        if executer is None:
            print(f"Executer type for {algorithm.upper()}?")
            print("1. CPU single-threaded")
            print("2. CPU multi-threaded")
            print("3. GPU")
            print("4. Hybrid (CPU + GPU)")
            executer_choice = input("Enter 1, 2, 3, or 4: ")
            executer = {'1': 'single', '2': 'multi', '3': 'gpu', '4': 'hybrid'}.get(executer_choice)
            if executer is None:
                print("Invalid executer choice.")
                return
        
        print(f"Running tests for {algorithm.upper()} with {executer} executer...")
        test_algorithms_main(algorithm_name=algorithm, executer_type=executer)

    elif choice == '2' or choice == 2:
        print('Running tests for fitness functions...')
        test_fitness_main()

    elif choice == '3' or choice == 3:
        csv_file = "datasets/Clas/bank.csv"

        algo = ['ga', 'aco', 'pso']
        exec = ['single', 'multi', 'gpu', 'hybrid']

        timelimits = [20, 40, 60]
        iterations = [300, 600, 900]

        total_runs = len(algo) * len(exec) * (len(timelimits) + len(iterations))
        runs_completed = 0

        for a, e in itertools.product(algo, exec):
            for t in timelimits:
                i = np.inf
                print(f"Running Clas with {a.upper()} on {e} executer for {t} seconds...")
                clas_main(csv_file=csv_file, algorithm=a, executer=e, timelimit=t, iterations=i, verbose=False)
                runs_completed += 1
                print(f"Completed {runs_completed}/{total_runs} runs.")
            for i in iterations:
                t = None
                print(f"Running Clas with {a.upper()} on {e} executer for {i} iterations...")
                clas_main(csv_file=csv_file, algorithm=a, executer=e, timelimit=t, iterations=i, verbose=False)
                runs_completed += 1
                print(f"Completed {runs_completed}/{total_runs} runs.")

        # Read de results file
        results_file = 'results/clas_results.csv'
        if not os.path.exists(results_file):
            print("No results file found. Please run the tests first.")
            return
        print(f"Results saved in {results_file}.")

    elif choice == '4' or choice == 4:
        base_path = 'datasets/TSP/'
        tsp_files = ['bays29.tsp', 'eil51.tsp', 'berlin52.tsp', 'eil76.tsp', 'eil101.tsp', 'tsp225.tsp', 'pcb442.tsp']

        algo = ['ga', 'aco', 'pso']
        exec = ['single', 'multi', 'gpu', 'hybrid']

        timelimits = [30, 60]
        iterations = [500, 1000]
        total_runs = len(algo) * len(exec) * len(tsp_files) * (len(timelimits) + len(iterations))
        runs_completed = 0

        for tsp_file in tsp_files:
            for a, e in itertools.product(algo, exec):
                for t in timelimits:
                    i = np.inf
                    print(f"Running TSP with {a.upper()} on {e} executer for {t} seconds on {tsp_file}...")
                    tsp_main(tsp_file=f"{base_path}{tsp_file}", algorithm=a, executer=e, timelimit=t, iterations=i, verbose=False)
                    runs_completed += 1
                    print(f"Completed {runs_completed}/{total_runs} runs.")
                for i in iterations:
                    t = None
                    print(f"Running TSP with {a.upper()} on {e} executer for {i} iterations on {tsp_file}...")
                    tsp_main(tsp_file=f"{base_path}{tsp_file}", algorithm=a, executer=e, timelimit=t, iterations=i, verbose=False)
                    runs_completed += 1
                    print(f"Completed {runs_completed}/{total_runs} runs.")

        # Read de results file
        results_file = 'results/tsp_results.csv'
        if not os.path.exists(results_file):
            print("No results file found. Please run the tests first.")
            return
        print(f"Results saved in {results_file}.")

    elif choice == '5' or choice == 5:
        if problem is None:
            problem = input("Problem type (tsp/clas, default: tsp): ") or 'tsp'
        
        if problem == 'tsp':
            base_path = 'datasets/TSP/'
            tsp_files = ['bays29.tsp', 'eil51.tsp', 'berlin52.tsp', 'eil76.tsp', 'eil101.tsp', 'tsp225.tsp', 'pcb442.tsp']

            timelimits = [30, 60, 120]
            exec = ['single', 'multi', 'gpu', 'hybrid']
            total_runs = len(tsp_files) * len(timelimits) * len(exec)
            runs_completed = 0

            for tsp_file in tsp_files:
                for e in exec:
                    for t in timelimits:
                        print(f"Running Cluster TSP on {tsp_file} with {e} executer for {t} seconds...")
                        cluster_main(problem_file=f"{base_path}{tsp_file}", executer=e, timelimit=t, problem=problem, verbose=False)
                        runs_completed += 1
                        print(f"Completed {runs_completed}/{total_runs} runs.")

            results_file = 'results/cluster_tsp_results.csv'
            if not os.path.exists(results_file):
                print("No results file found. Please run the tests first.")
                return
            print(f"Results saved in {results_file}.")
        elif problem == 'clas':
            pass

    elif choice == '6' or choice == 6:
        print("Exiting the test suite.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run various experiment test suites.')
    parser.add_argument('-t', '--test_type', type=str, choices=['1', '2', '3', '4', '5', '6'],
                        help='Type of test to run: 1-Algorithms, 2-Fitness, 3-Clas, 4-TSP, 5-Cluster, 6-Exit')
    parser.add_argument('-a', '--algorithm', type=str, choices=['ga', 'aco', 'pso'],
                        help='Algorithm name: ga, aco, pso')
    parser.add_argument('-e', '--executer', type=str, choices=['single', 'multi', 'gpu', 'hybrid'],
                        help='Executer type: single, multi, gpu, hybrid')
    parser.add_argument('-f', '--tsp_file', type=str,
                        help='TSP problem file path')
    parser.add_argument('-p', '--problem', type=str, choices=['tsp', 'clas'],
                        help='Problem type for cluster: tsp or clas')
    parser.add_argument('-pf', '--problem_file', type=str,
                        help='Problem file for cluster')
    parser.add_argument('-n', '--nodes', nargs='+',
                        help='List of nodes for cluster execution')
    parser.add_argument('-l', '--timelimit', type=int,
                        help='Time limit in seconds for execution')

    args = parser.parse_args()
    main()