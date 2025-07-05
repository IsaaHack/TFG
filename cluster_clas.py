from executers import ClusterExecuter
from mpi4py import MPI
import argparse
import numpy as np
from problems import ClasProblem
from algorithms import ACO, GA, PSO
import os
from sklearn.model_selection import train_test_split

from clas import read_csv_file, preprocess_bank_marketing_dataset, classify_weight

def main(csv_file, executer='gpu', timelimit=60, verbose=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Read the CSV file
    df = read_csv_file(csv_file)
    
    # Preprocess the dataset
    df = preprocess_bank_marketing_dataset(df)

    X = df.drop('y', axis=1).values
    y = df['y'].values

    X = X.astype('float32')
    y = y.astype('int32')

    # Split the dataset into training and testing sets
    if X.shape[0] > dataset_size*10/7:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=dataset_size, stratify=y, random_state=42
        )
    else:
        if verbose:
            print("Dataset has less than 2200 samples, using 70% for training and 30% for testing.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.7, stratify=y, random_state=42
        )

    dataset_size = X_train.shape[0]

    problem = ClasProblem(X_train, y_train)

    algorithm0 = GA(problem, population_size=50, seed=42, executer=executer)
    algorithm1 = ACO(problem, colony_size=50, seed=42, executer=executer, alpha=1.0, beta=1.5, evaporation_rate=0.03)
    algorithm2 = PSO(problem, swarm_size=50, seed=42, executer=executer, inertia_weight=0.9, cognitive_weight=0.4, social_weight=0.7)

    comm.barrier()  # Ensure all processes are synchronized before proceeding

    algorithms = [algorithm0, algorithm1, algorithm2]
    exec = ClusterExecuter(algorithms, type='master-slave')
    weights = exec.execute(comm, rank, size, timelimit=timelimit, verbose=verbose)
    
    # Gather results from all processes
    results = comm.gather((rank, weights), root=0)

    if rank == 0:
        # Process results
        if verbose:
            for r in results:
                print(f"Rank {r[0]}: Weights fitness = {problem.fitness(r[1])}")

        # Find the best weight
        best_weight = max(results, key=lambda x: problem.fitness(x[1]))
        if verbose:
            print("Best fitness:", problem.fitness(best_weight[1]))

        weights = best_weight[1]

        fit = problem.fitness(weights)
        clas_rate = problem.clas_rate(weights)
        red_rate = problem.red_rate(weights)
        pred = problem.predict(X_test, weights)
        accuracy = np.mean(pred == y_test) * 100

        # Filtrar las características seleccionadas
        selected_features = weights > 0.1
        feature_names = df.drop('y', axis=1).columns
        
        if verbose:
            # Imprimir todas las características junto con su peso y si son seleccionadas
            print("\nSelected Features and Weights:")
            for name, weight, selected in zip(feature_names, weights, selected_features):
                if selected:
                    print(f"{name}: {weight:.4f} - {classify_weight(weight)}")

            print("\nResults:")
            print(f"Fitness: {fit:.4f} %")
            print(f"Classification Rate: {clas_rate:.4f} %")
            print(f"Reduction Rate: {red_rate:.4f} %")
            print(f"Selected Features: {np.sum(selected_features)} out of {len(weights)}")
            print(f"Accuracy: {accuracy:.4f} %")
            print(f"Fitness Train: {accuracy*0.75 + red_rate*0.25:.4f} %")

        # Save the results to a CSV file
        results_file = 'results/cluster_clas_results.csv'
        if not os.path.exists('results'):
            os.makedirs('results')

        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                f.write("Size,Executer,Timelimit,Fitness,Classification Rate,Reduction Rate,Selected Features,Accuracy,Fitness Train\n")

        with open(results_file, 'a') as f:
            f.write(f"{dataset_size},{executer},{timelimit},{fit:.4f},{clas_rate:.4f},{red_rate:.4f},{accuracy:.4f},{accuracy*0.75 + red_rate*0.25:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a cluster script for Wine Quality classification'
    )
    parser.add_argument('csv_file', type=str, required=True, help='Path to the CSV file containing the dataset')
    parser.add_argument('-e', '--executer', type=str, default='gpu', choices=['single', 'multi', 'gpu', 'hybrid'], help='Execution type: single, multi, gpu, or hybrid (default: gpu)')
    parser.add_argument('-t', '--timelimit', type=int, default=60, help='Time limit for the algorithm in seconds (default: 60)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output (default: False)')
    args = parser.parse_args()
    
    main(executer=args.executer, timelimit=args.timelimit, verbose=args.verbose)