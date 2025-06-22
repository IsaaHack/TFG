import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from problems import ClasProblem
from algorithms import ACO, GA, PSO
from sklearn.model_selection import train_test_split
import numpy as np
from time import time

def read_csv_file(path):
    return pd.read_csv(path)

def preprocess_bank_marketing_dataset(df):
    try:
        pd.set_option('future.no_silent_downcasting', True)
    except Exception as e:
        pass

    # Codificar binarias como 0/1
    binary_cols = ['default', 'housing', 'loan']
    df[binary_cols] = df[binary_cols].replace({'no': 0, 'yes': 1})

    # Codificar target 'y'
    df['y'] = df['y'].replace({'no': 0, 'yes': 1})

    # Variables numéricas a escalar
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # One-hot encoding para categóricas (no binarias)
    categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df

def classify_weight(weight):
    relevancy_intervals = {
        'Low relevance': (0.1, 0.5),
        'Relevant': (0.5, 0.8),
        'Highly relevant': (0.8, 1.0),
        'No relevance': (0, 0.1)
    }

    for label, (low, high) in relevancy_intervals.items():
        if low < weight <= high:
            return label
    return 'Unknown relevance'

def main(csv_file, algorithm='aco', executer='gpu', timelimit=None, iterations=300, verbose=True):
    # Read the CSV file
    df = read_csv_file(csv_file)
    
    # Preprocess the dataset
    df = preprocess_bank_marketing_dataset(df)

    X = df.drop('y', axis=1).values
    y = df['y'].values

    X = X.astype('float32')
    y = y.astype('int32')

    dataset_size = 500

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

    # Create the problem instance
    problem = ClasProblem(X_train, y_train)

    if verbose:
        print("Dataset Information:")
        print("Number of samples:", X.shape[0])
        print("Number of features:", X.shape[1])
        print("Number of classes:", len(set(y)))

        print("Training set shape:", X_train.shape)
        print("Test set shape:", X_test.shape)

    # Initialize the algorithm based on user choice
    if algorithm == 'ga':
        algorithm_instance = GA(problem, population_size=50, seed=42, executer=executer)
    elif algorithm == 'aco':
        algorithm_instance = ACO(problem, colony_size=50, seed=42, executer=executer, 
                                 alpha=1.0, beta=1.5, evaporation_rate=0.03)
    elif algorithm == 'pso':
        algorithm_instance = PSO(problem, swarm_size=50, seed=42, executer=executer, 
                                 inertia_weight=0.9, cognitive_weight=0.4, social_weight=0.7)
    else:
        raise ValueError("Unsupported algorithm: {}".format(algorithm))
    
    # Fit the algorithm
    if verbose:
        print(f"\nRunning {algorithm.upper()} with {executer} executer...")

    start = time()
    weights = algorithm_instance.fit(iterations=iterations, timelimit=timelimit, verbose=verbose)
    end = time()
    if verbose:
        print(f"\n{algorithm.upper()} completed in {end - start:.2f} seconds.")
    fit = problem.fitness(weights)
    clas_rate = problem.clas_rate(weights)
    red_rate = problem.red_rate(weights)
    pred = problem.predict(X_test, weights)
    accuracy = (pred == y_test).mean() * 100

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
        print(f"Accuracy: {accuracy:.4f} %")

    # Save the results to a CSV file
    results_file = 'results/clas_results.csv'
    if not os.path.exists('results'):
            os.makedirs('results')
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("Algorithm,Executer,Iterations,Timelimit,Fitness,Classification Rate,Reduction Rate,Accuracy,Time\n")

    with open(results_file, 'a') as f:
        f.write(f"{algorithm},{executer},{iterations},{timelimit},{fit:.4f},{clas_rate:.4f},{red_rate:.4f},{accuracy:.4f},{end - start:.2f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract distance matrix from a TSPlib .tsp file using NumPy.")
    parser.add_argument('csv_file', help='TSPlib .tsp file path')
    parser.add_argument('-a','--algorithm', choices=['aco', 'ga', 'pso'], default='ga',
                        help='Algorithm to use (default: ga)')
    parser.add_argument('-e', '--executer', choices=['single', 'multi', 'gpu', 'hybrid'], default='gpu',
                        help='Execution type: single, multi, gpu, or hybrid (default: gpu)')
    parser.add_argument('-t', '--timelimit', type=int, default=None, help='Time limit for the algorithm in seconds (default: None)')
    parser.add_argument('-i', '--iterations', type=int, default=300, help='Number of iterations for the algorithm (default: 300)')
    args = parser.parse_args()
    if args.timelimit is not None:
        if args.timelimit <= 0:
            raise ValueError(f"Invalid timelimit: {args.timelimit}. It must be a positive integer.")
        else:
            args.iterations = np.inf
    main(args.csv_file, args.algorithm, args.executer, args.timelimit, args.iterations)