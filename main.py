from test.test_algorithms import main as test_algorithms_main
from test.test_fitness import main as test_fitness_main
from clas import main as clas_main
from tsp import main as tsp_main
from cluster_main import main as cluster_main
import argparse
import numpy as np
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt

def main(test_type=None, algorithm=None, executer=None, tsp_file=None,
         problem=None):
    
    if test_type is None:  # Modo interactivo
        print("What would you like to test?")
        print("1. Test algorithms")
        print("2. Test fitness functions")
        print("3. Clas experiments")
        print("4. TSP experiments")
        print("5. Cluster experiments")
        print("6. Analysis of results")
        print("7. Exit")
        choice = input("Enter 1, 2, 3, 4, 5, 6, or 7: ")
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

        timelimits = [30, 60]
        iterations = [300, 600]
        dataset_sizes = [500, 1000, 1500, 2000]

        total_runs = len(algo) * len(exec) * (len(timelimits) + len(iterations)) * len(dataset_sizes)
        runs_completed = 0

        for dataset_size in dataset_sizes:
            for a, e in itertools.product(algo, exec):
                for t in timelimits:
                    i = np.inf
                    print(f"Running Clas with {a.upper()} on {e} executer for {t} seconds with dataset size {dataset_size}...")
                    clas_main(csv_file=csv_file, algorithm=a, executer=e, timelimit=t, iterations=i, dataset_size=dataset_size, verbose=False)
                    runs_completed += 1
                    print(f"Completed {runs_completed}/{total_runs} runs.")
                for i in iterations:
                    t = None
                    print(f"Running Clas with {a.upper()} on {e} executer for {i} iterations with dataset size {dataset_size}...")
                    clas_main(csv_file=csv_file, algorithm=a, executer=e, timelimit=t, iterations=i, dataset_size=dataset_size, verbose=False)
                    runs_completed += 1
                    print(f"Completed {runs_completed}/{total_runs} runs.")

        # for a, e in itertools.product(algo, exec):
        #     for t in timelimits:
        #         i = np.inf
        #         print(f"Running Clas with {a.upper()} on {e} executer for {t} seconds...")
        #         clas_main(csv_file=csv_file, algorithm=a, executer=e, timelimit=t, iterations=i, verbose=False)
        #         runs_completed += 1
        #         print(f"Completed {runs_completed}/{total_runs} runs.")
        #     for i in iterations:
        #         t = None
        #         print(f"Running Clas with {a.upper()} on {e} executer for {i} iterations...")
        #         clas_main(csv_file=csv_file, algorithm=a, executer=e, timelimit=t, iterations=i, verbose=False)
        #         runs_completed += 1
        #         print(f"Completed {runs_completed}/{total_runs} runs.")

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
        print("Running analysis of results...")
        # Cargar datos
        df = pd.read_csv('results/tsp_results.csv')

        # Filtrar por Iterations = 500 o 1000
        df_filtered = df[df['Iterations'].isin([500, 1000])]

        # Crear un diccionario para almacenar los DataFrames por algoritmo
        excel_data = {}

        # Para cada algoritmo
        for algorithm in df_filtered['Algorithm'].unique():
            df_algo = df_filtered[df_filtered['Algorithm'] == algorithm]
            
            # Crear una lista para almacenar los datos ordenados
            data_frames = []

            # Primero Iterations = 500
            df_500 = df_algo[df_algo['Iterations'] == 500]
            for executer in ['single', 'multi', 'gpu', 'hybrid']:
                df_exec_500 = df_500[df_500['Executer'] == executer]
                if not df_exec_500.empty:
                    df_exec_500 = df_exec_500[['Iterations', 'Executer', 'Cities', 'Fitness', 'Gap', 'Time']]
                    data_frames.append(df_exec_500)
            
            # Después Iterations = 1000
            df_1000 = df_algo[df_algo['Iterations'] == 1000]
            for executer in ['single', 'multi', 'gpu', 'hybrid']:
                df_exec_1000 = df_1000[df_1000['Executer'] == executer]
                if not df_exec_1000.empty:
                    df_exec_1000 = df_exec_1000[['Iterations', 'Executer', 'Cities', 'Fitness', 'Gap', 'Time']]
                    data_frames.append(df_exec_1000)
            
            # Concatenar todos los resultados en un único DataFrame para este algoritmo
            final_df = pd.concat(data_frames, ignore_index=True)
            
            # Guardar en el diccionario
            excel_data[algorithm] = final_df

        # Guardar todos los DataFrames en un único archivo Excel (una hoja por algoritmo)
        with pd.ExcelWriter('results/tsp_results_summary.xlsx') as writer:
            for algorithm, data in excel_data.items():
                # Usar el nombre del algoritmo como nombre de la hoja
                sheet_name = algorithm[:31]  # máximo 31 caracteres para el nombre de hoja en Excel
                data.to_excel(writer, sheet_name=sheet_name, index=False)

        print("Archivo 'tsp_results_summary.xlsx' generado con éxito.")

        # Filtrar por Iterations = 1000
        df_filtered = df[df['Iterations'] == 1000]

        # Iterar sobre cada algoritmo
        for algorithm in df_filtered['Algorithm'].unique():
            # Filtrar los datos para el algoritmo actual
            df_algo = df_filtered[df_filtered['Algorithm'] == algorithm]
            
            # Crear una figura para el algoritmo
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Definir los ejecutores y colores
            executers = ['single', 'multi', 'gpu', 'hybrid']
            colors = ['b', 'g', 'r', 'c']

            # Gráfico normal (Tiempo vs Ciudades)
            for i, executer in enumerate(executers):
                df_exec = df_algo[df_algo['Executer'] == executer]
                df_exec = df_exec.sort_values('Cities')  # Ordenar por número de ciudades
                axes[0].plot(df_exec['Cities'], df_exec['Time'], label=executer, color=colors[i], marker='o')
            
            axes[0].set_title(f'{algorithm} - Time vs Cities (Normal Scale)')
            axes[0].set_xlabel('Number of Cities')
            axes[0].set_ylabel('Time (s)')
            axes[0].legend(title="Executer")

            # Gráfico en escala logarítmica
            for i, executer in enumerate(executers):
                df_exec = df_algo[df_algo['Executer'] == executer]
                df_exec = df_exec.sort_values('Cities')
                axes[1].plot(df_exec['Cities'], df_exec['Time'], label=executer, color=colors[i], marker='o')
            
            axes[1].set_title(f'{algorithm} - Time vs Cities (Logarithmic Scale)')
            axes[1].set_xlabel('Number of Cities')
            axes[1].set_ylabel('Time (s)')
            axes[1].set_yscale('log')
            axes[1].legend(title="Executer")
            
            plt.tight_layout()

            # Crear directorio si no existe
            output_dir = f'results/{algorithm.upper()}'
            os.makedirs(output_dir, exist_ok=True)

            # Guardar la figura
            output_path = f'{output_dir}/{algorithm}_time_vs_cities.png'
            plt.savefig(output_path)
            plt.close()

            print(f"Gráfica guardada en: {output_path}")

        # Filtrar por Iterations = 1000
        df_filtered = df[df['Iterations'] == 1000]

        # Iterar sobre cada algoritmo
        for algorithm in df_filtered['Algorithm'].unique():
            df_algo = df_filtered[df_filtered['Algorithm'] == algorithm]
            
            # Obtener los tiempos de 'single' como base
            df_single = df_algo[df_algo['Executer'] == 'single'][['Cities', 'Time']].rename(columns={'Time': 'Time_single'})

            # Merge con cada ejecutor para calcular speedup
            speedup_df = df_single.copy()
            
            for executer in ['multi', 'gpu', 'hybrid']:
                df_exec = df_algo[df_algo['Executer'] == executer][['Cities', 'Time']].rename(columns={'Time': f'Time_{executer}'})
                speedup_df = pd.merge(speedup_df, df_exec, on='Cities')

            # Calcular speedup
            for executer in ['multi', 'gpu', 'hybrid']:
                speedup_df[f'Speedup_{executer}'] = speedup_df['Time_single'] / speedup_df[f'Time_{executer}']

            # Plotear
            plt.figure(figsize=(8,6))
            colors = {'multi': 'g', 'gpu': 'r', 'hybrid': 'c'}
            
            for executer in ['multi', 'gpu', 'hybrid']:
                plt.plot(speedup_df['Cities'], speedup_df[f'Speedup_{executer}'],
                        marker='o', label=f'{executer}', color=colors[executer])
            
            plt.title(f'{algorithm} - Speedup vs Cities (Iterations=1000)')
            plt.xlabel('Number of Cities')
            plt.ylabel('Speedup over Single')
            plt.legend(title='Executer')
            plt.grid(True)
            plt.tight_layout()
            
            # Guardar gráfico
            output_dir = f'results/{algorithm.upper()}'
            os.makedirs(output_dir, exist_ok=True)
            output_path = f'{output_dir}/{algorithm}_speedup_vs_cities.png'
            plt.savefig(output_path)
            plt.close()
            
            print(f"Gráfico de speedup guardado en: {output_path}")

        # Filtrar por Timelimit = 60
        df_filtered = df[df['Timelimit'] == 60]

        # Crear carpeta base si no existe
        base_dir = 'results'

        # Iterar por algoritmo
        for algorithm in df_filtered['Algorithm'].unique():
            df_algo = df_filtered[df_filtered['Algorithm'] == algorithm]
            
            # Crear carpeta para el algoritmo
            algo_dir = os.path.join(base_dir, algorithm.upper(), 'timelimit_60')
            os.makedirs(algo_dir, exist_ok=True)
            
            # Iterar por cada archivo (Name)
            for name in df_algo['Name'].unique():
                df_name = df_algo[df_algo['Name'] == name]
                
                # Ordenar por ejecutor
                df_name = df_name.sort_values('Executer')
                
                executers = df_name['Executer'].values
                fitness = df_name['Fitness'].values
                gap = df_name['Gap'].values

                # Calcular óptimo estimado usando el primer valor
                fitness_ref = fitness[0]
                gap_ref = gap[0]
                optimo_estimado = fitness_ref / (1 + gap_ref / 100)

                # Determinar límites del eje Y para Fitness
                ymin_fitness = min(fitness.min(), optimo_estimado) * 0.98  # Margen 2%
                ymax_fitness = max(fitness.max(), optimo_estimado) * 1.02  # Margen 2%

                # Crear figura con 2 subplots: Fitness y Gap
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # --- Gráfico de Fitness ---
                bars_fitness = axes[0].bar(executers, fitness, color=['b', 'g', 'r', 'c'])
                axes[0].axhline(y=optimo_estimado, color='red', linestyle='--', label=f'Optimum ≈ {optimo_estimado:.2f}')
                axes[0].set_ylim(ymin_fitness, ymax_fitness)
                axes[0].set_title(f'{algorithm} - Fitness - {name}')
                axes[0].set_xlabel('Executer')
                axes[0].set_ylabel('Fitness')
                axes[0].grid(axis='y', linestyle='--', alpha=0.7)
                axes[0].legend()

                # Mostrar valores encima de las barras de Fitness
                for bar in bars_fitness:
                    height = bar.get_height()
                    axes[0].annotate(f'{height:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # desplazamiento 3 puntos arriba
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)

                # --- Gráfico de Gap ---
                axes[1].bar(executers, gap, color=['b', 'g', 'r', 'c'])
                axes[1].set_title(f'{algorithm} - Gap - {name}')
                axes[1].set_xlabel('Executer')
                axes[1].set_ylabel('Gap (%)')
                axes[1].grid(axis='y', linestyle='--', alpha=0.7)

                # Ajuste y guardado
                plt.tight_layout()
                plt.savefig(os.path.join(algo_dir, f'{name}_fitness_gap.png'))
                plt.close()

        print("Gráficas combinadas de Fitness y Gap guardadas en las carpetas correspondientes.")

        # Asegurar que la carpeta de destino existe
        output_dir = 'results/TSP'
        os.makedirs(output_dir, exist_ok=True)

        # Filtrar por Executer = 'gpu' y Iterations = 500 o 1000
        df_filtered = df[(df['Executer'] == 'gpu') & (df['Iterations'].isin([500, 1000]))]

        # Para cada archivo (Name)
        for name in df_filtered['Name'].unique():
            df_name = df_filtered[df_filtered['Name'] == name]
            
            # Crear la figura con dos subgráficas (una por número de iteraciones)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for idx, iterations in enumerate([500, 1000]):
                df_iter = df_name[df_name['Iterations'] == iterations]
                
                # Extraer valores para los tres algoritmos
                algorithms = ['ga', 'pso', 'aco']
                fitness_values = []
                
                for algo in algorithms:
                    df_algo = df_iter[df_iter['Algorithm'] == algo]
                    if not df_algo.empty:
                        fitness = df_algo['Fitness'].values[0]
                    else:
                        fitness = None  # Si no hay datos, se marca como None
                    fitness_values.append(fitness)
                
                # Cálculo del óptimo usando el fitness y gap de la primera fila disponible de este subconjunto
                if not df_iter.empty:
                    fitness_ref = df_iter.iloc[0]['Fitness']
                    gap_ref = df_iter.iloc[0]['Gap']
                    optimo = fitness_ref / (1 + gap_ref / 100)
                else:
                    optimo = 0  # Si no hay datos, establecer a 0 o continuar
                
                # Graficar
                axes[idx].bar(algorithms, fitness_values, color=['b', 'g', 'r'])
                axes[idx].set_title(f'{name} - Iterations: {iterations}')
                axes[idx].set_xlabel('Algorithm')
                axes[idx].set_ylabel('Fitness')
                
                # Dibujar línea roja horizontal en el óptimo calculado
                axes[idx].axhline(optimo, color='red', linestyle='--', label=f'Optimum: {optimo:.2f}')
                
                # Ajustar el rango del eje y para visualizar mejor las diferencias
                valid_fitness = [v for v in fitness_values if v is not None]
                min_fitness = min(valid_fitness + [optimo])
                max_fitness = max(valid_fitness + [optimo])
                axes[idx].set_ylim(min_fitness * 0.95, max_fitness * 1.05)  # Margen del 5%

                # Mostrar valores encima de las barras
                for i, v in enumerate(fitness_values):
                    if v is not None:
                        axes[idx].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
                
                axes[idx].legend()

            plt.tight_layout()
            plt.savefig(f'{output_dir}/{name}_fitness_comparison.png')
            plt.close()

        print("Gráficas guardadas en: results/TSP")

    elif choice == '7' or choice == 7:
        print("Exiting the test suite.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run various experiment test suites.')
    parser.add_argument('-t', '--test_type', type=str, choices=['1', '2', '3', '4', '5', '6', '7'],
                        default=None,
                        help='Type of test to run: 1-Algorithms, 2-Fitness, 3-Clas, 4-TSP, 5-Cluster, 6-Analysis, 7-Exit')
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
    main(test_type=args.test_type, 
         algorithm=args.algorithm, 
         executer=args.executer, 
         tsp_file=args.tsp_file, 
         problem=args.problem)