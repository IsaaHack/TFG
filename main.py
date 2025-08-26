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
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

def main(test_type=None, algorithm=None, executer=None, tsp_file=None,
         problem=None):
    
    if test_type is None:  # Modo interactivo
        print("What would you like to test?")
        print("1. Test algorithms")
        print("2. Test fitness functions")
        print("3. Clas experiments")
        print("4. TSP experiments")
        print("5. Clas Cluster experiments")
        print("6. TSP Cluster experiments")
        print("7. Analysis of results")
        print("8. Exit")
        choice = input("Enter 1, 2, 3, 4, 5, 6, 7 or 8: ")
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
        exec = ['single', 'multi', 'hybrid', 'gpu']

        timelimits = [60]
        iterations = [300]
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

        # Read de results file
        results_file = 'results/clas_results.csv'
        if not os.path.exists(results_file):
            print("No results file found. Please run the tests first.")
            return
        print(f"Results saved in {results_file}.")

    elif choice == '4' or choice == 4:
        csv_file = "datasets/Clas/bank.csv"

        exec = ['single', 'multi', 'hybrid', 'gpu']

        timelimits = [60]
        dataset_sizes = [500, 1000, 1500, 2000, 2500]

        total_runs = len(exec) * (len(timelimits)) * len(dataset_sizes)
        runs_completed = 0

        for dataset_size in dataset_sizes:
            for e in exec:
                for t in timelimits:
                    print(f"Running Cluster Clas with {e} executer for {t} seconds and dataset size {dataset_size}...")
                    cluster_main(problem='clas', problem_file=csv_file, executer=e, timelimit=t, dataset_size=dataset_size, verbose=False)
                    runs_completed += 1
                    print(f"Completed {runs_completed}/{total_runs} runs.")

        # Read de results file
        results_file = 'results/cluster_clas_results.csv'
        if not os.path.exists(results_file):
            print("No results file found. Please run the tests first.")
            return
        print(f"Results saved in {results_file}.")

    elif choice == '5' or choice == 5:
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

    elif choice == '6' or choice == 6:
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

    elif choice == '7' or choice == 7:
        print("Running analysis of results...")
        analysis_TSP()
        analysis_Clas()
        analysis_TSP_Statistic()
        analysis_Clas_Statistic()

    elif choice == '8' or choice == 8:
        print("Exiting the test suite.")
        return

def analysis_TSP():
    print("Starting analysis of TSP results...")

    # Cargar datos
    df = pd.read_csv('results/tsp_results.csv')

    # Filtrar por Iterations = 1000 o Timelimit = 60
    df_filtered = df[(df['Iterations'] == 1000) | (df['Iterations'] == 500) | (df['Timelimit'] == 60)]

    # Crear un diccionario para almacenar los DataFrames por algoritmo y condición
    excel_data = {}

    # Para cada algoritmo
    for algorithm in df_filtered['Algorithm'].unique():
        df_algo = df_filtered[df_filtered['Algorithm'] == algorithm]

        # Separar por Timelimit = 60
        df_time_60 = df_algo[df_algo['Timelimit'] == 60]
        data_time = []

        for executer in ['single', 'multi', 'gpu', 'hybrid']:
            df_exec = df_time_60[df_time_60['Executer'] == executer]
            if not df_exec.empty:
                df_exec = df_exec[['Timelimit', 'Executer', 'Cities', 'Fitness', 'Gap', 'Time']]
                data_time.append(df_exec)

        final_time_df = pd.concat(data_time, ignore_index=True) if data_time else pd.DataFrame()

        # Separar por Iterations = 500
        df_iter_500 = df_algo[df_algo['Iterations'] == 500]
        data_iter = []

        for executer in ['single', 'multi', 'gpu', 'hybrid']:
            df_exec = df_iter_500[df_iter_500['Executer'] == executer]
            if not df_exec.empty:
                df_exec = df_exec[['Iterations', 'Executer', 'Cities', 'Fitness', 'Gap', 'Time']]
                data_iter.append(df_exec)

        final_iter500_df = pd.concat(data_iter, ignore_index=True) if data_iter else pd.DataFrame()

        # Separar por Iterations = 1000
        df_iter_1000 = df_algo[df_algo['Iterations'] == 1000]
        data_iter = []

        for executer in ['single', 'multi', 'gpu', 'hybrid']:
            df_exec = df_iter_1000[df_iter_1000['Executer'] == executer]
            if not df_exec.empty:
                df_exec = df_exec[['Iterations', 'Executer', 'Cities', 'Fitness', 'Gap', 'Time']]
                data_iter.append(df_exec)

        final_iter_df = pd.concat(data_iter, ignore_index=True) if data_iter else pd.DataFrame()

        # Guardar en el diccionario con nombre de hoja específico
        if not final_time_df.empty:
            excel_data[f"{algorithm}_60s"] = final_time_df
        if not final_iter500_df.empty:
            excel_data[f"{algorithm}_500it"] = final_iter500_df
        if not final_iter_df.empty:
            excel_data[f"{algorithm}_1000it"] = final_iter_df

    # Guardar en un archivo Excel con múltiples hojas
    with pd.ExcelWriter('results/tsp_results_summary.xlsx') as writer:
        for sheet_name, data in excel_data.items():
            sheet_name = sheet_name[:31]  # Excel no permite nombres de hoja > 31 caracteres
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Archivo 'tsp_results_summary.xlsx' generado con éxito.")

    # Cargar df_cluster
    df_cluster = pd.read_csv('results/cluster_tsp_results.csv')

    # Filtrar por Timelimit 60 y 120
    df_filtered = df_cluster[df_cluster['Timelimit'].isin([60, 120])]

    # Crear un diccionario para almacenar los DataFrames por Timelimit y Executer
    excel_data = {}

    for timelimit in [60, 120]:
        df_time = df_filtered[df_filtered['Timelimit'] == timelimit]
        if not df_time.empty:
            # Seleccionar columnas relevantes
            df_time = df_time[['Timelimit', 'Executer', 'Cities', 'Fitness', 'Gap']]
            excel_data[f"Timelimit_{timelimit}"] = df_time

    # Guardar en Excel con múltiples hojas
    output_file = 'results/tsp_cluster_results_summary.xlsx'
    os.makedirs('results', exist_ok=True)
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, data in excel_data.items():
            # Limitar el nombre de hoja a 31 caracteres (restricción de Excel)
            sheet_name_safe = sheet_name[:31]
            data.to_excel(writer, sheet_name=sheet_name_safe, index=False)

    print(f"Archivo '{output_file}' generado con éxito.")

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
        output_dir = f'results/TSP/{algorithm.upper()}'
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
        output_dir = f'results/TSP/{algorithm.upper()}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{algorithm}_speedup_vs_cities.png'
        plt.savefig(output_path)
        plt.close()
        
        print(f"Gráfico de speedup guardado en: {output_path}")

    # Filtrar por Timelimit = 60
    df_filtered = df[df['Timelimit'] == 60]

    # Crear carpeta base si no existe
    base_dir = 'results/TSP'

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

    df_cluster = pd.read_csv('results/cluster_tsp_results.csv')

    # Filtrar por timelimit = 60 y con executer = 'gpu'
    df_filtered = df[(df['Timelimit'] == 60) & (df['Executer'] == 'gpu')]
    df_cluster_filtered = df_cluster[(df_cluster['Timelimit'] == 60) & (df_cluster['Executer'] == 'gpu')]

    # Asegurar que Name es el identificador
    names = df_filtered['Name'].unique()

    for name in names:
        # Subconjuntos
        subset_df = df_filtered[df_filtered['Name'] == name]
        subset_cluster = df_cluster_filtered[df_cluster_filtered['Name'] == name]
        
        # Construcción del dataframe comparativo
        data = []
        for _, row in subset_df.iterrows():
            data.append([row['Algorithm'], row['Fitness'], row['Gap']])
        
        for _, row in subset_cluster.iterrows():
            data.append([f"Cluster", row['Fitness'], row['Gap']])
        
        compare_df = pd.DataFrame(data, columns=['Algorithm', 'Fitness', 'Gap'])
        
        # ====== Calcular el óptimo ======
        if not compare_df.empty:
            fitness_ref = compare_df.iloc[0]['Fitness']
            gap_ref = compare_df.iloc[0]['Gap']
            optimo = fitness_ref / (1 + gap_ref / 100)
        else:
            optimo = 0

        # Crear figura con 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        colors = ['b', 'g', 'r', 'c']  # colores fijos
        
        # --- FITNESS ---
        bars_fit = axes[0].bar(compare_df['Algorithm'], compare_df['Fitness'], color=colors[:len(compare_df)])
        axes[0].set_title(f'Fitness - {name}')
        axes[0].set_ylabel('Fitness')
        axes[0].tick_params(axis='x', rotation=45)

        # Línea roja del óptimo
        axes[0].axhline(optimo, color='red', linestyle='--', linewidth=1.5, label=f'Optimum: {optimo:.2f}')
        axes[0].legend(loc='lower center')

        # Mostrar valores encima de las barras
        for bar, val in zip(bars_fit, compare_df['Fitness']):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height(),
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=9)

        # --- GAP ---
        bars_gap = axes[1].bar(compare_df['Algorithm'], compare_df['Gap'], color=colors[:len(compare_df)])
        axes[1].set_title(f'Gap - {name}')
        axes[1].set_ylabel('Gap')
        axes[1].tick_params(axis='x', rotation=45)

        # Mostrar valores encima de las barras
        for bar, val in zip(bars_gap, compare_df['Gap']):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height(),
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=9)

        # Ajustar y guardar
        plt.suptitle(f'Comparación de Fitness y Gap - {name}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{output_dir}/{name}_fitness_gap_fullcomparison.png')
        plt.close()

    print("Gráficas comparativas de Cluster guardadas en: results/TSP")

    # Asegurar que la carpeta existe
    output_dir_cluster = "results/TSP/Cluster"
    os.makedirs(output_dir_cluster, exist_ok=True)

    df_cluster_filtered = df_cluster[df_cluster['Timelimit'].isin([60, 120])]

    # Colores fijos por executer (mismos que la gráfica anterior)
    executer_colors = {
        "single": "b",   # azul
        "multi": "g",    # verde
        "gpu": "r",      # rojo
        "hybrid": "c"    # cian
    }

    # Iterar sobre cada Name
    for name in df_cluster_filtered['Name'].unique():
        df_name = df_cluster_filtered[df_cluster_filtered['Name'] == name]

        # Calcular rango común de Gap para ambos Timelimits
        min_gap = df_name['Gap'].min()
        max_gap = df_name['Gap'].max()
        margin = (max_gap - min_gap) * 0.05  # margen 5%
        ylim_min = max(min_gap - margin, 0)
        ylim_max = max_gap + margin

        # Crear figura con dos subplots (Timelimit = 60 y 120)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, t in enumerate([60, 120]):
            df_t = df_name[df_name['Timelimit'] == t]

            if df_t.empty:
                continue

            executers = df_t['Executer'].values
            gap_values = df_t['Gap'].values

            # Colores según executer
            colors = [executer_colors.get(exec, "gray") for exec in executers]

            # Graficar barras
            bars = axes[idx].bar(executers, gap_values, color=colors)

            axes[idx].set_title(f'{name} - Timelimit: {t}')
            axes[idx].set_xlabel('Executer')
            axes[idx].set_ylabel('Gap')
            axes[idx].tick_params(axis='x', rotation=30)
            if ylim_min < ylim_max:
                axes[idx].set_ylim(ylim_min, ylim_max)  # mismo rango para ambos subplots

            # Mostrar valores encima de las barras
            for bar, val in zip(bars, gap_values):
                axes[idx].text(bar.get_x() + bar.get_width()/2,
                            bar.get_height(),
                            f'{val:.2f}',
                            ha='center', va='bottom', fontsize=9)

        # Título general
        plt.suptitle(f'Comparación de Gap en Executers - {name}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{output_dir_cluster}/{name}_executer_gap_comparison.png")
        plt.close()

    print("Gráficas de Cluster guardadas en: results/TSP/Cluster")

def analysis_Clas():
    print("Starting analysis of Clas results...")
    
    # Cargar datos
    df = pd.read_csv('results/clas_results.csv')
    df_cluster = pd.read_csv('results/cluster_clas_results.csv')

    # Filtrar por Iterations = 300 o Timelimit = 60
    df_filtered = df[(df['Iterations'] == 300) | (df['Timelimit'] == 60)]

    # Crear un diccionario para almacenar los DataFrames por algoritmo y condición
    excel_data = {}

    # Para cada algoritmo
    for algorithm in df_filtered['Algorithm'].unique():
        df_algo = df_filtered[df_filtered['Algorithm'] == algorithm]

        # Separar por Timelimit = 60
        df_time_60 = df_algo[df_algo['Timelimit'] == 60]
        data_time = []

        for executer in ['single', 'multi', 'gpu', 'hybrid']:
            df_exec = df_time_60[df_time_60['Executer'] == executer]
            if not df_exec.empty:
                df_exec = df_exec[['Iterations', 'Executer', 'Size', 'Fitness Test', 'Accuracy', 'Time']]
                data_time.append(df_exec)

        final_time_df = pd.concat(data_time, ignore_index=True) if data_time else pd.DataFrame()

        # Separar por Iterations = 300
        df_iter_300 = df_algo[df_algo['Iterations'] == 300]
        data_iter = []

        for executer in ['single', 'multi', 'gpu', 'hybrid']:
            df_exec = df_iter_300[df_iter_300['Executer'] == executer]
            if not df_exec.empty:
                df_exec = df_exec[['Iterations', 'Executer', 'Size', 'Fitness Test', 'Accuracy', 'Time']]
                data_iter.append(df_exec)

        final_iter300_df = pd.concat(data_iter, ignore_index=True) if data_iter else pd.DataFrame()

        # Guardar en el diccionario con nombre de hoja específico
        if not final_time_df.empty:
            excel_data[f"{algorithm}_60s"] = final_time_df
        if not final_iter300_df.empty:
            excel_data[f"{algorithm}_300it"] = final_iter300_df
    
    # Guardar en un archivo Excel con múltiples hojas
    with pd.ExcelWriter('results/clas_results_summary.xlsx') as writer:
        for sheet_name, data in excel_data.items():
            sheet_name = sheet_name[:31]  # Excel no permite nombres de hoja > 31 caracteres
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Archivo 'clas_results_summary.xlsx' generado con éxito.")

    # Cargar df_cluster
    df_cluster = pd.read_csv('results/cluster_clas_results.csv')

    # Filtrar por Timelimit 60
    df_filtered = df_cluster[df_cluster['Timelimit'] == 60]
    # Crear un diccionario para almacenar los DataFrames por Timelimit y Executer
    excel_data = {}
    for timelimit in [60]:
        df_time = df_filtered[df_filtered['Timelimit'] == timelimit]
        if not df_time.empty:
            # Seleccionar columnas relevantes
            df_time = df_time[['Timelimit', 'Executer', 'Size', 'Fitness Test', 'Accuracy']]
            excel_data[f"Timelimit_{timelimit}"] = df_time
    # Guardar en Excel con múltiples hojas
    output_file = 'results/clas_cluster_results_summary.xlsx'
    os.makedirs('results', exist_ok=True)
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, data in excel_data.items():
            # Limitar el nombre de hoja a 31 caracteres (restricción de Excel)
            sheet_name_safe = sheet_name[:31]
            data.to_excel(writer, sheet_name=sheet_name_safe, index=False)
    print(f"Archivo '{output_file}' generado con éxito.")

    # Crear la carpeta de resultados si no existe
    output_dir = 'results/Clas'
    os.makedirs(output_dir, exist_ok=True)

    # Filtrar por Iterations = 300
    df_filtered = df[df['Iterations'] == 300]

    # Gráfica de rendimiento (Time vs Size) para cada algoritmo
    for algorithm in df_filtered['Algorithm'].unique():
        df_algo = df_filtered[df_filtered['Algorithm'] == algorithm]
        
        # Crear una figura para el algoritmo
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        executers = ['single', 'multi', 'gpu', 'hybrid']
        colors = ['b', 'g', 'r', 'c']

        for i, executer in enumerate(executers):
            df_exec = df_algo[df_algo['Executer'] == executer]
            df_exec = df_exec.sort_values('Size')
            axes[0].plot(df_exec['Size'], df_exec['Time'], label=executer, color=colors[i], marker='o')

        axes[0].set_title(f'{algorithm} - Time vs Size (Normal Scale)')
        axes[0].set_xlabel('Dataset Size')
        axes[0].set_ylabel('Time (s)')
        axes[0].legend(title="Executer")

        # Gráfico en escala logarítmica
        for i, executer in enumerate(executers):
            df_exec = df_algo[df_algo['Executer'] == executer]
            df_exec = df_exec.sort_values('Size')
            axes[1].plot(df_exec['Size'], df_exec['Time'], label=executer, color=colors[i], marker='o')

        axes[1].set_title(f'{algorithm} - Time vs Size (Logarithmic Scale)')
        axes[1].set_xlabel('Dataset Size')
        axes[1].set_ylabel('Time (s)')
        axes[1].set_yscale('log')
        axes[1].legend(title="Executer")
        plt.tight_layout()

        # Crear directorio si no existe
        output_dir_algo = f'results/Clas/{algorithm.upper()}'
        os.makedirs(output_dir_algo, exist_ok=True)
        # Guardar la figura
        output_path = f'{output_dir_algo}/{algorithm}_time_vs_size.png'
        plt.savefig(output_path)
        plt.close()

        print(f"Gráfica guardada en: {output_path}")

    # Filtrar por Iterations = 300
    df_filtered = df[df['Iterations'] == 300]

    # Grafica de speedup
    for algorithm in df_filtered['Algorithm'].unique():
        df_algo = df_filtered[df_filtered['Algorithm'] == algorithm]
        
        # Obtener los tiempos de 'single' como base
        df_single = df_algo[df_algo['Executer'] == 'single'][['Size', 'Time']].rename(columns={'Time': 'Time_single'})

        # Merge con cada ejecutor para calcular speedup
        speedup_df = df_single.copy()
        
        for executer in ['multi', 'gpu', 'hybrid']:
            df_exec = df_algo[df_algo['Executer'] == executer][['Size', 'Time']].rename(columns={'Time': f'Time_{executer}'})
            speedup_df = pd.merge(speedup_df, df_exec, on='Size')

        # Calcular speedup
        for executer in ['multi', 'gpu', 'hybrid']:
            speedup_df[f'Speedup_{executer}'] = speedup_df['Time_single'] / speedup_df[f'Time_{executer}']

        # Plotear
        plt.figure(figsize=(8,6))
        colors = {'multi': 'g', 'gpu': 'r', 'hybrid': 'c'}
        
        for executer in ['multi', 'gpu', 'hybrid']:
            plt.plot(speedup_df['Size'], speedup_df[f'Speedup_{executer}'],
                    marker='o', label=f'{executer}', color=colors[executer])
        
        plt.title(f'{algorithm} - Speedup vs Size (Iterations=300)')
        plt.xlabel('Dataset Size')
        plt.ylabel('Speedup over Single')
        plt.legend(title='Executer')
        plt.grid(True)
        plt.tight_layout()
        
        # Guardar gráfico
        output_dir_algo = f'results/Clas/{algorithm.upper()}'
        os.makedirs(output_dir_algo, exist_ok=True)
        output_path = f'{output_dir_algo}/{algorithm}_speedup_vs_size.png'
        plt.savefig(output_path)
        plt.close()
        
        print(f"Gráfico de speedup guardado en: {output_path}")

    # Filtrar por Timelimit = 60
    df_filtered = df[df['Timelimit'] == 60]

    # Crear carpeta base si no existe
    base_dir = 'results/Clas'

    # Gráficas de productividad (Fitness Test y Accuracy)
    for algorithm in df_filtered['Algorithm'].unique():
        df_algo = df_filtered[df_filtered['Algorithm'] == algorithm]
        
        # Crear carpeta para el algoritmo
        algo_dir = os.path.join(base_dir, algorithm.upper())
        os.makedirs(algo_dir, exist_ok=True)

        # Obtener ejecutores únicos
        executers = df_algo['Executer'].unique()
        
        # Crear figura con 2 subplots: Fitness Test y Accuracy
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Gráfico de Fitness Test ---
        for executer in executers:
            df_exec = df_algo[df_algo['Executer'] == executer].sort_values('Size')
            axes[0].plot(df_exec['Size'], df_exec['Fitness Test'], marker='o', label=executer)
        ymin_fitness = df_algo['Fitness Test'].min()
        axes[0].set_ylim(70, 100)
        axes[0].set_title(f'{algorithm} - Fitness Test vs Train Size')
        axes[0].set_xlabel('Train Size')
        axes[0].set_ylabel('Fitness Test')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].legend()

        # --- Gráfico de Accuracy ---
        for executer in executers:
            df_exec = df_algo[df_algo['Executer'] == executer].sort_values('Size')
            axes[1].plot(df_exec['Size'], df_exec['Accuracy'], marker='o', label=executer)
        y_min_accuracy = df_algo['Accuracy'].min()
        axes[1].set_ylim(70, 100)
        axes[1].set_title(f'{algorithm} - Accuracy vs Train Size')
        axes[1].set_xlabel('Train Size')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend()

        # Ajuste y guardado
        plt.tight_layout()
        plt.savefig(os.path.join(algo_dir, f'{algorithm}_fitness_accuracy_lines.png'))
        plt.close()

    print("Gráficas combinadas de Fitness Test y Accuracy guardadas en las carpetas correspondientes.")

    # FIltrar por Executer = 'gpu' y Iterations = 300
    df_filtered = df[(df['Executer'] == 'gpu') & (df['Iterations'] == 300)]

    # Crear figura con 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Lista de algoritmos a comparar
    algorithms = df_filtered['Algorithm'].unique()

    # --- Fitness Test ---
    for algorithm in algorithms:
        df_algo = df_filtered[df_filtered['Algorithm'] == algorithm].sort_values('Size')
        axes[0].plot(df_algo['Size'], df_algo['Fitness Test'], marker='o', label=algorithm.upper())

    ymin_fitness = df_filtered['Fitness Test'].min()
    axes[0].set_ylim(ymin_fitness, 100)
    axes[0].set_title('Fitness Test vs Train Size (GPU, Iter=300)')
    axes[0].set_xlabel('Train Size')
    axes[0].set_ylabel('Fitness Test')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # --- Accuracy ---
    for algorithm in algorithms:
        df_algo = df_filtered[df_filtered['Algorithm'] == algorithm].sort_values('Size')
        axes[1].plot(df_algo['Size'], df_algo['Accuracy'], marker='o', label=algorithm.upper())

    ymin_accuracy = df_filtered['Accuracy'].min()
    axes[1].set_ylim(ymin_accuracy, 100)
    axes[1].set_title('Accuracy vs Train Size (GPU, Iter=300)')
    axes[1].set_xlabel('Train Size')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    # Ajustar y guardar
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fitness_accuracy_gpu_comparison.png')
    plt.close()

    print("Gráfica comparativa de Fitness Test y Accuracy guardada en: results/Clas")

    # Cargar resultados del cluster
    df_cluster = pd.read_csv("results/cluster_clas_results.csv")

    # Filtrar por Timelimit = 60
    df_filtered = df_cluster[df_cluster['Timelimit'] == 60]

    # Crear carpeta base si no existe
    base_dir = 'results/Clas/Cluster'
    os.makedirs(base_dir, exist_ok=True)

    # Obtener ejecutores únicos
    executers = df_filtered['Executer'].unique()

    # Crear figura con 2 subplots: Fitness Test y Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Gráfico de Fitness Test ---
    for executer in executers:
        df_exec = df_filtered[df_filtered['Executer'] == executer].sort_values('Size')
        axes[0].plot(df_exec['Size'], df_exec['Fitness Test'], marker='o', label=executer)

    ymin_fitness = df_filtered['Fitness Test'].min()
    axes[0].set_ylim(ymin_fitness, 100)
    axes[0].set_title('Fitness Test vs Train Size (Cluster)')
    axes[0].set_xlabel('Train Size')
    axes[0].set_ylabel('Fitness Test')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # --- Gráfico de Accuracy ---
    for executer in executers:
        df_exec = df_filtered[df_filtered['Executer'] == executer].sort_values('Size')
        axes[1].plot(df_exec['Size'], df_exec['Accuracy'], marker='o', label=executer)

    ymin_accuracy = df_filtered['Accuracy'].min()
    axes[1].set_ylim(ymin_accuracy, 100)
    axes[1].set_title('Accuracy vs Train Size (Cluster)')
    axes[1].set_xlabel('Train Size')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    # Ajuste y guardado
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fitness_accuracy_lines_cluster.png"))
    plt.close()

    print("Gráficas de líneas (Cluster) guardadas en la carpeta correspondiente.")

    # Cargar resultados
    df_normal = pd.read_csv("results/clas_results.csv")          # CSV con GA, PSO, ACO
    df_cluster = pd.read_csv("results/cluster_clas_results.csv") # CSV del cluster

    # Filtrar por Timelimit = 60
    df_normal = df_normal[df_normal['Timelimit'] == 60]
    df_cluster = df_cluster[df_cluster['Timelimit'] == 60]

    # Seleccionar solo un ejecutor de referencia en los normales (ej. 'gpu')
    df_normal = df_normal[df_normal['Executer'] == 'gpu']
    df_cluster = df_cluster[df_cluster['Executer'] == 'gpu']

    # Añadir columna "Algorithm"
    df_cluster['Algorithm'] = "Cluster"
    df_normal['Algorithm'] = df_normal['Algorithm'].str.upper()

    # Unir datasets
    df_all = pd.concat([df_normal, df_cluster], ignore_index=True)

    # Crear carpeta base si no existe
    base_dir = 'results/Clas'
    os.makedirs(base_dir, exist_ok=True)

    # Crear figura con 2 subplots: Fitness Test y Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Gráfico de Fitness Test ---
    for algo in df_all['Algorithm'].unique():
        df_algo = df_all[df_all['Algorithm'] == algo].sort_values('Size')
        axes[0].plot(df_algo['Size'], df_algo['Fitness Test'], marker='o', label=algo)

    ymin_fitness = df_all['Fitness Test'].min()
    axes[0].set_ylim(ymin_fitness, 100)
    axes[0].set_title('Fitness Test vs Train Size (Cluster vs GA/PSO/ACO)')
    axes[0].set_xlabel('Train Size')
    axes[0].set_ylabel('Fitness Test')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # --- Gráfico de Accuracy ---
    for algo in df_all['Algorithm'].unique():
        df_algo = df_all[df_all['Algorithm'] == algo].sort_values('Size')
        axes[1].plot(df_algo['Size'], df_algo['Accuracy'], marker='o', label=algo)

    ymin_accuracy = df_all['Accuracy'].min()
    axes[1].set_ylim(ymin_accuracy, 100)
    axes[1].set_title('Accuracy vs Train Size (Cluster vs GA/PSO/ACO)')
    axes[1].set_xlabel('Train Size')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    # Ajuste y guardado
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fitness_accuracy_lines_cluster_vs_algos.png"))
    plt.close()

    print("Gráficas de comparación (Cluster vs GA/PSO/ACO) guardadas en la carpeta correspondiente.")

def analysis_TSP_Statistic():
    # --- 1. Cargar datos ---
    df_normal = pd.read_csv("results/tsp_results.csv")          # GA/PSO/ACO
    df_cluster = pd.read_csv("results/cluster_tsp_results.csv") # Cluster

    # Filtrar por Timelimit = 60
    df_normal = df_normal[df_normal['Timelimit'] == 60]
    df_cluster = df_cluster[df_cluster['Timelimit'] == 60]

    # Seleccionar un ejecutor de referencia para normal (GPU)
    df_normal = df_normal[df_normal['Executer'] == 'gpu']

    # Añadir columna Algorithm para cluster
    df_cluster['Algorithm'] = "CLUSTER"
    df_normal['Algorithm'] = df_normal['Algorithm'].str.upper()

    # Unir datasets
    df_all = pd.concat([df_normal, df_cluster], ignore_index=True)

    # --- 2. Filtrar tamaños comunes para análisis estadístico ---
    sizes_normal = df_normal['Cities'].unique()
    sizes_cluster = df_cluster['Cities'].unique()
    common_sizes = np.intersect1d(sizes_normal, sizes_cluster)  # tamaños comunes
    df_all_common = df_all[df_all['Cities'].isin(common_sizes)]

    # --- 3. Preparar tablas Fitness y Gap ---
    algos = df_all_common['Algorithm'].unique()
    sizes = common_sizes

    fit_table = pd.DataFrame(index=sizes, columns=algos)
    gap_table = pd.DataFrame(index=sizes, columns=algos)

    for size in sizes:
        df_size = df_all_common[df_all_common['Cities'] == size]
        for algo in algos:
            value_fit = df_size[df_size['Algorithm']==algo]['Fitness'].values
            value_gap = df_size[df_size['Algorithm']==algo]['Gap'].values
            fit_table.loc[size, algo] = value_fit[0] if len(value_fit)>0 else np.nan
            gap_table.loc[size, algo] = value_gap[0] if len(value_gap)>0 else np.nan

    # --- 4. Friedman test ---
    stat_fit, p_fit = friedmanchisquare(*[fit_table[algo].dropna().values for algo in algos])
    stat_gap, p_gap = friedmanchisquare(*[gap_table[algo].dropna().values for algo in algos])

    friedman_results = pd.DataFrame({
        'Metric': ['Fitness','Gap'],
        'Chi2': [stat_fit, stat_gap],
        'p-value': [p_fit, p_gap]
    })

    # --- 5. Post-hoc Nemenyi ---
    nemenyi_fit = None
    nemenyi_gap = None
    if p_fit < 0.05:
        nemenyi_fit = sp.posthoc_nemenyi_friedman(fit_table)
    if p_gap < 0.05:
        nemenyi_gap = sp.posthoc_nemenyi_friedman(gap_table)

    # --- 6. Ranking promedio ---
    def rank_algos(df_metric):
        ranks = []
        for idx, row in df_metric.iterrows():
            row_rank = row.rank(ascending=True, method='min')  # menor es mejor
            ranks.append(row_rank)
        return pd.DataFrame(ranks).mean()

    rank_fit = rank_algos(fit_table)
    rank_gap = rank_algos(gap_table)

    # --- 7. Guardar todo en un Excel ---
    output_dir = 'results/TSP'
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, "statistical_analysis.xlsx")

    with pd.ExcelWriter(excel_path) as writer:
        fit_table.to_excel(writer, sheet_name='Fitness Values')
        gap_table.to_excel(writer, sheet_name='Gap Values')
        friedman_results.to_excel(writer, sheet_name='Friedman Test', index=False)
        
        if nemenyi_fit is not None:
            nemenyi_fit.to_excel(writer, sheet_name='Nemenyi Fitness')
        if nemenyi_gap is not None:
            nemenyi_gap.to_excel(writer, sheet_name='Nemenyi Gap')
        
        rank_fit.to_frame('Ranking Fitness').to_excel(writer, sheet_name='Ranking Fitness')
        rank_gap.to_frame('Ranking Gap').to_excel(writer, sheet_name='Ranking Gap')

    print("Todos los resultados estadísticos del TSP guardados en:", excel_path)

def analysis_Clas_Statistic():
    # --- 1. Cargar datos ---
    df_normal = pd.read_csv("results/clas_results.csv")          # GA/PSO/ACO
    df_cluster = pd.read_csv("results/cluster_clas_results.csv") # Cluster

    # Filtrar por Timelimit = 60
    df_normal = df_normal[df_normal['Timelimit'] == 60]
    df_cluster = df_cluster[df_cluster['Timelimit'] == 60]

    # Seleccionar un ejecutor de referencia para normal (GPU)
    df_normal = df_normal[df_normal['Executer'] == 'gpu']
    # Añadir columna Algorithm para cluster
    df_cluster['Algorithm'] = "CLUSTER"
    df_normal['Algorithm'] = df_normal['Algorithm'].str.upper()

    # Unir datasets
    df_all = pd.concat([df_normal, df_cluster], ignore_index=True)

    # --- 2. Filtrar tamaños comunes para análisis estadístico ---
    sizes_normal = df_normal['Size'].unique()
    sizes_cluster = df_cluster['Size'].unique()
    common_sizes = np.intersect1d(sizes_normal, sizes_cluster)  # tamaños comunes
    df_all_common = df_all[df_all['Size'].isin(common_sizes)]

    # --- 3. Preparar tablas Accuracy y Fitness ---
    algos = df_all_common['Algorithm'].unique()
    sizes = common_sizes

    acc_table = pd.DataFrame(index=sizes, columns=algos)
    fit_table = pd.DataFrame(index=sizes, columns=algos)

    for size in sizes:
        df_size = df_all_common[df_all_common['Size'] == size]
        for algo in algos:
            value_acc = df_size[df_size['Algorithm']==algo]['Accuracy'].values
            value_fit = df_size[df_size['Algorithm']==algo]['Fitness Test'].values
            acc_table.loc[size, algo] = value_acc[0] if len(value_acc)>0 else np.nan
            fit_table.loc[size, algo] = value_fit[0] if len(value_fit)>0 else np.nan

    # --- 4. Friedman test ---
    stat_acc, p_acc = friedmanchisquare(*[acc_table[algo].dropna().values for algo in algos])
    stat_fit, p_fit = friedmanchisquare(*[fit_table[algo].dropna().values for algo in algos])

    friedman_results = pd.DataFrame({
        'Metric': ['Accuracy','Fitness Test'],
        'Chi2': [stat_acc, stat_fit],
        'p-value': [p_acc, p_fit]
    })

    # --- 5. Post-hoc Nemenyi ---
    nemenyi_acc = None
    nemenyi_fit = None
    if p_acc < 0.05:
        nemenyi_acc = sp.posthoc_nemenyi_friedman(acc_table)
    if p_fit < 0.05:
        nemenyi_fit = sp.posthoc_nemenyi_friedman(fit_table)

    # --- 6. Ranking promedio ---
    def rank_algos(df_metric):
        ranks = []
        for idx, row in df_metric.iterrows():
            row_rank = row.rank(ascending=False, method='min')
            ranks.append(row_rank)
        return pd.DataFrame(ranks).mean()

    rank_acc = rank_algos(acc_table)
    rank_fit = rank_algos(fit_table)

    # --- 7. Guardar todo en un Excel ---
    output_dir = 'results/Clas'
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, "statistical_analysis.xlsx")

    with pd.ExcelWriter(excel_path) as writer:
        acc_table.to_excel(writer, sheet_name='Accuracy Values')
        fit_table.to_excel(writer, sheet_name='Fitness Values')
        friedman_results.to_excel(writer, sheet_name='Friedman Test', index=False)
        
        if nemenyi_acc is not None:
            nemenyi_acc.to_excel(writer, sheet_name='Nemenyi Accuracy')
        if nemenyi_fit is not None:
            nemenyi_fit.to_excel(writer, sheet_name='Nemenyi Fitness')
        
        rank_acc.to_frame('Ranking Accuracy').to_excel(writer, sheet_name='Ranking Accuracy')
        rank_fit.to_frame('Ranking Fitness').to_excel(writer, sheet_name='Ranking Fitness')

    print("Todos los resultados estadísticos guardados en:", excel_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run various experiment test suites.')
    parser.add_argument('-t', '--test_type', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8'],
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