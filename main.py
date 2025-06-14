from test.test_algorithms import main as test_algorithms_main
from test.test_fitness import main as test_fitness_main

def main():
    print("What would you like to test?")
    print("1. Algorithms")
    print("2. Fitness functions")
    choice = input("Enter 1 or 2: ")
    if choice == '1':
        print('Welcome to the Algorithms test suite!')
        print("This will run tests for the algorithms implemented in the 'algorithms' module.")
        print('What algorithm would you like to test?')
        print("1. Genetic Algorithm (GA)")
        print("2. Ant Colony Optimization (ACO)")
        print("3. Particle Swarm Optimization (PSO)")
        algorithm_choice = input("Enter 1, 2, or 3: ")
        if algorithm_choice == '1':
            algorithm = 'ga'
        elif algorithm_choice == '2':
            algorithm = 'aco'
        elif algorithm_choice == '3':
            algorithm = 'pso'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            return
        
        print(f"Which executer type would you like to use for {algorithm.upper()}?")
        print("1. CPU single-threaded")
        print("2. CPU multi-threaded")
        print("3. GPU")
        print("4. Hybrid (CPU + GPU)")
        executer_choice = input("Enter 1, 2, 3, or 4: ")
        if executer_choice == '1':
            executer_type = 'single'
        elif executer_choice == '2':
            executer_type = 'multi'
        elif executer_choice == '3':
            executer_type = 'gpu'
        elif executer_choice == '4':
            executer_type = 'hybrid'
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
            return
        
        print(f"Running tests for {algorithm.upper()} with {executer_type} executer type...")

        test_algorithms_main(
            algorithm=algorithm,
            executer_type=executer_type
        )
    elif choice == '2':
        print('Running tests for fitness functions...')
        test_fitness_main()
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()