/**
 * @file utils.cpp
 * @brief Utility functions for feature selection and the Traveling Salesman Problem (TSP) with Python bindings via pybind11.
 *
 * This file provides a collection of C++ functions optimized for performance, including OpenMP parallelization,
 * to be used from Python for tasks such as:
 *   - Feature selection and evaluation (classification rate, reduction rate, fitness calculation)
 *   - Weighted nearest neighbor classification and prediction
 *   - TSP solution evaluation (fitness), crossover, mutation, and local search (2-opt)
 *   - Ant Colony Optimization (ACO) operations for TSP: pheromone update and solution construction
 *   - Utility functions for tour manipulation (swap sequence computation)
 *
 * The functions are exposed to Python using pybind11, allowing efficient integration with NumPy arrays.
 *
 * Main functionalities:
 *   - Classification and feature selection:
 *       - clas_rate_cpp, clas_rate: Compute weighted nearest neighbor classification rate.
 *       - red_rate_cpp, red_rate: Compute feature reduction rate.
 *       - fitness, fitness_omp: Combined fitness for feature selection (classification + reduction).
 *       - predict: Predict labels for test data using weighted nearest neighbor.
 *   - TSP optimization:
 *       - fitness_tsp, fitness_tsp_omp: Evaluate TSP tours (single and batch).
 *       - crossover_tsp: Perform order-based crossover for TSP populations.
 *       - mutation_tsp: Apply swap mutation to TSP individuals.
 *       - update_pheromones_tsp: Update pheromone matrix for ACO algorithms.
 *       - construct_solutions_tsp_inner, construct_solutions_tsp_inner_cpu: Construct TSP solutions using ACO probabilistic rules.
 *       - get_swap_sequence: Compute swap sequence to transform one tour into another.
 *       - two_opt: Apply 2-opt local search to improve TSP tours.
 *
 * All functions are designed for efficient use with NumPy arrays and support parallel execution where appropriate.
 *
 * @author IsaaHack
 * @date 2025
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>

namespace py = pybind11;
using namespace std;

/**
 * @brief Computes the classification rate using a weighted nearest neighbor approach.
 *
 * This function selects features whose weights are above a given threshold, computes
 * the weighted Euclidean distance between all pairs of training samples using only
 * the selected features, and predicts the label of each sample as the label of its
 * nearest neighbor (excluding itself). The classification rate is returned as a percentage
 * of correctly predicted labels.
 *
 * @param weights        Pointer to the array of feature weights.
 * @param w_size         Number of feature weights (size of weights array).
 * @param X_train        Pointer to the training data matrix (row-major, n samples x d features).
 * @param n              Number of training samples.
 * @param d              Number of features per sample.
 * @param y_train        Pointer to the array of training labels.
 * @param threshold      Threshold for selecting relevant features based on their weights.
 * @return float         Classification rate as a percentage (0.0f - 100.0f).
 */
float clas_rate_cpp(const float* weights, size_t w_size, const float* X_train, size_t n, size_t d, const int* y_train, 
                    const float threshold) {
    vector<float> weights_to_use;
    
    vector<int> valid_indices;

    for (size_t j = 0; j < w_size; ++j) {
        if (weights[j] >= threshold) {
            weights_to_use.push_back(weights[j]);
            valid_indices.push_back(j);
        }
    }

    vector<vector<float>> attributes_to_use(n, vector<float>(valid_indices.size()));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < valid_indices.size(); ++j) {
            attributes_to_use[i][j] = X_train[i * d + valid_indices[j]];
        }
    }

    vector<vector<float>> distances(n, vector<float>(n));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < attributes_to_use[i].size(); ++k) {
                float diff = attributes_to_use[i][k] - attributes_to_use[j][k];
                sum += weights_to_use[k] * diff * diff;
            }
            float dist = sqrt(sum);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        distances[i][i] = numeric_limits<float>::infinity();
    }

    vector<int> index_predictions(n);
    for (size_t i = 0; i < n; ++i) {
        index_predictions[i] = min_element(distances[i].begin(), distances[i].end()) - distances[i].begin();
    }

    vector<int> predictions_labels(n);
    for (size_t i = 0; i < n; ++i) {
        predictions_labels[i] = y_train[index_predictions[i]];
    }

    int correct = 0;
    for (size_t i = 0; i < n; ++i) {
        if (predictions_labels[i] == y_train[i]) {
            ++correct;
        }
    }
    
    return 100.0f * correct / n;
}


/**
 * @brief Computes the classification rate given model weights, training data, and labels.
 *
 * This function takes NumPy arrays for model weights, training features, and training labels,
 * along with a classification threshold, and computes the classification rate by delegating
 * to the `clas_rate_cpp` function.
 *
 * @param weights_np NumPy array (1D, float) containing the model weights.
 * @param X_train_np NumPy array (2D, float) containing the training feature matrix (samples x features).
 * @param y_train_np NumPy array (1D, int) containing the training labels.
 * @param threshold Float value representing the classification threshold.
 * @return The classification rate as a float.
 */
float clas_rate(py::array_t<float> weights_np, py::array_t<float> X_train_np, py::array_t<int> y_train_np, const float threshold) {
    auto weights = weights_np.unchecked<1>();
    auto X_train = X_train_np.unchecked<2>();
    auto y_train = y_train_np.unchecked<1>();

    size_t w_size = weights.shape(0);
    size_t n = X_train.shape(0);
    size_t d = X_train.shape(1);

    return clas_rate_cpp(weights.data(0), w_size, X_train.data(0, 0), n, d, y_train.data(0), threshold);
}

/**
 * Calculates the reduction rate of weights that exceed a specified threshold.
 *
 * @param weights Pointer to an array of float weights.
 * @param w_size The number of elements in the weights array.
 * @param threshold The threshold value to compare each weight against.
 * @return The reduction rate as a float, representing the proportion of weights above the threshold.
 */
float red_rate_cpp(const float* weights, size_t w_size, const float threshold) {
    using namespace std;
    size_t count = 0;
    for (size_t i = 0; i < w_size; ++i) {
        if (weights[i] < threshold) {
            ++count;
        }
    }
    return 100.0f * count / w_size;
}

/**
 * @brief Calculates the reduction rate of weights above a given threshold.
 *
 * This function takes a 1-dimensional NumPy array of floats representing weights,
 * and computes the reduction rate by delegating to the `red_rate_cpp` function.
 * Only weights greater than the specified threshold are considered.
 *
 * @param weights_np A 1D NumPy array (py::array_t<float>) containing the weights.
 * @param threshold The threshold value to determine which weights are counted.
 * @return The reduction rate as a float.
 */
float red_rate(py::array_t<float> weights_np, const float threshold) {
    auto weights = weights_np.unchecked<1>();
    size_t w_size = weights.shape(0);
    return red_rate_cpp(weights.data(0), w_size, threshold);
}

/**
 * @brief Computes the fitness value for a given set of weights, training data, and labels.
 *
 * The fitness is calculated as a weighted sum of the classification rate and the reduction rate,
 * controlled by the parameter alpha. The function expects input data as NumPy arrays and uses
 * helper functions `clas_rate_cpp` and `red_rate_cpp` to compute the respective rates.
 *
 * @param weights_np NumPy array of floats representing the model weights (1D).
 * @param X_train_np NumPy array of floats representing the training data (2D: samples x features).
 * @param y_train_np NumPy array of ints representing the training labels (1D).
 * @param alpha Weighting factor between classification and reduction rates (0 <= alpha <= 1).
 * @param threshold Threshold value used in the classification and reduction rate calculations.
 * @return The computed fitness value as a float.
 */
float fitness(py::array_t<float> weights_np, py::array_t<float> X_train_np, py::array_t<int> y_train_np, const float alpha, const float threshold) {
    using namespace std;
    auto weights = weights_np.unchecked<1>();
    auto X_train = X_train_np.unchecked<2>();
    auto y_train = y_train_np.unchecked<1>();

    size_t w_size = weights.shape(0);
    size_t n = X_train.shape(0);
    size_t d = X_train.shape(1);

    float clas = clas_rate_cpp(weights.data(0), w_size, X_train.data(0, 0), n, d, y_train.data(0), threshold);
    float red = red_rate_cpp(weights.data(0), w_size, threshold);

    return alpha * clas + (1-alpha) * red;
}

/**
 * @brief Computes the fitness values for a set of solutions in parallel using OpenMP.
 *
 * This function evaluates the fitness of multiple solutions (weight vectors) by combining
 * their classification rate and reduction rate, weighted by the parameter alpha. The computation
 * is parallelized across all solutions using OpenMP for improved performance.
 *
 * @param weights         Pointer to the array of all solution weights (flattened).
 * @param X_train         Pointer to the training data matrix (flattened, row-major).
 * @param y_train         Pointer to the array of training labels.
 * @param fitness_values  Pointer to the output array where computed fitness values will be stored.
 * @param n_sol           Number of solutions (weight vectors).
 * @param n               Number of training samples.
 * @param d               Number of features per sample.
 * @param w_size          Size of each weight vector.
 * @param alpha           Weighting factor for combining classification and reduction rates (0 <= alpha <= 1).
 * @param threshold       Threshold value used in classification and reduction calculations.
 */
void fitness_omp_cpp(const float* weights, const float* X_train, const int* y_train, float *fitness_values,
                  size_t n_sol, size_t n, size_t d, size_t w_size, const float alpha, const float threshold) {
    #pragma omp parallel for default(none) shared(weights, X_train, y_train, fitness_values, n_sol, n, d, w_size, alpha, threshold)
    for (size_t i = 0; i < n_sol; ++i) {
        float clas = clas_rate_cpp(&weights[i * w_size], w_size, X_train, n, d, y_train, threshold);
        float red = red_rate_cpp(&weights[i * w_size], w_size, threshold);

        fitness_values[i] = alpha * clas + (1 - alpha) * red;
    }
}

/**
 * @brief Computes the fitness values for a set of solutions using OpenMP parallelization.
 *
 * This function takes NumPy arrays (via pybind11) representing a set of weights (solutions),
 * training data, training labels, and an output array for fitness values. It extracts the
 * necessary information from the arrays and calls the underlying C++ implementation to compute
 * the fitness for each solution in parallel.
 *
 * @param weights_np        NumPy array of shape (n_sol, d) containing the weights for each solution.
 * @param X_train_np        NumPy array of shape (n, d) containing the training data.
 * @param y_train_np        NumPy array of shape (n,) containing the training labels.
 * @param fitness_values_np NumPy array of shape (n_sol,) to store the computed fitness values.
 * @param alpha             Regularization parameter or weighting factor.
 * @param threshold         Threshold value used in the fitness computation.
 */
void fitness_omp(py::array_t<float> weights_np, py::array_t<float> X_train_np, py::array_t<int> y_train_np,
                 py::array_t<float> fitness_values_np, const float alpha, const float threshold) {

    auto weights_info = weights_np.request();
    auto X_train_info = X_train_np.request();
    auto y_train_info = y_train_np.request();
    auto fitness_values_info = fitness_values_np.request();

    size_t w_size = X_train_info.shape[1];
    size_t n = X_train_info.shape[0];
    size_t d = X_train_info.shape[1];
    size_t n_sol = weights_info.shape[0];

    auto weights_ptr = static_cast<float *>(weights_info.ptr);
    auto X_train_ptr = static_cast<float *>(X_train_info.ptr);
    auto y_train_ptr = static_cast<int *>(y_train_info.ptr);
    auto fitness_values = static_cast<float *>(fitness_values_info.ptr);

    fitness_omp_cpp(weights_ptr, X_train_ptr, y_train_ptr, fitness_values,
                  n_sol, n, d, w_size, alpha, threshold);

}

/**
 * @brief Predicts class labels for test samples using a weighted nearest neighbor approach with feature selection.
 *
 * This function selects features whose weights are greater than or equal to a given threshold,
 * then computes the weighted Euclidean distance between each test sample and all training samples
 * using only the selected features. The label of the nearest training sample is assigned to each test sample.
 *
 * @param X_test_np      2D NumPy array (n_test, d): Test samples.
 * @param weights_np     1D NumPy array (d,): Feature weights.
 * @param X_train_np     2D NumPy array (n_train, d): Training samples.
 * @param y_train_np     1D NumPy array (n_train,): Training labels.
 * @param threshold      Minimum weight value for a feature to be selected.
 * @return py::array_t<int> 1D NumPy array (n_test,): Predicted labels for test samples.
 */
py::array_t<int> predict(py::array_t<float> X_test_np, py::array_t<float> weights_np, 
                         py::array_t<float> X_train_np, py::array_t<int> y_train_np,
                         const float threshold) 
                         {
    auto X_test = X_test_np.unchecked<2>();   // (n_test, d)
    auto weights = weights_np.unchecked<1>(); // (d,)
    auto X_train = X_train_np.unchecked<2>(); // (n_train, d)
    auto y_train = y_train_np.unchecked<1>(); // (n_train,)

    size_t n_test = X_test.shape(0);
    size_t n_train = X_train.shape(0);
    size_t d = X_test.shape(1);

    // Selección de atributos por threshold (como en Python)
    std::vector<size_t> selected_features;
    for (size_t k = 0; k < d; ++k) {
        if (weights(k) >= threshold) {
            selected_features.push_back(k);
        }
    }

    // Nueva dimensión
    size_t d_selected = selected_features.size();

    // Vector para predicciones
    std::vector<int> predictions(n_test);

    for (size_t i = 0; i < n_test; ++i) {
        float min_dist = std::numeric_limits<float>::infinity();
        int best_index = -1;

        for (size_t j = 0; j < n_train; ++j) {
            float dist = 0.0f;
            for (size_t idx = 0; idx < d_selected; ++idx) {
                size_t feature_idx = selected_features[idx];
                float diff = X_test(i, feature_idx) - X_train(j, feature_idx);
                // ponderar por sqrt(weight)² = weight
                dist += weights(feature_idx) * diff * diff;
            }
            dist = std::sqrt(dist);

            if (dist < min_dist) {
                min_dist = dist;
                best_index = j;
            }
        }

        predictions[i] = y_train(best_index);
    }

    // Convertir a py::array_t<int> para retornar a Python
    py::array_t<int> result(n_test);
    auto result_mutable = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n_test; ++i) {
        result_mutable(i) = predictions[i];
    }

    return result;
}

/**
 * @brief Calculates the fitness value for a given TSP (Traveling Salesman Problem) solution.
 *
 * This function computes the total distance of the tour specified by the solution array,
 * using the provided distance matrix. The fitness value is returned as the negative of the
 * total distance, suitable for optimization algorithms that maximize fitness.
 *
 * @param distances Pointer to a 1D array representing the distance matrix (row-major order).
 *                  The matrix should be of size n x n.
 * @param solution  Pointer to an array of integers representing the order of nodes visited in the tour.
 *                  The array should contain n elements, each representing a node index.
 * @param n         The number of nodes in the TSP instance.
 * @return float    The negative total distance of the tour (fitness value).
 */
float fitness_tsp_cpp(const float* distances, const int* solution, int n) {
    float fitness_value = 0.0f;

    // Calcular distancia entre nodos consecutivos
    for(int i = 0; i < n - 1; ++i) {
        int from = solution[i];
        int to = solution[i + 1];
        fitness_value += distances[from * n + to];
    }

    // Cerrar el ciclo (último al primero)
    int last = solution[n - 1];
    int first = solution[0];
    fitness_value += distances[last * n + first];

    return -fitness_value;
}

/**
 * @brief Calculates the fitness value for a given TSP solution.
 *
 * This function receives a distance matrix and a proposed solution (tour) for the
 * Traveling Salesman Problem (TSP), both as NumPy arrays from Python via pybind11.
 * It extracts the raw data pointers and calls the underlying C++ implementation
 * to compute the total distance (fitness) of the tour.
 *
 * @param distances A 2D NumPy array (flattened) of type float representing the distance matrix.
 * @param solution A 1D NumPy array of type int representing the order of cities in the tour.
 * @return The total distance (fitness) of the provided TSP solution as a float.
 */
float fitness_tsp(py::array_t<float> distances,
                   py::array_t<int> solution)
{
    py::buffer_info dist_info = distances.request();
    py::buffer_info sol_info = solution.request();

    const int n = sol_info.shape[0];

    // Obtener punteros a los datos
    const float *dist_ptr = static_cast<float *>(dist_info.ptr);
    const int *sol_ptr = static_cast<int *>(sol_info.ptr);

    // Llamar a la función C++
    return fitness_tsp_cpp(dist_ptr, sol_ptr, n);
}

/**
 * @brief Computes the fitness values for a batch of TSP solutions in parallel using OpenMP.
 *
 * This function evaluates the fitness (typically the total distance) of multiple
 * Traveling Salesman Problem (TSP) solutions. Each solution is represented as a sequence
 * of city indices. The computation is parallelized using OpenMP to improve performance
 * when handling a large number of solutions.
 *
 * @param distances      Pointer to a flattened 2D array (size n_cities * n_cities) representing the distance matrix between cities.
 * @param solutions      Pointer to a flattened 2D array (size n_sol * n_cities) where each row is a permutation of city indices representing a TSP solution.
 * @param fitness_values Pointer to an array (size n_sol) where the computed fitness values for each solution will be stored.
 * @param n_sol          Number of solutions to evaluate.
 * @param n_cities       Number of cities in each TSP solution.
 */
void fitness_tsp_omp_cpp(const float* distances, const int* solutions, float *fitness_values, int n_sol, int n_cities) {
    #pragma omp parallel for default(none) shared(distances, solutions, fitness_values, n_sol, n_cities)
    for (int i = 0; i < n_sol; ++i) {
        fitness_values[i] = fitness_tsp_cpp(distances, &solutions[i * n_cities], n_cities);
    }
}

/**
 * @brief Calculates the fitness values for a set of TSP solutions using OpenMP.
 *
 * This function receives NumPy arrays (via pybind11) representing the distance matrix,
 * a batch of TSP solutions, and an output array for fitness values. It extracts the
 * underlying buffer information and calls the C++ implementation to compute the fitness
 * (typically the total distance for each solution).
 *
 * @param distances      2D NumPy array (float) of shape (n_cities, n_cities) representing the distance matrix.
 * @param solutions      2D NumPy array (int) of shape (n_sol, n_cities) where each row is a permutation of city indices.
 * @param fitness_values 1D NumPy array (float) of shape (n_sol,) to store the computed fitness values for each solution.
 */
void fitness_tsp_omp(py::array_t<float> distances,
    py::array_t<int> solutions, py::array_t<float> fitness_values)
{
    py::buffer_info dist_info = distances.request();
    py::buffer_info sol_info = solutions.request();
    py::buffer_info fit_info = fitness_values.request();

    const int n_sol = sol_info.shape[0];
    const int n_cities = dist_info.shape[0];

    // Obtener punteros a los datos
    const float *dist_ptr = static_cast<float *>(dist_info.ptr);
    const int *sol_ptr = static_cast<int *>(sol_info.ptr);
    float *fit_ptr = static_cast<float *>(fit_info.ptr);

    fitness_tsp_omp_cpp(dist_ptr, sol_ptr, fit_ptr, n_sol, n_cities);
}

/**
 * @brief Realiza la operación de cruce (crossover) para el problema del viajante (TSP) sobre una población de soluciones.
 *
 * Esta función implementa un operador de cruce para el TSP, donde cada par de padres genera dos hijos intercambiando segmentos de sus rutas.
 * El segmento a intercambiar se define por los arrays de índices aleatorios `random_starts` y `random_ends`.
 * La posición 0 de cada ruta se mantiene fija en la ciudad 0.
 * El cruce se realiza en paralelo para mejorar el rendimiento.
 *
 * @param population      Array 2D (num_soluciones x num_ciudades) que representa la población de rutas.
 * @param random_starts   Array 1D de índices de inicio de los segmentos a intercambiar para cada cruce (debe ser >= 1).
 * @param random_ends     Array 1D de índices de fin de los segmentos a intercambiar para cada cruce (debe ser >= random_starts y < num_ciudades).
 * @return py::array_t<int>  La población modificada, donde cada par de padres ha sido reemplazado por sus hijos tras el cruce.
 *
 * @throws std::runtime_error Si la población no es 2D, los arrays de índices no son 1D o no tienen el mismo tamaño,
 *                            si los índices de segmento son inválidos, o si no hay suficientes soluciones para el número de cruces.
 */
py::array_t<int> crossover_tsp(
    py::array_t<int> population,
    py::array_t<int> random_starts,
    py::array_t<int> random_ends
) {
    // Obtener información sobre la población
    py::buffer_info pop_info = population.request();
    if (pop_info.ndim != 2)
        throw std::runtime_error("La población debe ser una matriz 2D");

    int num_sol   = pop_info.shape[0]; // número de soluciones (filas)
    int n_cities  = pop_info.shape[1]; // número de genes por solución
    int* pop_ptr = static_cast<int*>(pop_info.ptr);

    // Obtener los arrays de índices aleatorios
    py::buffer_info starts_info = random_starts.request();
    py::buffer_info ends_info   = random_ends.request();
    if (starts_info.ndim != 1 || ends_info.ndim != 1)
        throw std::runtime_error("Los índices aleatorios deben ser 1D");
    if (starts_info.size != ends_info.size)
        throw std::runtime_error("Los arrays de índices aleatorios deben tener el mismo tamaño");
    int num_crossovers = static_cast<int>(starts_info.size);

    int* starts_ptr = static_cast<int*>(starts_info.ptr);
    int* ends_ptr   = static_cast<int*>(ends_info.ptr);

    // Comprobar que existen al menos 2 padres por cruce
    if (2 * num_crossovers > num_sol)
        throw std::runtime_error("No hay suficientes soluciones en la población para el número de cruces dado.");

    // Realizar los cruces en paralelo sobre cada par de padres
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_crossovers; ++k) {
        int parent1 = 2 * k;
        int parent2 = parent1 + 1;

        int start = starts_ptr[k];
        int end   = ends_ptr[k];
        if (start < 1 || end >= n_cities || start > end)
            throw std::runtime_error("Índices de segmento inválidos o tocan la posición 0");

        // Vectores temporales para los hijos, inicializados con -1
        std::vector<int> child1(n_cities, -1);
        std::vector<int> child2(n_cities, -1);

        // --- Fijar la posición 0 a ciudad 0 ---
        child1[0] = 0;
        child2[0] = 0;

        // --- Copiar el segmento seleccionado para posiciones >=1 ---
        for (int i = start; i <= end; ++i) {
            child1[i] = pop_ptr[parent1 * n_cities + i];
            child2[i] = pop_ptr[parent2 * n_cities + i];
        }

        // --- Calcular los índices restantes (orden circular) ---
        std::vector<int> rem;
        for (int i = end + 1; i < n_cities; ++i) rem.push_back(i);
        for (int i = 1; i < start; ++i) rem.push_back(i);

        // --- Rellenar child1 ---
        std::vector<bool> used(n_cities, false);
        // Marcar genes presentes en el segmento de child1
        for (int i = start; i <= end; ++i) {
            int gene = child1[i];
            if (gene >= 0 && gene < n_cities)
                used[gene] = true;
        }
        int pos = 0;
        for (int j = 0; j < n_cities && pos < int(rem.size()); ++j) {
            int gene = pop_ptr[parent2 * n_cities + j];
            if (!used[gene] && gene != 0) {
                child1[rem[pos]] = gene;
                used[gene] = true;
                pos++;
            }
        }

        // --- Rellenar child2 ---
        used.assign(n_cities, false);
        for (int i = start; i <= end; ++i) {
            int gene = child2[i];
            if (gene >= 0 && gene < n_cities)
                used[gene] = true;
        }
        pos = 0;
        for (int j = 0; j < n_cities && pos < int(rem.size()); ++j) {
            int gene = pop_ptr[parent1 * n_cities + j];
            if (!used[gene] && gene != 0) {
                child2[rem[pos]] = gene;
                used[gene] = true;
                pos++;
            }
        }

        // --- Actualizar la población con los nuevos hijos ---
        for (int j = 0; j < n_cities; ++j) {
            pop_ptr[parent1 * n_cities + j] = child1[j];
            pop_ptr[parent2 * n_cities + j] = child2[j];
        }
    }

    return population;
}

/**
 * @brief Applies swap mutation to a subset of individuals in a TSP population.
 *
 * This function mutates selected individuals in a population for the Traveling Salesman Problem (TSP)
 * by swapping two cities in their tour representation. The indices of individuals to mutate and the
 * positions to swap are provided as input arrays. The mutation is performed in parallel for efficiency.
 *
 * @param population         2D NumPy array (n_individuals x n_cities) representing the population of tours.
 * @param individual_indices 1D NumPy array of length M containing indices of individuals to mutate.
 * @param indices1           1D NumPy array of length M containing the first swap position for each mutation.
 * @param indices2           1D NumPy array of length M containing the second swap position for each mutation.
 *
 * @throws std::runtime_error if input arrays have incorrect dimensions, mismatched lengths,
 *         or if any index is out of valid range.
 *
 * @note This function is intended to be called from Python via pybind11.
 * @note The mutation is performed in-place on the provided population array.
 * @note OpenMP is used for parallelization.
 */
void mutation_tsp(
    py::array_t<int> population,             // Array 2D (n_individuals x n_cities)
    py::array_t<int> individual_indices,       // Array 1D de longitud M: índices de individuos a mutar
    py::array_t<int> indices1,                 // Array 1D de longitud M: primer índice de swap para cada individuo
    py::array_t<int> indices2                  // Array 1D de longitud M: segundo índice de swap para cada individuo
) {
    // --- Obtener información de la población ---
    py::buffer_info pop_info = population.request();
    if (pop_info.ndim != 2)
        throw std::runtime_error("La población debe ser un array 2D");
    int n_individuals = pop_info.shape[0];
    int n_cities = pop_info.shape[1];
    int* pop_ptr = static_cast<int*>(pop_info.ptr);

    // --- Obtener información de los arrays aleatorios ---
    py::buffer_info ind_info = individual_indices.request();
    py::buffer_info i1_info = indices1.request();
    py::buffer_info i2_info = indices2.request();

    if (ind_info.ndim != 1 || i1_info.ndim != 1 || i2_info.ndim != 1)
        throw std::runtime_error("Los arrays de índices aleatorios deben ser unidimensionales");
    if (ind_info.size != i1_info.size || ind_info.size != i2_info.size)
        throw std::runtime_error("Los arrays aleatorios deben tener la misma longitud");

    int num_mutations = static_cast<int>(ind_info.size);
    int* ind_ptr = static_cast<int*>(ind_info.ptr);
    int* i1_ptr = static_cast<int*>(i1_info.ptr);
    int* i2_ptr = static_cast<int*>(i2_info.ptr);

    // --- Aplicar la swap mutation en paralelo ---
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_mutations; ++i) {
        int indiv = ind_ptr[i];
        if (indiv < 0 || indiv >= n_individuals)
            throw std::runtime_error("Índice de individuo fuera de rango");

        int pos1 = i1_ptr[i];
        int pos2 = i2_ptr[i];
        if (pos1 < 0 || pos1 >= n_cities || pos2 < 0 || pos2 >= n_cities)
            throw std::runtime_error("Índice de ciudad fuera de rango");

        // Calcular la posición base de la fila
        int base = indiv * n_cities;
        // Realizar el swap
        int temp = pop_ptr[base + pos1];
        pop_ptr[base + pos1] = pop_ptr[base + pos2];
        pop_ptr[base + pos2] = temp;
    }
}

/**
 * @brief Updates the pheromone matrix for the Traveling Salesman Problem (TSP) using Ant Colony Optimization principles.
 *
 * This function performs two main operations on the pheromone matrix:
 * 1. Evaporation: Reduces all pheromone values by a factor determined by the evaporation rate.
 * 2. Deposition: Increases pheromone values along the edges traversed by each solution in the colony,
 *    with the amount deposited inversely proportional to the solution's fitness value.
 *
 * Both evaporation and deposition steps are parallelized for performance.
 *
 * @param pheromones        A 2D square numpy array (n_cities x n_cities) representing the current pheromone levels.
 * @param colony            A 2D numpy array (n_solutions x solution_length) where each row is a sequence of city indices representing a solution.
 * @param fitness_values    A 1D numpy array (n_solutions) containing the fitness (cost) of each solution in the colony.
 * @param evaporation_rate  The rate at which pheromones evaporate (should be in the range [0, 1]).
 * @return py::array_t<double> The updated pheromone matrix (same object as input, modified in-place).
 */
py::array_t<double> update_pheromones_tsp(py::array_t<double> pheromones,
                                          py::array_t<int> colony,
                                          py::array_t<double> fitness_values,
                                          double evaporation_rate)
{
    auto buf_phero = pheromones.request();
    auto buf_colony = colony.request();
    auto buf_fitness = fitness_values.request();

    const int n_cities = buf_phero.shape[0];
    const int n_solutions = buf_colony.shape[0];
    const int solution_length = buf_colony.shape[1];

    double *phero_ptr = static_cast<double *>(buf_phero.ptr);
    const int *colony_ptr = static_cast<int *>(buf_colony.ptr);
    const double *fitness_ptr = static_cast<double *>(buf_fitness.ptr);

    // 1. Evaporación paralelizada con SIMD
    const int total = n_cities * n_cities;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; ++i)
    {
        phero_ptr[i] *= (1.0 - evaporation_rate);
    }

    // 2. Cálculo de depósitos optimizado
    std::vector<double> delta_pheromones(total, 0.0);

    #pragma omp parallel
    {
    #pragma omp for schedule(static) nowait
        for (int sol = 0; sol < n_solutions; ++sol)
        {
            const double deposit = -1.0f / fitness_ptr[sol];
            const int *solution = &colony_ptr[sol * solution_length];

            for (int j = 0; j < solution_length - 1; ++j)
            {
                const int city_from = solution[j];
                const int city_to = solution[j + 1];
                const int idx1 = city_from * n_cities + city_to;
                const int idx2 = city_to * n_cities + city_from;

                #pragma omp atomic
                delta_pheromones[idx1] += deposit;

                #pragma omp atomic
                delta_pheromones[idx2] += deposit;
            }
        }
    }

    // 3. Actualización final paralelizada
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; ++i)
    {
        phero_ptr[i] += delta_pheromones[i];
    }

    return pheromones;
}

/**
 * @brief Constructs solutions for the Traveling Salesman Problem (TSP) using an Ant Colony Optimization (ACO) approach on the CPU.
 *
 * This function iteratively builds solutions for a colony of artificial ants, each constructing a tour through the cities.
 * At each step, the next city is selected probabilistically based on pheromone levels, heuristic information, and a random matrix.
 * The function updates the solutions and visited arrays in-place.
 *
 * @param solutions         A py::array_t<int32_t> (shape: [colony_size, n_cities]) to store the constructed tours for each ant.
 * @param visited           A py::array_t<uint8_t> (shape: [colony_size, n_cities]) indicating which cities have been visited by each ant.
 * @param pheromones        A py::array_t<double> (shape: [n_cities, n_cities]) containing the pheromone levels between cities.
 * @param heuristic_matrix  A py::array_t<double> (shape: [n_cities, n_cities]) containing heuristic values (e.g., inverse distances) between cities.
 * @param rand_matrix       A py::array_t<double> (shape: [colony_size, n_cities - 1]) containing random values for probabilistic selection.
 * @param colony_size       The number of ants (solutions) to construct.
 * @param n_cities          The number of cities in the TSP instance.
 * @param alpha             The exponent applied to pheromone values to control their influence.
 * @param epsilon           A small value added to denominators to prevent division by zero.
 *
 * @note
 * - The function assumes that the first city for each ant is already set in the solutions array and marked as visited.
 * - The function modifies the solutions and visited arrays in-place.
 * - The function is intended to be called from Python via pybind11.
 */
void construct_solutions_tsp_inner_cpu(
    py::array_t<int32_t> solutions,
    py::array_t<uint8_t> visited,
    py::array_t<double> pheromones,
    py::array_t<double> heuristic_matrix,
    py::array_t<double> rand_matrix,
    int colony_size,
    int n_cities,
    double alpha,
    double epsilon)
{
    // Obtener acceso a los buffers de los arrays
    auto solutions_buf = solutions.request();
    auto visited_buf = visited.request();
    auto pheromones_buf = pheromones.request();
    auto heuristic_buf = heuristic_matrix.request();
    auto rand_buf = rand_matrix.request();

    // Obtener punteros a los datos
    int32_t* solutions_ptr = static_cast<int32_t*>(solutions_buf.ptr);
    uint8_t* visited_ptr = static_cast<uint8_t*>(visited_buf.ptr);
    double* pheromones_ptr = static_cast<double*>(pheromones_buf.ptr);
    double* heuristic_ptr = static_cast<double*>(heuristic_buf.ptr);
    double* rand_ptr = static_cast<double*>(rand_buf.ptr);

    for(int i = 0; i < colony_size; ++i) {
        for(int step = 1; step < n_cities; ++step) {
            
            const int current = solutions_ptr[i * n_cities + (step - 1)];
            double rand = rand_ptr[i * (n_cities - 1) + (step - 1)];
            
            double total = 0.0;
            double probabilities[n_cities];
            
            // Calcular probabilidades no normalizadas
            for(int city = 0; city < n_cities; ++city) {
                if(!visited_ptr[i * n_cities + city]) {
                    const int idx = current * n_cities + city;
                    const double ph = std::pow(pheromones_ptr[idx], alpha);
                    probabilities[city] = ph * heuristic_ptr[idx];
                    total += probabilities[city];
                } else {
                    probabilities[city] = 0.0;
                }
            }
            
            // Encontrar la siguiente ciudad
            int selected = -1;
            double cumulative = 0.0;
            const double inv_total = (total > 0) ? 1.0 / (total + epsilon) : 0.0;
            
            for(int city = 0; city < n_cities; ++city) {
                if(!visited_ptr[i * n_cities + city]) {
                    cumulative += probabilities[city] * inv_total;
                    if(cumulative >= rand && selected == -1) {
                        selected = city;
                    }
                }
            }
            
            // Fallback: seleccionar primera ciudad no visitada
            if(selected == -1) {
                for(int city = 0; city < n_cities; ++city) {
                    if(!visited_ptr[i * n_cities + city]) {
                        selected = city;
                        break;
                    }
                }
            }

            // Actualizar solución y visitados
            solutions_ptr[i * n_cities + step] = selected;
            visited_ptr[i * n_cities + selected] = 1;
        }
    }
}

/**
 * @brief Constructs solutions for the Traveling Salesman Problem (TSP) using an Ant Colony Optimization (ACO) approach.
 *
 * This function iteratively builds solutions for a colony of agents (ants), each constructing a tour by probabilistically
 * selecting the next city to visit based on pheromone levels and heuristic information. The selection is influenced by
 * the parameter alpha, which controls the importance of pheromone trails, and epsilon, which is used to avoid division by zero.
 * The function operates in parallel over the colony using OpenMP.
 *
 * @param solutions        A py::array_t<int32_t> (shape: [colony_size, n_cities]) to store the constructed tours for each ant.
 * @param visited          A py::array_t<uint8_t> (shape: [colony_size, n_cities]) indicating which cities have been visited by each ant.
 * @param pheromones       A py::array_t<double> (shape: [n_cities, n_cities]) representing the pheromone levels between cities.
 * @param heuristic_matrix A py::array_t<double> (shape: [n_cities, n_cities]) containing heuristic information (e.g., inverse distances).
 * @param rand_matrix      A py::array_t<double> (shape: [colony_size, n_cities - 1]) with random values for probabilistic selection.
 * @param colony_size      The number of ants (solutions) to construct.
 * @param n_cities         The number of cities in the TSP instance.
 * @param alpha            The exponent for pheromone influence in the probability calculation.
 * @param epsilon          A small value added to the denominator to prevent division by zero.
 */
void construct_solutions_tsp_inner(
    py::array_t<int32_t> solutions,
    py::array_t<uint8_t> visited,
    py::array_t<double> pheromones,
    py::array_t<double> heuristic_matrix,
    py::array_t<double> rand_matrix,
    int colony_size,
    int n_cities,
    double alpha,
    double epsilon)
{
    // Obtener acceso a los buffers de los arrays
    auto solutions_buf = solutions.request();
    auto visited_buf = visited.request();
    auto pheromones_buf = pheromones.request();
    auto heuristic_buf = heuristic_matrix.request();
    auto rand_buf = rand_matrix.request();

    // Obtener punteros a los datos
    int32_t* solutions_ptr = static_cast<int32_t*>(solutions_buf.ptr);
    uint8_t* visited_ptr = static_cast<uint8_t*>(visited_buf.ptr);
    double* pheromones_ptr = static_cast<double*>(pheromones_buf.ptr);
    double* heuristic_ptr = static_cast<double*>(heuristic_buf.ptr);
    double* rand_ptr = static_cast<double*>(rand_buf.ptr);

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < colony_size; ++i) {
        for(int step = 1; step < n_cities; ++step) {
            
            const int current = solutions_ptr[i * n_cities + (step - 1)];
            double rand = rand_ptr[i * (n_cities - 1) + (step - 1)];
            
            double total = 0.0;
            double probabilities[n_cities];
            
            // Calcular probabilidades no normalizadas
            for(int city = 0; city < n_cities; ++city) {
                if(!visited_ptr[i * n_cities + city]) {
                    const int idx = current * n_cities + city;
                    const double ph = std::pow(pheromones_ptr[idx], alpha);
                    probabilities[city] = ph * heuristic_ptr[idx];
                    total += probabilities[city];
                } else {
                    probabilities[city] = 0.0;
                }
            }
            
            // Encontrar la siguiente ciudad
            int selected = -1;
            double cumulative = 0.0;
            const double inv_total = (total > 0) ? 1.0 / (total + epsilon) : 0.0;
            
            for(int city = 0; city < n_cities; ++city) {
                if(!visited_ptr[i * n_cities + city]) {
                    cumulative += probabilities[city] * inv_total;
                    if(cumulative >= rand && selected == -1) {
                        selected = city;
                    }
                }
            }
            
            // Fallback: seleccionar primera ciudad no visitada
            if(selected == -1) {
                for(int city = 0; city < n_cities; ++city) {
                    if(!visited_ptr[i * n_cities + city]) {
                        selected = city;
                        break;
                    }
                }
            }

            // Actualizar solución y visitados
            solutions_ptr[i * n_cities + step] = selected;
            visited_ptr[i * n_cities + selected] = 1;
        }
    }
}

/**
 * @brief Computes the minimal sequence of swaps to transform one permutation into another.
 *
 * Given two 1D numpy arrays representing permutations of integers in the range [0, n-1],
 * this function calculates the sequence of swap operations needed to transform the first
 * permutation (`from_np`) into the second permutation (`to_np`). Each swap operation is
 * represented as a pair of indices (i, j) indicating that elements at positions i and j
 * should be swapped.
 *
 * @param from_np A 1D numpy array (py::array_t<int>) representing the initial permutation.
 * @param to_np   A 1D numpy array (py::array_t<int>) representing the target permutation.
 * @return std::vector<std::pair<int, int>> A vector of swap operations (index pairs).
 *
 * @note Both input arrays must be permutations of [0, n-1] and have the same length.
 * @note The function assumes C-style contiguous arrays and uses forcecast for type safety.
 */
std::vector<std::pair<int,int>> 
get_swap_sequence(py::array_t<int, py::array::c_style | py::array::forcecast> from_np,
                  py::array_t<int, py::array::c_style | py::array::forcecast> to_np)
{
    auto buf_f = from_np.request();
    auto buf_t = to_np.request();

    std::size_t n = buf_f.shape[0];
    int* from_ptr = static_cast<int*>(buf_f.ptr);
    int* to_ptr   = static_cast<int*>(buf_t.ptr);

    // Vector de posiciones (asumiendo elementos en [0, n-1])
    std::vector<int> pos_map(n);
    for (std::size_t i = 0; i < n; ++i) {
        pos_map[from_ptr[i]] = i;
    }

    std::vector<std::pair<int,int>> seq;

    // Trabajar directamente sobre el array original virtual
    std::vector<int> current_index(n);
    std::iota(current_index.begin(), current_index.end(), 0);

    for (std::size_t i = 0; i < n; ++i) {
        int target = to_ptr[i];
        int current_val = from_ptr[current_index[i]];
        
        if (current_val == target) continue;
        
        int j = pos_map[target];
        seq.emplace_back(i, j);
        
        // Actualizar solo las posiciones afectadas
        std::swap(current_index[i], current_index[j]);
        pos_map[current_val] = j;
        pos_map[target] = i;
    }
    return seq;
}

/**
 * @brief Performs a single 2-opt optimization pass on a batch of tours.
 *
 * This function takes a batch of tours and a distance matrix, and for each tour,
 * attempts to improve it by performing a single 2-opt move (i.e., reversing a subsegment
 * of the tour if it results in a shorter path). The optimization is parallelized
 * across tours using OpenMP.
 *
 * @param tours_np      A 2D NumPy array (n_tours x n) of integer indices representing the tours.
 *                      Each row corresponds to a tour (sequence of node indices).
 * @param distances_np  A 2D NumPy array (n x n) of floats representing the distance matrix.
 *                      distances_np[i, j] gives the distance from node i to node j.
 *
 * @note
 * - Only the first improving 2-opt move found for each tour is applied.
 * - The function modifies the input tours in-place.
 * - Assumes input arrays are contiguous and properly shaped.
 * - Parallelization is performed at the tour level.
 */
void two_opt(py::array_t<int, py::array::c_style | py::array::forcecast> tours_np,
             py::array_t<float, py::array::c_style | py::array::forcecast> distances_np) {
    // Obtener buffers y formas
    auto buf_t = tours_np.request();
    auto buf_d = distances_np.request();

    std::size_t n_tours = buf_t.shape[0];
    std::size_t n       = buf_t.shape[1];

    // Punteros a datos contiguos
    int*    tours_ptr     = static_cast<int*>(buf_t.ptr);
    float*  distances_ptr = static_cast<float*>(buf_d.ptr);

    // Paralelizar a nivel de tours
    #pragma omp parallel for schedule(guided) shared(tours_ptr, distances_ptr, n_tours, n)
    for (std::size_t t = 0; t < n_tours; ++t) {
        int* tour = tours_ptr + t * n;
        bool improved = false;

        for (std::size_t i = 1; i + 2 < n && !improved; ++i) {
            int a = tour[i - 1];
            int b = tour[i];

            float acost = distances_ptr[a * n + b];

            for (std::size_t k = i + 1; k + 1 < n && !improved; ++k) {
                int c = tour[k];
                int d = tour[k + 1];

                float delta = (acost + distances_ptr[c * n + d])
                            - (distances_ptr[a * n + c] + distances_ptr[b * n + d]);

                if (delta > 1e-6f) {
                    // invertir subarray [i..k]
                    std::reverse(tour + i, tour + k + 1);
                    improved = true;
                }
            }
        }
    }
}

PYBIND11_MODULE(utils, m) {
    m.doc() = "C++ utility module for optimization tasks, including classification and TSP-specific operations.";

    m.def("fitness", &fitness,
        "Computes the combined fitness function for feature selection, "
        "based on classification and reduction rates.",
        py::arg("weights"), py::arg("X_train"), py::arg("y_train"),
        py::arg("alpha"), py::arg("threshold"));

    m.def("fitness_omp", &fitness_omp,
        "Computes the combined fitness function using OpenMP parallelism "
        "for feature selection tasks.",
        py::arg("weights"), py::arg("X_train"), py::arg("y_train"), py::arg("fitness_values"),
        py::arg("alpha"), py::arg("threshold"));

    m.def("clas_rate", &clas_rate,
        "Computes the classification rate given a set of selected features.",
        py::arg("weights"), py::arg("X_train"), py::arg("y_train"), py::arg("threshold"));

    m.def("red_rate", &red_rate,
        "Computes the feature reduction rate based on a selection threshold.",
        py::arg("weights"), py::arg("threshold"));

    m.def("predict", &predict,
        "Performs prediction on test data using a 1-NN classifier with weighted features.",
        py::arg("X_test"), py::arg("weights"), py::arg("X_train"), py::arg("y_train"),
        py::arg("threshold"));

    m.def("fitness_tsp", &fitness_tsp,
        "Computes the fitness (i.e., total distance) of a single TSP tour.",
        py::arg("distances"), py::arg("solution"));

    m.def("fitness_tsp_omp", &fitness_tsp_omp,
        "Computes the fitness of multiple TSP tours using OpenMP parallelism.",
        py::arg("distances"), py::arg("solutions"), py::arg("fitness_values"));

    m.def("crossover_tsp", &crossover_tsp,
        "Performs crossover on a population of TSP solutions based on given start and end indices.",
        py::arg("population"), py::arg("random_starts"), py::arg("random_ends"));

    m.def("mutation_tsp", &mutation_tsp,
        "Performs mutation on a population of TSP solutions using swap operations.",
        py::arg("population"), py::arg("individual_indices"), py::arg("indices1"), py::arg("indices2"));

    m.def("update_pheromones_tsp", &update_pheromones_tsp,
        "Updates the pheromone matrix in ACO based on solution quality and evaporation.",
        py::arg("pheromones"), py::arg("colony"), py::arg("fitness_values"), py::arg("evaporation_rate"));

    m.def("construct_solutions_tsp_inner_cpu", &construct_solutions_tsp_inner_cpu,
        "Constructs TSP solutions using the ACO probabilistic model on CPU.",
        py::arg("solutions"), py::arg("visited"), py::arg("pheromones"),
        py::arg("heuristic_matrix"), py::arg("rand_matrix"), py::arg("colony_size"),
        py::arg("n_cities"), py::arg("alpha"), py::arg("epsilon"));

    m.def("construct_solutions_tsp_inner", &construct_solutions_tsp_inner,
        "Constructs TSP solutions using the ACO probabilistic model (parallel version).",
        py::arg("solutions"), py::arg("visited"), py::arg("pheromones"),
        py::arg("heuristic_matrix"), py::arg("rand_matrix"), py::arg("colony_size"),
        py::arg("n_cities"), py::arg("alpha"), py::arg("epsilon"));

    m.def("get_swap_sequence", &get_swap_sequence,
        "Generates the swap sequence required to transform one TSP tour into another.",
        py::arg("from_tour"), py::arg("to_tour"));

    m.def("two_opt", &two_opt,
        "Applies the 2-opt local search optimization to a batch of TSP tours.",
        py::arg("tours"), py::arg("distances"));
}
