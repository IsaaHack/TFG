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

// Definición de constantes
//const float CLASS_WEIGHT = 0.75f;
//const float RED_WEIGHT = 0.25f;
//const float THRESHOLD = 0.1f;

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

float clas_rate(py::array_t<float> weights_np, py::array_t<float> X_train_np, py::array_t<int> y_train_np, const float threshold) {
    auto weights = weights_np.unchecked<1>();
    auto X_train = X_train_np.unchecked<2>();
    auto y_train = y_train_np.unchecked<1>();

    size_t w_size = weights.shape(0);
    size_t n = X_train.shape(0);
    size_t d = X_train.shape(1);

    return clas_rate_cpp(weights.data(0), w_size, X_train.data(0, 0), n, d, y_train.data(0), threshold);
}

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

float red_rate(py::array_t<float> weights_np, const float threshold) {
    auto weights = weights_np.unchecked<1>();
    size_t w_size = weights.shape(0);
    return red_rate_cpp(weights.data(0), w_size, threshold);
}

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

void fitness_omp_cpp(const float* weights, const float* X_train, const int* y_train, float *fitness_values,
                  size_t n_sol, size_t n, size_t d, size_t w_size, const float alpha, const float threshold) {
    #pragma omp parallel for default(none) shared(weights, X_train, y_train, fitness_values, n_sol, n, d, w_size, alpha, threshold)
    for (size_t i = 0; i < n_sol; ++i) {
        float clas = clas_rate_cpp(&weights[i * w_size], w_size, X_train, n, d, y_train, threshold);
        float red = red_rate_cpp(&weights[i * w_size], w_size, threshold);

        fitness_values[i] = alpha * clas + (1 - alpha) * red;
    }
}

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

// Versión con punteros para integración con Numpy
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

// Wrapper para Python/Numpy
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

// Versión con OpenMP
void fitness_tsp_omp_cpp(const float* distances, const int* solutions, float *fitness_values, int n_sol, int n_cities) {
    #pragma omp parallel for default(none) shared(distances, solutions, fitness_values, n_sol, n_cities)
    for (int i = 0; i < n_sol; ++i) {
        fitness_values[i] = fitness_tsp_cpp(distances, &solutions[i * n_cities], n_cities);
    }
}

// Wrapper para Python/Numpy
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

// Función de cruce en C++
// - population: array de enteros, forma (num_sol, n_cities)
// - random_starts y random_ends: arrays de enteros de forma (num_crossovers,), con índices precomputados y ordenados
//
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

// Función mutation que realiza swap mutation usando índices aleatorios pasados desde Python.
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

// Exponer las funciones a Python con pybind11
PYBIND11_MODULE(utils, m) {
    m.doc() = "Módulo de utilidades CPP";
    m.def("fitness", &fitness, "Calcula la función fitness combinada", 
          py::arg("weights"), py::arg("X_train"), py::arg("y_train"),
          py::arg("alpha"), py::arg("threshold"));
    m.def("fitness_omp", &fitness_omp, "Calcula la función fitness con OpenMP",
            py::arg("weights"), py::arg("X_train"), py::arg("y_train"), py::arg("fitness_values"),
            py::arg("alpha"), py::arg("threshold"));
    m.def("clas_rate", &clas_rate, "Calcula la tasa de clasificación",
            py::arg("weights"), py::arg("X_train"), py::arg("y_train"), py::arg("threshold"));
    m.def("red_rate", &red_rate, "Calcula la tasa de reducción",
            py::arg("weights"), py::arg("threshold"));
    m.def("predict", &predict, "Realiza predicciones basadas en el modelo",
            py::arg("X_test"), py::arg("weights"), py::arg("X_train"), py::arg("y_train"),
            py::arg("threshold"));
    m.def("fitness_tsp", &fitness_tsp, "Calcula la función fitness para TSP",
            py::arg("distances"), py::arg("solution"));
    m.def("fitness_tsp_omp", &fitness_tsp_omp, "Calcula la función fitness para TSP usando OpenMP",
            py::arg("distances"), py::arg("solutions"), py::arg("fitness_values"));
    m.def("crossover_tsp", &crossover_tsp, "Realiza el cruce de soluciones TSP",
            py::arg("population"), py::arg("random_starts"), py::arg("random_ends"));
    m.def("mutation_tsp", &mutation_tsp, "Realiza la mutación de soluciones TSP",
            py::arg("population"), py::arg("individual_indices"), py::arg("indices1"), py::arg("indices2"));
    m.def("update_pheromones_tsp", &update_pheromones_tsp, "Actualiza las feromonas para TSP",
            py::arg("pheromones"), py::arg("colony"), py::arg("fitness_values"), py::arg("evaporation_rate"));
    m.def("construct_solutions_tsp_inner_cpu", &construct_solutions_tsp_inner_cpu, "Construye soluciones para TSP en CPU",
            py::arg("solutions"), py::arg("visited"), py::arg("pheromones"),
            py::arg("heuristic_matrix"), py::arg("rand_matrix"), py::arg("colony_size"),
            py::arg("n_cities"), py::arg("alpha"), py::arg("epsilon"));
    m.def("construct_solutions_tsp_inner", &construct_solutions_tsp_inner, "Construye soluciones para TSP",
            py::arg("solutions"), py::arg("visited"), py::arg("pheromones"),
            py::arg("heuristic_matrix"), py::arg("rand_matrix"), py::arg("colony_size"),
            py::arg("n_cities"), py::arg("alpha"), py::arg("epsilon"));
    m.def("get_swap_sequence", &get_swap_sequence, "Obtiene la secuencia de swaps entre dos tours",
            py::arg("from_tour"), py::arg("to_tour"));
    m.def("two_opt", &two_opt, "Optimiza tours TSP usando 2-opt",
            py::arg("tours"), py::arg("distances"));
}