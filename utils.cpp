#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace py = pybind11;
using namespace std;

// Definición de constantes
const float CLASS_WEIGHT = 0.75f;
const float RED_WEIGHT = 0.25f;
const float THRESHOLD = 0.1f;

float clas_rate_cpp(const float* weights, size_t w_size, const float* X_train, size_t n, size_t d, const int* y_train) {
    vector<float> weights_to_use;
    vector<vector<float>> attributes_to_use;

    for (size_t j = 0; j < w_size; ++j) {
        if (weights[j] >= THRESHOLD) {
            weights_to_use.push_back(weights[j]);
        }
    }

    for (size_t i = 0; i < n; ++i) {
        vector<float> row;
        for (size_t j = 0; j < w_size; ++j) {
            if (weights[j] >= THRESHOLD) {
                row.push_back(X_train[i * d + j]);
            }
        }
        attributes_to_use.push_back(row);
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

float clas_rate(py::array_t<float> weights_np, py::array_t<float> X_train_np, py::array_t<int> y_train_np) {
    auto weights = weights_np.unchecked<1>();
    auto X_train = X_train_np.unchecked<2>();
    auto y_train = y_train_np.unchecked<1>();

    size_t w_size = weights.shape(0);
    size_t n = X_train.shape(0);
    size_t d = X_train.shape(1);

    return clas_rate_cpp(weights.data(0), w_size, X_train.data(0, 0), n, d, y_train.data(0));
}

float red_rate_cpp(const float* weights, size_t w_size) {
    using namespace std;
    size_t count = 0;
    for (size_t i = 0; i < w_size; ++i) {
        if (weights[i] < 0.1f) {
            ++count;
        }
    }
    return 100.0f * count / w_size;
}

float red_rate(py::array_t<float> weights_np) {
    auto weights = weights_np.unchecked<1>();
    size_t w_size = weights.shape(0);
    return red_rate_cpp(weights.data(0), w_size);
}

float fitness(py::array_t<float> weights_np, py::array_t<float> X_train_np, py::array_t<int> y_train_np) {
    using namespace std;
    auto weights = weights_np.unchecked<1>();
    auto X_train = X_train_np.unchecked<2>();
    auto y_train = y_train_np.unchecked<1>();

    size_t w_size = weights.shape(0);
    size_t n = X_train.shape(0);
    size_t d = X_train.shape(1);

    float clas = clas_rate_cpp(weights.data(0), w_size, X_train.data(0, 0), n, d, y_train.data(0));
    float red = red_rate_cpp(weights.data(0), w_size);

    return CLASS_WEIGHT * clas + RED_WEIGHT * red;
}

void fitness_omp_cpp(const float* weights, const float* X_train, const int* y_train, float *fitness_values,
                  size_t n_sol, size_t n, size_t d, size_t w_size) {
    #pragma omp parallel for default(none) shared(weights, X_train, y_train, fitness_values, n_sol, n, d, w_size)
    for (size_t i = 0; i < n_sol; ++i) {
        float clas = clas_rate_cpp(&weights[i * w_size], w_size, X_train, n, d, y_train);
        float red = red_rate_cpp(&weights[i * w_size], w_size);

        fitness_values[i] = CLASS_WEIGHT * clas + RED_WEIGHT * red;
    }
}

void fitness_omp(py::array_t<float> weights_np, py::array_t<float> X_train_np, py::array_t<int> y_train_np,
                 py::array_t<float> fitness_values_np) {

    auto weights_info = weights_np.request();
    auto X_train_info = X_train_np.request();
    auto y_train_info = y_train_np.request();
    auto fitness_values_info = fitness_values_np.request();

    size_t w_size = X_train_info.shape[1];
    size_t n = X_train_info.shape[0];
    size_t d = X_train_info.shape[1];
    size_t n_sol = weights_info.shape[0];

    if (X_train_info.ndim != 2 || y_train_info.ndim != 1 || fitness_values_info.ndim != 1)
        throw std::runtime_error("Formato de entrada inválido");

    auto weights_ptr = static_cast<float *>(weights_info.ptr);
    auto X_train_ptr = static_cast<float *>(X_train_info.ptr);
    auto y_train_ptr = static_cast<int *>(y_train_info.ptr);
    auto fitness_values = static_cast<float *>(fitness_values_info.ptr);

    fitness_omp_cpp(weights_ptr, X_train_ptr, y_train_ptr, fitness_values,
                  n_sol, n, d, w_size);

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
    // Verificar propiedades de los arrays
    if (solution.ndim() != 1 || distances.ndim() != 2)
        throw std::runtime_error("Formato de entrada inválido");

    py::buffer_info dist_info = distances.request();
    py::buffer_info sol_info = solution.request();

    const int n = sol_info.shape[0];

    if (dist_info.shape[0] != n || dist_info.shape[1] != n)
        throw std::runtime_error("Dimensiones incompatibles");

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
    // Verificar propiedades de los arrays
    if (solutions.ndim() != 2 || distances.ndim() != 2)
    throw std::runtime_error("Formato de entrada inválido");

    py::buffer_info dist_info = distances.request();
    py::buffer_info sol_info = solutions.request();
    py::buffer_info fit_info = fitness_values.request();

    const int n_sol = sol_info.shape[0];
    const int n_cities = dist_info.shape[0];

    if (dist_info.shape[0] != n_cities || dist_info.shape[1] != n_cities)
        throw std::runtime_error("Dimensiones incompatibles");

    // Obtener punteros a los datos
    const float *dist_ptr = static_cast<float *>(dist_info.ptr);
    const int *sol_ptr = static_cast<int *>(sol_info.ptr);
    float *fit_ptr = static_cast<float *>(fit_info.ptr);

    fitness_tsp_omp_cpp(dist_ptr, sol_ptr, fit_ptr, n_sol, n_cities);
}

void crossover_blx(
    py::array_t<float> population,
    py::array_t<int> pairs_even_idx,
    py::array_t<int> pairs_odd_idx,
    py::array_t<float> rand_uniform1,
    py::array_t<float> rand_uniform2,
    float alpha
) {
    auto pop_info = population.request();
    auto even_info = pairs_even_idx.request();
    auto odd_info = pairs_odd_idx.request();
    auto rand1_info = rand_uniform1.request();
    auto rand2_info = rand_uniform2.request();

    if (pop_info.ndim != 2)
        throw std::runtime_error("Population must be 2D");

    int n_pairs = even_info.size;
    int n_vars = pop_info.shape[1];

    if (odd_info.size != n_pairs || rand1_info.shape[0] != n_pairs || rand2_info.shape[0] != n_pairs)
        throw std::runtime_error("Mismatch in number of pairs or random samples");

    float* pop_ptr = static_cast<float*>(pop_info.ptr);
    int* even_ptr = static_cast<int*>(even_info.ptr);
    int* odd_ptr = static_cast<int*>(odd_info.ptr);
    float* rand1_ptr = static_cast<float*>(rand1_info.ptr);
    float* rand2_ptr = static_cast<float*>(rand2_info.ptr);

    #pragma omp parallel for
    for (int i = 0; i < n_pairs; ++i) {
        int idx1 = even_ptr[i];
        int idx2 = odd_ptr[i];

        float* p1 = &pop_ptr[idx1 * n_vars];
        float* p2 = &pop_ptr[idx2 * n_vars];
        float* r1 = &rand1_ptr[i * n_vars];
        float* r2 = &rand2_ptr[i * n_vars];

        for (int j = 0; j < n_vars; ++j) {
            float cmin = std::min(p1[j], p2[j]);
            float cmax = std::max(p1[j], p2[j]);
            float I = cmax - cmin;

            float lower = cmin - alpha * I;
            float upper = cmax + alpha * I;

            float o1 = lower + r1[j] * (upper - lower);
            float o2 = lower + r2[j] * (upper - lower);

            p1[j] = (o1 < 0.0f) ? 0.0f : (o1 > 1.0f ? 1.0f : o1);
            p2[j] = (o2 < 0.0f) ? 0.0f : (o2 > 1.0f ? 1.0f : o2);
        }
    }
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

        // Obtener los índices del segmento (ya ordenados)
        int start = starts_ptr[k];
        int end   = ends_ptr[k];
        if (start < 0 || end >= n_cities || start > end)
            throw std::runtime_error("Índices de segmento inválidos.");

        // Vectores temporales para los hijos
        vector<int> child1(n_cities, -1);
        vector<int> child2(n_cities, -1);

        // --- Copiar el segmento seleccionado ---
        for (int i = start; i <= end; ++i) {
            child1[i] = pop_ptr[parent1 * n_cities + i];
            child2[i] = pop_ptr[parent2 * n_cities + i];
        }

        // --- Calcular los índices restantes (orden circular) ---
        vector<int> rem;
        // Desde end+1 hasta el final
        for (int i = end + 1; i < n_cities; ++i)
            rem.push_back(i);
        // Desde 0 hasta start-1
        for (int i = 0; i < start; ++i)
            rem.push_back(i);

        // --- Rellenar child1 ---
        // Marcar genes presentes en el segmento de child1 (tomados del padre1)
        vector<bool> used(n_cities, false);
        for (int i = start; i <= end; ++i) {
            int gene = child1[i];
            if (gene >= 0 && gene < n_cities)
                used[gene] = true;
        }
        int pos = 0;
        // Iterar en el orden en que aparecen en parent2
        for (int j = 0; j < n_cities && pos < int(rem.size()); ++j) {
            int gene = pop_ptr[parent2 * n_cities + j];
            if (!used[gene]) {
                child1[rem[pos]] = gene;
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
        // Iterar en el orden en que aparecen en parent1
        for (int j = 0; j < n_cities && pos < int(rem.size()); ++j) {
            int gene = pop_ptr[parent1 * n_cities + j];
            if (!used[gene]) {
                child2[rem[pos]] = gene;
                pos++;
            }
        }

        // --- Actualizar la población con los nuevos hijos ---
        for (int j = 0; j < n_cities; ++j) {
            pop_ptr[parent1 * n_cities + j] = child1[j];
            pop_ptr[parent2 * n_cities + j] = child2[j];
        }
    } // fin de for paralelo

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

// Exponer las funciones a Python con pybind11
PYBIND11_MODULE(utils, m) {
    m.doc() = "Módulo de utilidades CPP";
    m.def("fitness", &fitness, "Calcula la función fitness combinada", 
          py::arg("weights"), py::arg("X_train"), py::arg("y_train"));
    m.def("fitness_omp", &fitness_omp, "Calcula la función fitness con OpenMP",
            py::arg("weights"), py::arg("X_train"), py::arg("y_train"), py::arg("fitness_values"));
    m.def("clas_rate", &clas_rate, "Calcula la tasa de clasificación",
            py::arg("weights"), py::arg("X_train"), py::arg("y_train"));
    m.def("red_rate", &red_rate, "Calcula la tasa de reducción",
            py::arg("weights"));
    m.def("fitness_tsp", &fitness_tsp, "Calcula la función fitness para TSP",
            py::arg("distances"), py::arg("solution"));
    m.def("fitness_tsp_omp", &fitness_tsp_omp, "Calcula la función fitness para TSP usando OpenMP",
            py::arg("distances"), py::arg("solutions"), py::arg("fitness_values"));
    m.def("crossover_blx", &crossover_blx, "Realiza el cruce BLX",
            py::arg("population"), py::arg("pairs_even_idx"), py::arg("pairs_odd_idx"),
            py::arg("rand_uniform1"), py::arg("rand_uniform2"), py::arg("alpha"));
    m.def("crossover_tsp", &crossover_tsp, "Realiza el cruce de soluciones TSP",
            py::arg("population"), py::arg("random_starts"), py::arg("random_ends"));
    m.def("mutation_tsp", &mutation_tsp, "Realiza la mutación de soluciones TSP",
            py::arg("population"), py::arg("individual_indices"), py::arg("indices1"), py::arg("indices2"));
}