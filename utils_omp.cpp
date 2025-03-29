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

// Función clas_rate con punteros
float clas_rate(const float* weights, size_t w_size, const float* X_train, size_t n, size_t d, const int* y_train) {
    vector<float> weights_to_use;
    vector<vector<float>> attributes_to_use;

    // Filtrar pesos
    for (size_t j = 0; j < w_size; ++j) {
        if (weights[j] >= 0.1f) {
            weights_to_use.push_back(weights[j]);
        }
    }

    // Filtrar atributos
    for (size_t i = 0; i < n; ++i) {
        vector<float> row;
        for (size_t j = 0; j < w_size; ++j) {
            if (weights[j] >= 0.1f) {
                row.push_back(X_train[i * d + j]);
            }
        }
        attributes_to_use.push_back(row);
    }

    vector<vector<float>> distances(n, vector<float>(n, numeric_limits<float>::infinity()));

    // Paralelizar cálculo de distancias
    #pragma omp parallel for
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

    vector<int> index_predictions(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        index_predictions[i] = min_element(distances[i].begin(), distances[i].end()) - distances[i].begin();
    }

    vector<int> predictions_labels(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        predictions_labels[i] = y_train[index_predictions[i]];
    }

    int correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < n; ++i) {
        if (predictions_labels[i] == y_train[i]) {
            correct++;
        }
    }

    return 100.0f * correct / n;
}

// Función red_rate con punteros
float red_rate(const float* weights, size_t w_size) {
    size_t count = 0;
    #pragma omp parallel for reduction(+:count)
    for (size_t i = 0; i < w_size; ++i) {
        if (weights[i] < 0.1f) {
            count++;
        }
    }
    return 100.0f * count / w_size;
}

// Función fitness con compatibilidad para Python
float fitness_omp(py::array_t<float> weights_np, py::array_t<float> X_train_np, py::array_t<int> y_train_np) {
    auto weights = weights_np.unchecked<1>();
    auto X_train = X_train_np.unchecked<2>();
    auto y_train = y_train_np.unchecked<1>();

    size_t w_size = weights.shape(0);
    size_t n = X_train.shape(0);
    size_t d = X_train.shape(1);

    float clas = clas_rate(weights.data(0), w_size, X_train.data(0, 0), n, d, y_train.data(0));
    float red = red_rate(weights.data(0), w_size);

    return 0.75f * clas + 0.25f * red;
}

// Versión C++ con punteros y OpenMP
float fitness_tsp_omp_cpp(const float *distances, const int *solution, int n){
    float fitness_value = 0.0f;

    #pragma omp parallel for reduction(+ : fitness_value)
    for (int i = 0; i < n-1; ++i)
    {
        int from = solution[i];
        int to = solution[(i + 1) % n];
        fitness_value += distances[from * n + to];
    }

    return -fitness_value;
}

// Wrapper para Python/Numpy
float fitness_tsp_omp(py::array_t<float> distances,
                              py::array_t<int> solution)
{
    // Verificar propiedades de los arrays
    if (solution.ndim() != 1 || distances.ndim() != 2)
        throw std::runtime_error("Formato de entrada inválido");

    py::buffer_info dist_info = distances.request();
    py::buffer_info sol_info = solution.request();

    const int n = sol_info.shape[0];

    if (dist_info.shape[0] != n || dist_info.shape[1] != n)
        throw std::runtime_error("Dimensiones de distancias no coinciden");

    // Obtener punteros a los datos
    const float *dist_ptr = static_cast<float *>(dist_info.ptr);
    const int *sol_ptr = static_cast<int *>(sol_info.ptr);

    return fitness_tsp_omp_cpp(dist_ptr, sol_ptr, n);
}

PYBIND11_MODULE(utils_omp, m) {
    m.doc() = "Módulo de utilidades con OpenMP";
    // Exponer fitness_omp
    m.def("fitness_omp", &fitness_omp, "Calcula la función fitness con OpenMP",
            py::arg("weights"), py::arg("X_train"), py::arg("y_train"));

    // Exponer fitness_tsp_omp
    m.def("fitness_tsp_omp", &fitness_tsp_omp, "Calcula la función fitness para TSP con OpenMP",
          py::arg("distances"), py::arg("solution"));
}