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

// Paralelización de clas_rate
double clas_rate(const vector<double>& weights, const vector<vector<double>>& X_train, const vector<int>& y_train) {
    vector<double> weights_to_use;
    vector<vector<double>> attributes_to_use;
    size_t n = X_train.size();

    // Filtrar pesos y atributos (paralelizado)
    for (size_t j = 0; j < weights.size(); ++j) {
        if (weights[j] >= 0.1) {
            weights_to_use.push_back(weights[j]);
        }
    }

    // Filtrar atributos con múltiples hilos
    for (size_t i = 0; i < n; ++i) {
        vector<double> row;
        for (size_t j = 0; j < weights.size(); ++j) {
            if (weights[j] >= 0.1) {
                row.push_back(X_train[i][j]);
            }
        }
        attributes_to_use.push_back(row);
    }

    vector<vector<double>> distances(n, vector<double>(n, numeric_limits<double>::infinity()));

    // Paralelizar cálculo de distancias
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < attributes_to_use[i].size(); ++k) {
                double diff = attributes_to_use[i][k] - attributes_to_use[j][k];
                sum += weights_to_use[k] * diff * diff;
            }
            double dist = sqrt(sum);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    // Ajustar diagonal a infinito
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        distances[i][i] = numeric_limits<double>::infinity();
    }

    // Paralelizar búsqueda del vecino más cercano
    vector<int> index_predictions(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        index_predictions[i] = min_element(distances[i].begin(), distances[i].end()) - distances[i].begin();
    }

    // Obtener etiquetas de predicción en paralelo
    vector<int> predictions_labels(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        predictions_labels[i] = y_train[index_predictions[i]];
    }

    // Calcular tasa de acierto en paralelo
    int correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < n; ++i) {
        if (predictions_labels[i] == y_train[i]) {
            correct++;
        }
    }

    return 100.0 * correct / n;
}

// Paralelización de red_rate
double red_rate(const vector<double>& weights) {
    size_t count = 0;

    // Contar pesos en paralelo con `reduction`
    #pragma omp parallel for reduction(+:count)
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] < 0.1) {
            count++;
        }
    }

    return 100.0 * count / weights.size();
}

// Función fitness combinada
double fitness_omp(const vector<double>& weights, const vector<vector<double>>& X_train, const vector<int>& y_train) {
    double clas = clas_rate(weights, X_train, y_train);
    double red = red_rate(weights);
    
    return 0.75 * clas + 0.25 * red;
}

// Versión C++ con punteros y OpenMP
float fitness_tsp_omp_cpp(const float *distances, const int *solution, int n){
    float fitness_value = 0.0f;

    #pragma omp parallel for reduction(+ : fitness_value)
    for (int i = 0; i < n-1; ++i)
    {
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

PYBIND11_MODULE(utils3, m) {
    m.doc() = "Módulo de utilidades con OpenMP";
    
    m.def("clas_rate", &clas_rate, "Calcula la tasa de acierto", 
          py::arg("weights"), py::arg("X_train"), py::arg("y_train"));
    
    m.def("red_rate", &red_rate, "Calcula la tasa de reducción", 
          py::arg("weights"));
    
    // Exponer fitness_omp
    m.def("fitness_omp", &fitness_omp, "Calcula la función fitness con OpenMP",
          py::arg("weights"), py::arg("X_train"), py::arg("y_train"));

    // Exponer fitness_tsp_omp
    m.def("fitness_tsp_omp", &fitness_tsp_omp, "Calcula la función fitness para TSP con OpenMP",
          py::arg("distances"), py::arg("solution"));
}