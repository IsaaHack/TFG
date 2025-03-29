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


double clas_rate(const vector<double>& weights, const vector<vector<double>>& X_train, const vector<int>& y_train) {
    vector<double> weights_to_use;
    vector<vector<double>> attributes_to_use;

    // Filtrar pesos y atributos
    for (size_t j = 0; j < weights.size(); ++j) {
        if (weights[j] >= 0.1) {
            weights_to_use.push_back(weights[j]);
        }
    }

    for (size_t i = 0; i < X_train.size(); ++i) {
        vector<double> row;
        for (size_t j = 0; j < weights.size(); ++j) {
            if (weights[j] >= 0.1) {
                row.push_back(X_train[i][j]);
            }
        }
        attributes_to_use.push_back(row);
    }

    size_t n = attributes_to_use.size();
    vector<vector<double>> distances(n, vector<double>(n, numeric_limits<double>::infinity()));

    // Calcular distancias Euclidianas ponderadas
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

    // Asegurarse de que la diagonal tenga valores infinitos
    // para que no se considere a sí mismo como el vecino más cercano
    for (size_t i = 0; i < n; ++i) {
        distances[i][i] = numeric_limits<double>::infinity();
    }

    // Encontrar los índices con menor distancia (excluyendo la diagonal)
    vector<int> index_predictions(n);
    for (size_t i = 0; i < n; ++i) {
        index_predictions[i] = min_element(distances[i].begin(), distances[i].end()) - distances[i].begin();
    }

    // Obtener etiquetas de predicción
    vector<int> predictions_labels(n);
    for (size_t i = 0; i < n; ++i) {
        predictions_labels[i] = y_train[index_predictions[i]];
    }

    // Calcular tasa de acierto
    int correct = 0;
    for (size_t i = 0; i < n; ++i) {
        if (predictions_labels[i] == y_train[i]) {
            ++correct;
        }
    }
    
    return 100.0 * correct / n;
}

double red_rate(const vector<double>& weights) {
    size_t count = 0;
    for (double w : weights) {
        if (w < 0.1) {
            ++count;
        }
    }
    return 100.0 * count / weights.size();
}

double fitness(const vector<double>& weights, const vector<vector<double>>& X_train, const vector<int>& y_train) {
    double clas = clas_rate(weights, X_train, y_train);
    double red = red_rate(weights);
    return 0.75 * clas + 0.25 * red;
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

// Exponer las funciones a Python con pybind11
PYBIND11_MODULE(utils, m) {
    m.doc() = "Módulo de utilidades para calcular clas_rate, red_rate y fitness";
    m.def("clas_rate", &clas_rate, "Calcula la tasa de acierto", 
          py::arg("weights"), py::arg("X_train"), py::arg("y_train"));
    m.def("red_rate", &red_rate, "Calcula la tasa de reducción", 
          py::arg("weights"));
    m.def("fitness", &fitness, "Calcula la función fitness combinada", 
          py::arg("weights"), py::arg("X_train"), py::arg("y_train"));
    m.def("fitness_tsp", &fitness_tsp, "Calcula la función fitness para TSP",
            py::arg("distances"), py::arg("solution"));
}