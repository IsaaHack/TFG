#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace py = pybind11;
using namespace std;

double parallel_clas_rate(const vector<double>& weights,
                          const vector<vector<double>>& X_train,
                          const vector<int>& y_train) {
    vector<double> weights_to_use;
    vector<vector<double>> attributes_to_use;
    
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
    
    int n = attributes_to_use.size();
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int chunk = n / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? n : (rank + 1) * chunk;
    
    int local_correct = 0;
    for (int i = start; i < end; ++i) {
        double min_dist = numeric_limits<double>::infinity();
        int min_index = -1;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double sum = 0.0;
            for (size_t k = 0; k < attributes_to_use[i].size(); ++k) {
                double diff = attributes_to_use[i][k] - attributes_to_use[j][k];
                sum += weights_to_use[k] * diff * diff;
            }
            double dist = sqrt(sum);
            if (dist < min_dist) {
                min_dist = dist;
                min_index = j;
            }
        }
        if (y_train[min_index] == y_train[i]) {
            local_correct++;
        }
    }
    
    int total_correct = 0;
    MPI_Reduce(&local_correct, &total_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double global_clas_rate = 0.0;
    if (rank == 0) {
        global_clas_rate = 100.0 * total_correct / n;
    }
    
    MPI_Bcast(&global_clas_rate, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return global_clas_rate;
}

double parallel_red_rate(const vector<double>& weights) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = weights.size() / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? weights.size() : (rank + 1) * chunk;

    int local_count = 0;
    for (int i = start; i < end; ++i) {
        if (weights[i] < 0.1) {
            local_count++;
        }
    }

    int total_count = 0;
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_red_rate = 0.0;
    if (rank == 0) {
        global_red_rate = 100.0 * total_count / weights.size();
    }

    MPI_Bcast(&global_red_rate, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return global_red_rate;
}

void mpi_initialize() {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }
}

void mpi_finalize() {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

double fitness_mpi(const vector<double>& weights,
                   const vector<vector<double>>& X_train,
                   const vector<int>& y_train) {

    mpi_initialize();
    double clas = parallel_clas_rate(weights, X_train, y_train);
    double red = parallel_red_rate(weights);
    return 0.75 * clas + 0.25 * red;
    mpi_finalize();
}

double fitness_tsp_mpi(const vector<vector<double>>& distances,
                          const vector<int>& solution) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = solution.size() / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? solution.size() : (rank + 1) * chunk;

    double fitness_value = 0.0;
    for (int i = start; i < end; ++i) {
        int from = solution[i];
        int to = solution[(i + 1) % solution.size()];
        fitness_value += distances[from][to];
    }

    double total_fitness_value = 0.0;
    MPI_Reduce(&fitness_value, &total_fitness_value, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return -total_fitness_value;
}

PYBIND11_MODULE(utils2, m) {
    m.def("fitness_mpi", &fitness_mpi, "Función de evaluación de fitness con MPI");
    m.def("mpi_initialize", &mpi_initialize, "Inicializa MPI");
    m.def("mpi_finalize", &mpi_finalize, "Finaliza MPI");
    m.def("fitness_tsp_mpi", &fitness_tsp_mpi, "Función de fitness para TSP con MPI");
}