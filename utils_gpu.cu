#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>

namespace py = pybind11;
using namespace std;

const float THRESHOLD = 0.1;
const float CLASS_WEIGHT = 0.75;
const float RED_WEIGHT = 0.25;

__global__ void warmup_kernel() {
    // Kernel vacío para calentar la GPU
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        tid = 0; // No hacer nada, solo calentar
    }
}

__global__ void tsp_fitness_kernel_multiple(
    const float* distances,
    const int* solutions,
    float* fitness,
    int n,
    int num_solutions
) {
    // Índice global del hilo
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int *solution = solutions + tid * n;  

    if (tid < num_solutions) {
        float sum = 0.0f;

        for(int i = 0; i < n-1; ++i) {
            // Obtener índices de la solución
            int from = __ldg(&solution[i]);
            int to = __ldg(&solution[i + 1]);  // Manejo circular automático
    
            // Acceso directo a memoria global
            sum += __ldg(&distances[from * n + to]);
        }

        //Hacer el camino circular
        int from = __ldg(&solution[n - 1]);
        int to = __ldg(&solution[0]);
        sum += __ldg(&distances[from * n + to]);
    
        fitness[tid] = -sum;
    }
    
}

// Función wrapper que acepta punteros de GPU
void fitness_tsp_cuda(py::capsule distances_capsule, py::capsule solution_capsule, py::capsule fitness_capsule, int n, int num_solutions) {
    // Obtener punteros desde los objetos de GPU
    float* d_distances = static_cast<float*>(distances_capsule.get_pointer());
    int* d_solution = static_cast<int*>(solution_capsule.get_pointer());
    float* d_fitness = static_cast<float*>(fitness_capsule.get_pointer());

    // Lanzar kernel
    int blockSize = 256;
    int gridSize = (num_solutions + blockSize - 1) / blockSize;
    tsp_fitness_kernel_multiple<<<gridSize, blockSize>>>(d_distances, d_solution, d_fitness, n, num_solutions);
}

// Kernel para filtrar pesos y características
__global__ void filter_kernel(const float* weights, int* valid_indices, int* valid_count, int total_features) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Cargar datos en memoria compartida
    if (idx < total_features) {
        sdata[tid] = (weights[idx] >= THRESHOLD) ? 1 : 0;
    }
    __syncthreads();

    if (idx < total_features) {
        if (sdata[tid]) {
            int pos = atomicAdd(valid_count, 1);
            valid_indices[pos] = idx;
        }
    }
}

__global__ void obtain_matrix_filtered_kernel(const float *X, float *X_filtered,
                                              const float *weights, float *weights_filtered,
                                              const int *valid_indices, int num_samples,
                                              int num_features, int valid_features)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_samples){
        for (int j = 0; j < valid_features; ++j){
            int feature_idx = __ldg(&valid_indices[j]); // Lectura optimizada
            X_filtered[i * valid_features + j] = __ldg(&X[i * num_features + feature_idx]);
            weights_filtered[j] = __ldg(&weights[feature_idx]);
        }
    }
}

// Kernel para calcular distancias euclidianas ponderadas
__global__ void distance_kernel(const float* __restrict__ X, const float* __restrict__ weights,
                               float* distances, int num_samples,
                               int num_features, const int* __restrict__ valid_indices,
                               int valid_features) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_samples && j < num_samples && i < j) {
        float sum = 0.0f;

        // Cache X[i * num_features + feature_idx] en registros
        for (int k = 0; k < valid_features; ++k) {
            int feature_idx = __ldg(&valid_indices[k]); // Lectura optimizada
            float xi = __ldg(&X[i * num_features + feature_idx]);
            float xj = __ldg(&X[j * num_features + feature_idx]);
            float w  = __ldg(&weights[feature_idx]);
            float diff = xi - xj;
            sum += w * diff * diff;
        }

        float dist = sqrtf(sum);
        distances[i * num_samples + j] = dist;
        distances[j * num_samples + i] = dist;
    }
}

// Kernel para calcular distancias euclidianas
__global__ void distance_kernel(const float* __restrict__ X, const float* __restrict__ weights,
                               float* distances, int num_samples,
                               int num_features) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_samples && j < num_samples && i < j) {
        float sum = 0.0f;

        // Cache X[i * num_features + feature_idx] en registros
        for (int k = 0; k < num_features; ++k) {
            float xi = __ldg(&X[i * num_features + k]);
            float xj = __ldg(&X[j * num_features + k]);
            float w  = __ldg(&weights[k]);
            
            float diff = xi - xj;
            sum += w * diff * diff;
        }

        float dist = sqrtf(sum);
        distances[i * num_samples + j] = dist;
        distances[j * num_samples + i] = dist;
    }
}

// Kernel para encontrar vecinos más cercanos
__global__ void nearest_neighbor_kernel(const float* __restrict__ distances, int* predictions, int num_samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_samples) {
        float min_dist = INFINITY;
        int min_idx = -1;
        for (int j = 0; j < num_samples; ++j) {
            float dist = distances[j * num_samples + i];
            if (i != j && dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }
        predictions[i] = min_idx;
    }
}

// Kernel para calcular precisión
__global__ void accuracy_kernel(const int* __restrict__ predictions, const int* __restrict__ y_train, int* correct, int num_samples) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int correct_count = 0;
    if (i < num_samples) {
        if (y_train[i] == y_train[predictions[i]]) {
            correct_count = 1;
        }

        sdata[tid] = correct_count;
        __syncthreads();

        // Reducción en memoria compartida
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            atomicAdd(correct, sdata[0]);
        }
    }
}

float clas_rate_cuda(const thrust::device_vector<float>& d_weights,
                    const thrust::device_vector<float>& d_X_train,
                    const thrust::device_vector<int>& d_y_train,
                    int num_samples, int num_features) {
    // 1. Filtrar características relevantes
    thrust::device_vector<int> d_valid_indices(num_features);
    thrust::device_vector<int> d_valid_count(1, 0);

    int blockSize = 256;
    int gridSize = (num_features + blockSize - 1) / blockSize;
    size_t smSize = blockSize * sizeof(int);

    filter_kernel<<<gridSize, blockSize, smSize>>>(thrust::raw_pointer_cast(d_weights.data()),
                                                    thrust::raw_pointer_cast(d_valid_indices.data()),
                                                    thrust::raw_pointer_cast(d_valid_count.data()),
                                                    num_features);

    cudaDeviceSynchronize();
    int valid_features = d_valid_count[0];
    
    // 2. Obtener matriz y pesos filtrados
    thrust::device_vector<float> d_X_filtered(num_samples * valid_features);
    thrust::device_vector<float> d_weights_filtered(valid_features);
    gridSize = (num_samples + blockSize - 1) / blockSize;
    obtain_matrix_filtered_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_X_train.data()),
                                                            thrust::raw_pointer_cast(d_X_filtered.data()),
                                                            thrust::raw_pointer_cast(d_weights.data()),
                                                            thrust::raw_pointer_cast(d_weights_filtered.data()),
                                                            thrust::raw_pointer_cast(d_valid_indices.data()),
                                                            num_samples, num_features, valid_features);

    // 3. Calcular matriz de distancias
    thrust::device_vector<float> d_distances(num_samples * num_samples, INFINITY);
    dim3 block(16, 16);
    dim3 grid((num_samples + block.x - 1) / block.x,
             (num_samples + block.y - 1) / block.y);

    // distance_kernel<<<grid, block>>>(thrust::raw_pointer_cast(d_X_train.data()),
    //                                thrust::raw_pointer_cast(d_weights.data()),
    //                                thrust::raw_pointer_cast(d_distances.data()),
    //                                num_samples, num_features,
    //                                thrust::raw_pointer_cast(d_valid_indices.data()),
    //                                valid_features);
    distance_kernel<<<grid, block>>>(thrust::raw_pointer_cast(d_X_filtered.data()),
                                    thrust::raw_pointer_cast(d_weights_filtered.data()),
                                    thrust::raw_pointer_cast(d_distances.data()),
                                    num_samples, valid_features);

    // 4. Encontrar vecinos más cercanos
    thrust::device_vector<int> d_predictions(num_samples);
    gridSize = (num_samples + blockSize - 1) / blockSize;
    nearest_neighbor_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_distances.data()),
                                                   thrust::raw_pointer_cast(d_predictions.data()),
                                                   num_samples);

    // 5. Calcular precisión
    thrust::device_vector<int> d_correct(1, 0);
    smSize = blockSize * sizeof(int);
    accuracy_kernel<<<gridSize, blockSize, smSize>>>(thrust::raw_pointer_cast(d_predictions.data()),
                                                    thrust::raw_pointer_cast(d_y_train.data()),
                                                    thrust::raw_pointer_cast(d_correct.data()),
                                                    num_samples);

    cudaDeviceSynchronize();
    return 100.0f * d_correct[0] / num_samples;
}

struct IsRedundant {
    __host__ __device__ bool operator()(float w) const {
        return w < THRESHOLD;
    }
};

float red_rate_cuda(const thrust::device_vector<float>& d_weights) {
    auto count = thrust::count_if(thrust::device,
                                 d_weights.begin(),
                                 d_weights.end(),
                                 IsRedundant{});
    return 100.0f * count / d_weights.size();
}

float fitness_cuda(
    py::capsule weights_capsule,
    py::capsule X_train_capsule,
    py::capsule y_train_capsule,
    int num_samples,
    int num_features
) {
    // Obtener punteros desde CuPy y envolver en thrust::device_ptr
    thrust::device_ptr<float> d_weights_prt(
        static_cast<float*>(weights_capsule.get_pointer())
    );
    thrust::device_ptr<float> d_X_train_prt(
        static_cast<float*>(X_train_capsule.get_pointer())
    );
    thrust::device_ptr<int> d_y_train_prt(
        static_cast<int*>(y_train_capsule.get_pointer())
    );

    // Crear vectores de dispositivo
    thrust::device_vector<float> d_weights(d_weights_prt, d_weights_prt + num_features);
    thrust::device_vector<float> d_X_train(d_X_train_prt, d_X_train_prt + num_samples * num_features);
    thrust::device_vector<int> d_y_train(d_y_train_prt, d_y_train_prt + num_samples);

    float clas = 0.0f;
    float red = 0.0f;

    // Llamar a kernels CUDA directamente con los punteros
    clas = clas_rate_cuda(d_weights, d_X_train, d_y_train, num_samples, num_features);
    red = red_rate_cuda(d_weights);

    return CLASS_WEIGHT * clas + RED_WEIGHT * red;
}

// Función para crear cápsula desde un puntero entero (dirección de memoria)
py::capsule create_capsule(size_t ptr_address) {
    void* ptr = reinterpret_cast<void*>(ptr_address);
    return py::capsule(ptr, [](void*){ /* No liberar, manejado por CuPy */ });
}

PYBIND11_MODULE(utils_gpu, m) {
    m.def("fitness_tsp_cuda", &fitness_tsp_cuda, "Calcular fitness del TSP usando CUDA",
            py::arg("distances"), py::arg("solution"), py::arg("fitness"),
            py::arg("n"), py::arg("num_solutions"));
    m.def("fitness_cuda", &fitness_cuda, "Calcular fitness usando CUDA",
          py::arg("weights"), py::arg("X_train"), py::arg("y_train"),
          py::arg("num_samples"), py::arg("num_features"));
    m.def("warmup", &warmup_kernel, "Calentar la GPU");
    m.def("create_capsule", &create_capsule, "Crear cápsula CUDA",
          py::arg("ptr_address"));
}