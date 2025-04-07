#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <cmath>

namespace py = pybind11;
using namespace std;

#define TILE_SIZE 256
#define NUM_STREAMS 2

const double THRESHOLD = 0.1;
const double CLASS_WEIGHT = 0.75;
const double RED_WEIGHT = 0.25;

__global__ void warmup_kernel() {
    // Kernel vacío para calentar la GPU
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        tid = 0; // No hacer nada, solo calentar
    }
}

__global__ void tsp_fitness_kernel_simple(
    const float* distances,
    const int* solution,
    float* partial_sums,
    int n
) {
    // Índice global del hilo
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Verificar límites
    if (tid >= n) return;

    // Obtener índices de la solución
    int from = solution[tid];
    int to = solution[(tid + 1) % n];  // Manejo circular automático

    // Acceso directo a memoria global
    partial_sums[tid] = __ldg(&distances[from * n + to]);
}

__global__ void tsp_fitness_kernel(const float* __restrict__ distances, const int* __restrict__ solution, float* __restrict__ partial_sums, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Precarga en memoria compartida
    __shared__ int shared_sol[TILE_SIZE + 1];

    if (tid < n) {
        shared_sol[threadIdx.x] = solution[tid];
        if (threadIdx.x == blockDim.x - 1 && tid + 1 < n)
            shared_sol[blockDim.x] = solution[tid + 1];
    }
    __syncthreads();

    if (tid < n) {
        int from = shared_sol[threadIdx.x];
        int to = (threadIdx.x == blockDim.x - 1) ?
        solution[(tid + 1) % n] : shared_sol[threadIdx.x + 1];

        partial_sums[tid] = __ldg(&distances[from * n + to]);
    }
}

// Función wrapper que acepta punteros de GPU
float fitness_tsp_cuda(py::capsule distances_capsule, py::capsule solution_capsule, int n) {
    // Obtener punteros desde los objetos de GPU
    float* d_distances = static_cast<float*>(distances_capsule.get_pointer());
    int* d_solution = static_cast<int*>(solution_capsule.get_pointer());

    // Reservar memoria para resultados
    float* d_partial;
    cudaMalloc(&d_partial, n * sizeof(float));

    // Lanzar kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    tsp_fitness_kernel_simple<<<gridSize, blockSize>>>(d_distances, d_solution, d_partial, n);

    // Copiar resultado a CPU y sumar
    vector<float> h_partial(n);
    cudaMemcpy(h_partial.data(), d_partial, n * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for(float val : h_partial) total += val;

    // Liberar memoria
    cudaFree(d_partial);

    return -total;
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
            float w  = 0.0f;
            if(weights[k] >= THRESHOLD)
                w = __ldg(&weights[k]);
            
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
    int *s_y_train = sdata;
    int *s_predictions = sdata + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cargar datos en memoria compartida
    if (i < num_samples) {
        s_y_train[tid] = y_train[i];
        s_predictions[tid] = y_train[predictions[i]];
    }
    __syncthreads();

    int correct_count = 0;
    if (i < num_samples) {
        if (s_y_train[tid] == s_predictions[tid]) {
            correct_count = 1;
        }

        int *s_correct = sdata;
        s_correct[tid] = correct_count;
        __syncthreads();

        // Reducción en memoria compartida
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_correct[tid] += s_correct[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            atomicAdd(correct, s_correct[0]);
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
    // thrust::host_vector<int> h_valid_indices(d_valid_indices);

    // for (int i = 0; i < num_features; ++i) {
    //     printf("valid_indices[%d] = %d\n", i, h_valid_indices[i]);
    // }

    // 2. Calcular matriz de distancias
    thrust::device_vector<float> d_distances(num_samples * num_samples, INFINITY);
    dim3 block(16, 16);
    dim3 grid((num_samples + block.x - 1) / block.x,
             (num_samples + block.y - 1) / block.y);

    distance_kernel<<<grid, block>>>(thrust::raw_pointer_cast(d_X_train.data()),
                                   thrust::raw_pointer_cast(d_weights.data()),
                                   thrust::raw_pointer_cast(d_distances.data()),
                                   num_samples, num_features,
                                   thrust::raw_pointer_cast(d_valid_indices.data()),
                                   valid_features);
    // distance_kernel<<<grid, block>>>(thrust::raw_pointer_cast(d_X_train.data()),
    //                                 thrust::raw_pointer_cast(d_weights.data()),
    //                                 thrust::raw_pointer_cast(d_distances.data()),
    //                                 num_samples, num_features);

    // 3. Encontrar vecinos más cercanos
    thrust::device_vector<int> d_predictions(num_samples);
    gridSize = (num_samples + blockSize - 1) / blockSize;
    nearest_neighbor_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_distances.data()),
                                                   thrust::raw_pointer_cast(d_predictions.data()),
                                                   num_samples);

    // 4. Calcular precisión
    thrust::device_vector<int> d_correct(1, 0);
    smSize = blockSize * 2 * sizeof(int);
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
            py::arg("distances"), py::arg("solution"), py::arg("n"));
    m.def("fitness_cuda", &fitness_cuda, "Calcular fitness usando CUDA",
          py::arg("weights"), py::arg("X_train"), py::arg("y_train"),
          py::arg("num_samples"), py::arg("num_features"));
    m.def("warmup", &warmup_kernel, "Calentar la GPU");
    m.def("create_capsule", &create_capsule, "Crear cápsula CUDA",
          py::arg("ptr_address"));
}