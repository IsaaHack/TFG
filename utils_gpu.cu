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
#include <omp.h>
#include "./utils.cpp"
#include <chrono>

using namespace std::chrono;
namespace py = pybind11; 
using namespace std;

// Función para crear cápsula desde un puntero entero (dirección de memoria)
py::capsule create_capsule(size_t ptr_address) {
    void* ptr = reinterpret_cast<void*>(ptr_address);
    return py::capsule(ptr, [](void*){ /* No liberar, manejado por CuPy */ });
}

void s_solutions_hybrid(int &n_sol_cpu, int &n_sol_gpu, const int n_sol, const float speedup_factor) {
    // Calcular la distribución de soluciones entre CPU y GPU
    n_sol_gpu = round(n_sol * speedup_factor / (1 + speedup_factor));
    n_sol_cpu = n_sol - n_sol_gpu;

    if (n_sol_gpu <= 0) {
        n_sol_gpu = 0;
        n_sol_cpu = n_sol;
    } else if (n_sol_cpu <= 0) {
        n_sol_cpu = 0;
        n_sol_gpu = n_sol;
    }
    
}

__global__ void warmup_kernel() {
    // Kernel vacío para calentar la GPU
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        tid = 0; // No hacer nada, solo calentar
    }
}

__global__ void tsp_fitness_kernel_multiple(
    cudaTextureObject_t tex_distances,
    const int* solutions,
    float* fitness,
    int n,
    int num_solutions
) {
    // Índice global del hilo
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_solutions) {
        float sum = 0.0f;

        for (int city_idx = 0; city_idx < n; ++city_idx) {
            const int current_city = solutions[city_idx * num_solutions + tid];
            const int next_city = solutions[((city_idx + 1) % n) * num_solutions + tid];
            sum += tex2D<float>(tex_distances, current_city, next_city);
        }
        fitness[tid] = -sum;
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

void fitness_tsp_cuda_2(
    float* distances,
    int* solution,
    float* fitness,
    int n,
    int num_solutions
) {

    // Lanzar kernel
    int blockSize = 256;
    int gridSize = (num_solutions + blockSize - 1) / blockSize;
    tsp_fitness_kernel_multiple<<<gridSize, blockSize>>>(distances, solution, fitness, n, num_solutions);
}

float fitness_tsp_hybrid(
    py::array_t<float> distances_np,
    py::capsule distances_capsule,
    py::array_t<int> solution_np,
    py::array_t<float> fitness_np,
    int n,
    int num_solutions,
    float speedup_factor
) {
    // --- Obtener información de arrays ---
    auto distances_info = distances_np.request();
    auto solution_info = solution_np.request();
    auto fitness_info = fitness_np.request();

    auto distances_ptr = static_cast<float *>(distances_info.ptr);
    auto solution_ptr = static_cast<int *>(solution_info.ptr);
    auto fitness_ptr = static_cast<float *>(fitness_info.ptr);
    float* d_distances = static_cast<float*>(distances_capsule.get_pointer());

    // --- Calcular distribución CPU/GPU ---
    // int num_solutions_cpu = round(num_solutions / speedup_factor);
    // int num_solutions_gpu = num_solutions - num_solutions_cpu;

    // if (num_solutions_gpu <= 0) {
    //     num_solutions_gpu = 1;
    //     num_solutions_cpu = num_solutions - 1;
    // } else if (num_solutions_cpu <= 0) {
    //     num_solutions_cpu = 1;
    //     num_solutions_gpu = num_solutions - 1;
    // }
    int num_solutions_cpu = 0;
    int num_solutions_gpu = 0;
    s_solutions_hybrid(num_solutions_cpu, num_solutions_gpu, num_solutions, speedup_factor);

    float time_gpu, time_cpu;

    // if (num_solutions_cpu != 0) {
    //     //omp_set_nested(1);
    // }

    // --- Ejecución paralela CPU y GPU ---
    #pragma omp parallel sections shared(d_distances, solution_ptr, fitness_ptr, n, num_solutions_gpu, num_solutions_cpu)
    {
        // --- Sección GPU ---
        #pragma omp section
        {
            if (num_solutions_gpu != 0) {
                auto start = high_resolution_clock::now();

                // --- Preparar memoria en GPU ---
                int* solutions_gpu_ptr = nullptr;
                cudaMalloc((void**)&solutions_gpu_ptr, num_solutions_gpu * n * sizeof(int));
                cudaMemcpy(solutions_gpu_ptr, solution_ptr, num_solutions_gpu * n * sizeof(int), cudaMemcpyHostToDevice);

                auto fitness_gpu = thrust::device_vector<float>(num_solutions_gpu);

                fitness_tsp_cuda_2(
                    d_distances,
                    solutions_gpu_ptr,
                    thrust::raw_pointer_cast(fitness_gpu.data()),
                    n,
                    num_solutions_gpu
                );

                cudaMemcpy(fitness_ptr, fitness_gpu.data().get(), num_solutions_gpu * sizeof(float), cudaMemcpyDeviceToHost);

                // --- Liberar memoria de la GPU ---
                cudaFree(solutions_gpu_ptr);

                auto end = high_resolution_clock::now();
                duration<double> elapsed = end - start;
                time_gpu = elapsed.count() / num_solutions_gpu;
            }
        }

        // --- Sección CPU ---
        #pragma omp section
        {
            if (num_solutions_cpu != 0) {
                auto start = high_resolution_clock::now();

                omp_set_nested(1);
                int max_threads = omp_get_max_threads() / 2;
                omp_set_num_threads(std::max(1, max_threads - 1));

                fitness_tsp_omp_cpp(
                    distances_ptr,
                    &solution_ptr[num_solutions_gpu * n],
                    &fitness_ptr[num_solutions_gpu],
                    num_solutions_cpu,
                    n
                );

                omp_set_num_threads(max_threads);
                omp_set_nested(0);

                auto end = high_resolution_clock::now();
                duration<double> elapsed = end - start;
                time_cpu = elapsed.count() / num_solutions_cpu;
            }
        }
    }

    float new_speedup_factor = 1;
    if (num_solutions_cpu != 0 && num_solutions_gpu != 0) {
        new_speedup_factor = time_cpu / time_gpu;
    } else if (num_solutions_cpu == 0 || num_solutions_gpu == 0) {
        new_speedup_factor = speedup_factor;
    }

    return new_speedup_factor;
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

float fitness_hybrid(
    py::array_t<float> weights_np,
    py::array_t<float> X_train_np, 
    py::array_t<int> y_train_np,
    py::capsule X_train_capsule,
    py::capsule y_train_capsule,
    py::array_t<float> fitness_values_np,
    int num_samples,
    int num_features,
    float speedup_factor
) {
    // --- Obtener información de arrays ---
    auto weights_info = weights_np.request();
    auto X_train_info = X_train_np.request();
    auto y_train_info = y_train_np.request();
    auto fitness_values_info = fitness_values_np.request();

    if (X_train_info.ndim != 2 || y_train_info.ndim != 1 || fitness_values_info.ndim != 1)
        throw std::runtime_error("Formato de entrada inválido");

    auto weights_ptr = static_cast<float *>(weights_info.ptr);
    auto X_train_ptr = static_cast<float *>(X_train_info.ptr);
    auto y_train_ptr = static_cast<int *>(y_train_info.ptr);
    auto fitness_values = static_cast<float *>(fitness_values_info.ptr);

    size_t w_size = X_train_info.shape[1];
    size_t n = X_train_info.shape[0];
    size_t d = X_train_info.shape[1];
    size_t n_sol = weights_info.shape[0];

    // --- Obtener punteros de CuPy y crear vectores de dispositivo ---
    thrust::device_ptr<float> d_X_train_prt(static_cast<float*>(X_train_capsule.get_pointer()));
    thrust::device_ptr<int> d_y_train_prt(static_cast<int*>(y_train_capsule.get_pointer()));

    thrust::device_vector<float> d_X_train(d_X_train_prt, d_X_train_prt + num_samples * num_features);
    thrust::device_vector<int> d_y_train(d_y_train_prt, d_y_train_prt + num_samples);

    // --- Dividir trabajo entre CPU y GPU ---
    // int num_weights_cpu = round(n_sol / speedup_factor);
    // int num_weights_gpu = n_sol - num_weights_cpu;

    // if (num_weights_gpu <= 0) {
    //     num_weights_gpu = 1;
    //     num_weights_cpu = n_sol - 1;
    // } else if (num_weights_cpu <= 0) {
    //     num_weights_cpu = 1;
    //     num_weights_gpu = n_sol - 1;
    // }
    int num_weights_cpu = 0;
    int num_weights_gpu = 0;
    s_solutions_hybrid(num_weights_cpu, num_weights_gpu, n_sol, speedup_factor);

    float* weights_cpu = weights_ptr + num_weights_gpu * num_features;

    float time_gpu, time_cpu;

    // --- Secciones paralelas: GPU y CPU ---
    #pragma omp parallel sections num_threads(2) shared(weights_cpu, X_train_ptr, y_train_ptr, fitness_values)
    {
        // --- Sección GPU ---
        #pragma omp section
        {
            auto start = high_resolution_clock::now();

            // --- Reservar y copiar pesos a la GPU ---
            float* weights_gpu_ptr = nullptr;
            cudaMalloc((void**)&weights_gpu_ptr, num_weights_gpu * num_features * sizeof(float));
            cudaMemcpy(weights_gpu_ptr, weights_ptr, num_weights_gpu * num_features * sizeof(float), cudaMemcpyHostToDevice);

            for (int i = 0; i < num_weights_gpu; ++i) {
                float clas = 0.0f;
                float red = 0.0f;

                thrust::device_vector<float> temp_weight_vector(
                    weights_gpu_ptr + i * num_features,
                    weights_gpu_ptr + (i + 1) * num_features
                );

                clas = clas_rate_cuda(temp_weight_vector, d_X_train, d_y_train, n, d);
                red = red_rate_cuda(temp_weight_vector);

                cudaDeviceSynchronize();

                fitness_values[i] = CLASS_WEIGHT * clas + RED_WEIGHT * red;
            }

            // --- Liberar memoria de la GPU ---
            cudaFree(weights_gpu_ptr);

            auto end = high_resolution_clock::now();
            duration<double> elapsed = end - start;
            time_gpu = elapsed.count() / num_weights_gpu;
        }

        // --- Sección CPU ---
        #pragma omp section
        {
            auto start = high_resolution_clock::now();
            
            omp_set_nested(1);
            int max_threads = omp_get_max_threads() / 2;
            omp_set_num_threads(std::max(1, max_threads - 1));

            fitness_omp_cpp(
                weights_cpu,
                X_train_ptr,
                y_train_ptr,
                &fitness_values[num_weights_gpu],
                num_weights_cpu,
                n,
                d,
                w_size
            );

            omp_set_nested(0);
            omp_set_num_threads(max_threads);

            auto end = high_resolution_clock::now();
            duration<double> elapsed = end - start;
            time_cpu = elapsed.count() / num_weights_cpu;
            
        }
    }

    

    float new_speedup_factor = 1;
    if (num_weights_cpu != 0 && num_weights_gpu != 0) {
        new_speedup_factor = time_cpu / time_gpu;
    } else if (num_weights_cpu == 0 || num_weights_gpu == 0) {
        new_speedup_factor = speedup_factor;
    }

    return new_speedup_factor; 
}

PYBIND11_MODULE(utils_gpu, m) {
    m.def("fitness_tsp_cuda", &fitness_tsp_cuda, "Calcular fitness del TSP usando CUDA",
            py::arg("distances"), py::arg("solution"), py::arg("fitness"),
            py::arg("n"), py::arg("num_solutions"));
    m.def("fitness_tsp_hybrid", &fitness_tsp_hybrid, "Calcular fitness híbrido del TSP usando CUDA y CPU",
            py::arg("distances"), py::arg("distances_capsule"),
            py::arg("solution"), py::arg("fitness"),
            py::arg("n"), py::arg("num_solutions"), py::arg("speedup_factor"));
    m.def("fitness_cuda", &fitness_cuda, "Calcular fitness usando CUDA",
          py::arg("weights"), py::arg("X_train"), py::arg("y_train"),
          py::arg("num_samples"), py::arg("num_features"));
    m.def("fitness_hybrid", &fitness_hybrid, "Calcular fitness híbrido usando CUDA y CPU",
            py::arg("weights"), py::arg("X_train"), py::arg("y_train"),
            py::arg("X_train_capsule"), py::arg("y_train_capsule"),
            py::arg("fitness_values"), py::arg("num_samples"),
            py::arg("num_features"), py::arg("speedup_factor"));
    m.def("warmup", &warmup_kernel, "Calentar la GPU");
    m.def("create_capsule", &create_capsule, "Crear cápsula CUDA",
          py::arg("ptr_address"));
}