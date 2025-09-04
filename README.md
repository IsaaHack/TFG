# ⚡ Metaheuristics Library with Hardware Acceleration and Cluster Support

🌍 Available in: [🇬🇧 English](README.md) | [🇪🇸 Español](README_ES.md)

Framework for running **metaheuristic algorithms** (GA, PSO, ACO) with **GPU acceleration** and experimental support for distributed execution on clusters via **MPI**.

Includes practical examples of:
- 🔹 **Feature selection** in classification problems
- 🔹 **Traveling Salesman Problem (TSP)**

---

## 📦 Requirements (MANDATORY)

⚠️ The library currently **only works with NVIDIA GPU**. Pure CPU mode is not available.

- 🖥️ **NVIDIA GPU** with Compute Capability ≥ 7.0 (recommended 8.0+)
- 🔧 **Updated NVIDIA driver** (`nvidia-smi`)
- 🎯 **CUDA Toolkit 12.x** (`nvcc --version`)
- 🐍 **Python 3.8+**
- ⚙️ **GCC/G++ ≥ 9** (compatible with CUDA version)
- 🛠️ **Make** (to compile modules)
- 📦 Python dependencies (`requirements.txt`)
- 🌐 (Optional, cluster) **OpenMPI**

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Verify environment:
```bash
nvidia-smi
nvcc --version
```

---

## ⚙️ Compilation

Compile extensions (mandatory before use):
```bash
./build.sh
```
Generates shared libraries (`.so`) used internally.  
Recompile if CUDA is updated or GPU changes:
```bash
make clean && ./build.sh
```

---

## 🗂️ Project Structure

```
├── algorithms/    # Algorithms (GA, PSO, ACO)
├── problems/      # Problem definitions (CLAS, TSP)
├── cpp/           # C++/CUDA code
├── results/       # Outputs (CSV, charts, Excel)
├── datasets/      # Datasets
├── docs/          # Sphinx documentation
├── clas.py        # Example: Classification + Feature Selection
├── tsp.py         # Example: TSP with TSPLIB instances
├── cluster.py     # Example: Distributed execution (MPI)
├── cluster_main.py# High-level distributed execution (MPI)
├── build.sh       # Build script
├── Makefile       # GPU build process
└── requirements.txt
```

---

## 🚀 Quick Installation

1. Clone repository  
2. Install Python dependencies  
3. Compile GPU modules  
```bash
./build.sh
```
4. Run examples:

Classification:
```bash
python clas.py datasets/Clas/bank.csv -a ga -e gpu --iterations 100
```

TSP:
```bash
python tsp.py datasets/TSP/berlin52.tsp -a pso -e gpu --iterations 1000
```

---

## 🧩 API Examples

### Classification Example
```python
import pandas as pd
from problems.clas_problem import ClasProblem
from algorithms import GA
from clas import preprocess_bank_marketing_dataset

df = pd.read_csv("datasets/Clas/bank.csv")
df = preprocess_bank_marketing_dataset(df)
X = df.drop(columns=["y"]).values
y = df["y"].values

problem = ClasProblem(X, y, alpha=0.5, threshold=0.1)
ga = GA(problem, population_size=64, executer='gpu')
best_solution = ga.fit(iterations=100)
best_fitness = problem.fitness(best_solution)
print("Best fitness:", best_fitness)
```

### TSP Example
```python
from problems.tsp_problem import TSPProblem
from algorithms.aco import ACO
from tsp import read_tsp_file, extract_coords, build_distance_matrix
import numpy as np

meta, node_lines, edge_lines = read_tsp_file("datasets/TSP/berlin52.tsp")
coords_arr = extract_coords(node_lines) if node_lines else None
dist_matrix = build_distance_matrix(meta, coords_arr, edge_lines)

dist_matrix_np = np.array(dist_matrix, dtype=np.float32)

problem = TSPProblem(dist_matrix_np)
aco = ACO(problem, colony_size=100, executer='gpu')
best_path = aco.fit(iterations=1000)
best_distance = problem.fitness(best_path)
print("Best distance:", best_distance)
```

---

## 🖥️ Executors Available

- ⚡ **gpu** → GPU (recommended)  
- 🧩 **cluster** → Distributed with MPI (experimental)  
- 🐌 **single** → Sequential CPU (functional but slow)  
- 🔀 **multi** → Multithreaded CPU (functional but slow)  
- ⚠️ **hybrid** → CPU+GPU (not recommended)

---

## 🛠️ Customization

### ➕ New Algorithm
1. Inherit from `Algorithm`  
2. Implement:
   - `__init__` (pass `problem`, `executer`)  
   - `fit` (main logic)  
   - `fit_mpi` (optional, MPI support)  
3. (Optional) Integrate with existing problems via `patch_problem` in `problems/problem.py`

### ➕ New Problem
1. Create class in `problems/`  
2. Implement evaluation methods (`fitness`, solution generation, etc.)  
3. (Optional) Add support in executors (see `ClasProblem`, `TSPProblem`)  

---

## 🐞 Troubleshooting

- ❌ `CUDA driver not found` → Update driver / restart  
- ❌ `.so import error` → `make clean && ./build.sh`  
- ❌ `mpirun: command not found` → Install OpenMPI (`sudo apt install openmpi-bin`)  

---

## 📖 Documentation

Generate with Sphinx (if configured):
```bash
cd docs
make html
```
Open in browser: `docs/_build/html/index.html`

---

## 📊 Status

- ✅ **GPU**: stable and recommended  
- ⚠️ **Cluster (MPI)**: functional but experimental  
- ⚠️ **CPU single/multi**: functional but slow  
- ❌ **Hybrid**: not recommended  

---

## 📜 License

MIT (see `LICENSE`)

---

## 👤 Author

**Isaac Brao Aissaoni**, 2025