# ⚡ Librería de Metaheurísticas con Aceleración por Hardware y Soporte para Clúster

🌍 Disponible en: [🇬🇧 English](README.md) | [🇪🇸 Español](README_ES.md)

Framework para ejecutar **algoritmos metaheurísticos** (GA, PSO, ACO) con **aceleración GPU** y soporte experimental para ejecución distribuida en clústeres mediante **MPI**.

Incluye ejemplos prácticos de:
- 🔹 **Selección de características** en problemas de clasificación
- 🔹 **Traveling Salesman Problem (TSP)**

---

## 📦 Requisitos (OBLIGATORIOS)

⚠️ Actualmente la librería **solo funciona con GPU NVIDIA**. No hay modo CPU puro.

- 🖥️ **GPU NVIDIA** con Compute Capability ≥ 7.0 (recomendado 8.0+)  
- 🔧 **Driver NVIDIA** actualizado (`nvidia-smi`)  
- 🎯 **CUDA Toolkit 12.x** (`nvcc --version`)  
- 🐍 **Python 3.8+**  
- ⚙️ **GCC/G++ ≥ 9** (compatible con CUDA)  
- 🛠️ **Make** (para compilar módulos)  
- 📦 Dependencias Python (`requirements.txt`)  
- 🌐 (Opcional, solo cluster) **OpenMPI**  

Instalación dependencias Python:
```bash
pip install -r requirements.txt
```

Verificar entorno:
```bash
nvidia-smi
nvcc --version
```

---

## ⚙️ Compilación

Compilar extensiones (obligatorio antes de usar):
```bash
./build.sh
```
Generará librerías compartidas (`.so`) usadas internamente.  
Reconstruir si se actualiza CUDA o cambia GPU:
```bash
make clean && ./build.sh
```

---

## 🗂️ Estructura del proyecto

```
├── algorithms/    # Algoritmos (GA, PSO, ACO)
├── problems/      # Definiciones de problemas (CLAS, TSP)
├── cpp/           # Código C++/CUDA
├── results/       # Salidas (CSV, gráficos, Excel)
├── datasets/      # Conjuntos de datos
├── docs/          # Documentación Sphinx
├── clas.py        # Ejemplo: Clasificación + selección de características
├── tsp.py         # Ejemplo: TSP con instancias TSPLIB
├── cluster.py     # Ejemplo: Ejecución distribuida (MPI)
├── cluster_main.py# Ejecución distribuida de alto nivel (MPI)
├── build.sh       # Script de compilación
├── Makefile       # Proceso de build GPU
└── requirements.txt
```

---

## 🚀 Instalación rápida

1. Clonar repositorio  
2. Instalar dependencias Python  
3. Compilar módulos GPU  
```bash
./build.sh
```

4. Ejecutar ejemplos:

Clasificación:
```bash
python clas.py datasets/Clas/bank.csv -a ga -e gpu --iterations 100
```

TSP:
```bash
python tsp.py datasets/TSP/berlin52.tsp -a pso -e gpu --iterations 1000
```

---

## 🧩 Ejemplos de API

### Clasificación
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
print("Mejor fitness:", best_fitness)
```

### TSP
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
print("Mejor distancia:", best_distance)
```

---

## 🖥️ Ejecutores disponibles

- ⚡ **gpu** → GPU (recomendado)  
- 🧩 **cluster** → Distribuido con MPI (experimental)  
- 🐌 **single** → CPU secuencial (funcional pero lento)  
- 🔀 **multi** → CPU multihilo (funcional pero lento)  
- ⚠️ **hybrid** → CPU+GPU (no recomendado)

---

## 🛠️ Personalización

### ➕ Nuevo algoritmo
1. Heredar de `Algorithm`  
2. Implementar:
   - `__init__` (pasar `problem`, `executer`)  
   - `fit` (lógica principal)  
   - `fit_mpi` (opcional)  
3. (Opcional) Integrar con problemas existentes con `patch_problem`

### ➕ Nuevo problema
1. Crear clase en `problems/`  
2. Implementar métodos (`fitness`, generación de soluciones, etc.)  
3. (Opcional) Añadir soporte en ejecutores existentes

---

## 🐞 Resolución de problemas

- ❌ `CUDA driver not found` → Actualizar driver / reiniciar  
- ❌ Error al importar `.so` → `make clean && ./build.sh`  
- ❌ `mpirun: command not found` → Instalar OpenMPI (`sudo apt install openmpi-bin`)  

---

## 📖 Documentación

```bash
cd docs
make html
```
Abrir: `docs/_build/html/index.html`

---

## 📊 Estado

- ✅ **GPU**: estable y recomendado  
- ⚠️ **Cluster (MPI)**: funcional pero experimental  
- ⚠️ **CPU single/multi**: funciona pero lento  
- ❌ **Hybrid**: no recomendado

---

## 📜 Licencia

MIT (ver `LICENSE`)

---

## 👤 Autor

**Isaac Brao Aissaoni**, 2025