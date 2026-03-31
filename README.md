# MetaIBM v3.2.0

**MetaIBM** is a Python-based individual-based / agent-based modelling package for simulating **metacommunity ecological and evolutionary dynamics** across multiple spatial scales. The package organizes the model into four core abstractions:

- `individual` — the basic biological unit
- `habitat` — local microsite environment
- `patch` — a collection of habitats
- `metacommunity` — a collection of patches

MetaIBM adopts a package-oriented structure centered on the `metaibm` package and a lightweight bootstrap module for running experiment scripts from the `experiments/` directory.

---

## Highlights in v3.2.0

- **Package-oriented layout** with core code in `metaibm/`
- **Explicit package exports** through `metaibm/__init__.py`
- **Bootstrap-based path initialization** using `experiments/bootstrap_metaibm.py`
- **Experiment scripts** separated from core library code
- Continued support for landscape construction, selection, reproduction, dispersal, disturbance, and data export workflows

---

## Project Layout

```text
MetaIBM/
├── metaibm/
│   ├── __init__.py
│   ├── individual.py
│   ├── habitat.py
│   ├── patch.py
│   └── metacommunity.py
├── experiments/
│   ├── bootstrap_metaibm.py
│   ├── model.py
│   └── mpi_running.py
├── test/
├── docs/
│   └── MetaIBM users manual.docx
└── README.md
```

### Directory roles

#### `metaibm/`

Core package code.

- `individual.py` — defines the individual-level data structure, including genotype, phenotype, mutation, and individual attributes.
- `habitat.py` — defines habitat-level data structures and processes, including microsites, environment, survival, reproduction, germination, dormancy, and disturbance.
- `patch.py` — organizes one or more habitats into a patch and provides patch-level aggregation and dispersal utilities.
- `metacommunity.py` — manages multiple patches and provides metacommunity-scale initialization, dispersal, colonization, disturbance, visualization, and data export.
- `__init__.py` — re-exports the four core classes for package-style imports.

#### `experiments/`

Runnable experiment scripts.

- `bootstrap_metaibm.py` — ensures the project root is available on `sys.path`, so `metaibm` can be imported reliably when experiment scripts are run directly from the `experiments/` directory.
- `model.py` — single-run simulation script that constructs the metacommunity, initializes mainlands, executes the time loop, and writes outputs.
- `mpi_running.py` — MPI-based batch launcher for sweeping parameter combinations and invoking `model.main(...)` across multiple ranks.

#### `docs/`

Documentation resources.

- `MetaIBM users manual.docx` — detailed user manual covering package concepts, ecological processes, data structures, simulation workflow, output, and HPC usage.

#### `test/`
Runnable scripts for testing the fixed bug acompanying each updated code in the core package - metaibm.

---

## Core Package API

The package exports the four main classes directly from `metaibm`:

```python
from metaibm import individual, habitat, patch, metacommunity
```

Equivalent explicit imports are also supported:

```python
from metaibm.individual import individual
from metaibm.habitat import habitat
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity
```

---

## Installation

### Recommended environment

Use a dedicated Conda environment (or any isolated Python environment) with the scientific Python stack.

Typical dependencies used by the project include:

- `numpy`
- `matplotlib`
- `pandas`
- `seaborn`
- `mpi4py` (for MPI execution)

Example with Conda:

```bash
conda create -n metaibm python=3.11 numpy matplotlib pandas seaborn mpi4py
conda activate metaibm
```

> On Windows, MPI execution may additionally depend on an installed MPI runtime (for example MSMPI), and your `mpi4py` installation should match the active MPI implementation.

---

## How imports work in v3.2.0

When running scripts inside `experiments/`, the package import path is initialized by:

```python
import bootstrap_metaibm as _bootstrap
```

This bootstrap module computes the project root and inserts it into `sys.path`, allowing the script to import:

```python
import metaibm
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity
```

This design avoids repeating path-setup code in multiple experiment scripts and keeps the package entry path explicit and maintainable.

---

## Running a single simulation

From the project root:

```bash
python experiments/model.py
```

Or from the `experiments/` directory:

```bash
cd experiments
python model.py
```

### What `model.py` does

The single-run model script:

1. imports `bootstrap_metaibm.py`
2. imports the `metaibm` package and core classes
3. constructs a metacommunity landscape
4. creates two mainland source communities
5. initializes species pools
6. runs the ecological/evolutionary time loop
7. writes logs, compressed CSV output, and figures

---

## Running MPI batch experiments

From the `experiments/` directory:

```bash
mpiexec -np 16 python mpi_running.py
```

### What `mpi_running.py` does

The MPI launcher:

1. builds a parameter grid over replicate index, reproduction mode, mutation rate, patch disturbance rate, and environment value
2. allocates jobs across MPI ranks
3. imports `model`
4. executes `model.main(...)` for each assigned parameter combination
5. writes outputs into parameter-specific directories

This makes the project suitable for large parameter sweeps and HPC workflows.

---

## Minimal package usage example

Below is a minimal example showing how to import the package classes in user code:

```python
from metaibm import patch, metacommunity

meta = metacommunity(metacommunity_name='demo_meta')
p = patch(patch_name='patch1', patch_index=0, location=(0, 0))
meta.add_patch(patch_name='patch1', patch_object=p)
print(meta.metacommunity_name)
```

---

## Ecological processes represented in MetaIBM

MetaIBM supports model construction and simulation across multiple ecological and evolutionary processes, including:

- hierarchical spatial structure (`individual → habitat → patch → metacommunity`)
- environmental gradients
- individual genotype / phenotype representation
- natural selection (environmental filtering)
- asexual and sexual reproduction
- mutation
- colonization from mainland sources
- dispersal within and among patches
- dormancy processes
- disturbance processes
- visualization and compressed tabular output

---

## Output generated by the default experiment workflow

The default single-run workflow in `model.py` writes:

- log files (`*.log`)
- compressed CSV files (`*.csv.gz`) for species distribution and phenotype values through time
- final species distribution figures
- final phenotype distribution figures

---

## Recommended import style for future development

For all new code in v3.2.0 and later, prefer direct package imports:

```python
import bootstrap_metaibm as _bootstrap
import metaibm as metaIBM

from metaibm.patch import patch
from metaibm.metacommunity import metacommunity
```

This keeps experiment scripts aligned with the package layout and avoids dependence on legacy module facades.

---

## Documentation

For a more complete conceptual and technical description, see:

- `docs/MetaIBM users manual.docx`

The user manual documents:

- package concepts and spatial scales
- individual, habitat, patch, and metacommunity data structures
- ecological and evolutionary process modules
- simulation examples
- data output
- HPC / MPI usage

---

## List of Versions History

**MetaIBM v3.2.0**

MetaIBM **v3.2.0** introduces dispersal-kernel, including uniform distribution (by default), gaussian distribution (sigma), exponential distribution (rho), cauchy distribution, power_law distribution, updates metacommunity-level logic in dispersal among patches (the old code still works) and adds dedicated experiment and test scripts for improved validation and future development.



**MetaIBM v3.1.0**

MetaIBM **v3.1.0** adopts a package-oriented structure centered on the `metaibm` package and a lightweight bootstrap module for running experiment scripts from the `experiments/` directory. This README describes the package-oriented layout using `metaibm/` as the core library and `bootstrap_metaibm.py` as the preferred path initialization helper for experiment scripts. 



---

## License

MetaIBM v3.2.0 is distributed under a **source-available academic and non-commercial research license**.

- **Free** for academic, educational, and non-commercial research use
- **Paid commercial license required** for any commercial or for-profit use

For commercial licensing inquiries, please contact the author.

## Citation

If you use MetaIBM in academic work, please cite:

Jian-Hao Lin, Yu-Juan Quan, Bo-Ping Han,
MetaIBM: A Python-based library for individual-based modelling of eco-evolutionary dynamics in spatial-explicit metacommunities,
Ecological Modelling,
Volume 492,
2024,
110730,
ISSN 0304-3800,
https://doi.org/10.1016/j.ecolmodel.2024.110730.
(https://www.sciencedirect.com/science/article/pii/S0304380024001182)
