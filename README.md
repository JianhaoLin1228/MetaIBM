# MetaIBM v3.4.0

**MetaIBM** is a Python-based individual-based / agent-based modelling package for simulating **metacommunity ecological and evolutionary dynamics** across multiple spatial scales. The package organizes the model into four core abstractions plus a top-level driver:

- `individual` — the basic biological unit
- `habitat` — local microsite environment
- `patch` — a collection of habitats
- `metacommunity` — a collection of patches
- `simulator` — drives a CSV-described landscape through a user-defined schedule of ecological / evolutionary processes

MetaIBM adopts a package-oriented structure centered on the `metaibm` package and a lightweight bootstrap module for running experiment scripts from the `experiments/` directory (advanced users) or model scripts from the `playgrounds/` directory (general / rookie users).

---

## Highlights in v3.4.0

- New `metaibm/simulator.py` resolves a DSL-style schedule into Python calls on `metacommunity` objects, so users can write a model as a list of schedule items instead of hand-coding the time loop.
- New `playgrounds/` directory for general and rookie users with a runnable reference model `playgrounds/model-simulator-GRFE.py` driven entirely by two CSV files.
- New CSV-driven landscape API: `mainland.csv` (one row per species in the mainland) and `metacommunity_N=*_is_same_heterogeneity=*.csv` (one row per habitat in the islands), built into empty metacommunities by two schedule-callable simulator methods.
- New executable user-contract documentation at `test/test_simulator_user_freedom_and_contracts.py`, listing which inputs general users are free to vary and which conventions they must keep stable.
- Legacy v3.1.0–v3.3.0 tests live under `test/lecacy_v3.1.0-v3.3.0/`; legacy v3.3.1 tests live under `test/lecacy_v3.3.1/`.
- Documentation reorganized into `docs-users/` (user manual, quick start, release notes) and `docs-developer/` (per-class API docs).
- Fixes a miscalculation of the expected number of sexual offspring.

---

## Installation

MetaIBM is pure Python and has no `setup.py` / `pyproject.toml`; install the dependencies into a Python environment and run the scripts directly.

### Dependencies

- `numpy` (>= 1.24)
- `matplotlib` (>= 3.7)
- `pandas` (>= 2.0)
- `seaborn` (>= 0.12)
- `mpi4py` (>= 3.1) — only required for MPI-based batch experiments (`experiments/mpi_running.py`)

Exact pins are listed in `requirements.txt`.

### Recommended: Conda environment

```bash
conda create -n metaibm python=3.11 numpy matplotlib pandas seaborn mpi4py
conda activate metaibm
```

or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Platform notes

- **Windows** — install Anaconda, then `conda install numpy matplotlib pandas seaborn`. For MPI runs, install Microsoft C++ Build Tools and Microsoft MPI (MSMPI) first, then `conda install -c conda-forge mpi4py` (or `pip install mpi4py`).
- **Linux** — install Anaconda and add it to `PATH` in `~/.bashrc`, then `conda install numpy matplotlib pandas seaborn`. For MPI runs, build / install Open MPI and then `conda install -c conda-forge mpi4py`.
- **macOS (Apple Silicon)** — install Anaconda and the standard scientific stack as above. For MPI runs, `brew install open-mpi` and `pip3 install mpi4py`.

### Getting the code

MetaIBM is not on PyPI; clone or download the repository and run scripts from inside it:

```bash
git clone <repo-url> MetaIBM
cd MetaIBM
python playgrounds/model-simulator-GRFE.py
```

The bootstrap module in each script directory (`experiments/`, `playgrounds/`, `test/`) puts the project root on `sys.path`, so no extra install step is needed for the `metaibm` package itself.

See `docs-users/MetaIBM users manual.md` (Section 2: Installation; Section 7.1: MPI installation) for the fully detailed, per-platform walkthrough.

---

## Project Layout

```text
MetaIBM/
├── metaibm/
│   ├── __init__.py
│   ├── individual.py
│   ├── habitat.py
│   ├── patch.py
│   ├── metacommunity.py
│   └── simulator.py
├── experiments/
│   ├── bootstrap_metaibm.py
│   ├── model.py
│   ├── model-sloss.py
│   ├── model-sloss-GRFE.py
│   ├── model-sloss-global-habitat-network.py
│   ├── mpi_running.py
│   ├── patch_habitat_layouts.csv
│   ├── 32x32_habitats_env1.csv
│   └── 32x32_habitats_env2.csv
├── playgrounds/
│   ├── bootstrap_metaibm.py
│   ├── model-simulator-GRFE.py
│   ├── mainland.csv
│   └── metacommunity_N=*_is_same_heterogeneity=*.csv
├── test/
│   ├── bootstrap_metaibm.py
│   ├── test_simulator_user_freedom_and_contracts.py
│   ├── lecacy_v3.3.1/
│   └── lecacy_v3.1.0-v3.3.0/
├── extension/
│   ├── __init__.py
│   └── global_habitat_network.py
├── docs-users/
│   ├── MetaIBM users manual.md
│   ├── QUICK_START.md
│   └── MetaIBM_v*.*.*_release_notes_EN.md
├── docs-developer/
│   ├── metaibm-individual.md
│   ├── metaibm-habitat.md
│   ├── metaibm-patch.md
│   ├── metaibm-metacommunity.md
│   ├── metaibm-simulator.md
│   └── extension-global-habitat-network.md
└── README.md
```

### Directory roles

#### `metaibm/`

Core package code.

- `individual.py` — defines the individual-level data structure, including genotype, phenotype, mutation, and individual attributes.
- `habitat.py` — defines habitat-level data structures and processes, including microsites, environment, survival, reproduction, germination, dormancy, and disturbance.
- `patch.py` — organizes one or more habitats into a patch and provides patch-level aggregation and dispersal utilities.
- `metacommunity.py` — manages multiple patches and provides metacommunity-scale initialization, dispersal, colonization, disturbance, visualization, and data export.
- `simulator.py` — resolves a schedule of `{'target': ..., 'method': ..., 'params': ..., 'start': ..., 'end': ...}` items into method calls on registered metacommunity objects; provides CSV-driven `build_empty_mainland_from_species_csv` and `build_empty_metacommunity_from_patch_habitat_csv` builders.
- `__init__.py` — re-exports the five core classes for package-style imports.

#### `experiments/`

Hand-coded experiment scripts for advanced users (no simulator DSL).

- `bootstrap_metaibm.py` — ensures the project root is on `sys.path`.
- `model.py`, `model-sloss.py`, `model-sloss-GRFE.py`, `model-sloss-global-habitat-network.py` — single-run simulation scripts that construct the metacommunity, initialize mainlands, run the time loop by hand, and write outputs.
- `mpi_running.py` — MPI-based batch launcher for sweeping parameter combinations.
- `patch_habitat_layouts.csv`, `32x32_habitats_env1.csv`, `32x32_habitats_env2.csv` — landscape configuration consumed by `model-sloss-GRFE.py`.

#### `playgrounds/`

Entry point for general and rookie users: schedule-and-CSV driven models that go through `metaibm.simulator`.

- `model-simulator-GRFE.py` — reference single-run script. Loads landscape from CSV, builds a schedule of ecological / evolutionary process items, hands them to a `simulator()` instance.
- `mainland.csv` — one row per species; defines the mainland's species pool and per-species mainland habitat.
- `metacommunity_N=*_is_same_heterogeneity=*.csv` — one row per habitat; defines the islands' patch / habitat layout and environment gradients (with `N` patches and same / different heterogeneity).

#### `test/`

Standalone validation scripts (no pytest framework — run each directly).

- `test_simulator_user_freedom_and_contracts.py` — executable documentation of the user-facing boundary of `metaibm.simulator.simulator`: which CSV columns and `pheno_names_ls` choices the user is free to vary, and which conventions (consecutive integer `species_id`, required columns, rectangular grid, etc.) the user must keep stable.
- `lecacy_v3.3.1/` — legacy v3.3.1 tests (landscape initialization, GRFE SLOSS).
- `lecacy_v3.1.0-v3.3.0/` — legacy tests for dispersal kernels, environment offsets, dead selection, global habitat network, and non-square grid regression.

#### `extension/`

Modular add-on features that can be mounted onto the core package when needed. The global-habitat-network extension is auto-installed at import time and adds habitat-level dispersal across the whole landscape.

#### `docs-users/`

User-facing documentation.

- `MetaIBM users manual.md` / `.docx` — detailed user manual.
- `QUICK_START.md` — minimal walkthrough of the simulator + CSV workflow.
- `MetaIBM_v*.*.*_release_notes_EN.md` — per-version release notes.

#### `docs-developer/`

Per-class API documentation (attributes + methods) for each core class and the global-habitat-network extension. Consult these before reading the large source files.

---

## Core Package API

The package exports the five main classes directly from `metaibm`:

```python
from metaibm import individual, habitat, patch, metacommunity, simulator
```

Equivalent explicit imports are also supported:

```python
from metaibm.individual import individual
from metaibm.habitat import habitat
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity
from metaibm.simulator import simulator
```

---

## How imports work

When running scripts inside `experiments/`, `playgrounds/`, or `test/`, the package import path is initialized by:

```python
import bootstrap_metaibm as _bootstrap
```

This bootstrap module computes the project root and inserts it into `sys.path`, allowing the script to import:

```python
import metaibm
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity
from metaibm.simulator import simulator
```

Each of `experiments/`, `playgrounds/`, and `test/` has its own copy of `bootstrap_metaibm.py`, so any script in those directories can be run from there directly.

---

## Running a simulation

### For general / rookie users — playgrounds (simulator + CSV)

```bash
cd playgrounds
python model-simulator-GRFE.py
```

`model-simulator-GRFE.py`:

1. imports `bootstrap_metaibm.py` and the `metaibm` package (including `simulator`)
2. reads landscape configuration from `mainland.csv` and a `metacommunity_N=*_is_same_heterogeneity=*.csv` file
3. assembles a schedule of ecological / evolutionary process items (`{'target', 'method', 'params', 'start', 'end'}`)
4. registers the schedule and `global_params` on a `simulator()` instance
5. runs the time loop through `simulator.run(...)`
6. writes logs, compressed CSV output, and figures

See `docs-users/QUICK_START.md` for the full walkthrough and `test/test_simulator_user_freedom_and_contracts.py` for the user contract on CSV columns and naming.

### For advanced users — experiments (hand-coded loop)

```bash
python experiments/model.py
cd experiments && python model.py
```

### MPI batch experiments

From the `experiments/` directory:

```bash
mpiexec -np 16 python mpi_running.py
```

The MPI launcher builds a parameter grid (replicate, reproduction mode, mutation rate, disturbance rate, environment value), allocates jobs across ranks, and calls `model.main(...)` for each parameter combination. Suitable for large parameter sweeps and HPC workflows.

---

## Minimal package usage example

```python
from metaibm import patch, metacommunity

meta = metacommunity(metacommunity_name='demo_meta')
p = patch(patch_name='patch1', patch_index=0, location=(0, 0))
meta.add_patch(patch_name='patch1', patch_object=p)
print(meta.metacommunity_name)
```

For the simulator-driven workflow, see `playgrounds/model-simulator-GRFE.py` and `docs-users/QUICK_START.md`.

---

## Ecological processes represented in MetaIBM

- hierarchical spatial structure (`individual → habitat → patch → metacommunity`)
- environmental gradients (including CSV-defined gradients and Gaussian Random Field, Exponential)
- individual genotype / phenotype representation
- natural selection (environmental filtering)
- asexual and sexual reproduction
- mutation
- colonization from mainland sources
- dispersal within and among patches (uniform, gaussian, exponential, cauchy, power-law kernels)
- global habitat-network dispersal (via the `extension/global_habitat_network.py` extension)
- dormancy processes
- disturbance processes
- visualization and compressed tabular output

---

## Output generated by the default workflow

- log files (`*.log`)
- compressed CSV files (`*.csv.gz`) for species distribution and phenotype values through time
- final species distribution figures
- final phenotype distribution figures

---

## Recommended import style

For all new code in v3.4.0 and later, prefer direct package imports:

```python
import bootstrap_metaibm as _bootstrap
import metaibm as metaIBM

from metaibm.patch import patch
from metaibm.metacommunity import metacommunity
from metaibm.simulator import simulator
```

This keeps experiment and playground scripts aligned with the package layout and avoids dependence on legacy module facades.

---

## Documentation

- `docs-users/QUICK_START.md` — minimal walkthrough of the simulator + CSV workflow.
- `docs-users/MetaIBM users manual.md` — full user manual.
- `docs-users/MetaIBM_v3.4.0_release_notes_EN.md` — release notes for the current version (earlier versions also available).
- `docs-developer/metaibm-individual.md`, `metaibm-habitat.md`, `metaibm-patch.md`, `metaibm-metacommunity.md`, `metaibm-simulator.md`, `extension-global-habitat-network.md` — per-class API documentation.

---

## List of Versions History

**MetaIBM v3.4.0**
MetaIBM **v3.4.0** introduces `metaibm/simulator.py` as a schedule-driven top-level driver and a new `playgrounds/` directory for general and rookie users. Landscapes are now described by two CSV files (`mainland.csv` for the mainland species pool and `metacommunity.csv` for the islands' patch / habitat layout and environment gradients) and built by schedule-callable simulator methods. The user-facing boundary is documented executably in `test/test_simulator_user_freedom_and_contracts.py`. Legacy v3.3.1 tests are kept under `test/lecacy_v3.3.1/`. Also fixes a miscalculation of the expected number of sexual offspring.

**MetaIBM v3.3.1**
MetaIBM **v3.3.1** updates `experiments/model-SLOSS-GREF.py` to read landscape layouts of patch and habitat in the simulated landscape. `patch_habitat_layouts.csv` is the values of patch and habitat X-Y location; `32x32_habitats_env1.csv` is the environmental gradients of env. axis 1; `32x32_habitats_env2.csv` is the environmental gradients of env. axis 2.

**MetaIBM v3.3.0**
MetaIBM **v3.3.0** introduces the **global-habitat-network** extension for habitat-level dispersal across the whole landscape, adds the dedicated extension module `extension/global_habitat_network.py`, and supports extension installation into `metaibm/metacommunity.py` through `install_global_habitat_network_methods(metacommunity)`. This version continues the extension-oriented and package-based development direction of MetaIBM.

**MetaIBM v3.2.0**

MetaIBM **v3.2.0** introduces **dispersal-kernel**, including uniform distribution (by default), gaussian distribution (sigma), exponential distribution (rho), cauchy distribution, power_law distribution, updates metacommunity-level logic in dispersal among patches (the old code still works) and adds dedicated experiment and test scripts for improved validation and future development.

**MetaIBM v3.1.0**

MetaIBM **v3.1.0** adopts a **package-oriented structure** centered on the `metaibm` package and a lightweight bootstrap module for running experiment scripts from the `experiments/` directory. This README describes the package-oriented layout using `metaibm/` as the core library and `bootstrap_metaibm.py` as the preferred path initialization helper for experiment scripts.



## List of Highlights in History

## Highlights in v3.3.1

- `model-SLOSS-GREF.py` is designed to be able to read landscape configuration from `xxx.csv` file
- `patch_habitat_layouts.csv` is the values of patch and habitat layouts in the simulated landscape.
- `32x32_habitats_env1.csv` is the values of gradients of environmental axis 1.
- `32x32_habitats_env2.csv` is the values of gradients of environmental axis 2.

## Highlights in v3.3.0

- **global-habitat-network workflow** for habitat-level dispersal across the whole landscape
- **extension-based implementation** through `extension/global_habitat_network.py`
- **metacommunity integration** by installing extension methods into `metaibm/metacommunity.py`
- continued support for kernel-based dispersal methods, with the global habitat network designed to work with `uniform`, `gaussian`, `exponential`, `cauchy`, and `power_law` dispersal kernels

## Highlights in v3.2.0

- **Package-oriented layout** with core code in `metaibm/`
- **Explicit package exports** through `metaibm/__init__.py`
- **Bootstrap-based path initialization** using `experiments/bootstrap_metaibm.py`
- **Experiment scripts** separated from core library code
- Continued support for landscape construction, selection, reproduction, dispersal, disturbance, and data export workflows

---

## License

MetaIBM is distributed under a **source-available academic and non-commercial research license**.

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
