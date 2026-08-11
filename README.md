# MetaIBM v3.4.3

----------------------------------------------------------------------------------------------------------------
Online tutorial notebooks are now available! Just run them in your browser — no installation required: 

- Example 1 — islands / mainland eco-evolutionary tutorial: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JianhaoLin1228/MetaIBM/v3.4.3?labpath=examples%2Fexample.ipynb)
- Example 2 — eco-evolutionary dynamics on alternative stable states (ATS): [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JianhaoLin1228/MetaIBM/v3.4.3?labpath=examples%2Fexample2%2Fexample2.ipynb)

----------------------------------------------------------------------------------------------------------------

**MetaIBM** is a Python-based individual-based / agent-based modelling package for simulating **metacommunity ecological and evolutionary dynamics** across multiple spatial scales. The package organizes the model into four core abstractions plus a top-level driver:

- `individual` — the basic biological unit
- `habitat` — local microsite environment
- `patch` — a collection of habitats
- `metacommunity` — a collection of patches
- `simulator` — drives a CSV-described landscape through a user-defined schedule of ecological / evolutionary processes

MetaIBM adopts a package-oriented structure centered on the `metaibm` package and a lightweight bootstrap module for running experiment scripts from the `experiments/` directory (advanced users) or model scripts from the `playgrounds/` directory (general / rookie users).

---

## Highlights in v3.4.3

- **New tutorial `examples/example2/` — eco-evolutionary dynamics on alternative stable states (ATS)** under rapid environmental change. Documentation / example release only: the `metaibm` package is unchanged from v3.4.2.
- **The question:** whether two species with different thermal optima produce a **hysteresis loop** — a landscape whose composition depends on the direction the climate came from, not only on the climate itself — and how mutation, reproduction mode, disturbance, propagule supply and environmental heterogeneity move its edges. Framed by Scheffer et al. (2001) (fold bifurcation, hysteresis) and Dakos et al. (2019) (standing variation can flatten the fold; adaptation can delay or advance a shift).
- **The model:** two frozen mainland source pools (sp1 cold-adapted at 0.2, sp2 warm-adapted at 0.8, each burned in for 100 steps) rain propagules into 100 patches on a 10 × 10 grid, coupled by weak dispersal and thinned by patch disturbance. The environment steps by ±0.1 every 100 steps between step 99 and step 700 of 800, so the same climate can be walked up (warming) and down (cooling).
- **The design:** 3 × 3 × 2 = 18 runs — `(reproduce_mode, mutation_rate)` pairs × `patch_dist_rate` × climate direction.
- **The results:** trait lag behind the moving optimum (§3.1), the hysteresis loop and its tipping points from warming vs cooling runs at the same environment (§3.2), and two ways the loop disappears — heavy propagule rain and a patchy environment (§3.3).
- **Files:** `example2.ipynb` (annotated end-to-end notebook, runs the grid with `multiprocessing`), `ats.py` (the same model without any plotting; `main()` is one parameter combination, same shape as `experiments/model.py`), `mpi_running.py` (MPI launcher for the 18-run grid), `tmp_nb_code2.py` (notebook helpers for loading recorded tables and drawing the figures), `bootstrap_metaibm.py`.
- **Batch run:** `cd examples/example2 && mpiexec -np 18 python mpi_running.py` — one run per rank, jobs allocated longest-processing-time-first; each run writes `species_distribution_over_time.csv.gz`, `phenotype_distribution_over_time.csv.gz` and `logger.log` into a folder that encodes its own parameters.

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
├── examples/
│   ├── bootstrap_metaibm.py
│   ├── example.ipynb
│   ├── tmp_nb_code.py
│   └── example2/
│       ├── bootstrap_metaibm.py
│       ├── example2.ipynb
│       ├── tmp_nb_code2.py
│       ├── ats.py
│       └── mpi_running.py
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

#### `examples/`

Tutorial material, runnable in the browser through the Binder badges at the top of this README.

- `example.ipynb` — annotated end-to-end notebook: builds the mainland and the islands, runs a 100 time-step mainland burn-in to generate standing genetic variation, then runs the metacommunity loop (colonization → selection → reproduction → dispersal → germination → clear-up) and plots the results.
- `example2/example2.ipynb` — second tutorial: eco-evolutionary dynamics on **alternative stable states** under rapid environmental change (two thermal specialists, 100 patches, the same climate walked up and down, a 3 × 3 × 2 design).
- `example2/ats.py` — the model of example2 with no plotting; `main()` runs one parameter combination. `example2/mpi_running.py` — MPI launcher for the whole grid.
- `tmp_nb_code.py`, `example2/tmp_nb_code2.py` — notebook helper modules (tables, figures, loading recorded output).
- `bootstrap_metaibm.py` (one copy per notebook directory) — ensures the project root is on `sys.path`.

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

When running scripts inside `experiments/`, `playgrounds/`, `examples/`, or `test/`, the package import path is initialized by:

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

Each of `experiments/`, `playgrounds/`, `examples/`, `examples/example2/`, and `test/` has its own copy of `bootstrap_metaibm.py`, so any script in those directories can be run from there directly.

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

The example2 (ATS) grid has its own launcher with the same structure:

```bash
cd examples/example2
mpiexec -np 18 python mpi_running.py
```

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

For all new code in v3.4.1 and later, prefer direct package imports:

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
- `examples/example.ipynb`, `examples/example2/example2.ipynb` — annotated tutorial notebooks (also runnable in the browser via the Binder badges at the top).
- `docs-users/MetaIBM_v3.4.3_release_notes_EN.md` — most recent release notes file (earlier per-version files also available); the v3.4.1 and v3.4.2 changes are summarized in this README.
- `docs-developer/metaibm-individual.md`, `metaibm-habitat.md`, `metaibm-patch.md`, `metaibm-metacommunity.md`, `metaibm-simulator.md`, `extension-global-habitat-network.md` — per-class API documentation.

---

## List of Versions History

**MetaIBM v3.4.3**
MetaIBM **v3.4.3** adds the second tutorial `examples/example2/` — eco-evolutionary dynamics on **alternative stable states** under rapid environmental change: two thermal specialists rain propagules into 100 patches whose climate is walked up (warming) and down (cooling), and the two directions are compared at the same environment to measure the hysteresis loop, its tipping points, and how mutation rate, reproduction mode, disturbance, propagule supply and environmental heterogeneity move them. The tutorial ships as the notebook `example2.ipynb`, a plotting-free model script `ats.py`, an MPI launcher `mpi_running.py` for the 18-run grid, and the notebook helper module `tmp_nb_code2.py`. The `metaibm` package itself is unchanged from v3.4.2.

**MetaIBM v3.4.2**
MetaIBM **v3.4.2** fixes an object-reference (aliasing) problem in dispersal, colonization from the mainland, and local germination: because these processes sampled from their source pools without removing what they took, one and the same `individual` object (or offspring marker) could end up referenced from several microsites, or from a pool and a microsite simultaneously. Every affected process now takes an `is_remove` argument (default `False`, so v3.4.1 behaviour is preserved); with `is_remove=True` the process samples without replacement and deletes each pick from its source pool or mainland microsite. Two new helpers, `habitat.sample_offspring_without_replacement()` and `patch.sample_offspring_without_replacement()`, implement the multi-pool sampling-and-removal step. The `individual` class deliberately keeps Python's default identity-based equality (no `__eq__`), which the removal step depends on.

**MetaIBM v3.4.1**
MetaIBM **v3.4.1** introduces `metaibm/simulator.py` as a schedule-driven top-level driver and a new `playgrounds/` directory for general and rookie users. Landscapes are now described by two CSV files (`mainland.csv` for the mainland species pool and `metacommunity.csv` for the islands' patch / habitat layout and environment gradients) and built by schedule-callable simulator methods. The user-facing boundary is documented executably in `test/test_simulator_user_freedom_and_contracts.py`. Legacy v3.3.1 tests are kept under `test/lecacy_v3.3.1/`. This version also ships the online tutorial notebook `examples/example.ipynb`, which now begins with a 100 time-step burn-in simulation in the mainland so the species pool carries standing genetic variation before colonization of the islands starts. Also fixes a miscalculation of the expected number of sexual offspring.

**MetaIBM v3.3.1**
MetaIBM **v3.3.1** updates `experiments/model-SLOSS-GREF.py` to read landscape layouts of patch and habitat in the simulated landscape. `patch_habitat_layouts.csv` is the values of patch and habitat X-Y location; `32x32_habitats_env1.csv` is the environmental gradients of env. axis 1; `32x32_habitats_env2.csv` is the environmental gradients of env. axis 2.

**MetaIBM v3.3.0**
MetaIBM **v3.3.0** introduces the **global-habitat-network** extension for habitat-level dispersal across the whole landscape, adds the dedicated extension module `extension/global_habitat_network.py`, and supports extension installation into `metaibm/metacommunity.py` through `install_global_habitat_network_methods(metacommunity)`. This version continues the extension-oriented and package-based development direction of MetaIBM.

**MetaIBM v3.2.0**

MetaIBM **v3.2.0** introduces **dispersal-kernel**, including uniform distribution (by default), gaussian distribution (sigma), exponential distribution (rho), cauchy distribution, power_law distribution, updates metacommunity-level logic in dispersal among patches (the old code still works) and adds dedicated experiment and test scripts for improved validation and future development.

**MetaIBM v3.1.0**

MetaIBM **v3.1.0** adopts a **package-oriented structure** centered on the `metaibm` package and a lightweight bootstrap module for running experiment scripts from the `experiments/` directory. This README describes the package-oriented layout using `metaibm/` as the core library and `bootstrap_metaibm.py` as the preferred path initialization helper for experiment scripts.



## List of Highlights in History

## Highlights in v3.4.2

- **Fixes an object-reference (aliasing) problem** in dispersal, colonization from the mainland, and local germination. Before v3.4.2 these processes only *read* from their source pools without ever removing what they took, so the very same `individual` object (or the same offspring marker) could be taken by several processes within one time-step and end up **referenced from more than one microsite at once**, or exist in a source pool and in an occupied microsite at the same time. Because a microsite stores a *reference* rather than a copy, every such duplicate aged, mutated and died as one and the same organism.
- **New `is_remove` switch (default `False`)** on every affected metacommunity-level process. With `is_remove=True` the process samples **without replacement**: whatever is dispersed, germinated, or shipped from the mainland is deleted from its source pool (or from its mainland microsite), so each `individual` object / offspring marker is consumed exactly once per time-step.
- `is_remove` is available on colonization from the mainland (`meta_colonize_from_propagules_rains`, `pairwise_sexual_colonization_from_prpagules_rains`), on all four dispersal-among-patches methods, on all four dispersal-within-patch methods, and on all four local-germination methods (three object-pipeline entries + the marker-pipeline entry).
- **New sampling helper `sample_offspring_without_replacement(num, pool_name)`** on both `habitat` and `patch`. It samples across one or several pools at once (`('offspring_pool',)`, `('offspring_pool', 'dormancy_pool')`, or `('offspring_marker_pool',)`), removes each pick from the pool it actually came from, and returns the picked objects / markers. It caps the sample size at the pool size, so on the `is_remove=True` path an over-large dispersal number returns everything available instead of raising `ValueError`; the legacy `is_remove=False` path still calls `random.sample()` directly and is unchanged in this respect.
- **Backward compatible by default.** `is_remove=False` keeps exactly the v3.4.1 sampling behaviour, so existing model scripts and published results are unaffected unless the flag is switched on explicitly.
- `individual` must keep Python's default identity-based equality: the class deliberately defines **no `__eq__`**, because the removal step relies on `list.remove()` deleting the one object that was actually sampled rather than the first value-equal individual in the pool. This constraint is now recorded at the top of `metaibm/individual.py`.
- **Not yet covered:** the two dispersal methods of the global-habitat-network extension (`extension/global_habitat_network.py`) still sample with `random.sample()` without removal and take no `is_remove` argument.

## Highlights in v3.4.1

- The tutorial notebook `examples/example.ipynb` now runs a **100 time-step burn-in simulation in the mainland** before the metacommunity loop starts, so the mainland species pool accumulates standing genetic variation instead of starting from genetically identical founders.
- The burn-in chains the same mainland-level processes as the main loop (selection → sexual reproduction with mutation → germination → pool clear-up), so propagule rains delivered to the islands already carry within-species phenotypic variance from time-step 0.
- Online tutorial notebook runnable in the browser through Binder — no installation required.
- New `metaibm/simulator.py` resolves a DSL-style schedule into Python calls on `metacommunity` objects, so users can write a model as a list of schedule items instead of hand-coding the time loop.
- New `playgrounds/` directory for general and rookie users with a runnable reference model `playgrounds/model-simulator-GRFE.py` driven entirely by two CSV files.
- New CSV-driven landscape API: `mainland.csv` (one row per species in the mainland) and `metacommunity_N=*_is_same_heterogeneity=*.csv` (one row per habitat in the islands), built into empty metacommunities by two schedule-callable simulator methods.
- New executable user-contract documentation at `test/test_simulator_user_freedom_and_contracts.py`, listing which inputs general users are free to vary and which conventions they must keep stable.
- Legacy v3.1.0–v3.3.0 tests live under `test/lecacy_v3.1.0-v3.3.0/`; legacy v3.3.1 tests live under `test/lecacy_v3.3.1/`.
- Documentation reorganized into `docs-users/` (user manual, quick start, release notes) and `docs-developer/` (per-class API docs).
- Fixes a miscalculation of the expected number of sexual offspring.

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
