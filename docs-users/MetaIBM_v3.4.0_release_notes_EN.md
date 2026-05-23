# MetaIBM v3.4.0 Release Notes

This version focuses on the introduction of `metaibm/simulator.py` as a schedule-driven top-level driver and a new `playgrounds/` directory in which general and rookie users can write complete MetaIBM models from two CSV files and a list of schedule items.

## Overview

**MetaIBM v3.4.0** introduces a major change in how end users write a model. Instead of hand-coding the time loop and per-step ecological / evolutionary processes (as in `experiments/`), users describe a model as:

1. two CSV files that define the landscape (a mainland species pool and an islands metacommunity), and
2. an ordered list of schedule items (`{'target', 'method', 'params', 'start', 'end', ...}`) describing the ecological / evolutionary processes that should run at each time step.

A new `simulator` class in `metaibm/simulator.py` resolves that schedule into Python calls on registered metacommunity objects. The user-facing boundary (which inputs are free, which conventions must be obeyed) is now documented executably in `test/test_simulator_user_freedom_and_contracts.py`.

This release is important because it:

- adds a schedule-driven top-level driver (`metaibm/simulator.py`) for MetaIBM models
- adds a CSV-driven landscape API for "islands-mainland" models (`mainland.csv` + `metacommunity.csv`)
- adds a dedicated entry point for general and rookie users (`playgrounds/`)
- documents the simulator's user contract executably (`test/test_simulator_user_freedom_and_contracts.py`)
- reorganizes documentation into `docs-users/` and `docs-developer/`
- fixes a miscalculation of the expected number of sexual offspring

## Main Changes

### 1. New top-level driver: `metaibm/simulator.py`

The new `simulator` class manages one or more registered metacommunity objects (e.g. `mainland`, `islands`) and drives them through a schedule of items, where each item looks like:

```python
{'target': '<meta_name or "simulator">',
 'method': '<method name on the target>',
 'params': {...},
 'start': <int>, 'end': <int>,
 # optional: 'enabled', 'interval', '@name' parameter references
}
```

The simulator's responsibilities include:

- registering metacommunity objects under user-provided names
- holding non-eco-evo `global_params`
- advancing `time_step` and dispatching schedule items that are currently active
- providing two schedule-callable, CSV-driven landscape builders (see below)
- providing logging, timing, and output utilities

This change is important because it:

- moves the time loop and dispatch logic out of user scripts and into the package
- makes models describable as data (a schedule list) rather than as imperative code
- makes "what runs when" explicit and editable

### 2. CSV-driven landscape API

Two new schedule-callable simulator methods build empty metacommunities from CSV files:

- `simulator.build_empty_mainland_from_species_csv(meta_name, mainland_csv, pheno_names_ls, environment_types_name, environment_variation_ls, patch_name, patch_index, patch_location, dormancy_pool_max_size)`
- `simulator.build_empty_metacommunity_from_patch_habitat_csv(meta_name, metacommunity_csv, environment_types_name, environment_variation_ls, dormancy_pool_max_size)`

Two CSV file conventions are now standard:

- `mainland.csv` — one row per species; each row describes one species and its single mainland habitat (`species_id`, the user-chosen phenotype-mean columns, `hab_x_loc`, `hab_y_loc`, `hab_length`, `hab_width`).
- `metacommunity_N=*_is_same_heterogeneity=*.csv` — one row per habitat in the islands; describes the patch / habitat layout and the per-habitat environment values along each user-chosen environment axis.

The builders create only the empty patch / habitat / environment skeleton; individuals appear later, either via `mainland.meta_initialize(...)` or via colonization from the mainland.

### 3. New `playgrounds/` directory for general and rookie users

The new `playgrounds/` directory is the schedule-and-CSV entry point for general and rookie users:

- `playgrounds/model-simulator-GRFE.py` — runnable reference model. Loads the landscape from two CSV files, assembles a schedule (asexual / sexual / mixed variants), hands it to a `simulator()` instance, and runs the time loop through `simulator.run(...)`.
- `playgrounds/mainland.csv` — example mainland species pool.
- `playgrounds/metacommunity_N=*_is_same_heterogeneity=*.csv` — example islands metacommunities at multiple sizes (N = 1, 4, 16, 64, 256) and both `is_same_heterogeneity=True` and `is_same_heterogeneity=False`.
- `playgrounds/bootstrap_metaibm.py` — local copy of the bootstrap module so `python model-simulator-GRFE.py` runs directly from the `playgrounds/` directory.

The existing `experiments/` directory continues to host hand-coded experiment scripts (no simulator DSL) for advanced users.

### 4. Executable user-contract documentation

The new file `test/test_simulator_user_freedom_and_contracts.py` documents — in executable form — the user-facing boundary of `metaibm.simulator.simulator`:

- which inputs the user is free to vary (species count and phenotype values in `mainland.csv`; patch/habitat layout, environment values, and habitat size in `metacommunity.csv`; column names through `pheno_names_ls` and `environment_types_name`; schedule-level construction of mainland and islands)
- which conventions the user must keep stable (consecutive positive integer `species_id`; required CSV columns matching the chosen `pheno_names_ls` / `environment_types_name`; rectangular patch grid in plotting; same habitat grid shape per patch; 2D trait / environment use)

### 5. Documentation reorganization

Documentation has been split into two top-level directories:

- `docs-users/` — user-facing documentation: full user manual (`MetaIBM users manual.md` / `.docx`), `QUICK_START.md` (simulator + CSV walkthrough), per-version release notes.
- `docs-developer/` — per-class API documentation: `metaibm-individual.md`, `metaibm-habitat.md`, `metaibm-patch.md`, `metaibm-metacommunity.md`, `metaibm-simulator.md`, `extension-global-habitat-network.md`.

### 6. Legacy test reorganization

Legacy tests are now stored under version-tagged subdirectories so that the top of `test/` only carries the live contracts and validation scripts for the current release:

- `test/lecacy_v3.3.1/` — legacy v3.3.1 tests (landscape initialization, GRFE SLOSS).
- `test/lecacy_v3.1.0-v3.3.0/` — legacy tests for dispersal kernels, environment offsets, dead selection, global habitat network, and non-square grid regression.

### 7. Bug fix: expected number of sexual offspring

A miscalculation in the expected number of sexual offspring was fixed. Models that used the sexual or mixed reproduction modes are affected; previously-reported absolute counts in those modes are not directly comparable to v3.4.0 outputs.

## Files Included in This Release

### New files

- `metaibm/simulator.py`
- `playgrounds/bootstrap_metaibm.py`
- `playgrounds/model-simulator-GRFE.py`
- `playgrounds/mainland.csv`
- `playgrounds/metacommunity_N=1_is_same_heterogeneity=True.csv`
- `playgrounds/metacommunity_N=1_is_same_heterogeneity=False.csv`
- `playgrounds/metacommunity_N=4_is_same_heterogeneity=True.csv`
- `playgrounds/metacommunity_N=4_is_same_heterogeneity=False.csv`
- `playgrounds/metacommunity_N=16_is_same_heterogeneity=True.csv`
- `playgrounds/metacommunity_N=16_is_same_heterogeneity=False.csv`
- `playgrounds/metacommunity_N=64_is_same_heterogeneity=True.csv`
- `playgrounds/metacommunity_N=64_is_same_heterogeneity=False.csv`
- `playgrounds/metacommunity_N=256_is_same_heterogeneity=True.csv`
- `playgrounds/metacommunity_N=256_is_same_heterogeneity=False.csv`
- `test/test_simulator_user_freedom_and_contracts.py`
- `test/lecacy_v3.3.1/` (moved from the top of `test/`)
- `docs-users/QUICK_START.md`
- `docs-developer/metaibm-individual.md`
- `docs-developer/metaibm-habitat.md`
- `docs-developer/metaibm-patch.md`
- `docs-developer/metaibm-metacommunity.md`
- `docs-developer/metaibm-simulator.md`
- `docs-developer/extension-global-habitat-network.md`

### Modified files

- `metaibm/__init__.py` — re-exports the new `simulator` class.
- `metaibm/habitat.py` — fix for the expected number of sexual offspring.
- `README.md` — describes the simulator + CSV workflow, the new `playgrounds/` entry point, the docs reorganization, and the new version history entry.

## Release Value Summary

Compared with v3.3.1, **v3.4.0** provides value in the following areas:

1. **Functionality**

   - schedule-driven top-level driver (`simulator`) decouples the time loop from user scripts
   - CSV-driven landscape API standardizes islands-mainland model setup

2. **Usability**

   - dedicated `playgrounds/` entry point lowers the barrier for general and rookie users
   - `docs-users/QUICK_START.md` provides a focused walkthrough of the new workflow

3. **Reproducibility**

   - models become describable as data (schedule + CSV), easier to store, diff, and rerun
   - the user contract for CSV inputs is documented executably under `test/`

4. **Maintainability**

   - documentation is cleanly split between users (`docs-users/`) and developers (`docs-developer/`)
   - legacy tests are version-tagged under `test/lecacy_v*` and no longer clutter the top of `test/`

5. **Correctness**

   - the expected number of sexual offspring is now calculated correctly

## Release Tags

```text
MetaIBM v3.4.0
```

## Short Release Summary

```text
MetaIBM v3.4.0 introduces metaibm/simulator.py — a schedule-driven top-level driver — and a new playgrounds/ directory in which general and rookie users describe a complete islands-mainland model with two CSV files (mainland.csv and metacommunity.csv) and a list of schedule items. The user contract is documented executably in test/test_simulator_user_freedom_and_contracts.py, documentation is reorganized into docs-users/ and docs-developer/, legacy v3.3.1 tests are moved under test/lecacy_v3.3.1/, and a miscalculation of the expected number of sexual offspring is fixed.
```

## Notes

This release note focuses on the introduction of the **schedule-driven simulator** and the **CSV-driven landscape API**, together with the new **`playgrounds/` entry point** for general and rookie users. It is intended to document the transition from hand-coded experiment scripts (still available under `experiments/`) toward a schedule-and-CSV-based modelling workflow in MetaIBM.
