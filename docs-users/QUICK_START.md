# MetaIBM Quick Start

This guide shows how to drive a full MetaIBM run through `metaibm/simulator.py`. After reading it you should be able to:

1. Describe a mainland and an islands metacommunity in two CSV files and let the simulator build both for you.
2. Write the per-time-step "schedule" — the ordered list of ecological / evolutionary processes the simulator executes.
3. Wire it all together with a `simulator()` instance and run it.

A complete, runnable reference implementation lives at `playgrounds/model-simulator-GRFE.py`; the executable user-contract tests live at `test/test_simulator_user_freedom_and_contracts.py`. This document distills both.

---

## Table of contents

- [1. Building mainland and islands from CSV files](#1-building-mainland-and-islands-from-csv-files)
  - [1.1 The two CSV files](#11-the-two-csv-files)
  - [1.2 mainland.csv — required columns and rules](#12-mainlandcsv--required-columns-and-rules)
  - [1.3 metacommunity.csv — required columns and rules](#13-metacommunitycsv--required-columns-and-rules)
  - [1.4 Calling the two builders from Python](#14-calling-the-two-builders-from-python)
  - [1.5 What the simulator lets you change (freedom)](#15-what-the-simulator-lets-you-change-freedom)
  - [1.6 What you must obey (contracts)](#16-what-you-must-obey-contracts)
- [2. Building schedule items](#2-building-schedule-items)
  - [2.1 The shape of one schedule item](#21-the-shape-of-one-schedule-item)
  - [2.2 Gating fields: `enabled`, `start`, `end`, `interval`](#22-gating-fields-enabled-start-end-interval)
  - [2.3 The `'@name'` parameter reference](#23-the-name-parameter-reference)
  - [2.4 A minimal schedule, end to end](#24-a-minimal-schedule-end-to-end)
  - [2.5 Recommended ordering](#25-recommended-ordering)
- [3. Using the simulator](#3-using-the-simulator)
  - [3.1 Required `global_params`](#31-required-global_params)
  - [3.2 The five-line run pattern](#32-the-five-line-run-pattern)
  - [3.3 Outputs](#33-outputs)
  - [3.4 Logging and timing](#34-logging-and-timing)

---

## 1. Building mainland and islands from CSV files

### 1.1 The two CSV files

A standard "islands-mainland" MetaIBM model is described by two CSV files:

| File | One row (information) = | Used by |
|------|-----------|---------|
| `mainland.csv` | one *species* and the single mainland *habitat* that hosts it | `simulator.build_empty_mainland_from_species_csv` |
| `metacommunity.csv` | one *habitat* belonging to one *patch* in the islands metacommunity | `simulator.build_empty_metacommunity_from_patch_habitat_csv` |

Both files are read with `pandas.read_csv`, so any pandas-readable CSV (comma-separated, header row on line 1, UTF-8) works. Both builders create **empty** metacommunities — they set up the patch/habitat skeleton and the environment field, but do not place any individuals. Individuals appear later, either via `mainland.meta_initialize(...)` or via colonization from the mainland into the islands.

### 1.2 mainland.csv — required columns and rules

Required columns (column names are fixed, except the phenotype columns):

| Column | Type | Meaning |
|--------|------|---------|
| `species_id` | int | species identifier (see contract below) |
| `<pheno_name_1>`, `<pheno_name_2>`, ... | float | the mean environment / mean phenotype value of this species along each trait axis. The column names are user-chosen and must match the `pheno_names_ls` argument you pass to the builder. |
| `hab_x_loc` | int | x coordinate of this species' mainland habitat |
| `hab_y_loc` | int | y coordinate of this species' mainland habitat |
| `hab_length` | int | number of microsite rows in this species' mainland habitat |
| `hab_width` | int | number of microsite columns in this species' mainland habitat |

Example (the file shipped under `playgrounds/mainland.csv`):

```csv
species_id,phenotype_axis1,phenotype_axis2,hab_x_loc,hab_y_loc,hab_length,hab_width
1,0.2,0.2,0,0,20,20
2,0.2,0.4,0,1,20,20
```

Rules:

- One row per species. The number of rows = the number of species in the mainland.
- `species_id` **must be consecutive positive integers `1, 2, ..., n`** (no gaps, no zero, no negative values, no strings). The builder sorts rows by `species_id` and names the resulting habitats `'h0', 'h1', ..., 'h{n-1}'` in that sorted order; the rest of the package assumes IDs derived from `'sp' + str(i+1)`.
- Row order in the file does **not** matter — the builder sorts internally.
- The phenotype-column names you write in the CSV must match the `pheno_names_ls` you pass to the builder. Same length, same names.

### 1.3 metacommunity.csv — required columns and rules

Required columns (column names are fixed, except the environment columns):

| Column | Type | Meaning |
|--------|------|---------|
| `patch_id` | str | patch name; rows sharing this value belong to the same patch |
| `patch_index` | int | integer index of the patch within the metacommunity |
| `patch_location_x` | float | x coordinate of the patch in the landscape |
| `patch_location_y` | float | y coordinate of the patch in the landscape |
| `habitat_id` | str | habitat name (unique within a patch) |
| `habitat_index` | int | integer index of the habitat within its patch |
| `habitat_x_location` | int | x coordinate of the habitat inside its patch |
| `habitat_y_location` | int | y coordinate of the habitat inside its patch |
| `<env_name_1>`, `<env_name_2>`, ... | float | mean value of each environment axis at this habitat. The column names are user-chosen and must match `environment_types_name`. |
| `hab_length` | int | number of microsite rows in this habitat |
| `hab_width` | int | number of microsite columns in this habitat |

(Any extra columns the CSV happens to carry — e.g. `patch_num`, `patch_size`, `global_habitat_num` in the shipped files — are ignored by the builder.)

Example (the first two habitats from `playgrounds/metacommunity_N=4_is_same_heterogeneity=True.csv`):

```csv
patch_id,patch_index,patch_location_x,patch_location_y,habitat_id,habitat_index,habitat_x_location,habitat_y_location,env_axis1,env_axis2,hab_length,hab_width
patch1,0,8.0,8.0,h0,0,1,1,0.814,0.235,10,10
patch1,0,8.0,8.0,h1,1,1,2,0.753,0.209,10,10
```

Rules:

- One row per habitat. Patches are derived as the unique `(patch_id, patch_index, patch_location_x, patch_location_y)` rows (`drop_duplicates`).
- Patches are added in order of `(patch_index, patch_id)`; habitats within a patch are added in order of `(habitat_index, habitat_id)`.
- Every habitat inside a single patch **must share the same `hab_length` x `hab_width`** — the patch-level reshape in `patch.get_patch_microsites_environment_values` assumes a uniform habitat shape.
- For the plotting helpers (`plot_species_distribution`, etc.) the patch layout must form a **rectangular grid**, and every patch must use the **same habitat grid shape**. The plotters also infer one representative `hab_length` / `hab_width` from the first row of the CSV.
- Environment-column names in the CSV must match `environment_types_name`. Same length, same names.

### 1.4 Calling the two builders from Python

Both builders are methods on the `simulator` instance. They both register the resulting metacommunity on `sim.meta_objects` under the `meta_name` you pass.

```python
from metaibm.simulator import simulator

sim = simulator()

# 1. Mainland (one patch, many single-species habitats)
sim.build_empty_mainland_from_species_csv(
    meta_name='mainland',
    mainland_csv='mainland.csv',
    pheno_names_ls=('phenotype_axis1', 'phenotype_axis2'),
    environment_types_name=('env_axis1', 'env_axis2'),
    environment_variation_ls=[0.025, 0.025],
    patch_name='patch0', patch_index=0, patch_location=(0, 0),
    dormancy_pool_max_size=0,
)

# 2. Islands metacommunity (many patches, many habitats per patch)
sim.build_empty_metacommunity_from_patch_habitat_csv(
    meta_name='islands',
    metacommunity_csv='metacommunity_N=4_is_same_heterogeneity=True.csv',
    environment_types_name=('env_axis1', 'env_axis2'),
    environment_variation_ls=[0.025, 0.025],
    dormancy_pool_max_size=0,
)

# Both metacommunities are now in:
sim.meta_objects['mainland']
sim.meta_objects['islands']
```

Both methods are *schedule-callable* as well — instead of calling them directly, you can drop them in the schedule as one-shot items at `time_step=0` (this is what `playgrounds/model-simulator-GRFE.py` does). See [§2.4](#24-a-minimal-schedule-end-to-end).

### 1.5 What the simulator lets you change (freedom)

The CSV-driven builders intentionally expose the following knobs to the user:

- **Species count** — row count of `mainland.csv`.
- **Phenotype values per species** — phenotype columns of `mainland.csv`.
- **Mainland habitat size and location** — `hab_length`, `hab_width`, `hab_x_loc`, `hab_y_loc` per row.
- **Patch count and layout** — unique `(patch_id, patch_location_x, patch_location_y)` rows in `metacommunity.csv`.
- **Habitat layout per patch** — `(habitat_id, habitat_x_location, habitat_y_location)` rows.
- **Environment values** — env columns per habitat row.
- **Islands habitat size** — `hab_length`, `hab_width` per habitat row.
- **Schedule-level construction** — both builders can be invoked as schedule items, not only directly.
- **2D column names** — `pheno_names_ls` and `environment_types_name` are user-chosen (e.g. `('trait_a', 'trait_b')`, `('env_temp', 'env_moisture')`).
- **Metacommunity name** — `meta_name` parameter on both builders (used as the lookup key in `sim.meta_objects` and as the `target` in schedule items).
- **CSV row order** — normalized internally; you do not have to pre-sort `mainland.csv`.

### 1.6 What you must obey (contracts)

The flip side of the freedoms above — these rules are enforced or assumed by the rest of the package and are pinned by the executable tests in `test/test_simulator_user_freedom_and_contracts.py`:

- `species_id` must be consecutive positive integers `1, 2, ..., n` (not non-contiguous, not arbitrary start, not strings).
- `mainland.csv` must contain every column listed in `pheno_names_ls`.
- `metacommunity.csv` must contain every column listed in `environment_types_name`.
- Every habitat inside a single patch must share the same `hab_length` x `hab_width`.
- Patch layout must form a rectangular grid for the plotting helpers (`nunique(patch_location_x)` x `nunique(patch_location_y)`).
- Every patch must use the same habitat grid shape for the plotting helpers.
- Plotting uses one representative `hab_length` / `hab_width` taken from the first CSV row.
- In the schedule: any item that consumes a metacommunity (`prime_*`, ecological items, recorders, plotters) must run **after** the builder that registers it.

---

## 2. Building schedule items

The simulator does not hard-code an ecological loop. Instead, you give it a Python list of *schedule items*; on each time step it walks the list in order and dispatches each item to either the simulator itself or to one of the metacommunity objects you registered.

### 2.1 The shape of one schedule item

Every item is a plain dict:

```python
{
    'target': '<simulator-or-meta-name>',  # required
    'method': '<method-name>',             # required
    'params': { ... },                     # optional, default {}
    'enabled': True,                       # optional gating fields
    'start':   0,
    'end':     None,
    'interval': 1,
}
```

| Field | Meaning |
|-------|---------|
| `target` | `'simulator'` to call a method on the `simulator` instance itself, or the `meta_name` of a registered metacommunity to call a method on that object. |
| `method` | The name of the method to call on the chosen target. |
| `params` | A dict of keyword arguments for the method. Values that are strings starting with `'@'` are resolved (see [§2.3](#23-the-name-parameter-reference)). |
| `enabled` | If `False`, the item is skipped. Default `True`. |
| `start` | First time step at which the item may run. Default `0`. |
| `end` | Last time step at which the item may run. Default `None` (no upper bound). |
| `interval` | Run every `interval` steps counting from `start`. Default `1` (every step). |

### 2.2 Gating fields: `enabled`, `start`, `end`, `interval`

The simulator checks them via `simulator.should_run`:

- `'start': 0, 'end': 0` → fires only at `time_step = 0` (one-shot init / priming).
- `'start': all_time_step - 1, 'end': all_time_step - 1` → fires only at the last step (end-of-run plots).
- `'interval': 100` → fires every 100 steps starting at `start`.
- No gating fields → fires every step from step 0 onwards.

### 2.3 The `'@name'` parameter reference

Inside `params`, any string value beginning with `'@'` is replaced at dispatch time with the registered metacommunity object whose name follows the `'@'`. This is how the islands schedule references the mainland to receive propagule rains:

```python
{'target': 'islands', 'method': 'meta_colonize_from_propagules_rains',
 'params': {'mainland_obj': '@mainland', 'propagules_rain_num': 200}},
```

The lookup is done by `simulator._resolve_value` and recurses into lists, tuples, and dicts, so `'@mainland'` can appear at any depth inside a params value.

### 2.4 A minimal schedule, end to end

The schedule below builds both metacommunities at `time_step=0` from CSV, primes the recorder file, runs one tick of dead-selection + asexual reproduction on the mainland, and flushes the log:

```python
schedule = [
    # one-shot: build mainland from CSV
    {'target': 'simulator', 'method': 'build_empty_mainland_from_species_csv',
     'params': {'meta_name': 'mainland', 'mainland_csv': 'mainland.csv',
                'pheno_names_ls': ('phenotype_axis1', 'phenotype_axis2'),
                'environment_types_name': ('env_axis1', 'env_axis2'),
                'environment_variation_ls': [0.025, 0.025],
                'patch_name': 'patch0', 'patch_index': 0, 'patch_location': (0, 0),
                'dormancy_pool_max_size': 0},
     'start': 0, 'end': 0},

    # one-shot: initialize the mainland (place individuals)
    {'target': 'mainland', 'method': 'meta_initialize',
     'params': {'traits_num': 2, 'pheno_names_ls': ('phenotype_axis1', 'phenotype_axis2'),
                'pheno_var_ls': (0.025, 0.025), 'geno_len_ls': (20, 20),
                'reproduce_mode': 'asexual',
                'species_2_phenotype_ls': [[0.2, 0.2], [0.2, 0.4]]},
     'start': 0, 'end': 0},

    # every-step: mainland dead selection
    {'target': 'mainland', 'method': 'meta_dead_selection',
     'params': {'base_dead_rate': 0.1, 'fitness_wid': 0.5, 'method': 'niche_gaussian'}},

    # every-step: mainland reproduction
    {'target': 'mainland', 'method': 'meta_mainland_asexual_birth_mutate_germinate',
     'params': {'asexual_birth_rate': 0.5, 'mutation_rate': 1e-4,
                'pheno_var_ls': (0.025, 0.025)}},

    # every-step: flush this step's accumulated log
    {'target': 'simulator', 'method': 'flush_step_log'},
]
```

For a full islands-mainland example (builders, priming, dispersal, recorders, end-of-run plots) see `build_schedule_asexual` and `build_schedule_sexual` in `playgrounds/model-simulator-GRFE.py`.

### 2.5 Recommended ordering

A schedule is just a Python list, so ordering matters. The conventional order used in `playgrounds/model-simulator-GRFE.py` is:

1. **One-shot construction** (`start=0, end=0`)
   - `simulator.build_empty_mainland_from_species_csv`
   - `mainland.meta_initialize`
   - `simulator.build_empty_metacommunity_from_patch_habitat_csv`
2. **One-shot priming of `csv.gz` recorders** (`start=0, end=0`, `mode='w'` headers)
   - `simulator.prime_optimum_sp_distribution`, `simulator.prime_environment_distribution`
3. **Mainland eco-evo loop** — dead selection → reproduction.
4. **Islands eco-evo loop** — dead selection → reproduction → colonization from mainland → dispersal within patch → dispersal among patches → germination + birth → disturbance → cleanup.
5. **Per-step utility items** — `flush_step_log`, `print_progress` (use `interval` to throttle).
6. **Per-step recorders** — `record_species_distribution`, `record_phenotype_distribution` (use `interval`).
7. **End-of-run plots** (`start=all_time_step-1, end=all_time_step-1`).

The hard rule is the one from [§1.6](#16-what-you-must-obey-contracts): a metacommunity must be built before anything else touches it.

---

## 3. Using the simulator

### 3.1 Required `global_params`

`simulator.set_global_params(global_params)` accepts a flat dict. The keys consumed by the simulator itself:

| Key | Type | Required | Used by |
|-----|------|----------|---------|
| `all_time_steps` | int | yes | `run`, `run_one_time_step`, `print_progress` — total number of time steps |
| `is_logging` | bool | yes | `open_logger`, `write_logger` — `True` writes to `<goal_path>/logger.log`, `False` prints to stdout |
| `is_timing` | bool | no (default `True`) | `run` — wraps the schedule loop in a wall-clock timer and writes total runtime to the log |
| `goal_path` | str | yes (set via `set_goal_path`) | logger, recorders, plotters — output directory |
| `root_path` | str | optional | set by `set_goal_path` for your own reference |

You may add any other key you like — it is just a dict — but those are the ones the simulator reads.

### 3.2 The five-line run pattern

The end-to-end pattern, lifted from `playgrounds/model-simulator-GRFE.py`:

```python
from metaibm.simulator import simulator

sim = simulator()
goal_path = sim.set_goal_path('./results', 'my_experiment', 'rep=0')
sim.set_global_params({'all_time_steps': 5000, 'goal_path': goal_path,
                       'is_logging': True, 'is_timing': True})
sim.set_schedule_per_time_step(schedule)  # the list from §2
sim.run()
```

`sim.run()`:
1. Opens the logger.
2. Optionally starts the timer.
3. For `time_step` in `0..all_time_steps-1`, executes every schedule item whose `should_run` returns `True`.
4. Optionally writes the total runtime to the log.
5. Closes the logger.
6. Returns `sim.finalize()` — a dict with `meta_objects`, `global_params`, and `schedule_one_time_step`.

If you want to add to the schedule incrementally instead of passing a full list, use `sim.add_schedule_item_per_time_step(item)`.

### 3.3 Outputs

All recorder and plotter file names are resolved through `goal_path`:

- **Recorders** (`record_species_distribution`, `record_phenotype_distribution`, `prime_*`) — write gzipped CSV under `<goal_path>/<file_name>` (or to the absolute path if you give one). Use the `prime_*` items at `time_step=0` to write the header row (`mode='w'`); use the `record_*` items every step (or every `interval` steps) to append rows (`mode='a'`).
- **Plotters** (`plot_species_distribution`, `plot_species_phenotype_distribution`, `plot_environment_distribution`) — write JPGs under `<goal_path>/time_step=<t>-<file_name>`. The `time_step=<t>-` prefix is added automatically.
- **Logger** — `<goal_path>/logger.log` when `is_logging=True`.

### 3.4 Logging and timing

The logger has a two-stage buffer:

1. Most schedule items return a short string describing what they did. `simulator.run_one_schedule_item` appends that string to `self.current_step_log` automatically.
2. Put `{'target': 'simulator', 'method': 'flush_step_log'}` at the end of the per-step schedule to write the accumulated `current_step_log` for that step (to `logger.log` if `is_logging=True`, otherwise to stdout).

For progress monitoring during a long run, add:

```python
{'target': 'simulator', 'method': 'print_progress',
 'params': {'target': 'islands', 'rank': rank, 'job_num': job_num, 'task_idx': task_idx},
 'interval': 1000},
```

This prints one line per `interval` steps with the current step, the live individual count, and the empty-site count.

---

## See also

- `docs-developer/metaibm-simulator.md` — per-method reference for every public and private method on `simulator`.
- `docs-developer/metaibm-metacommunity.md`, `docs-developer/metaibm-patch.md`, `docs-developer/metaibm-habitat.md`, `docs-developer/metaibm-individual.md` — reference for the underlying eco-evo classes the schedule dispatches to.
- `docs-developer/extension-global-habitat-network.md` — the global-habitat-network methods (`dispersal_among_patches_in_global_habitat_network_*`, etc.) used in the islands dispersal block.
- `playgrounds/model-simulator-GRFE.py` — a complete asexual + sexual example.
- `test/test_simulator_user_freedom_and_contracts.py` — executable proof of every freedom and contract listed in [§1.5](#15-what-the-simulator-lets-you-change-freedom) and [§1.6](#16-what-you-must-obey-contracts).
