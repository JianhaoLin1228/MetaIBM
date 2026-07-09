# Class `simulator`

Source: `metaibm/simulator.py`

> **Note:** The `simulator` class is the top-level driver for a MetaIBM run. It owns one or more registered `metacommunity`-like objects, a flat dictionary of non-eco-evo global parameters, and an ordered list of *schedule items* (dicts describing one eco-evo process each) that are executed every time step. The same schedule list is reused across all time steps; per-item fields `enabled`, `start`, `end`, `interval` control when each item fires.

---

## Attributes

### Instance Attributes

| Attribute | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `meta_objects` | dictionary of registered metacommunity-like objects, keyed by the user-provided name (typically `meta_object.metacommunity_name`) | dict | `{'mainland': meta_obj, 'islands': meta_obj}` |
| `global_params` | non-eco-evo global parameters used by the simulator itself (e.g. `all_time_steps`, `is_logging`, `is_timing`, `root_path`, `goal_path`) | dict | `{'all_time_steps': 5000, 'is_logging': True, 'is_timing': True, 'root_path': '...', 'goal_path': '...'}` |
| `schedule_one_time_step` | ordered list of schedule items (eco-evo process specs); each item is a dict with keys `target`, `method`, `params`, and optional `enabled`, `start`, `end`, `interval` | list of dicts | `[{'target': 'islands', 'method': 'meta_dead_selection', 'params': {...}, 'start': 0, 'interval': 1}, ...]` |
| `time_step` | current time step counter, updated at the start of every `run_one_time_step` call | int | 0, 1, ..., 4999 |
| `logger_file` | open file handle for the logger, or `None` if logging is off or not yet opened | file object or `None` | `<_io.TextIOWrapper name='.../logger.log' mode='a'>` |
| `current_step_log` | accumulated log text for the current time step; appended to by `append_step_log` and flushed by `flush_step_log` | str | `'time_step=12 \n... \n'` |
| `start_time` | wall-clock time at the start of `run()`; set only when `global_params['is_timing'] == True` | float or `None` | `1716000000.123` |
| `end_time` | wall-clock time at the end of `run()`; set only when `global_params['is_timing'] == True` | float or `None` | `1716000123.456` |

---

## Schedule item format

Every entry of `schedule_one_time_step` is a dict with the following fields:

| Field | Required | Description | Type | Example |
|-----------|-------------|-------------|-----------|-----------|
| `target` | yes | dispatch target: the literal string `'simulator'` (calls a simulator-level method), or the name of a registered metacommunity-like object in `self.meta_objects` | str | `'simulator'`, `'mainland'`, `'islands'` |
| `method` | yes | name of the method to invoke on the dispatch target | str | `'meta_dead_selection'`, `'flush_step_log'` |
| `params` | no | keyword arguments passed to the method; values prefixed with `'@'` are resolved to the matching registered metacommunity object (see `_resolve_value`) | dict | `{'base_dead_rate': 0.05, 'mainland_obj': '@mainland'}` |
| `enabled` | no | when `False`, the item is skipped; default `True` | bool | `True`, `False` |
| `start` | no | first time step at which the item may run; default `0` | int | `0`, `100` |
| `end` | no | last time step at which the item may run; default `None`. <span style="color:red">When `None`, no upper bound is applied (item may run for the entire simulation). </span> | int or `None` | `None`, `4999` |
| `interval` | no | run every `interval` steps counting from `start`; default `1`. <span style="color:red">`interval=None` is interpreted as "always run" by the current code.</span> | int or `None` | `1`, `10`, `None` |

---

## Methods

### General / Construction

#### `__init__`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Initializes an empty simulator: `meta_objects = {}`, `global_params = {}`, `schedule_one_time_step = []`, `time_step = 0`, `logger_file = None`, `current_step_log = ""`, `start_time = None`, `end_time = None`.

---

#### `add_metacommunity_obj`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `name` | user-provided registration name (typically `meta_object.metacommunity_name`); becomes the lookup key in `self.meta_objects` and is referenced by `target` and `'@name'` params in schedule items | str | `'mainland'`, `'islands'` |
| `obj` | the metacommunity-like object to register | `metacommunity` | `meta_object` |

**Returns:** `None`

**Description:**

Registers a metacommunity-like object under `name` by assigning `self.meta_objects[name] = obj`. Called internally by the `build_empty_*` constructors; can also be called directly by user code.

---

#### `build_empty_mainland_from_species_csv`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `meta_name` | name to register the new metacommunity under | str | `'mainland'` |
| `mainland_csv` | path to a species CSV. One row = one species = one mainland habitat. Required columns: `species_id`, `hab_x_loc`, `hab_y_loc`, `hab_length`, `hab_width`, plus one column per name in `pheno_names_ls` holding the mean environment value for that trait. | str | `'mainland.csv'` |
| `pheno_names_ls` | list of phenotype (trait) column names to read from the CSV as per-habitat mean environment values | list of str | `['phenotype1', 'phenotype2']` |
| `environment_types_name` | names of the environment axes registered on every habitat (length must match the per-habitat mean list built from `pheno_names_ls`) | list of str | `['env1', 'env2']` |
| `environment_variation_ls` | per-axis standard deviation of within-habitat environment noise | list of float | `[0.025, 0.025]` |
| `patch_name` | name of the single patch that holds all mainland habitats; default `'patch0'` | str | `'patch0'` |
| `patch_index` | integer index of that patch within the metacommunity; default `0` | int | `0` |
| `patch_location` | `(X, Y)` location of that patch; default `(0, 0)` | tuple | `(0, 0)` |
| `dormancy_pool_max_size` | max capacity of each habitat's dormancy pool; default `0` (no dormancy) | int | `0`, `1000` |

**Returns:** `str` (log text describing each habitat that was registered)

**Description:**

Builds an "empty" mainland metacommunity (one patch, many habitats, no individuals yet) from a species CSV:
1. Reads the CSV and sorts by `species_id`.
2. Creates a new `metacommunity(metacommunity_name=meta_name)` and one `patch(patch_name, patch_index, patch_location)`.
3. For each CSV row (one species), creates one habitat named `'h<row_index>'` with the per-row location, size, and mean environment values read via `pheno_names_ls`. The variance vector `environment_variation_ls` is shared across all habitats.
4. Adds the patch to the metacommunity and registers the metacommunity on the simulator via `add_metacommunity_obj`.
5. Returns the accumulated log string (also appended to `current_step_log` by `run_one_schedule_item` when called from the schedule).

Schedule-callable.

---

#### `build_empty_metacommunity_from_patch_habitat_csv`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `meta_name` | name to register the new metacommunity under | str | `'islands'` |
| `metacommunity_csv` | path to a patch-habitat CSV. One row = one habitat. Required columns: `patch_id`, `patch_index`, `patch_location_x`, `patch_location_y`, `habitat_id`, `habitat_index`, `habitat_x_location`, `habitat_y_location`, `hab_length`, `hab_width`, plus one column per name in `environment_types_name`. Patches are inferred from unique `(patch_id, patch_index, patch_location_x, patch_location_y)` rows. | str | `'metacommunity.csv'` |
| `environment_types_name` | names of the environment axes registered on every habitat; also names of the CSV columns holding per-habitat mean values | list of str | `['env1', 'env2']` |
| `environment_variation_ls` | per-axis standard deviation of within-habitat environment noise | list of float | `[0.025, 0.025]` |
| `dormancy_pool_max_size` | max capacity of each habitat's dormancy pool; default `0` | int | `0`, `1000` |

**Returns:** `str` (log text describing each habitat that was registered)

**Description:**

Builds an "empty" multi-patch metacommunity (islands-style layout) from a single patch-habitat CSV:
1. Reads the CSV.
2. Derives the unique patch register rows (sorted by `patch_index`, then `patch_id`); creates one `patch` per row.
3. For each patch, iterates over its rows (sorted by `habitat_index`, then `habitat_id`) and adds each habitat with the per-row location, size, and mean environment values.
4. Adds every patch to the metacommunity and registers the metacommunity on the simulator via `add_metacommunity_obj`.
5. Returns the accumulated log string.

Schedule-callable.

---

#### `set_global_params`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `global_params` | dict of non-eco-evo runtime parameters; replaces `self.global_params` wholesale | dict | `{'all_time_steps': 5000, 'is_logging': True, 'is_timing': True}` |

**Returns:** `None`

**Description:**

Replaces `self.global_params` with the supplied dict. Expected keys consumed elsewhere in this class: `all_time_steps` (int, required by `run`/`run_one_time_step`/`print_progress`), `is_logging` (bool, required by `open_logger`/`write_logger`), `is_timing` (bool, optional with default `True` via `.get`), `root_path` / `goal_path` (str, set by `set_goal_path` and used by loggers, recorders, and plotters).

---

#### `_mkdir_if_not_exist`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `goal_path` | directory path to create | str | `'./results/run01/'` |

**Returns:** `None`

**Description:**

Creates the folder hierarchy at `goal_path` (`os.makedirs(..., exist_ok=True)`). <span style="color:red">Not currently called by any other simulator method (`set_goal_path` calls `os.makedirs` directly); kept as a public utility.</span>

---

#### `set_goal_path`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `root_path` | root directory for all simulator outputs | str | `'./results'` |
| `*multi_layer_folder` | any number of additional subfolder names joined under `root_path` | str(s) | `'sloss', 'run01'` |

**Returns:** `str` (the resulting `goal_path`)

**Description:**

Joins `root_path` with `*multi_layer_folder` into `goal_path`, stores both in `self.global_params['root_path']` and `self.global_params['goal_path']`, creates `goal_path` on disk, and returns it. Used by the logger, recorders, and plotters as the default output directory.

---

### Schedule

#### `set_schedule_per_time_step`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `schedule_one_time_step` | ordered list of schedule items (see [Schedule item format](#schedule-item-format)) | list of dicts | `[{'target': 'islands', 'method': '...', 'params': {...}}, ...]` |

**Returns:** `None`

**Description:**

Replaces `self.schedule_one_time_step` with the provided list. The list is iterated every time step by `run_one_time_step`.

---

#### `add_schedule_item_per_time_step`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `item` | one schedule item dict (see [Schedule item format](#schedule-item-format)) | dict | `{'target': 'simulator', 'method': 'flush_step_log', 'params': {}}` |

**Returns:** `None`

**Description:**

Appends one item to `self.schedule_one_time_step`. Useful for building the schedule incrementally.

---

### Logging

#### `open_logger`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `mode` | file open mode; default `'a'` (append) | str | `'a'`, `'w'` |

**Returns:** `None`

**Description:**

If `global_params['is_logging'] == True`, opens `<goal_path>/logger.log` in the given mode and stores the file handle on `self.logger_file`. If logging is off, does nothing. Called automatically at the start of `run()`.

---

#### `close_logger`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Closes `self.logger_file` if it was opened. Called automatically at the end of `run()`.

---

#### `append_step_log`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `log_info` | text to append to the per-step log buffer | str | `'meta_dead_selection: ... \n'` |

**Returns:** `None`

**Description:**

Concatenates `log_info` to `self.current_step_log`. Called by `run_one_schedule_item` whenever a dispatched method returns a string.

---

#### `write_logger`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `log_info` | text to write | str | `'time_step=12 \n... \n'` |

**Returns:** `None`

**Description:**

If `global_params['is_logging'] == True`, writes `log_info` to `self.logger_file` via `print(..., file=self.logger_file)`. If `is_logging == False`, prints to stdout instead.

---

#### `flush_step_log`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Writes the accumulated `self.current_step_log` for the current time step via `write_logger`. <span style="color:red">Does not reset `self.current_step_log` to `""` after flushing; the reset happens at the start of the next `run_one_time_step`.</span> Schedule-callable (typically the last item in a per-step schedule).

---

### Main loop

#### `run`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `dict` (the return value of `finalize()`)

**Description:**

Runs the full simulation:
1. Opens the logger (`open_logger()`).
2. If `global_params.get('is_timing', True) == True`, records `self.start_time`.
3. Iterates `time_step` from `0` to `global_params['all_time_steps'] - 1`, calling `run_one_time_step(time_step)` for each.
4. If timing is on, records `self.end_time` and writes a `'total simulation time: ... s'` line via `write_logger`.
5. Closes the logger and returns `self.finalize()`.

---

#### `run_one_time_step`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `time_step` | current time step index | int | 0, 1, ..., `all_time_steps - 1` |

**Returns:** `None`

**Description:**

Sets `self.time_step = time_step`, resets `self.current_step_log` to `'time_step=<t> \n'`, then iterates `self.schedule_one_time_step` in order. For each item, calls `should_run(item, time_step)`; if it returns `True`, calls `run_one_schedule_item(item)`.

---

#### `should_run`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `item` | a schedule item (see [Schedule item format](#schedule-item-format)) | dict | `{'target': '...', 'method': '...', 'start': 10, 'interval': 5}` |
| `time_step` | current time step index | int | 0, 1, ... |

**Returns:** `bool`

**Description:**

Decides whether a schedule item should run at the current step. Reads the item's optional gating fields with defaults `enabled=True`, `start=0`, `end=None`, `interval=1`. Returns `False` if `enabled` is false, if `time_step < start`, or if `end is not None and time_step > end`. If `interval is None`, returns `True`. Otherwise returns `(time_step - start) % interval == 0`.

---

#### `run_one_schedule_item`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `item` | a schedule item (see [Schedule item format](#schedule-item-format)) | dict | `{'target': 'islands', 'method': 'meta_dead_selection', 'params': {...}}` |

**Returns:** `None`

**Description:**

Executes one schedule item:
1. Reads `target`, `method`, and `params` from the item; resolves `params` via `_resolve_params` (so `'@name'` strings become the registered metacommunity object).
2. If `target == 'simulator'`, looks up the method on `self` and calls it with the resolved params.
3. Otherwise looks up `self.meta_objects[target]` and calls the method on that object with the resolved params. (Simulator-level methods read `self.time_step` directly when they need the current step, so it is not passed via params.)
4. If the returned value is a string, appends it to `self.current_step_log` via `append_step_log`.

---

### Parameter resolution helpers

#### `_resolve_params`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `params` | the `params` dict from a schedule item | dict | `{'base_dead_rate': 0.05, 'mainland_obj': '@mainland'}` |

**Returns:** `dict` (same keys, each value resolved via `_resolve_value`)

**Description:**

Resolves every value in a params dict via `self._resolve_value`. Called by `run_one_schedule_item` before dispatching.

---

#### `_resolve_value`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `value` | the value to resolve | any | `'@mainland'`, `0.05`, `['@a', '@b']` |

**Returns:** the resolved value (any type)

**Description:**

Resolves one parameter value. A string beginning with `'@'` is interpreted as a reference to a registered metacommunity-like object and replaced with `self.meta_objects[value[1:]]`. Lists, tuples, and dicts are recursed into so that `'@name'` references can appear at any depth. All other values are returned unchanged.

---

### Recorders

#### `_build_record_path`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `file_name` | output file name; absolute path or relative to `goal_path` | str | `'sp_dis.csv.gz'`, `'/tmp/sp_dis.csv.gz'` |

**Returns:** `str` (resolved absolute or joined path)

**Description:**

If `file_name` is absolute, returns it as-is. Otherwise joins it under `self.global_params['goal_path']`. Used by all recorder methods.

---

#### `_get_columns`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `target` | name of a registered metacommunity-like object | str | `'islands'` |

**Returns:** `list` of length 3: `[cols_patch_id, cols_hab_id, cols_microsite_id]`

**Description:**

Builds the multi-row column header used by the gzipped CSV recorders by calling `self.meta_objects[target].columns_patch_habitat_microsites_id()`.

---

#### `prime_optimum_sp_distribution`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `target` | name of a registered metacommunity-like object | str | `'islands'` |
| `base_dead_rate` | base death rate used to compute per-microsite optimum species id | float | `0.05` |
| `fitness_wid` | fitness width parameter used to compute per-microsite optimum species id | float | `0.5` |
| `species_2_phenotype_ls` | <span style="color:red">list of mean preset phenotype value lists per species (same convention as elsewhere in the package: species ID is derived as `'sp' + str(index_in_list + 1)`)</span> | list of list of float | `[[0.2, 0.4], [0.5, 0.6]]` |
| `file_name` | output file (absolute or relative to `goal_path`) | str | `'optimum_sp.csv.gz'` |

**Returns:** `None`

**Description:**

Schedule-callable. Writes the `mode='w'` header row of optimum-species-id values for one registered object, by calling `meta_obj.get_meta_microsites_optimum_sp_id_val(...)` and then `meta_obj.meta_distribution_data_all_time_to_csv_gz(...)` with `index=['optimun_sp_id_values']` and the multi-row column header from `_get_columns`. Typically used once at simulation start, before the per-step `record_*` calls append rows.

---

#### `prime_environment_distribution`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `target` | name of a registered metacommunity-like object | str | `'islands'` |
| `environment_name` | environment axis to record | str | `'env1'` |
| `index_label` | label written in the row index column for this header row | str | `'env1_values'` |
| `file_name` | output file (absolute or relative to `goal_path`) | str | `'env1_dis.csv.gz'` |

**Returns:** `None`

**Description:**

Schedule-callable. Writes the `mode='w'` header row containing one environment axis's per-microsite values for one registered object. Like `prime_optimum_sp_distribution`, typically used once at simulation start.

---

#### `record_species_distribution`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `target` | name of a registered metacommunity-like object | str | `'islands'` |
| `file_name` | output file (absolute or relative to `goal_path`) | str | `'sp_dis.csv.gz'` |
| `mode` | file mode passed through to the writer; default `'a'` (append) | str | `'a'`, `'w'` |

**Returns:** `None`

**Description:**

Schedule-callable. Records the species distribution at `self.time_step` for one registered object, by calling `meta_obj.get_meta_microsites_individuals_sp_id_values()` and appending a row labeled `'time_step<t>'` via `meta_obj.meta_distribution_data_all_time_to_csv_gz(...)`.

---

#### `record_phenotype_distribution`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `target` | name of a registered metacommunity-like object | str | `'islands'` |
| `trait_name` | phenotype/trait to record | str | `'phenotype1'` |
| `file_name` | output file (absolute or relative to `goal_path`) | str | `'pheno1_dis.csv.gz'` |
| `mode` | file mode passed through to the writer; default `'a'` (append) | str | `'a'`, `'w'` |

**Returns:** `None`

**Description:**

Schedule-callable. Records the phenotype distribution of one trait at `self.time_step` for one registered object, appending a row labeled `'time_step<t>'`.

---

### Plotters

#### `_build_plot_path`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `file_name` | output file name; absolute path or relative to `goal_path` | str | `'sp_dis.jpg'`, `'/tmp/sp_dis.jpg'` |

**Returns:** `str` (resolved path with a `'time_step=<t>-'` basename prefix)

**Description:**

Composes the plot file path. If `file_name` is absolute, keeps its directory and prefixes `'time_step=<t>-'` to the basename. Otherwise joins under `self.global_params['goal_path']` with the same prefix. The `<t>` is read from `self.time_step` so every per-step plot is uniquely named.

---

#### `plot_species_distribution`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `target` | name of a registered metacommunity-like object | str | `'islands'` |
| `sub_row` | number of subplot rows in the figure grid | int | 2 |
| `sub_col` | number of subplot columns in the figure grid | int | 3 |
| `hab_num_x_axis_in_patch` | habitats per patch along the figure X axis | int | 1 |
| `hab_num_y_axis_in_patch` | habitats per patch along the figure Y axis | int | 1 |
| `hab_y_len` | habitat grid length along Y | int | 20 |
| `hab_x_len` | habitat grid length along X | int | 20 |
| `vmin` | color scale lower bound | float / int | 1 |
| `vmax` | color scale upper bound | float / int | 6 |
| `cmap` | matplotlib colormap name, resolved via `plt.get_cmap(cmap)` | str | `'tab10'` |
| `file_name` | output file (absolute or relative to `goal_path`); will be prefixed with `'time_step=<t>-'` | str | `'sp_dis.jpg'` |

**Returns:** `None`

**Description:**

Schedule-callable. Writes one JPG of the species distribution at `self.time_step` for one registered object, by delegating to `meta_obj.meta_show_species_distribution(...)` with the resolved colormap and the prefixed file path.

---

#### `plot_species_phenotype_distribution`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `target` | name of a registered metacommunity-like object | str | `'islands'` |
| `trait_name` | trait to plot | str | `'phenotype1'` |
| `sub_row` | number of subplot rows in the figure grid | int | 2 |
| `sub_col` | number of subplot columns in the figure grid | int | 3 |
| `hab_num_x_axis_in_patch` | habitats per patch along the figure X axis | int | 1 |
| `hab_num_y_axis_in_patch` | habitats per patch along the figure Y axis | int | 1 |
| `hab_y_len` | habitat grid length along Y | int | 20 |
| `hab_x_len` | habitat grid length along X | int | 20 |
| `cmap` | matplotlib colormap name | str | `'viridis'` |
| `file_name` | output file (absolute or relative to `goal_path`); prefixed with `'time_step=<t>-'` | str | `'pheno1_dis.jpg'` |

**Returns:** `None`

**Description:**

Schedule-callable. Writes one JPG of the phenotype distribution of one trait at `self.time_step` for one registered object. <span style="color:red">No `vmin`/`vmax` parameters here; color limits are determined by the underlying `meta_show_species_phenotype_distribution` implementation.</span>

---

#### `plot_environment_distribution`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `target` | name of a registered metacommunity-like object | str | `'islands'` |
| `environment_name` | environment axis to plot | str | `'env1'` |
| `sub_row` | number of subplot rows in the figure grid | int | 2 |
| `sub_col` | number of subplot columns in the figure grid | int | 3 |
| `hab_num_x_axis_in_patch` | habitats per patch along the figure X axis | int | 1 |
| `hab_num_y_axis_in_patch` | habitats per patch along the figure Y axis | int | 1 |
| `hab_y_len` | habitat grid length along Y | int | 20 |
| `hab_x_len` | habitat grid length along X | int | 20 |
| `mask_loc` | <span style="color:red">mask locations passed through to `meta_show_environment_distribution`; </span> | str or None | `'upper','lower', or None` |
| `cmap` | matplotlib colormap name | str | `'viridis'` |
| `file_name` | output file (absolute or relative to `goal_path`); prefixed with `'time_step=<t>-'` | str | `'env1_dis.jpg'` |

**Returns:** `None`

**Description:**

Schedule-callable. Writes one JPG of one environment axis at `self.time_step` for one registered object, by delegating to `meta_obj.meta_show_environment_distribution(...)`.

---

### Progress / finalize

#### `print_progress`

**Parameters:**

| Parameter | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `self` | self | self | self |
| `target` | name of a registered metacommunity-like object; default `None` (prints only the step counter) | str or `None` | `'islands'`, `None` |
| `rank` | MPI process rank, for parallel runs; default `0` | int | 0, 1, ... |
| `job_num` | total number of tasks in the sweep; default `1` | int | 16 |
| `task_idx` | this task's index in the sweep; default `1` | int | 1, ..., `job_num` |

**Returns:** `None`

**Description:**

Schedule-callable. Prints a one-line progress message for the current time step. When `target is None`, prints `'time_step/all_time_step=<t>/<T>'`. Otherwise also queries `meta_obj.get_meta_individual_num()` and `meta_obj.show_meta_empty_sites_num()` and prints them alongside the rank / task / step counters.

---

#### `finalize`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `dict` with keys `'meta_objects'`, `'global_params'`, `'schedule_one_time_step'`

**Description:**

Returns a lightweight dictionary describing the final state of the simulator: the registered metacommunity objects, the global parameters dict, and the schedule list. Called automatically at the end of `run()`; its return value is the return value of `run()`.
