# Class `metacommunity`

Source: `metaibm/metacommunity.py`

> **Note:** Additional methods and attributes are monkey-patched onto this class by the global-habitat-network extension (`extension/global_habitat_network.py`), installed at the bottom of `metacommunity.py`. Those are documented separately.

---

## Attributes

### Instance Attributes

| Attribute | Description | Type | Example |
|-----------|-------------|-----------|-----------|
| `set` | dictionary storing patch objects, keyed by patch name (patch id) | dict | `{'patch1': patch_obj, 'patch2': patch_obj}` |
| `patch_num` | count of patches in the metacommunity (initialized to 0 when no patches were added) | int | 1, 2, ... |
| `metacommunity_name` | string name of the metacommunity | str | `'mainland'` or  `'metacommunity'` |
| `patch_id_ls` | ordered list of patch IDs aligned with `patch.index` (0, 1, 2, ..., N) | list of strings | `['patch1','patch2'..]` |
| `patch_id_2_index_dir` | dictionary mapping patch_id (name) to patch.index for quick lookup | dict | `{'patch1': index, 'patch2': index}` |
| `pairwise_patch_distance_matrix` | numpy matrix of Euclidean distances between all patch pairs; row/column i corresponds to the patch with `index == i` | np.matrix | np.matrix[i, j] = distance(patchi, patchj) |

---

## Methods

### General

#### `get_data`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `dict`, {'patch_id': {'h_id': h_object.set, ...}, ...}

**Description:**

Returns a dictionary aggregating data from all patches by calling `get_data()` on each patch object. Keys are patch names, values are patch data dictionaries. 

---

#### `update_disp_current_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.matrix`

**Description:**

Calculates the Euclidean distance between all patch pairs based on their `location` coordinates and stores the result in `pairwise_patch_distance_matrix`. Called automatically when patches are added. Returns the distance matrix.

---

#### `add_patch`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `patch_name` | string identifier for the patch |
| `patch_object` | a `patch` object to add |

**Returns:** `None`

**Description:**

Adds a patch to the metacommunity. Stores it in `self.set`, increments `patch_num`, updates `patch_id_ls` and `patch_id_2_index_dir`, recalculates the distance matrix via `update_disp_current_matrix()`, and initializes global habitat network fields (extension).

---

### Visualization

#### `reshape_habitat_data_in_patch`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `df` | DataFrame containing habitat data to reshape |
| `hab_num_x_axis_in_patch` | number of habitats along the x-axis within a patch |
| `hab_num_y_axis_in_patch` | number of habitats along the y-axis within a patch |
| `hab_y_len` | length (rows) of each habitat |
| `hab_x_len` | width (columns) of each habitat |
| `mask_loc` | `'lower'`, `'upper'`, or `None` (defalut); controls triangular masking for heatmaps |

**Returns:** `tuple` (reshape_data_col_row, mask_data_col_row)

**Description:**

Reshapes flat habitat data into a 2D array suitable for heatmap visualization, arranging habitats in their spatial positions with gaps between them. Also produces a corresponding mask array for optional triangular masking. 

<span style="color:red"> 'A possible input for df: </span>

<span style="color:red">`df = pd.DataFrame(patch_object.get_patch_microsites_individals_sp_id_values())`</span>

<span style="color:red">A possible output with df above: </span>

<span style="color:red">``reshape_data_col_row` is patch-scale sp distribution data for plotting, arranged by habitat location in a patch (`hab_num_x_axis_in_patch`, `hab_num_y_axis_in_patch`) with gap (1) to distinguish boundary of habitats within this patch.' </span>

<span style="color:red">``mask_data_col_row`  is upper or lower triangle indentity matrix</span>

---

#### `meta_show_species_distribution`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `sub_row` | number of subplot rows |
| `sub_col` | number of subplot columns |
| `hab_num_x_axis_in_patch` | habitats per row within each patch |
| `hab_num_y_axis_in_patch` | habitats per column within each patch |
| `hab_y_len` | habitat length (rows) |
| `hab_x_len` | habitat width (columns) |
| `vmin` | minimum value for heatmap color scale |
| `vmax` | maximum value for heatmap color scale |
| `cmap` | colormap name for the heatmap |
| `file_name` | output file path for the saved figure |

**Returns:** `int` (0)

**Description:**

Visualizes species distribution across all patches using seaborn heatmaps. Each patch is shown as a subplot. Species IDs are plotted as numeric values with `NaN` for empty microsites. Saves the figure to `file_name`.

---

#### `meta_show_species_phenotype_distribution`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `trait_name` | string; which phenotype to visualize |
| `sub_row` | number of subplot rows |
| `sub_col` | number of subplot columns |
| `hab_num_x_axis_in_patch` | habitats per row within each patch |
| `hab_num_y_axis_in_patch` | habitats per column within each patch |
| `hab_y_len` | habitat length (rows) |
| `hab_x_len` | habitat width (columns) |
| `cmap` | colormap name |
| `file_name` | output file path |

**Returns:** `int` (0)

**Description:**

Visualizes the phenotype distribution for a specific trait across all patches using heatmaps. Phenotype values are shown per microsite with `NaN` for empty sites.

<span style="color:red">sns.heatmap : ``vmin=0` `vmax=0.8` </span>

---

#### `meta_show_environment_distribution`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `environment_name` | string; which environment variable to visualize |
| `sub_row` | number of subplot rows |
| `sub_col` | number of subplot columns |
| `hab_num_x_axis_in_patch` | habitats per row within each patch |
| `hab_num_y_axis_in_patch` | habitats per column within each patch |
| `hab_y_len` | habitat length (rows) |
| `hab_x_len` | habitat width (columns) |
| `mask_loc` | `'lower'`, `'upper'`, or `None` |
| `cmap` | colormap name |
| `file_name` | output file path |

**Returns:** `int` (0)

**Description:**

Visualizes an environmental variable across all patches using heatmaps with optional triangular masking.

<span style="color:red">sns.heatmap : ``vmin=0` `vmax=0.8` </span>

---

#### `meta_show_two_environment_distribution`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `environment1_name` | string; first environment variable |
| `environment2_name` | string; second environment variable |
| `sub_row` | number of subplot rows |
| `sub_col` | number of subplot columns |
| `hab_num_x_axis_in_patch` | habitats per row within each patch |
| `hab_num_y_axis_in_patch` | habitats per column within each patch |
| `hab_y_len` | habitat length (rows) |
| `hab_x_len` | habitat width (columns) |
| `mask_loc1` | mask location for first environment |
| `mask_loc2` | mask location for second environment |
| `cmap1` | colormap for first environment |
| `cmap2` | colormap for second environment |
| `file_name` | output file path |

**Returns:** `int` (0)

**Description:**

Overlays two environmental variables on the same heatmap using triangular masking (e.g. lower triangle for environment1, upper triangle for environment2). Allows side-by-side comparison of two environment gradients.

<span style="color:red">sns.heatmap : ``vmin=0` `vmax=0.8` </span>

---

### Initialization

#### `meta_initialize`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `traits_num` | number of traits per individual |
| `pheno_names_ls` | list of phenotype names |
| `pheno_var_ls` | list of phenotypic variance per trait |
| `geno_len_ls` | list of genotype lengths per trait |
| `reproduce_mode` | reproduction mode: `'asexual'`, `'sexual'`, or `'mixed'` |
| `species_2_phenotype_ls` | <span style="color:red">list of mean phenotype value lists, one per species; the species ID is derived as `'sp' + str(index_in_list + 1)`</span> |

**Returns:** `str` (log_info)

**Description:**

Initializes all patches in the metacommunity by calling each patch's `patch_initialize()`. Fills all microsites across the entire metacommunity with individuals. Returns a log string confirming initialization.

---

### Metacommunity Queries

#### `get_meta_individual_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the total number of living individuals across all patches in the metacommunity.

---

#### `get_meta_empty_sites_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuples (patch_id, h_id, len_id, wid_id)

**Description:**

Returns a list of all empty microsite positions across the entire metacommunity. Each tuple includes the patch name, habitat name, and row/column indices.

---

#### `show_meta_empty_sites_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the total count of empty microsites across the entire metacommunity.

---

#### `get_meta_pairwise_empty_sites_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuple pairs

**Description:**

Returns a list of paired empty microsite positions `[((patch_id, hab_id, row, col), (patch_id, hab_id, row, col)), ...]` across the metacommunity. Used for sexual colonization.

---

#### `meta_get_occupied_location_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuples (patch_id, h_id, row_id, col_id)

**Description:**

Returns a list of all occupied microsite positions across the metacommunity.

---

#### `get_meta_pairwise_occupied_sites_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuple pairs

**Description:**

Returns a list of female-male paired occupied positions across the metacommunity. Used for sexual reproduction pairing.

---

### Pool Counts

#### `meta_offspring_pool_individual_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the total count of individual objects in `offspring_pool` across all patches and habitats.

---

#### `meta_immigrant_pool_individual_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the total count of individual objects in `immigrant_pool` across all patches and habitats.

---

#### `meta_dormancy_pool_individual_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the total count of individual objects in `dormancy_pool` across all patches and habitats.

---

#### `meta_offspring_marker_pool_marker_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the total count of marker tuples in `offspring_marker_pool` across all patches and habitats.

---

#### `meta_immigrant_marker_pool_marker_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the total count of marker tuples in `immigrant_marker_pool` across all patches and habitats.

---

### Environment

#### `meta_offset_environmental_values`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `env_name_ls` | list of environment type names to modify |
| `delta_mean_ls` | list of additive offsets for environment means |
| `delta_var_ls` | list of additive offsets for environment variances (optional) |

**Returns:** `str` (log_info)

**Description:**

Shifts environmental values across all patches in the metacommunity by delegating to each patch's `patch_offset_environmental_values()`. Returns a log string describing the changes applied.

---

### Selection

#### `meta_dead_selection`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `base_dead_rate` | float; base death rate |
| `fitness_wid` | float; fitness width parameter |
| `method` | selection method string (e.g. `'niche_gaussian'`, `'neutral_uniform'`) |

**Returns:** `str` (log_info)

**Description:**

Applies mortality selection across all patches by delegating to each patch's `patch_dead_selection()`. Returns a log string reporting the total dead count, remaining individuals, and empty sites.

---

### Direct Reproduction + Germination (Mainland)

#### `meta_mainland_asexual_birth_mutate_germinate`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per individual |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `str` (log_info)

**Description:**

Performs asexual reproduction with mutation and immediate germination across all patches (mainland mode, no dispersal or dormancy). Returns a log string with the germination count.

---

#### `meta_mainland_sexual_birth_mutate_germinate`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `sexual_birth_rate` | float; birth rate per breeding pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `str` (log_info)

**Description:**

Performs sexual reproduction with mutation and immediate germination across all patches (mainland mode). Returns a log string with the germination count.

---

#### `meta_mainland_mixed_birth_mutate_germinate`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per high-fitness asexual parent |
| `sexual_birth_rate` | float; birth rate per low-fitness sexual pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `str` (log_info)

**Description:**

Performs mixed asexual + sexual reproduction with mutation and immediate germination across all patches (mainland mode). Returns a log string with the germination count.

---

### Colonization from Mainland

#### `meta_colonize_from_propagules_rains`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `mainland_obj` | a `metacommunity` object representing the mainland source |
| `propagules_rain_num` | float; expected number of propagules per patch |

**Returns:** `str` (log_info)

**Description:**

Colonizes the metacommunity from a mainland source via asexual propagule rain. For each patch, probabilistically rounds the propagule count, then randomly samples occupied sites from the mainland and empty sites from the target patch. Creates deep copies of mainland individuals and places them into target microsites. Returns a log string with the colonization count.

---

#### `pairwise_sexual_colonization_from_prpagules_rains`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `mainland_obj` | a `metacommunity` object representing the mainland source |
| `propagules_rain_num` | float; expected number of propagule pairs per patch |

**Returns:** `str` (log_info)

**Description:**

Colonizes via sexual propagule pairs. Selects female-male pairs from mainland occupied sites and places them into paired empty sites in the target metacommunity. Counts pairs as 2 individuals. Returns a log string with the colonization count.

---

### Reproduction into Offspring Pool (Object Pipeline)

#### `meta_asex_reproduce_mutate_into_offspring_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per individual |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `str` (log_info)

**Description:**

Creates asexual offspring as full `individual` objects with mutations and stores in offspring pools across all patches. Returns a log string with the born count and pool size.

---

#### `meta_sex_reproduce_mutate_into_offspring_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `sexual_birth_rate` | float; birth rate per breeding pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `str` (log_info)

**Description:**

Creates sexual offspring as full `individual` objects with mutations and stores in offspring pools across all patches. Returns a log string with the born count and pool size.

---

#### `meta_mix_reproduce_mutate_into_offspring_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per high-fitness asexual parent |
| `sexual_birth_rate` | float; birth rate per low-fitness sexual pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `str` (log_info)

**Description:**

Creates mixed asexual + sexual offspring as full `individual` objects with mutations and stores in offspring pools across all patches. Returns a log string with the born count and pool size.

---

### Reproduction into Offspring Marker Pool

#### `meta_asex_reproduce_calculation_into_offspring_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per individual |

**Returns:** `str` (log_info)

**Description:**

Generates asexual offspring markers (lightweight tuples) across all patches. Actual individual creation is deferred until germination. Used when dormancy is off (default workflow). Returns a log string with marker count.

---

#### `meta_sex_reproduce_calculation_with_offspring_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `sexual_birth_rate` | float; birth rate per breeding pair |

**Returns:** `str` (log_info)

**Description:**

Generates sexual offspring markers across all patches. Returns a log string with marker count.

---

#### `meta_mix_reproduce_calculation_with_offspring_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per high-fitness asexual parent |
| `sexual_birth_rate` | float; birth rate per low-fitness sexual pair |

**Returns:** `str` (log_info)

**Description:**

Generates mixed reproduction offspring markers across all patches. Returns a log string with marker count.

---

### Dispersal Helpers

#### `matrix_around`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `matrix` | numpy matrix with potentially fractional values |

**Returns:** `numpy.matrix`

**Description:**

Probabilistically rounds fractional values in a matrix. For each element, the decimal part is treated as a probability of rounding up (+1). `NaN` values become 0. Returns the rounded integer matrix.

---

#### `dispersal_kernel_strength`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `d_ij` | float; Euclidean distance between two patches |
| `method` | dispersal kernel type: `'uniform'`, `'gaussian'`, `'exponential'`, `'cauchy'`, or `'power_law'` |
| `**kwargs` | method-specific parameters (see below) |

**Returns:**  `float`

**Description:**

Calculates the relative dispersal strength (Dij) between two patches (i, j) based on their distance. Supported methods:
- `'uniform'`: `D = 1.0` (equal for all distances)
- `'gaussian'`: `D = exp(-(d^2) / (2 * sigma^2))` (requires `sigma`)
- `'exponential'`: `D = rho * exp(-rho * d)` (requires `rho`)
- `'cauchy'`: `D = 1 / (1 + (d/gamma)^2)` (requires `gamma`; long-tailed)
- `'power_law'`: `D = (1 + d/r0)^(-alpha)` (requires `alpha`, `r0`; long-tailed)

---

#### `calculate_dispersal_kernel_strength_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `method` | dispersal kernel method |
| `**kwargs` | kernel-specific parameters |

**Returns:** `numpy.matrix`

**Description:**

Builds an unnormalized dispersal kernel strength matrix where `D[i,j] = dispersal_kernel_strength(distance[i,j])`. Diagonal elements are 0 (no self-dispersal). Shape: `(patch_num, patch_num)`.

---

#### `normalized_calculate_dispersal_kernel_strength_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `method` | dispersal kernel method |
| `**kwargs` | kernel-specific parameters |

**Returns:** `numpy.matrix`

**Description:**

Returns the row-normalized dispersal kernel matrix. Each row sums to 1, representing the probability distribution of emigrant destinations from a given patch.

---

#### `get_disp_among_rate_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `total_disp_among_rate` | float `m`; the fraction of individuals that emigrate |
| `method` | dispersal kernel method |
| `**kwargs` | kernel-specific parameters |

**Returns:** `numpy.matrix`

**Description:**

Builds the full dispersal rate matrix. Off-diagonal: `m * normalized_kernel[i,j]` (probability that an emigrant from patch i goes to patch j). Diagonal: `1 - m` (probability of staying). Each row sums to 1.

---

#### `get_offspring_marker_pool_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.matrix`

**Description:**

Returns a diagonal matrix with each patch's total offspring marker pool count on the diagonal. Used for calculating among-patch dispersal flows.

---

#### `get_offspring_pool_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.matrix`

**Description:**

Returns a diagonal matrix with each patch's total offspring pool count on the diagonal.

---

#### `get_dormance_pool_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.matrix`

**Description:**

Returns a diagonal matrix with each patch's total dormancy pool count on the diagonal.

---

#### `get_patch_empty_sites_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.matrix`

**Description:**

Returns a diagonal matrix with each patch's total empty microsite count on the diagonal.

---

#### `get_emigrants_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `total_disp_among_rate` | float; emigration fraction |
| `method` | dispersal kernel method |
| `**kwargs` | kernel-specific parameters |

**Returns:** `numpy.matrix`

**Description:**

Calculates expected emigrant numbers from each patch. Formula: `(offspring_pool_num + dormancy_pool_num) * dispersal_rate_matrix`. Element `[i,j]` represents expected emigrants from patch i to patch j.

---

#### `get_immigrants_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `total_disp_among_rate` | float; emigration fraction |
| `method` | dispersal kernel method |
| `**kwargs` | kernel-specific parameters |

**Returns:** `numpy.matrix`

**Description:**

Allocates immigrants across patches proportionally to available empty sites. Formula: `(emigrants / total_emigrants_per_column) * empty_sites_num`. Ensures patches don't receive more immigrants than they have empty sites.

---

#### `get_dispersal_among_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `total_disp_among_rate` | float; emigration fraction |
| `method` | dispersal kernel method |
| `**kwargs` | kernel-specific parameters |

**Returns:** `numpy.matrix`

**Description:**

Computes actual dispersal numbers as `min(emigrants, immigrants)` per element, then probabilistically rounds. Represents the realized number of individuals moving between each pair of patches.

---

### Dispersal Among Patches

#### `dispersal_among_patches_from_offspring_pool_and_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `total_disp_among_rate` | float; emigration fraction |
| `method` | dispersal kernel method (default `'uniform'`) |
| `**kwargs` | kernel-specific parameters |

**Returns:** `str` (log_info)

**Description:**

Moves individuals from offspring + dormancy pools directly into empty microsites of other patches. <span style="color:red">Uses `get_dispersal_among_num_matrix()` which takes `min(emigrants, immigrants)` to cap dispersal by available empty sites or available offspring.</span> Samples migrants from source pools, places them into random empty microsites of target patches. Skips if `patch_num < 2`. Returns a log string with the dispersal count.

---

#### `dispersal_aomng_patches_from_offspring_pool_to_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `total_disp_among_rate` | float; emigration fraction |
| `method` | dispersal kernel method |
| `**kwargs` | kernel-specific parameters |

**Returns:** `str` (log_info)

**Description:**

Moves offspring individual objects from source patches' `offspring_pool` to destination patches' `immigrant_pool` (staged for later local germination or local recolonization). <span style="color:red">Uses `offspring_pool_num * disp_rate_matrix` directly (does NOT cap by empty sites — capping happens later during local germination or local recolonization).</span> Distributes immigrants evenly across habitats within each target patch. Returns a log string with the count.

---

#### `dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `total_disp_among_rate` | float; emigration fraction |
| `method` | dispersal kernel method |
| `**kwargs` | kernel-specific parameters |

**Returns:** `str` (log_info)

**Description:**

Disperses lightweight marker tuples between patches (marker pipeline). <span style="color:red">Uses `offspring_marker_pool_num * disp_rate_matrix` directly (does NOT cap by empty sites).</span> Samples markers from source patches' `offspring_marker_pool` and distributes to destination patches' `immigrant_marker_pool`. Returns a log string with the marker count.

---

#### `dispersal_among_patches_from_offsrping_pool_and_dormancy_pool_to_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `total_disp_among_rate` | float; emigration fraction |
| `method` | dispersal kernel method |
| `**kwargs` | kernel-specific parameters |

**Returns:** `str` (log_info)

**Description:**

Moves individuals from combined offspring + dormancy pools to destination patches' `immigrant_pool` (staged for later germination). <span style="color:red">Uses `get_emigrants_matrix()` directly (does NOT cap by empty sites).</span> Returns a log string with the count.

---

### Dispersal Within Patch

#### `meta_dispersal_within_patch_from_offspring_marker_to_immigrant_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; fraction of offspring markers that disperse to other habitats within the same patch |

**Returns:** `str` (log_info)

**Description:**

Executes within-patch dispersal of offspring markers across all patches. Moves markers from `offspring_marker_pool` to `immigrant_marker_pool` between habitats. Returns a log string with the marker count dispersed.

---

#### `meta_dispersal_within_patch_from_offspring_to_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; within-patch dispersal rate |

**Returns:** `str` (log_info)

**Description:**

Executes within-patch dispersal of offspring individuals across all patches. Moves individuals from `offspring_pool` to `immigrant_pool` between habitats. Returns a log string with the count.

---

#### `meta_dispersal_within_patch_from_offspring_and_dormancy_to_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; within-patch dispersal rate |

**Returns:** `str` (log_info)

**Description:**

Executes within-patch dispersal from combined offspring + dormancy pools to immigrant pools across all patches. Returns a log string with the count.

---

#### `meta_dispersal_within_patch_from_offspring_and_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; within-patch dispersal rate |

**Returns:** `str` (log_info)

**Description:**

Executes within-patch dispersal of offspring + dormancy pool individuals directly into empty microsites across all patches (no intermediate staging in immigrant pool). Returns a log string with the dispersal count and metacommunity state.

---

### Local Germination (or local recolonization)

#### `meta_local_germinate_from_offspring_and_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `str` (log_info)

**Description:**

Places individuals from offspring + dormancy pools into empty microsites across all patches (object pipeline). Returns a log string with the germination count.

---

#### `meta_local_germinate_from_offspring_and_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `str` (log_info)

**Description:**

Places individuals from offspring + immigrant pools into empty microsites across all patches. Returns a log string with the germination count.

---

#### `meta_local_germinate_from_offspring_immigrant_and_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `str` (log_info)

**Description:**

Places individuals from all three pools (offspring, immigrant, dormancy) into empty microsites across all patches. Returns a log string with the germination count.

---

#### `meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `str` (log_info)

**Description:**

Realizes births from the marker pipeline. For each habitat, combines offspring and immigrant markers, shuffles with empty sites, then for each marker reads the birth origin `(birth_patch_id, birth_hab_id, reproduce_mode)` and calls the appropriate reproduction method on the birth habitat to create the actual `individual` object. Supports modes: `'asexual'`, `'sexual'`, `'mix_asexual'`, `'mix_sexual'`. Returns a log string with the birth/germination count.

---

### Dormancy

#### `meta_dormancy_process_from_offspring_pool_to_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `str` (log_info)

**Description:**

Processes dormancy across all patches: moves offspring into dormancy pools subject to capacity constraints. Returns a log string reporting survival count, elimination count, new dormancy entries, total dormancy, and pool sizes.

---

#### `meta_dormancy_process_from_offspring_pool_and_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `str` (log_info)

**Description:**

Processes dormancy from combined offspring + immigrant pools across all patches. Returns a log string with dormancy statistics.

---

### Pool Cleanup

#### `meta_clear_up_offspring_and_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Empties offspring and immigrant pools across all patches by delegating to each patch's cleanup method.

---

#### `meta_clear_up_offspring_marker_and_immigrant_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Empties offspring marker and immigrant marker pools across all patches by delegating to each patch's cleanup method.

---

### Disturbance

#### `meta_disturbance_process_in_patches`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `patch_dist_rate` | float; probability of disturbance per patch (0 to 1) |

**Returns:** `str` (log_info)

**Description:**

Applies patch-level disturbance stochastically. Each patch is disturbed with probability `patch_dist_rate`. Disturbed patches have all individuals killed, pools cleared, and community structure reset. Returns a log string listing which patches were disturbed.

---

#### `meta_disturbance_process_in_habitat`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `hab_dist_rate` | float; probability of disturbance per habitat (0 to 1) |

**Returns:** `str` (log_info)

**Description:**

Applies habitat-level disturbance stochastically. Each habitat within each patch is disturbed independently with probability `hab_dist_rate`. Returns a log string listing which habitats were disturbed.

---

### Data Extraction and Saving

#### `get_meta_microsites_optimum_sp_id_val`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `d` | float; base death rate for survival calculation |
| `w` | float; fitness width parameter |
| `species_2_phenotype_ls` | <span style="color:red">list of preset (mean) phenotype value lists, one per species in an order of sp_id</span> |

**Returns:** `numpy.ndarray` (shape: 1 x total_microsites)

**Description:**

Calculates the optimal species ID (species with highest survival rate) for every <span style="color:red">habitat</span> across the metacommunity based on <span style="color:red">each habitat's mean environment values</span> and species preset phenotype configurations. <span style="color:red">All microsites within a habitat share the same optimal species value.</span> Returns a flattened 1D array reshaped to `(1, total_microsites)`.

---

#### `get_meta_microsite_environment_values`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `environment_name` | string; which environment variable to extract |
| `digits` | int; decimal precision for rounding (default 3) |

**Returns:** `numpy.ndarray` (shape: 1 x total_microsites)

**Description:**

Extracts and rounds environmental values for a specific environment type from all microsites across the metacommunity. Returns a `(1, total_microsites)` array.

---

#### `get_meta_microsites_individuals_sp_id_values`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.ndarray` (shape: 1 x total_microsites)

**Description:**

Extracts numeric species IDs from all microsites across the metacommunity. Uses regex to extract the numeric part from species ID strings. Returns a `(1, total_microsites)` integer array with `NaN` for empty sites.

---

#### `get_meta_microsites_individuals_phenotype_values`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `trait_name` | string; which phenotype to extract |
| `digits` | int; decimal precision for rounding (default 3) |

**Returns:** `numpy.ndarray` (shape: 1 x total_microsites)

**Description:**

Extracts and rounds phenotype values for a specific trait from all microsites across the metacommunity. Returns a `(1, total_microsites)` array with `NaN` for empty sites.

---

#### `columns_patch_habitat_microsites_id`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `tuple` (columns_patch_id, columns_habitat_id, columns_microsite_id)

**Description:**

Generates three arrays of column identifiers for data tables. Each array has length `total_microsites` and contains the patch ID, habitat ID, or microsite ID for every microsite in the metacommunity. Used as column headers when saving data to CSV.

---

#### `meta_distribution_data_all_time_to_csv_gz`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `dis_data_all_time` | numpy array; the data to save (rows = time steps, columns = microsites) |
| `file_name` | string; output file path (typically `.csv.gz`) |
| `index` | list; row index labels (e.g. time step identifiers) |
| `columns` | list; column names |
| `mode` | `'w'` (write/overwrite) or `'a'` (append) |

**Returns:** `pandas.DataFrame`

**Description:**

Saves distribution data to a compressed CSV file (`.csv.gz`). Creates a pandas DataFrame from the data array with the specified index and columns, then writes to disk. Returns the DataFrame.
