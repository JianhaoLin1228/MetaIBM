# Class `patch`

Source: `metaibm/patch.py`

---

## Attributes

### Instance Attributes

| Attribute | Description |
|-----------|-------------|
| `name` | string identifier of the patch (e.g. `'patch0'`) |
| `index` | integer index of the patch within the metacommunity |
| `set` | dictionary storing habitat objects, keyed by habitat name (e.g. `{'h1': habitat_obj, 'h2': habitat_obj}`) |
| `hab_num` | count of habitats in the patch (initialized to 0) |
| `hab_id_ls` | ordered list of habitat name identifiers |
| `location` | tuple `(X, Y)` coordinates of the patch in the landscape |

---

## Methods

### General

#### `get_data`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `dict`

**Description:**

Returns a dictionary aggregating data from all habitats in the patch. Each key is a habitat name and each value is the habitat's internal data set (`habitat.set`).

---

#### `add_habitat`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `hab_name` | string name for the new habitat |
| `hab_index` | integer index for the habitat within this patch |
| `hab_location` | tuple `(X, Y)` location of the habitat |
| `num_env_types` | number of environment types |
| `env_types_name` | list of environment type names |
| `mean_env_ls` | list of mean values for each environment type |
| `var_env_ls` | list of variance values for each environment type |
| `length` | number of rows in the habitat's microsite grid |
| `width` | number of columns in the habitat's microsite grid |
| `dormancy_pool_max_size` | maximum capacity of the habitat's dormancy pool |

**Returns:** `None`

**Description:**

Creates a new `habitat` object with the specified parameters and adds it to `self.set`. Increments `hab_num` and appends the habitat name to `hab_id_ls`.

---

#### `patch_initialize`

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

**Returns:** `int` (0)

**Description:**

Initializes all habitats in the patch by calling each habitat's `hab_initialize()` with the given trait configuration. Fills all microsites with individuals.

---

### Selection

#### `patch_dead_selection`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `base_dead_rate` | float; base death rate |
| `fitness_wid` | float; fitness width parameter |
| `method` | selection method string (e.g. `'niche_gaussian'`, `'neutral_uniform'`) |

**Returns:** `int` (counter)

**Description:**

Applies mortality selection across all habitats in the patch by calling each habitat's `hab_dead_selection()`. Returns the total number of dead individuals across all habitats.

---

### Patch Queries

#### `get_patch_individual_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the total number of living individuals across all habitats by summing each habitat's `indi_num`.

---

#### `get_patch_empty_sites_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuples (h_id, len_id, wid_id)

**Description:**

Returns a list of all empty microsite positions across all habitats. Each tuple contains the habitat name and the row/column indices of the empty microsite.

---

#### `patch_empty_sites_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the total count of empty microsites across all habitats in the patch.

---

#### `get_patch_pairwise_empty_sites_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuple pairs

**Description:**

Returns a list of paired empty microsite positions `[((h_id, row, col), (h_id, row, col)), ...]` across all habitats. Used for sexual colonization requiring paired placement.

---

#### `get_patch_pairwise_occupied_sites_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuple pairs

**Description:**

Returns a list of female-male paired occupied positions across all habitats. Used for sexual reproduction pairing.

---

#### `get_patch_occupied_sites_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuples (h_id, len_id, wid_id)

**Description:**

Returns a list of all occupied microsite positions across all habitats. Each tuple contains the habitat name and the row/column indices.

---

#### `get_patch_offspring_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of markers

**Description:**

Returns the combined list of all offspring marker tuples from all habitats' `offspring_marker_pool`.

---

#### `get_patch_offspring_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of `individual`

**Description:**

Returns the combined list of all offspring `individual` objects from all habitats' `offspring_pool`.

---

#### `get_patch_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of `individual`

**Description:**

Returns the combined list of all dormant `individual` objects from all habitats' `dormancy_pool`.

---

#### `get_patch_offspring_and_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of `individual`

**Description:**

Returns the combined list of all offspring and dormant `individual` objects from all habitats.

---

### Data Extraction

#### `get_patch_microsites_individals_sp_id_values`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.ndarray` (shape: hab_num x hab_size)

**Description:**

Extracts numeric species ID values from all microsites across the patch. Uses regex to extract the numeric part from species ID strings (e.g. `'sp1'` -> `1`). Returns a 2D numpy array with `NaN` for empty microsites.

---

#### `get_patch_microsites_individals_phenotype_values`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `trait_name` | string; the phenotype name to extract (e.g. `'phenotype1'`) |

**Returns:** `numpy.ndarray` (shape: hab_num x hab_size)

**Description:**

Extracts phenotype values for a specific trait from all microsites across the patch. Returns a 2D numpy array with `NaN` for empty microsites.

---

#### `get_patch_microsites_environment_values`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `environment_name` | string; the environment type name to extract |

**Returns:** `numpy.ndarray` (shape: hab_num x hab_size)

**Description:**

Extracts environmental values for a specific environment type from all microsites across the patch. Returns a 2D numpy array reshaped by habitat count.

---

#### `get_patch_microsites_optimum_sp_id_value_array`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `d` | float; base death rate for survival calculation |
| `w` | float; fitness width parameter |
| `species_2_phenotype_ls` | <span style="color:red">list of mean phenotype value lists, one per species</span> |

**Returns:** `numpy.ndarray` (shape: hab_num x hab_size)

**Description:**

Calculates the optimal species ID (highest survival rate) for each <span style="color:red">habitat</span> based on <span style="color:red">the habitat's mean environment values (`mean_env_ls`)</span> and all species' phenotype configurations. <span style="color:red">The calculation is per-habitat (not per-microsite): all microsites within a habitat share the same optimal species value.</span> Returns a 2D array of optimal species IDs per habitat.

---

### Environment

#### `patch_offset_environmental_values`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `env_name_ls` | list of environment type names to modify |
| `delta_mean_ls` | list of additive offsets for environment means |
| `delta_var_ls` | list of additive offsets for environment variances (optional) |

**Returns:** `int` (0)

**Description:**

Shifts environmental values across all habitats in the patch by calling each habitat's `hab_offset_environment_values()` with the specified deltas.

---

### Direct Reproduction + Germination (Mainland)

#### `patch_asexual_birth_germinate`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per individual |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (counter)

**Description:**

Performs asexual reproduction and immediate germination across all habitats by delegating to each habitat's `hab_asexual_reprodece_germinate()`. Returns the total birth count.

---

#### `patch_sexual_birth_germinate`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `sexual_birth_rate` | float; birth rate per breeding pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (counter)

**Description:**

Performs sexual reproduction and immediate germination across all habitats. Returns the total birth count.

---

#### `patch_mixed_birth_germinate`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per high-fitness asexual parent |
| `sexual_birth_rate` | float; birth rate per low-fitness sexual pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (counter)

**Description:**

Performs mixed asexual + sexual reproduction and immediate germination across all habitats. Returns the total birth count.

---

### Reproduction into Offspring Marker Pool

#### `patch_asex_reproduce_calculation_into_offspring_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per individual |

**Returns:** `int` (counter)

**Description:**

Generates asexual offspring markers across all habitats by delegating to each habitat's marker pool method. Stores lightweight tuples in each habitat's `offspring_marker_pool`. Returns total marker count.

---

#### `patch_sex_reproduce_calculation_into_offspring_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `sexual_birth_rate` | float; birth rate per breeding pair |

**Returns:** `int` (counter)

**Description:**

Generates sexual offspring markers across all habitats. Returns total marker count.

---

#### `patch_mix_reproduce_calculation_into_offspring_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per high-fitness asexual parent |
| `sexual_birth_rate` | float; birth rate per low-fitness sexual pair |

**Returns:** `int` (counter)

**Description:**

Generates mixed reproduction offspring markers across all habitats. Returns total marker count.

---

### Reproduction into Offspring Pool (Object Pipeline)

#### `patch_asex_reproduce_mutate_into_offspring_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per individual |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (counter)

**Description:**

Creates asexual offspring as full `individual` objects with mutations and stores in each habitat's `offspring_pool`. Returns total offspring count across all habitats.

---

#### `patch_sex_reproduce_mutate_into_offspring_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `sexual_birth_rate` | float; birth rate per breeding pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (counter)

**Description:**

Creates sexual offspring as full `individual` objects with mutations and stores in each habitat's `offspring_pool`. Returns total offspring count.

---

#### `patch_mix_reproduce_mutate_into_offspring_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per high-fitness asexual parent |
| `sexual_birth_rate` | float; birth rate per low-fitness sexual pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (counter)

**Description:**

Creates both asexual and sexual offspring as full `individual` objects with mutations and stores in each habitat's `offspring_pool`. Returns total offspring count.

---

### Dispersal Among Patches into Habitat Pools

#### `split`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `ls` | list to split |
| `n` | int; number of sublists to create |

**Returns:** `list` of lists

**Description:**

Splits a list into `n` roughly equal sublists after shuffling. Helper method for distributing individuals or markers evenly among habitats.

---

#### `migrants_to_patch_into_habs_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `migrants_indi_obj_to_patch_ls` | list of migrant `individual` objects arriving at this patch |

**Returns:** `int` (0)

**Description:**

Distributes migrant individual objects arriving at the patch evenly across habitats' `immigrant_pool` using the `split()` helper.

---

#### `migrants_marker_to_patch_into_habs_immigrant_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `migrants_to_j_offspring_marker_ls` | list of migrant marker tuples arriving at this patch |

**Returns:** `int` (0)

**Description:**

Distributes migrant marker tuples arriving at the patch evenly across habitats' `immigrant_marker_pool` using the `split()` helper.

---

### Dispersal Within Patch

#### `patch_matrix_around`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `matrix` | numpy matrix with potentially fractional values |

**Returns:** `numpy.matrix`

**Description:**

Probabilistically rounds fractional values in a matrix. For each element, the decimal part is treated as a probability of rounding up (+1). `NaN` values become 0. Returns the rounded integer matrix.

---

#### `get_patch_dispersal_within_rate_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; fraction of offspring that disperse to other habitats within the patch |

**Returns:** `numpy.matrix`

**Description:**

Generates an N x N dispersal rate matrix (N = number of habitats). Diagonal elements are `(1 - disp_within_rate)` (probability of staying), off-diagonal elements are `disp_within_rate / (N - 1)` (probability of dispersing to each other habitat). Represents uniform within-patch dispersal.

---

#### `get_patch_habs_offspring_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.matrix`

**Description:**

Returns a diagonal matrix with each habitat's offspring pool count on the diagonal. Used for calculating within-patch dispersal flows.

---

#### `get_patch_habs_offspring_marker_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.matrix`

**Description:**

Returns a diagonal matrix with each habitat's offspring marker pool count on the diagonal.

---

#### `get_patch_habs_dormancy_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.matrix`

**Description:**

Returns a diagonal matrix with each habitat's dormancy pool count on the diagonal.

---

#### `get_patch_habs_empty_sites_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `numpy.matrix`

**Description:**

Returns a diagonal matrix with each habitat's empty microsite count on the diagonal.

---

#### `get_habs_emigrants_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; dispersal rate within the patch |

**Returns:** `numpy.matrix`

**Description:**

Calculates the expected number of emigrants from each habitat based on `(offspring + dormancy) * dispersal_rate_matrix`.

---

#### `get_habs_immigrants_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; dispersal rate within the patch |

**Returns:** `numpy.matrix`

**Description:**

Allocates immigrants to habitats proportionally to their available empty sites. Distributes emigrant flow based on `(emigrants / total_emigrants_per_column) * empty_sites`.

---

#### `get_dispersal_within_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; dispersal rate within the patch |

**Returns:** `numpy.matrix`

**Description:**

Computes the final within-patch dispersal matrix as `min(emigrants, immigrants)` per element, then probabilistically rounds. Represents the actual number of individuals moving between each pair of habitats.

---

#### `patch_dipersal_within_from_offspring_and_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; dispersal rate within the patch |

**Returns:** `int` (counter)

**Description:**

Executes within-patch dispersal from offspring and dormancy pools directly into empty microsites of target habitats. Samples migrants from source habitats' combined pools and places them into target habitats. Returns total number of dispersed individuals.

---

#### `patch_dispersal_within_from_offspring_marker_pool_to_immigrant_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; dispersal rate within the patch |

**Returns:** `int` (counter)

**Description:**

Executes within-patch dispersal of offspring markers. Samples markers from source habitats' `offspring_marker_pool` and adds them to target habitats' `immigrant_marker_pool`. Returns total markers dispersed.

---

#### `patch_dispersal_within_from_offspring_pool_to_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; dispersal rate within the patch |

**Returns:** `int` (counter)

**Description:**

Executes within-patch dispersal of offspring individuals from `offspring_pool` to target habitats' `immigrant_pool`. Returns total individuals dispersed.

---

#### `patch_dispersal_within_from_offspring_pool_and_dormancy_pool_to_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `disp_within_rate` | float; dispersal rate within the patch |

**Returns:** `int` (counter)

**Description:**

Executes within-patch dispersal from combined offspring and dormancy pools to target habitats' `immigrant_pool`. Returns total individuals dispersed.

---

### Local Germination

#### `patch_local_germinate_from_offspring_and_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int` (counter)

**Description:**

Germinates individuals from offspring and dormancy pools into empty microsites across all habitats. Delegates to each habitat's germination method. Returns total germination count.

---

#### `patch_local_germinate_from_offspring_and_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int` (counter)

**Description:**

Germinates individuals from offspring and immigrant pools into empty microsites across all habitats (competition between local and external colonists). Returns total germination count.

---

#### `patch_local_germinate_from_offspring_immigrant_and_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int` (counter)

**Description:**

Germinates individuals from all three pools (offspring, immigrant, dormancy) into empty microsites across all habitats. Returns total germination count.

---

### Dormancy

#### `patch_dormancy_process_from_offspring_pool_to_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `tuple` (survival_counter, eliminate_counter, new_dormancy_counter, all_dormancy_num)

**Description:**

Processes dormancy across all habitats: moves offspring into dormancy pools subject to capacity constraints. Aggregates counts from each habitat. Returns a tuple of totals: (survivors from old pool, eliminated from old pool, new dormancy entries, final total dormancy count).

---

#### `patch_dormancy_process_from_offspring_pool_and_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `tuple` (survival_counter, eliminate_counter, new_dormancy_counter, all_dormancy_num)

**Description:**

Processes dormancy from combined offspring and immigrant pools across all habitats. Returns the same tuple format as above.

---

### Pool Cleanup

#### `patch_clear_up_offspring_and_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Clears offspring and immigrant pools across all habitats by delegating to each habitat's cleanup method.

---

#### `patch_clear_up_offspring_marker_and_immigrant_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Clears offspring marker and immigrant marker pools across all habitats by delegating to each habitat's cleanup method.

---

### Disturbance

#### `patch_disturbance_process`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Executes a complete disturbance event across all habitats in the patch. Delegates to each habitat's `habitat_disturbance_process()`, which kills all individuals, clears all pools, and resets community structure.
