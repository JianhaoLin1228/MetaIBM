# Class `habitat`

Source: `metaibm/habitat.py`

---

## Attributes

### Instance Attributes

| Attribute | Description |
|-----------|-------------|
| `name` | string identifier of the habitat (e.g. `'h1'`) |
| `index` | integer index of the habitat within its parent patch |
| `location` | tuple `(X, Y)` coordinates of the habitat in the landscape |
| `num_env_types` | number of environmental factors in this habitat |
| `env_types_name` | list of environment type names (e.g. `['temperature', 'altitude']`) |
| `mean_env_ls` | list of mean values for each environment type |
| `var_env_ls` | list of variance (std) values for each environment type; micro-environmental values follow `N(mean, var)` |
| `length` | number of rows in the microsite grid |
| `width` | number of columns in the microsite grid |
| `size` | total number of microsites (`length * width`) |
| `set` | dictionary containing the microsite grid and environment landscapes. Keys include each environment type name (mapped to a `[length x width]` numpy array of environmental values) and `'microsite_individuals'` (a `[length x width]` 2D list where each cell holds an `individual` object or `None`) |
| `indi_num` | current count of individuals occupying microsites |
| `offspring_pool` | list of offspring `individual` objects awaiting germination (object pipeline) |
| `immigrant_pool` | list of immigrant `individual` objects received from dispersal (object pipeline) |
| `dormancy_pool` | list of dormant `individual` objects |
| `offspring_marker_pool` | list of lightweight tuples `(patch_name, hab_name, reproduce_mode)` representing offspring (marker pipeline) |
| `immigrant_marker_pool` | list of lightweight marker tuples received from dispersal (marker pipeline) |
| `species_category` | nested dict `{species_id: {gender: [(row, col), ...]}}` tracking individual locations by species and gender |
| `occupied_site_pos_ls` | list of `(row, col)` tuples for occupied microsites |
| `empty_site_pos_ls` | list of `(row, col)` tuples for empty microsites |
| `dormancy_pool_max_size` | maximum capacity of the dormancy pool |
| `reproduction_mode_threhold` | float (0.897); fitness threshold separating asexual parents (>= threshold) from sexual parents (< threshold) in mixed reproduction mode |
| `asexual_parent_pos_ls` | list of `(row, col)` positions of high-fitness individuals selected for asexual reproduction in mixed mode |
| `species_category_for_sexual_parents_pos` | dict tracking low-fitness individual positions by species/gender for sexual reproduction in mixed mode |

---

## Methods

### Environment

#### `hab_reset_environment_values`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `env_name_ls` | list of environment type names to reset |
| `env_mean_val_ls` | list of new mean values for each specified environment |
| `env_var_val_ls` | list of new variance values for each specified environment |

**Returns:** `int` (0)

**Description:**

Completely regenerates the environmental landscape for specified environment types. For each specified environment, creates a new `[length x width]` numpy array with values sampled from `N(new_mean, new_var)` and updates `mean_env_ls` and `var_env_ls`.

---

#### `hab_offset_environment_values`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `env_name_ls` | list of environment type names to modify |
| `delta_mean_ls` | list of additive offsets to apply to each environment's mean |
| `delta_var_ls` | list of additive offsets to apply to each environment's variance (optional, defaults to no change) |

**Returns:** `int` (0)

**Description:**

Shifts environmental values incrementally by adding delta offsets to the current mean and variance, then calls `hab_reset_environment_values()` with the adjusted values. Used for simulating environmental change over time. <span style="color:red">Note: internally, `new_env_mean_ls` is computed by `zip(self.mean_env_ls, delta_mean_ls)`. This assumes `delta_mean_ls` has the same length and order as `self.mean_env_ls` (i.e., covers all environment types). Passing a subset may produce incorrect pairings.</span>

---

### Individual Management

#### `add_individual`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `indi_object` | an `individual` object to place |
| `len_id` | row index of the target microsite |
| `wid_id` | column index of the target microsite |

**Returns:** `None`

**Description:**

Places an individual object at microsite position `(len_id, wid_id)`. Updates `microsite_individuals` grid, moves the position from `empty_site_pos_ls` to `occupied_site_pos_ls`, increments `indi_num`, and registers the individual in `species_category`. Raises an error if the microsite is already occupied.

---

#### `del_individual`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `len_id` | row index of the microsite |
| `wid_id` | column index of the microsite |

**Returns:** `None`

**Description:**

Removes the individual at microsite position `(len_id, wid_id)`. Sets the grid cell to `None`, moves the position from `occupied_site_pos_ls` to `empty_site_pos_ls`, decrements `indi_num`, and removes the individual from `species_category`. Raises an error if the microsite is already empty.

---

#### `habitat_disturbance_process`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Simulates a complete habitat disturbance event (e.g. fire, flooding). Removes all individuals by resetting the microsite grid to empty, clears all position tracking lists, resets `indi_num` to 0 and `species_category` to `{}`, and clears all reproduction pools and parent lists.

---

### Habitat Queries

#### `get_hab_pairwise_empty_site_pos_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuples

**Description:**

Returns a list of paired empty microsite positions: `[((row1, col1), (row2, col2)), ...]`. Randomly shuffles empty sites and pairs consecutive ones. Returns empty list if fewer than 2 empty sites. Used for sexual colonization which requires placing pairs.

---

#### `get_hab_pairwise_occupied_site_pos_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuples

**Description:**

Returns a list of female-male pairs from occupied microsites: `[((female_row, female_col), (male_row, male_col)), ...]`. Groups individuals by species, separately shuffles females and males within each species, then zips them into pairs. Returns empty list if fewer than 2 individuals.

---

#### `get_microsite_env_val_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `len_id` | row index of the microsite |
| `wid_id` | column index of the microsite |

**Returns:** `list`

**Description:**

Returns a list of environment values at a specific microsite `(len_id, wid_id)`, one value per environment type in the order of `env_types_name`. E.g. `[0.52, 0.78]` for a habitat with two environment types.

---

### Initialization

#### `hab_initialize`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `traits_num` | number of traits per individual |
| `pheno_names_ls` | list of phenotype names |
| `pheno_var_ls` | list of phenotypic variance (std) per trait |
| `geno_len_ls` | list of genotype lengths per trait |
| `reproduce_mode` | reproduction mode string: `'asexual'`, `'sexual'`, or `'mixed'` |
| `species_2_phenotype_ls` | <span style="color:red">list of mean phenotype value lists, one per species; the species ID is derived as `'sp' + str(index_in_list + 1)`</span> |

**Returns:** `int` (0)

**Description:**

Fills all microsites with newly created individuals. For each microsite, creates an `individual` with species ID determined by matching the habitat's `mean_env_ls` to an entry in `species_2_phenotype_ls` <span style="color:red">(via `list.index()`)</span>, gender set to `'female'` if asexual mode or randomly assigned if sexual mode, and randomly initialized genotype/phenotype via `random_init_indi()`. <span style="color:red">Note: `'mixed'` mode is not explicitly handled — only `'asexual'` and `'sexual'` are checked, so passing `'mixed'` may leave `gender` undefined.</span>

---

### Selection

#### `survival_rate`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `d` | float; base death rate (disturbance strength) |
| `phenotype_ls` | list of the individual's phenotype values for each trait |
| `env_val_ls` | list of the microsite's environment values for each type |
| `w` | float; fitness width parameter (default 0.5), controls strength of stabilizing selection |
| `method` | `'niche_gaussian'` (default) or `'neutral_uniform'` |

**Returns:** `float`

**Description:**

Calculates the probability that an individual survives to the next generation. With `'niche_gaussian'` method: `survival = (1-d) * product_over_traits(exp(-((phenotype_i - env_i) / w)^2))`, implementing stabilizing selection toward local environment. With `'neutral_uniform'` method: returns `(1-d)` regardless of phenotype-environment match.

---

#### `hab_dead_selection`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `base_dead_rate` | float; base death rate `d` passed to `survival_rate()` |
| `fitness_wid` | float; fitness width `w` passed to `survival_rate()` |
| `method` | selection method string passed to `survival_rate()` |

**Returns:** `int` (counter)

**Description:**

Applies mortality selection across all occupied microsites. For each individual: calculates survival rate, then kills the individual with probability `(1 - survival_rate)`. Survivors with high fitness (>= 0.897 threshold) are assigned to `asexual_parent_pos_ls`; survivors with lower fitness are assigned to `species_category_for_sexual_parents_pos`. Returns the total number of dead individuals.

---

### Reproduction Helpers (for mainland / dispersal)

#### `hab_asex_reproduce_mutate_with_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait for mutation |
| `num` | int; number of offspring to produce |

**Returns:** `list` of `individual`

**Description:**

Creates `num` asexual offspring by randomly sampling parents from occupied microsites. For each offspring: deep-copies the parent, recalculates phenotype as `mean(genotype) + N(0, var)`, then applies mutation. Returns the list of new individual objects.

---

#### `hab_sexual_pairwise_parents_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuples

**Description:**

Returns a list of female-male parent pairs `[(female_pos, male_pos), ...]` from `species_category`. For each species, randomly shuffles females and males independently and zips them into pairs.

---

#### `hab_sexual_pairwise_parents_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the number of available sexual breeding pairs (i.e. `len(hab_sexual_pairwise_parents_ls())`).

---

#### `hab_sex_reproduce_mutate_with_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait for mutation |
| `num` | int; number of offspring to produce |

**Returns:** `list` of `individual`

**Description:**

Creates `num` sexual offspring from randomly sampled parent pairs. For each offspring: deep-copies the female parent, randomly assigns gender, inherits one haploid genotype from each parent (diploid recombination), recalculates phenotype from the new genotype `mean(genotype) + N(0, var)`, then applies mutation. Returns the list of new individual objects.

---

#### `hab_mixed_sexual_pairwise_parents_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list` of tuples

**Description:**

Returns female-male parent pairs only from low-fitness individuals stored in `species_category_for_sexual_parents_pos` (individuals below the 0.897 threshold). Used exclusively in mixed reproduction mode.

---

#### `hab_mixed_sexual_pairwse_parents_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the number of available sexual breeding pairs from low-fitness individuals in mixed mode.

---

#### `hab_mixed_asexual_parent_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int`

**Description:**

Returns the number of high-fitness asexual parents in `asexual_parent_pos_ls` (individuals at or above the 0.897 threshold in mixed mode).

---

#### `hab_mix_asex_reproduce_mutate_with_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait for mutation |
| `num` | int; number of offspring to produce |

**Returns:** `list` of `individual`

**Description:**

Creates `num` asexual offspring from high-fitness parents in `asexual_parent_pos_ls`. Same mechanism as `hab_asex_reproduce_mutate_with_num` but restricted to high-fitness parents only (mixed mode).

---

#### `hab_mix_sex_reproduce_mutate_with_num`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait for mutation |
| `num` | int; number of offspring to produce |

**Returns:** `list` of `individual`

**Description:**

Creates `num` sexual offspring from low-fitness parent pairs in `species_category_for_sexual_parents_pos`. Same mechanism as `hab_sex_reproduce_mutate_with_num` but restricted to low-fitness parents only (mixed mode).

---

### Direct Reproduction + Germination

#### `hab_asexual_reprodece_germinate`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per individual |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (counter)

**Description:**

Performs asexual reproduction and immediately places offspring into empty microsites (no pooling). The number of offspring is `min(empty_sites, indi_num * asexual_birth_rate)`. Offspring and empty sites are randomly shuffled and matched. Returns the count of germinated offspring.

---

#### `hab_sexual_reprodece_germinate`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `sexual_birth_rate` | float; birth rate per breeding pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (counter)

**Description:**

Performs sexual reproduction and immediately places offspring into empty microsites. The number of offspring is `min(empty_sites, pairwise_parents_num * sexual_birth_rate)`. Returns the count of germinated offspring.

---

#### `hab_mixed_reproduce_germinate`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per asexual parent (high-fitness) |
| `sexual_birth_rate` | float; birth rate per sexual breeding pair (low-fitness) |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (counter)

**Description:**

Performs mixed asexual + sexual reproduction and immediately germinates offspring. Calculates expected offspring from both pathways <span style="color:red">(using `np.around()` for rounding instead of probabilistic rounding)</span>; if not enough empty sites, scales both proportionally. <span style="color:red">Unlike the asexual and sexual versions, this method does not explicitly shuffle the empty sites list before placement.</span> Returns total count of germinated individuals.

---

### Reproduction into Offspring Marker Pool

#### `hab_asex_reproduce_calculation_into_offspring_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `patch_name` | string; name of the parent patch (stored in the marker tuple) |
| `asexual_birth_rate` | float; birth rate per individual |

**Returns:** `int` (pool size)

**Description:**

Calculates the expected number of asexual offspring (`indi_num * asexual_birth_rate`) with probabilistic rounding of the fractional part, then creates that many lightweight marker tuples `(patch_name, hab_name, 'asexual')` in `offspring_marker_pool`. Actual individual creation is deferred until germination. Returns the marker count.

---

#### `hab_sex_reproduce_calculation_into_offspring_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `patch_name` | string; name of the parent patch |
| `sexual_birth_rate` | float; birth rate per breeding pair |

**Returns:** `int` (pool size)

**Description:**

Calculates the expected number of sexual offspring (<span style="color:red">`indi_num * sexual_birth_rate`</span>) with probabilistic rounding, then creates marker tuples `(patch_name, hab_name, 'sexual')` in `offspring_marker_pool`. Returns the marker count. <span style="color:red">Note: unlike `hab_sex_reproduce_mutate_into_offspring_pool` which uses `pairwise_parents_num`, this method uses `indi_num` as the base.</span>

---

#### `hab_mix_reproduce_calculation_into_offspring_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `patch_name` | string; name of the parent patch |
| `asexual_birth_rate` | float; birth rate per high-fitness asexual parent |
| `sexual_birth_rate` | float; birth rate per low-fitness sexual pair |

**Returns:** `int` (pool size)

**Description:**

Calculates expected offspring from both asexual (high-fitness) and sexual (low-fitness) pathways with probabilistic rounding. Creates marker tuples with modes `'mix_asexual'` and `'mix_sexual'` respectively in `offspring_marker_pool`. Returns total marker count.

---

### Reproduction into Offspring Pool (Object Pipeline)

#### `hab_asex_reproduce_mutate_into_offspring_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per individual |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (pool size)

**Description:**

Creates asexual offspring as full `individual` objects with mutations and stores them in `offspring_pool`. The offspring count is `indi_num * asexual_birth_rate` with probabilistic rounding of the fractional part. Returns the number of individuals in the pool.

---

#### `hab_sex_reproduce_mutate_into_offspring_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `sexual_birth_rate` | float; birth rate per breeding pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (pool size)

**Description:**

Creates sexual offspring as full `individual` objects with mutations and stores them in `offspring_pool`. The offspring count is based on `pairwise_parents_num * sexual_birth_rate` with probabilistic rounding. Returns the number of individuals in the pool.

---

#### `hab_mix_reproduce_mutate_into_offspring_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `asexual_birth_rate` | float; birth rate per high-fitness asexual parent |
| `sexual_birth_rate` | float; birth rate per low-fitness sexual pair |
| `mutation_rate` | float; per-locus mutation probability |
| `pheno_var_ls` | list of phenotypic variance per trait |

**Returns:** `int` (pool size)

**Description:**

Creates both asexual offspring (from high-fitness parents) and sexual offspring (from low-fitness parents) as full `individual` objects with mutations and stores them all in `offspring_pool`. Each pathway uses probabilistic rounding. Returns total pool size.

---

### Local Germination

#### `hab_local_germinate_from_offspring_and_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int` (counter)

**Description:**

Places individuals from both `offspring_pool` and `dormancy_pool` into empty microsites. Combines both pools, randomly shuffles both empty sites and candidate individuals, then matches them by position. Returns the count of successfully germinated individuals.

---

#### `hab_local_germinate_from_offspring_and_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int` (counter)

**Description:**

Places individuals from both `offspring_pool` and `immigrant_pool` into empty microsites. Combines both pools, randomly shuffles, and matches to empty sites. Returns the count of successfully germinated individuals.

---

#### `hab_local_germinate_from_offspring_immigrant_and_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `int` (counter)

**Description:**

Places individuals from all three pools (`offspring_pool`, `immigrant_pool`, `dormancy_pool`) into empty microsites. The most comprehensive germination method. Returns the count of successfully germinated individuals.

---

### Dormancy

#### `hab_dormancy_process_from_offspring_pool_to_dormancy_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `tuple` (survival_in_dormancy_pool_num, eliminate_from_dormancy_pool_num, offspring_num, dormancy_pool_size)

**Description:**

Moves offspring into the dormancy pool subject to capacity constraints. Handles four scenarios:
1. `dormancy_pool_max_size == 0`: no dormancy, all offspring discarded.
2. Offspring count exceeds max: samples `max_size` individuals from offspring as new dormancy pool<span style="color:red">; the old dormancy pool is entirely replaced</span>.
3. Offspring + existing dormancy <= max: adds all offspring to dormancy pool<span style="color:red">; old dormancy individuals are all kept</span>.
4. Offspring + existing dormancy > max: <span style="color:red">randomly samples `(max_size - offspring_num)` survivors from the old dormancy pool, then adds all new offspring</span>.

Clears offspring and immigrant pools afterward. <span style="color:red">Note: the return value `eliminate_from_dormancy_pool_num` has inconsistent semantics across scenarios — in scenario 2 it equals `dormancy_pool_max_size` (not the old pool size), so it does not strictly represent the count of eliminated old pool members.</span>

---

#### `hab_dormancy_process_from_offspring_pool_and_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `tuple` (survival_in_dormancy_pool_num, eliminate_from_dormancy_pool_num, new_dormancy_num, dormancy_pool_size)

**Description:**

Combines offspring and immigrant pools into the dormancy pool subject to capacity constraints. Handles multiple priority scenarios: offspring first, then immigrants, then existing dormancy pool survivors. Returns a tuple tracking how many survived, were eliminated, newly entered dormancy, and the final pool size.

---

### Pool Cleanup

#### `hab_clear_up_offspring_and_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Empties both `offspring_pool` and `immigrant_pool` by resetting them to empty lists.

---

#### `hab_clear_up_offspring_marker_and_immigrant_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `None`

**Description:**

Empties both `offspring_marker_pool` and `immigrant_marker_pool` by resetting them to empty lists.
