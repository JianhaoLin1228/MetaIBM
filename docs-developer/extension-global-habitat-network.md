# Extension: Global Habitat Network

Source: `extension/global_habitat_network.py`

This extension monkey-patches the `metacommunity` class to add a **global habitat network** — a fully connected network of all habitats across all patches. It enables habitat-level (rather than patch-level) dispersal, where dispersal rates between any two habitats in different patches are determined by a distance-based dispersal kernel.

Installed automatically at the bottom of `metaibm/metacommunity.py`:

```python
from extension.global_habitat_network import install_global_habitat_network_methods
install_global_habitat_network_methods(metacommunity)
```

---

## Instance Attributes (added to `metacommunity`)

| Attribute | Description | Type | e.g. |
|-----------|-------------|------|------|
| `global_habitat_id_ls` | ordered list of all habitat identifiers across the metacommunity; each element is a `(patch_id, habitat_id)` tuple | list | `[('patch1', 'hab1'), ('patch1', 'hab2'), ('patch2', 'hab1')]` |
| `global_habitat_id_2_index_dir` | maps each `(patch_id, habitat_id)` tuple to its row/column index in the distance matrix | dict | `{('patch1', 'hab1'): 0, ('patch1', 'hab2'): 1}` |
| `global_habitat_distance_matrix` | symmetric matrix of Euclidean distances between all pairs of habitats; shape `(N, N)` where `N = len(global_habitat_id_ls)` | np.matrix | |

---

## Methods

### `global_habitat_id_idx_registry`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |
| `patch_object` | a `patch` object whose habitats should be registered |

**Returns:** `int` (0)

**Description:**

Registers all habitats in `patch_object` into the global habitat network. For each habitat in `patch_object.set`, appends `(patch_id, habitat_id)` to `global_habitat_id_ls` and records its index in `global_habitat_id_2_index_dir`. Must be called before updating the distance matrix.

---

### `update_global_habitat_distance_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |

**Returns:** `np.matrix` — the updated distance matrix

**Description:**

Builds the full `global_habitat_distance_matrix` from scratch. Computes the Euclidean distance between every pair of habitats using their `location` attribute `(x, y)`. If no habitats are registered, returns an empty matrix.

---

### `incremental_update_global_habitat_distance_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |

**Returns:** `np.matrix` — the updated distance matrix

**Description:**

Incrementally expands the distance matrix when new habitats have been registered since the last update. Reuses the existing sub-matrix for previously known habitats and only computes distances involving the newly added ones. Falls back to `update_global_habitat_distance_matrix` if the current matrix is empty.

---

### `get_global_habitat_network_dormancy_pool_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |

**Returns:** `np.matrix` — diagonal matrix of shape `(N, N)`

**Description:**

Returns a diagonal matrix where element `[i, i]` is the number of dormancy propagules (`len(habitat.dormancy_pool)`) in the i-th habitat of the global network. Off-diagonal elements are 0.

---

### `get_global_habitat_network_offspring_pool_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |

**Returns:** `np.matrix` — diagonal matrix of shape `(N, N)`

**Description:**

Returns a diagonal matrix where element `[i, i]` is the number of offspring (`len(habitat.offspring_pool)`) in the i-th habitat of the global network.

---

### `get_global_habitat_network_offspring_marker_pool_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |

**Returns:** `np.matrix` — diagonal matrix of shape `(N, N)`

**Description:**

Returns a diagonal matrix where element `[i, i]` is the number of offspring markers (`len(habitat.offspring_marker_pool)`) in the i-th habitat of the global network.

---

### `get_global_habitat_network_empty_sites_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |

**Returns:** `np.matrix` — diagonal matrix of shape `(N, N)`

**Description:**

Returns a diagonal matrix where element `[i, i]` is the number of empty microsites (`len(habitat.empty_site_pos_ls)`) in the i-th habitat of the global network.

---

### `calculate_global_habitat_network_dispersal_kernel_strength_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |
| `method` | str; dispersal kernel method name (default `'uniform'`). Supported: `'uniform'`, `'gaussian'`, `'exponential'`, `'cauchy'`, `'power_law'` |
| `**kwargs` | additional keyword arguments passed to `dispersal_kernel_strength` (e.g. `sigma`, `rho`, `gamma`, `alpha`, `r0`) |

**Returns:** `np.matrix` — unnormalized dispersal strength matrix of shape `(N, N)`

**Description:**

Converts the distance matrix into an unnormalized dispersal kernel strength matrix. For each pair of habitats `(i, j)`:
- If `i == j` (same habitat): strength is 0.
- If habitats are in the same patch: strength is 0 (within-patch dispersal is handled separately).
- Otherwise: strength is `dispersal_kernel_strength(distance_ij, method, **kwargs)`.

---

### `normalized_calculate_global_habitat_network_dispersal_kernel_strength_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |
| `method` | str; dispersal kernel method name (default `'uniform'`) |
| `**kwargs` | additional keyword arguments passed to the kernel |

**Returns:** `np.matrix` — row-normalized dispersal strength matrix of shape `(N, N)`

**Description:**

Returns the dispersal kernel strength matrix normalized so that each row sums to 1. Each element `[i, j]` represents the proportion of dispersal from habitat `i` directed to habitat `j`.

---

### `global_habitat_dispersal_among_rate_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |
| `total_disp_among_rate` | float (0 to 1); the total fraction of propagules that disperse among patches |
| `method` | str; dispersal kernel method name (default `'uniform'`) |
| `**kwargs` | additional keyword arguments passed to the kernel |

**Returns:** `np.matrix` — dispersal rate matrix of shape `(N, N)`

**Description:**

Builds the dispersal rate matrix `M` where:
- Off-diagonal `M[i, j] = total_disp_among_rate * normalized_kernel[i, j]` — the fraction of propagules from habitat `i` that disperse to habitat `j`.
- Diagonal `M[i, i] = 1 - total_disp_among_rate` — the fraction of propagules that remain local.

---

### `get_global_habitat_network_emigrants_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |
| `total_disp_among_rate` | float; total among-patch dispersal rate |
| `method` | str; dispersal kernel method name (default `'uniform'`) |
| `**kwargs` | additional keyword arguments passed to the kernel |

**Returns:** `np.matrix` — emigrants matrix of shape `(N, N)`

**Description:**

Computes the expected number of emigrants from each habitat to every other habitat. Element `[i, j]` is the expected number of propagules dispersing from habitat `i` to habitat `j`. Calculated as:

```
(offspring_pool_num_matrix + dormancy_pool_num_matrix) * dispersal_rate_matrix
```

---

### `get_global_habitat_network_immigrants_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |
| `total_disp_among_rate` | float; total among-patch dispersal rate |
| `method` | str; dispersal kernel method name (default `'uniform'`) |
| `**kwargs` | additional keyword arguments passed to the kernel |

**Returns:** `np.matrix` — immigrants matrix of shape `(N, N)`

**Description:**

Allocates empty microsites in each destination habitat to incoming propagules proportionally. Element `[i, j]` is the number of empty microsites in habitat `j` allocated to propagules from habitat `i`. Calculated as:

```
(emigrants_matrix / column_sum(emigrants_matrix)) * empty_sites_num_matrix
```

---

### `get_global_habitat_network_disp_among_num_matrix`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |
| `total_disp_among_rate` | float; total among-patch dispersal rate |
| `method` | str; dispersal kernel method name (default `'uniform'`) |
| `**kwargs` | additional keyword arguments passed to the kernel |

**Returns:** `np.matrix` — realized dispersal number matrix of shape `(N, N)`

**Description:**

Computes the realized number of propagules that disperse from habitat `i` to habitat `j`. Takes the element-wise minimum of the emigrants matrix and the immigrants matrix (supply vs. available space), then rounds probabilistically via `matrix_around`.

---

### `dispersal_among_patches_in_global_habitat_network_from_offspring_pool_to_immigrant_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |
| `total_disp_among_rate` | float; total among-patch dispersal rate |
| `method` | str; dispersal kernel method name (default `'uniform'`) |
| `**kwargs` | additional keyword arguments passed to the kernel |

**Returns:** `str` — log message

**Description:**

Executes the **object pipeline** dispersal across the global habitat network. For each destination habitat `j`, randomly samples individual objects from the `offspring_pool` of each source habitat `i` (in a different patch) according to the offspring dispersal matrix, then appends them (shuffled) to habitat `j`'s `immigrant_pool`. If the metacommunity has fewer than 2 patches, no dispersal occurs.

---

### `dispersal_among_patches_in_global_habitat_network_from_offspring_marker_pool_to_immigrant_marker_pool`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | metacommunity instance |
| `total_disp_among_rate` | float; total among-patch dispersal rate |
| `method` | str; dispersal kernel method name (default `'uniform'`) |
| `**kwargs` | additional keyword arguments passed to the kernel |

**Returns:** `str` — log message

**Description:**

Executes the **marker pipeline** dispersal across the global habitat network. Same logic as the offspring-pool version, but operates on `offspring_marker_pool` and `immigrant_marker_pool`. Used when dormancy is off for more memory-efficient dispersal.

---

## Installer Function

### `install_global_habitat_network_methods(cls)`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `cls` | the class to install methods onto (typically `metacommunity`) |

**Returns:** the modified class

**Description:**

Monkey-patches all global habitat network attributes and methods onto `cls`:
1. Wraps `cls.__init__` to call `_init_global_habitat_network_fields(self)` after the original `__init__`, adding the three instance attributes.
2. Attaches all methods listed above directly onto the class.
