# Class `individual`

Source: `metaibm/individual.py`

---

## Attributes

### Instance Attributes

| Attribute | Description | Type | e.g. |
|-----------|-------------|-----------|-----------|
| `species_id` | species identifier of the individual object | str | `'sp1'`, `'sp2'`, `'sp3'` |
| `gender` | gender of the individual object for sexual reproduction | str | `'female'` or `'male'` |
| `traits_num` | number of traits the individual has | int | 2 |
| `pheno_names_ls` | a list of trait (or phenotype) names, length equals `traits_num` | list | `['phenotype1', 'phenotype2']` |
| `genotype_set` | dictionary mapping each phenotype name to a diploid genotype (list of two binary numpy arrays). Each array has length `geno_len` with values 0 or 1. Initialized to `None` until `random_init_indi()` is called. | dict | `{'phenotype1': [array([1,0,1,...]), array([0,1,0,...])]}` |
| `phenotype_set` | dictionary mapping each phenotype name to its phenotype value (float). Initialized to `None` until `random_init_indi()` is called. | dict | `{'phenotype1': 0.523, 'phenotype2': 0.641}` |
| `age` | age of the individual object with `time_step` as units | int | 0, 1, ..., or 5000, ... |

---

## Methods

### `random_init_indi`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `mean_pheno_val_ls` | list of floats; the mean phenotype values for each trait in the species population (e.g. `[0.5, 0.6]`) |
| `pheno_var_ls` | list of floats; the standard deviation of phenotypic Gaussian noise for each trait (e.g. `[0.1, 0.15]`) |
| `geno_len_ls` | list of ints; the number of loci per haploid genotype for each trait (e.g. `[10, 20]`) |

**Returns:** `int` (0)

**Description:**

Randomly initializes the individual's genotype and phenotype for all traits. For each trait:
1. Creates two independent haploid genotype arrays (diploid), each of length `geno_len`. The number of 1-alleles in each array is `int(mean * geno_len)`, with positions chosen randomly.
2. Stores the two arrays as `[genotype_1, genotype_2]` in `genotype_set[pheno_name]`.
3. Samples the phenotype from a Gaussian distribution: `phenotype = mean + N(0, var)` and stores it in `phenotype_set[pheno_name]`.

---

### `get_indi_phenotype_ls`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |

**Returns:** `list`

**Description:**

Returns an ordered list of the individual's phenotype values, in the same order as `pheno_names_ls`. E.g. `[0.523, 0.641]` for a two-trait individual.

---

### `mutation`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `self` | self |
| `rate` | float (0 to 1); per-locus mutation probability. Each allele in both haploid genotypes is independently flipped (0->1 or 1->0) with this probability. |
| `pheno_var_ls` | list of floats; the standard deviation of Gaussian noise added when recalculating phenotype after mutation, one per trait |

**Returns:** `int` (0)

**Description:**

Applies stochastic point mutations to the individual's genotypes and recalculates phenotypes if any mutations occurred. For each trait:
1. Iterates over every locus in both haploid genotype arrays. Each locus is flipped (0<->1) independently with probability `rate`.
2. If at least one mutation occurred in either genotype for that trait, the phenotype is recalculated as: `phenotype = mean(both_genotypes) + N(0, var)`, where `mean(both_genotypes)` is the average allele value across all loci in both haploid arrays.
