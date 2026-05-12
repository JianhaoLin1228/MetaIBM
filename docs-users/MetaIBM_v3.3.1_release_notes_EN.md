# MetaIBM v3.3.1 Release Notes

This version focuses on the update of `experiments/model-sloss-GRFE.py` for CSV-file-based landscape configuration initialization.

## Overview

**MetaIBM v3.3.1** introduces an important update centered on **CSV-file-based landscape initialization** in `model-sloss-GRFE.py`.

This release is important because it:

- allows the metacommunity landscape to be initialized from external CSV files
- supports explicit initialization of patch and habitat layout information
- supports explicit initialization of environmental gradient information
- improves reproducibility and flexibility for landscape-configuration-driven experiments

## Main Changes

### 1. Updated `model-sloss-GRFE.py`

The most important change in this release is the update of `model-sloss-GRFE.py` so that it can read landscape configuration from CSV files.

This means:

- patch and habitat layout information can now be loaded directly from file
- environmental gradient information can now be initialized directly from file
- simulation scenarios can be configured more transparently and reproducibly through external landscape-definition files

### 2. Added support for patch / habitat layout initialization

New landscape configuration input:

- `patch_habitat_layouts.csv`

This file stores:

- the patch layout in the simulated metacommunity landscape
- the habitat layout associated with each patch

This addition is important because:

- the spatial structure of the simulated landscape can now be separated from hard-coded script logic
- users can modify landscape organization without rewriting model code
- experiment design becomes easier to manage and reproduce

### 3. Added support for environmental gradient initialization

New environmental gradient input files:

- `32x32_habitats_env1.csv`
- `32x32_habitats_env2.csv`

These files store:

- the gradient values of environmental axis 1
- the gradient values of environmental axis 2

This addition is important because:

- environmental gradients can now be defined explicitly in external files
- users can initialize landscape-scale environment values in a more controlled way
- future experiments can more easily compare alternative environmental configurations

## Files Included in This Release

### Modified files

- `experiments/model-sloss-GRFE.py`

### New files

- `pexperiments/atch_habitat_layouts.csv`
- `experiments/32x32_habitats_env1.csv`
- `experiments/32x32_habitats_env2.csv`

## Release Value Summary

Compared with the previous state of the project, **v3.3.1** provides value in the following areas:

1. **Functionality**
   
   - supports CSV-file-based initialization of landscape layout and environmental gradients

2. **Reproducibility**
   
   - makes simulation landscape configuration easier to store, reuse, and document

3. **Flexibility**
   
   - allows users to modify patch / habitat layout and environmental gradients without directly editing model code

4. **Experiment support**
   
   - improves the usability of `model-sloss-GRFE.py` for landscape-configuration-driven simulation experiments

## Release Tags

```text
MetaIBM v3.3.1
```

## Short Release Summary

```text
MetaIBM v3.3.1 updates model-sloss-GRFE.py to initialize metacommunity landscape configuration from CSV files, including patch and habitat layout information as well as environmental gradient values for environmental axis 1 and axis 2.
```

## Notes

This release note focuses on the update of `model-sloss-GRFE.py` for **CSV-file-based landscape configuration initialization**.
It is intended to document the transition from script-internal initialization toward more explicit file-based landscape setup in MetaIBM.
