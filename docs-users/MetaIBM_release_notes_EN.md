# MetaIBM Release Notes

---

## v3.3.1

This version focuses on the update of `experiments/model-sloss-GRFE.py` for CSV-file-based landscape configuration initialization.

### Overview

**MetaIBM v3.3.1** introduces an important update centered on **CSV-file-based landscape initialization** in `model-sloss-GRFE.py`.

This release is important because it:

- allows the metacommunity landscape to be initialized from external CSV files
- supports explicit initialization of patch and habitat layout information
- supports explicit initialization of environmental gradient information
- improves reproducibility and flexibility for landscape-configuration-driven experiments

### Main Changes

#### 1. Updated `model-sloss-GRFE.py`

The most important change in this release is the update of `model-sloss-GRFE.py` so that it can read landscape configuration from CSV files.

This means:

- patch and habitat layout information can now be loaded directly from file
- environmental gradient information can now be initialized directly from file
- simulation scenarios can be configured more transparently and reproducibly through external landscape-definition files

#### 2. Added support for patch / habitat layout initialization

New landscape configuration input:

- `patch_habitat_layouts.csv`

This file stores:

- the patch layout in the simulated metacommunity landscape
- the habitat layout associated with each patch

This addition is important because:

- the spatial structure of the simulated landscape can now be separated from hard-coded script logic
- users can modify landscape organization without rewriting model code
- experiment design becomes easier to manage and reproduce

#### 3. Added support for environmental gradient initialization

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

### Files Included in This Release

#### Modified files

- `experiments/model-sloss-GRFE.py`

#### New files

- `pexperiments/atch_habitat_layouts.csv`
- `experiments/32x32_habitats_env1.csv`
- `experiments/32x32_habitats_env2.csv`

### Release Value Summary

Compared with the previous state of the project, **v3.3.1** provides value in the following areas:

1. **Functionality**

   - supports CSV-file-based initialization of landscape layout and environmental gradients

2. **Reproducibility**

   - makes simulation landscape configuration easier to store, reuse, and document

3. **Flexibility**

   - allows users to modify patch / habitat layout and environmental gradients without directly editing model code

4. **Experiment support**

   - improves the usability of `model-sloss-GRFE.py` for landscape-configuration-driven simulation experiments

### Short Release Summary

```text
MetaIBM v3.3.1 updates model-sloss-GRFE.py to initialize metacommunity landscape configuration from CSV files, including patch and habitat layout information as well as environmental gradient values for environmental axis 1 and axis 2.
```

### Notes

This release note focuses on the update of `model-sloss-GRFE.py` for **CSV-file-based landscape configuration initialization**.
It is intended to document the transition from script-internal initialization toward more explicit file-based landscape setup in MetaIBM.

---

## v3.3.0

> This version focuses on the `global-habitat-network` workflow and its extension-based installation into `metaibm/metacommunity.py`.

### Overview

**MetaIBM v3.3.0** introduces a significant update centered on the **global-habitat-network** workflow as an extension-based habitat-level dispersal module.

This release is important because it:

- introduces the `global-habitat-network` workflow for habitat-level dispersal across the whole landscape
- provides a dedicated extension module for the new functionality
- supports lightweight installation of extension methods into `metaibm/metacommunity.py`
- continues the package-oriented and extension-oriented development direction of MetaIBM

### Main Changes

#### 1. global-habitat-network workflow

The most important change in this release is the introduction of the **global-habitat-network** workflow.

This means:

- dispersal can now be represented explicitly among habitats across the whole metacommunity landscape (with habitats as nodes)
- habitat-level connectivity can be constructed through a global habitat registry and distance-based kernel workflow
- habitat-network-based dispersal can be integrated into the metacommunity simulation pipeline

Main capabilities include:

- global habitat registry and indexing
- global habitat distance matrix construction
- habitat-level dispersal kernel strength calculation
- habitat-level migration-rate matrix construction
- offspring-marker redistribution through the global habitat network

#### 2. Added extension module: `extension/global_habitat_network.py`

New extension module:

- `extension/global_habitat_network.py`

This addition is important because:

- the habitat-network workflow is implemented as an optional extension rather than forcing changes into all core code paths
- the project keeps its modular structure by separating advanced dispersal logic into the `extension/` directory
- future development can continue through extension-oriented growth without overloading the base `metaibm` package

From a release and maintenance perspective, this indicates that:

- MetaIBM is expanding toward extension-based feature installation
- habitat-network dispersal is treated as a specialized but reusable module
- future extensions can follow the same installation pattern

#### 3. Extension installation in `metaibm/metacommunity.py`

To install the extension methods into the core package workflow, the following lines should be added in `metaibm/metacommunity.py`:

```python
#******************************* install extension module
from extension.global_habitat_network import install_global_habitat_network_methods
install_global_habitat_network_methods(metacommunity)
```

This installation pattern is important because:

- it keeps the extension loading process explicit and lightweight
- it makes the `metacommunity` class the main integration point for the habitat-network workflow
- users can install advanced functionality only when needed

After installation, the `metacommunity` class is extended with the habitat-network-related methods defined in `extension/global_habitat_network.py`.

#### 4. Method compatibility

The global-habitat-network extension is intended to work with the dispersal-kernel-based workflow already supported in MetaIBM.

Relevant methods include:

- `uniform`
- `gaussian (sigma)`
- `exponential (rho)`
- `cauchy`
- `power_law`

In **v3.3.0**, the main emphasis is the integration of **habitat-level network dispersal** into the extension-based architecture.

### Files Included in This Release

#### New files

- `extension/global_habitat_network.py`

### Release Value Summary

Compared with the previous state of the project, **v3.3.0** provides value in the following areas:

1. **Functionality**

   - introduces global habitat network dispersal at the habitat level

2. **Modularity**

   - implements the new workflow as an extension module

3. **Installation clarity**

   - defines an explicit extension installation pattern in `metaibm/metacommunity.py`

4. **Future development**

   - strengthens the extension-oriented architecture for future feature growth

### Short Release Summary

```text
MetaIBM v3.3.0 introduces the global-habitat-network workflow as an extension-based habitat-level dispersal module, adds the dedicated extension module extension/global_habitat_network.py, and supports explicit installation into metaibm/metacommunity.py through install_global_habitat_network_methods(metacommunity).
```

### Notes

This release note focuses on the **global-habitat-network** extension workflow and its installation pattern.

---

## v3.2.0

> This document is based on the changes that were explicitly confirmed in this conversation as already merged into `main`.
> The key confirmed change included in the main branch is the merge of the `feature/dispersal-kernel` branch into `main`.

### Overview

**MetaIBM v3.2.0** introduces a significant update centered on **dispersal-kernel-related functionality**, together with supporting experiment and test scripts.

This release is important because it:

- officially integrates the `feature/dispersal-kernel` branch into `main`
- extends metacommunity-level logic related to dispersal behavior
- adds an experiment script for scenario exploration and model demonstration
- adds dedicated test files to improve validation and future maintainability

### Main Changes

#### 1. Dispersal-kernel functionality merged into the main branch

The most important change in this release is the formal integration of the `feature/dispersal-kernel` branch into `main`.

This means:

- dispersal-kernel-related implementation is no longer isolated in a feature branch
- `main` now contains the official version of this functionality
- future development, testing, and releases can proceed directly from the main branch baseline

#### 2. Updated `metaibm/metacommunity.py`

Modified file:

- `metaibm/metacommunity.py`

This file is one of the core code changes in **v3.2.0**.

From a release and maintenance perspective, this indicates that:

- metacommunity-level dispersal logic has been extended
- the main implementation path now includes dispersal-kernel-related behavior, including 'uniform' (by default), 'guassian', 'exponential', 'power_law', 'cauchy' distribution
- future refinement of dispersal mechanisms should continue from this file

#### 3. Added experiment script: `experiments/model-sloss.py`

New file:

- `experiments/model-sloss.py`

This addition shows that the release includes not only core implementation changes, but also an experiment-oriented script layer.

Potential uses include:

- designing experiments involving dispersal kernels
- demonstrating behavior under specific simulation scenarios
- supporting future model analysis, reproducibility, or research workflows

#### 4. Added test files for validation and regression support

New files:

- `test/test_dispersal_kernel_realistic.py`
- `test/test_dispersal_kernel_verbose.py`

This is an important signal for the maturity of the feature:

- the new functionality is not just implemented, but also accompanied by tests
- the project is improving its validation and inspection workflow
- future refactoring and optimization will be easier and safer with these tests in place

From the file names, the likely intent is:

- `test_dispersal_kernel_realistic.py`: validation in realistic or near-realistic scenarios
- `test_dispersal_kernel_verbose.py`: verbose output for debugging, inspection, or detailed behavior tracing

### Files Included in This Release

#### New files

- `experiments/model-sloss.py`
- `test/test_dispersal_kernel_realistic.py`
- `test/test_dispersal_kernel_verbose.py`

#### Modified files

- `metaibm/metacommunity.py`

### Release Value Summary

Compared with the previous state of the main branch, **v3.2.0** provides value in the following areas:

1. **Functionality**
   - dispersal-kernel-related logic is now officially part of the main branch

2. **Experiment support**
   - a dedicated experiment script has been added for scenario analysis and demonstration

3. **Testing and validation**
   - two new test files improve reliability, traceability, and maintainability

4. **Version management**
   - the feature is no longer isolated in a feature branch
   - the merged implementation now serves as a stronger baseline for future releases

### Short Release Summary

```text
MetaIBM v3.2.0 introduces dispersal-kernel-related functionality into the main branch, updates metacommunity-level logic, and adds dedicated experiment and test scripts for improved validation and future development.
```

### Notes

This document describes only the changes that were explicitly confirmed in this conversation as already merged into `main`.

If additional branches (for example, a larger structural refactor branch) are merged later, it is recommended to prepare a separate update note for a future version such as `v3.3.0` or above.

---

## v3.1.0

> This release is primarily a structural refactor based on **v2.9.12**.
> The main goal of this version is to improve modularity and maintainability by splitting the original monolithic implementation into core modules inside the `metaibm` package.

### Overview

**MetaIBM v3.1.0** is a refactoring-oriented release built on top of **v2.9.12**.

The central change in this version is the decomposition of the original `metacommunity_IBM.py` implementation into four core modules:

- `individual.py`
- `habitat.py`
- `patch.py`
- `metacommunity.py`

These modules are now placed inside the core package directory:

- `metaibm/`

In addition, the package-level import interface is organized through:

- `metaibm/__init__.py`

This release focuses on code organization rather than introducing a major new ecological feature.

### Main Changes

#### 1. Refactored the original `metacommunity_IBM.py`

The previous single-file implementation in `metacommunity_IBM.py` has been reorganized into a more modular package structure.

This improves:

- readability of the codebase
- separation of responsibilities across core classes
- long-term maintainability
- future extensibility for additional functionality

#### 2. Split the implementation into four core files

The original monolithic logic was separated into four dedicated modules:

- `individual.py`
  Contains the core implementation related to individual-level entities and behaviors.

- `habitat.py`
  Contains habitat-level structures and processes.

- `patch.py`
  Contains patch-level organization and logic.

- `metacommunity.py`
  Contains metacommunity-level structures and higher-level orchestration.

This refactor aligns the code structure more closely with the conceptual hierarchy of the MetaIBM framework.

#### 3. Introduced the `metaibm` core package directory

The refactored core files are now grouped under the package directory:

```text
metaibm/
```

This change provides a clearer and more standard Python package structure for the project.

Benefits include:

- better internal organization
- easier import management
- cleaner future extension of the library
- improved support for package-style usage

#### 4. Added package import support through `__init__.py`

The file:

- `metaibm/__init__.py`

is included to define the package import interface.

This allows the package to expose core modules and classes in a cleaner and more maintainable way.

In practice, this means:

- imports can be standardized through the package entry point
- internal module relationships become easier to manage
- future refactoring can preserve a more stable external import style

### File Structure Impact

#### Core modules introduced in this release

- `metaibm/individual.py`
- `metaibm/habitat.py`
- `metaibm/patch.py`
- `metaibm/metacommunity.py`
- `metaibm/__init__.py`

#### Legacy implementation basis

- refactored from `metacommunity_IBM.py` in **v2.9.12**

### Release Significance

The significance of **v3.1.0** is mainly architectural.

Compared with **v2.9.12**, this release provides:

1. **Improved modularity**
   The codebase is no longer centered around a single large implementation file.

2. **Clearer conceptual mapping**
   Individual, habitat, patch, and metacommunity levels are now represented by dedicated modules.

3. **Better maintainability**
   Future debugging, extension, and refactoring become easier.

4. **Package-oriented structure**
   The introduction of the `metaibm` package and `__init__.py` makes the project more aligned with standard Python package design.

### Short Release Summary

```text
MetaIBM v3.1.0 is a structural refactor based on v2.9.12. The original metacommunity_IBM.py implementation has been split into four core modules—individual.py, habitat.py, patch.py, and metacommunity.py—organized within the metaibm package, with __init__.py defining the package import interface.
```

### Notes

This version is primarily a **refactoring and package-organization release**.

If later versions introduce new ecological functionality, algorithms, or testing/experiment workflows on top of this structure, those changes can be documented separately in subsequent version notes such as **v3.2.0** and beyond.
