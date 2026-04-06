# MetaIBM v3.3.0 Release Notes

> This version focuses on the `global-habitat-network` workflow and its extension-based installation into `metaibm/metacommunity.py`.

---

## Overview

**MetaIBM v3.3.0** introduces a significant update centered on the **global-habitat-network** workflow as an extension-based habitat-level dispersal module.

This release is important because it:

- introduces the `global-habitat-network` workflow for habitat-level dispersal across the whole landscape
- provides a dedicated extension module for the new functionality
- supports lightweight installation of extension methods into `metaibm/metacommunity.py`
- continues the package-oriented and extension-oriented development direction of MetaIBM

---

## Main Changes

### 1. global-habitat-network workflow

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

---

### 2. Added extension module: `extension/global_habitat_network.py`

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

---

### 3. Extension installation in `metaibm/metacommunity.py`

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

---

### 4. Method compatibility

The global-habitat-network extension is intended to work with the dispersal-kernel-based workflow already supported in MetaIBM.

Relevant methods include:

- `uniform`
- `gaussian (sigma)`
- `exponential (rho)`
- `cauchy`
- `power_law`

In **v3.3.0**, the main emphasis is the integration of **habitat-level network dispersal** into the extension-based architecture.

---

## Files Included in This Release

### New files

- `extension/global_habitat_network.py`

---

## Release Value Summary

Compared with the previous state of the project, **v3.3.0** provides value in the following areas:

1. **Functionality**
   
   - introduces global habitat network dispersal at the habitat level

2. **Modularity**
   
   - implements the new workflow as an extension module

3. **Installation clarity**
   
   - defines an explicit extension installation pattern in `metaibm/metacommunity.py`

4. **Future development**
   
   - strengthens the extension-oriented architecture for future feature growth

---

## Suggested Release Title

If you want to use this text in GitHub Releases, the following title is recommended:

```text
MetaIBM v3.3.0
```

---

## Suggested Short Release Summary

A concise summary for a GitHub Release page:

```text
MetaIBM v3.3.0 introduces the global-habitat-network workflow as an extension-based habitat-level dispersal module, adds the dedicated extension module extension/global_habitat_network.py, and supports explicit installation into metaibm/metacommunity.py through install_global_habitat_network_methods(metacommunity).
```

---

## Notes

This release note focuses on the **global-habitat-network** extension workflow and its installation pattern.
