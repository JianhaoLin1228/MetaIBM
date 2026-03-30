# MetaIBM v3.1.0 Release Notes

> This release is primarily a structural refactor based on **v2.9.12**.
> The main goal of this version is to improve modularity and maintainability by splitting the original monolithic implementation into core modules inside the `metaibm` package.

---

## Overview

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

---

## Main Changes

### 1. Refactored the original `metacommunity_IBM.py`

The previous single-file implementation in `metacommunity_IBM.py` has been reorganized into a more modular package structure.

This improves:

- readability of the codebase
- separation of responsibilities across core classes
- long-term maintainability
- future extensibility for additional functionality

---

### 2. Split the implementation into four core files

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

---

### 3. Introduced the `metaibm` core package directory

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

---

### 4. Added package import support through `__init__.py`

The file:

- `metaibm/__init__.py`

is included to define the package import interface.

This allows the package to expose core modules and classes in a cleaner and more maintainable way.

In practice, this means:

- imports can be standardized through the package entry point
- internal module relationships become easier to manage
- future refactoring can preserve a more stable external import style

---

## File Structure Impact

### Core modules introduced in this release

- `metaibm/individual.py`
- `metaibm/habitat.py`
- `metaibm/patch.py`
- `metaibm/metacommunity.py`
- `metaibm/__init__.py`

### Legacy implementation basis

- refactored from `metacommunity_IBM.py` in **v2.9.12**

---

## Release Significance

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

---

## Suggested Release Title

```text
MetaIBM v3.1.0
```

---

## Suggested Short Release Summary

```text
MetaIBM v3.1.0 is a structural refactor based on v2.9.12. The original metacommunity_IBM.py implementation has been split into four core modules—individual.py, habitat.py, patch.py, and metacommunity.py—organized within the metaibm package, with __init__.py defining the package import interface.
```

---

## Notes

This version is primarily a **refactoring and package-organization release**.

If later versions introduce new ecological functionality, algorithms, or testing/experiment workflows on top of this structure, those changes can be documented separately in subsequent version notes such as **v3.2.0** and beyond.
