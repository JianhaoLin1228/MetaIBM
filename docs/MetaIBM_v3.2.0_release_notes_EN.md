# MetaIBM v3.2.0 Release Notes

> This document is based on the changes that were explicitly confirmed in this conversation as already merged into `main`.
> The key confirmed change included in the main branch is the merge of the `feature/dispersal-kernel` branch into `main`.

---

## Overview

**MetaIBM v3.2.0** introduces a significant update centered on **dispersal-kernel-related functionality**, together with supporting experiment and test scripts.

This release is important because it:

- officially integrates the `feature/dispersal-kernel` branch into `main`
- extends metacommunity-level logic related to dispersal behavior
- adds an experiment script for scenario exploration and model demonstration
- adds dedicated test files to improve validation and future maintainability

---

## Main Changes

### 1. Dispersal-kernel functionality merged into the main branch

The most important change in this release is the formal integration of the `feature/dispersal-kernel` branch into `main`.

This means:

- dispersal-kernel-related implementation is no longer isolated in a feature branch
- `main` now contains the official version of this functionality
- future development, testing, and releases can proceed directly from the main branch baseline

---

### 2. Updated `metaibm/metacommunity.py`

Modified file:

- `metaibm/metacommunity.py`

This file is one of the core code changes in **v3.2.0**.

From a release and maintenance perspective, this indicates that:

- metacommunity-level dispersal logic has been extended
- the main implementation path now includes dispersal-kernel-related behavior, including 'uniform' (by default), 'guassian', 'exponential', 'power_law', 'cauchy' distribution
- future refinement of dispersal mechanisms should continue from this file

---

### 3. Added experiment script: `experiments/model-sloss.py`

New file:

- `experiments/model-sloss.py`

This addition shows that the release includes not only core implementation changes, but also an experiment-oriented script layer.

Potential uses include:

- designing experiments involving dispersal kernels
- demonstrating behavior under specific simulation scenarios
- supporting future model analysis, reproducibility, or research workflows

---

### 4. Added test files for validation and regression support

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

---

## Files Included in This Release

### New files

- `experiments/model-sloss.py`
- `test/test_dispersal_kernel_realistic.py`
- `test/test_dispersal_kernel_verbose.py`

### Modified files

- `metaibm/metacommunity.py`

---

## Release Value Summary

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

---

## Suggested Release Title

If you want to use this text in GitHub Releases, the following title is recommended:

```text
MetaIBM v3.2.0
```

---

## Suggested Short Release Summary

A concise summary for a GitHub Release page:

```text
MetaIBM v3.2.0 introduces dispersal-kernel-related functionality into the main branch, updates metacommunity-level logic, and adds dedicated experiment and test scripts for improved validation and future development.
```

---

## Notes

This document describes only the changes that were explicitly confirmed in this conversation as already merged into `main`.

If additional branches (for example, a larger structural refactor branch) are merged later, it is recommended to prepare a separate update note for a future version such as `v3.3.0` or above.
