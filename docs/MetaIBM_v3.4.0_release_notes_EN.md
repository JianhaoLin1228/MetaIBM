## MetaIBM v3.4.0 Release Notes

This version focuses on the introduction of explicit genetic recombination into the sexual reproduction workflow and on improving the internal correctness of colonization and offspring-generation processes.

### Overview

**MetaIBM v3.4.0** introduces an important update centered on **genetic recombination support** in sexual reproduction across the MetaIBM core package and experiment workflow.  
This release is important because it:
- introduces individual-level gamete generation for sexual inheritance
- supports explicit recombination control through `recomb_method` and `recomb_rate`
- adds interval-based multi-crossover recombination for trait-level genotype inheritance
- BUG-FIXING: improves the correctness of mainland-to-metacommunity colonization by using deep-copied individuals
- BUG-FIXING: aligns pure-sexual offspring-marker calculation with sexual parent-pair counts
- improves the package’s ability to simulate standing genetic variation (burning in mainland in model.py) and recombination-driven eco-evolutionary dynamics

### Main Changes

#### 1. Added explicit recombination at the individual level

The most important change in this release is the addition of explicit gamete-generation logic to the `individual` class. 
This means:
- sexual reproduction no longer relies only on simple haplotype sampling logic embedded in higher-level modules
- each trait can now generate a gamete through a dedicated recombination interface
- both simple segregation and interval-based multi-crossover recombination are supported
- recombination behavior can now be controlled directly through `recomb_method` and `recomb_rate`

#### 2. Integrated recombination through habitat, patch, and metacommunity workflows

Recombination support is now propagated consistently through the full reproductive pipeline. 
This means:
- sexual and mixed sexual reproduction in `habitat.py` now explicitly call gamete-generation methods
- recombination parameters are passed through `patch.py` and `metacommunity.py`
- mainland burn-in, offspring-marker generation, and local germination-and-birth workflows can all use the same recombination settings
- experimental scripts can now test recombination effects more explicitly and consistently

#### 3. Improved colonization correctness

Mainland-derived colonists are now deep-copied before being inserted into the metacommunity. 
This addition is important because:
- mainland source individuals and colonizing individuals are now independent objects
- unintended object-state sharing between mainland and metacommunity is avoided
- mutation, aging, and phenotype updates in the metacommunity no longer risk altering mainland source individuals indirectly

#### 4. Corrected pure-sexual offspring-marker accounting

The pure-sexual offspring-marker workflow has been aligned with the number of sexual parent pairs. 
This means:
- pure-sexual offspring-marker counts now use sexual pair counts rather than total individual abundance
- marker-pool generation is now consistent with pure-sexual offspring generation and germination logic
- reproductive bookkeeping is more coherent across marker-pool and realized-offspring workflows

### Files Included in This Release

#### Modified files
- metaibm/individual.py
- metaibm/habitat.py
- metaibm/patch.py
- metaibm/metacommunity.py
- experiments/model-sloss-GRFE.py

#### New files
- no required new core data files in this release

### Release Value Summary

Compared with the previous state of the project, **v3.4.0** provides value in the following areas:
- **Functionality**
- adds explicit recombination support to sexual inheritance workflows
- **Correctness**
- fixes object-sharing risk during mainland colonization and improves pure-sexual offspring-marker accounting
- **Consistency**
- unifies recombination parameter passing across individual, habitat, patch, metacommunity, and experiment layers
- **Research support**
- improves support for experiments involving standing genetic variation, recombination, and spatial eco-evolutionary dynamics

### Release Tags
MetaIBM v3.4.0

### Short Release Summary
MetaIBM v3.4.0 introduces explicit genetic recombination into sexual reproduction, propagates recombination parameters through the full simulation workflow, improves colonization correctness through deep-copied individuals, and aligns pure-sexual offspring-marker accounting with sexual parent-pair counts.

### Notes

This release note focuses on the introduction of **explicit genetic recombination support** and related correctness improvements in reproductive and colonization workflows.  
It is intended to document the transition from inheritance logic based mainly on higher-level reproduction routines toward a more explicit and controllable recombination-aware workflow in MetaIBM.
