#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test whether del_individual() behaves reliably when triggered by
hab_dead_selection(..., method="niche_gaussian").

Run from test/:
    python test_hab_dead_selection_niche.py
"""

import traceback
import numpy as np

import bootstrap_metaibm as _bootstrap

from metaibm.habitat import habitat
from metaibm.individual import individual


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def make_habitat(length=1, width=2, env_mean=0.0, env_var=0.0):
    return habitat(
        hab_name="h1",
        hab_index=0,
        hab_location=(0, 0),
        num_env_types=1,
        env_types_name=["environment"],
        mean_env_ls=[env_mean],
        var_env_ls=[env_var],
        length=length,
        width=width,
        dormancy_pool_max_size=0,
    )


def make_individual_with_mean(mean_value, gender="female"):
    """
    Create an individual whose phenotype is centered around mean_value.
    Using pheno_var_ls=[0.0] makes phenotype deterministic:
        phenotype = mean + gauss(0, 0) = mean
    """
    indi = individual(
        species_id="sp1",
        traits_num=1,
        pheno_names_ls=["phenotype"],
        gender=gender,
    )
    indi.random_init_indi(
        mean_pheno_val_ls=[mean_value],
        pheno_var_ls=[0.0],
        geno_len_ls=[10],
    )
    return indi


def assert_species_category_has_no_position(h, pos):
    """
    Current del_individual() removes position from species_category lists,
    but does not delete empty keys automatically.
    So we only assert that the deleted position is gone from all lists.
    """
    for sp_id, gender_map in h.species_category.items():
        for gender, pos_list in gender_map.items():
            assert pos not in pos_list, f"Deleted position {pos} still exists in species_category[{sp_id}][{gender}]"


def run_test(fn):
    try:
        fn()
        print(f"[PASS] {fn.__name__}")
    except Exception as e:
        print(f"[FAIL] {fn.__name__}: {e}")
        traceback.print_exc()
        return False
    return True


# -------------------------------------------------
# Tests
# -------------------------------------------------
def test_niche_selection_all_die_updates_state():
    """
    All individuals are badly mismatched with the environment,
    so all should die and del_individual() should clean state correctly.
    """
    h = make_habitat(length=2, width=3, env_mean=0.0, env_var=0.0)

    # Fill habitat with phenotype=1.0 individuals in environment=0.0
    for row in range(h.length):
        for col in range(h.width):
            indi = make_individual_with_mean(1.0)
            h.add_individual(indi, row, col)

    assert h.indi_num == 6
    assert len(h.occupied_site_pos_ls) == 6
    assert len(h.empty_site_pos_ls) == 0

    old_uniform = np.random.uniform
    try:
        # force random threshold to be high so very low survival_rate definitely dies
        np.random.uniform = lambda *args, **kwargs: np.array([0.999999])

        dead_num = h.hab_dead_selection(
            base_dead_rate=0.0,
            fitness_wid=0.1,
            method="niche_gaussian",
        )
    finally:
        np.random.uniform = old_uniform

    assert dead_num == 6
    assert h.indi_num == 0
    assert len(h.occupied_site_pos_ls) == 0
    assert len(h.empty_site_pos_ls) == 6

    for row in range(h.length):
        for col in range(h.width):
            assert h.set["microsite_individuals"][row][col] is None
            assert (row, col) in h.empty_site_pos_ls

    # ensure deleted positions do not remain in species_category lists
    for row in range(h.length):
        for col in range(h.width):
            assert_species_category_has_no_position(h, (row, col))


def test_niche_selection_none_die_keeps_state():
    """
    All individuals perfectly match the environment,
    and random threshold is fixed at 0.0, so nobody dies.
    """
    h = make_habitat(length=2, width=3, env_mean=0.0, env_var=0.0)

    for row in range(h.length):
        for col in range(h.width):
            indi = make_individual_with_mean(0.0)
            h.add_individual(indi, row, col)

    old_uniform = np.random.uniform
    try:
        # force threshold to 0.0, matched individuals with d=0 and phenotype=env have survival ~1
        np.random.uniform = lambda *args, **kwargs: np.array([0.0])

        dead_num = h.hab_dead_selection(
            base_dead_rate=0.0,
            fitness_wid=0.5,
            method="niche_gaussian",
        )
    finally:
        np.random.uniform = old_uniform

    assert dead_num == 0
    assert h.indi_num == 6
    assert len(h.occupied_site_pos_ls) == 6
    assert len(h.empty_site_pos_ls) == 0

    for row in range(h.length):
        for col in range(h.width):
            assert h.set["microsite_individuals"][row][col] is not None
            assert (row, col) in h.occupied_site_pos_ls


def test_niche_selection_partial_die_only_deletes_target():
    """
    Habitat 1x2:
    - left individual matches environment -> should survive
    - right individual mismatches environment badly -> should die

    This is the most direct test for del_individual() reliability under niche selection.
    """
    h = make_habitat(length=1, width=2, env_mean=0.0, env_var=0.0)

    survivor = make_individual_with_mean(0.0)
    victim = make_individual_with_mean(1.0)

    h.add_individual(survivor, 0, 0)
    h.add_individual(victim, 0, 1)

    assert h.indi_num == 2
    assert (0, 0) in h.occupied_site_pos_ls
    assert (0, 1) in h.occupied_site_pos_ls

    old_uniform = np.random.uniform
    try:
        # threshold=0.5:
        # survivor (perfect match) should have survival close to 1 -> survive
        # victim (bad mismatch, w=0.1) should have survival close to 0 -> die
        np.random.uniform = lambda *args, **kwargs: np.array([0.5])

        dead_num = h.hab_dead_selection(
            base_dead_rate=0.0,
            fitness_wid=0.1,
            method="niche_gaussian",
        )
    finally:
        np.random.uniform = old_uniform

    assert dead_num == 1
    assert h.indi_num == 1

    # survivor remains
    assert h.set["microsite_individuals"][0][0] is not None
    assert (0, 0) in h.occupied_site_pos_ls
    assert (0, 0) not in h.empty_site_pos_ls

    # victim removed
    assert h.set["microsite_individuals"][0][1] is None
    assert (0, 1) not in h.occupied_site_pos_ls
    assert (0, 1) in h.empty_site_pos_ls

    # deleted position must not remain in species_category lists
    assert_species_category_has_no_position(h, (0, 1))


# -------------------------------------------------
# Main runner
# -------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_niche_selection_all_die_updates_state,
        test_niche_selection_none_die_keeps_state,
        test_niche_selection_partial_die_only_deletes_target,
    ]

    passed = 0
    failed = 0

    print("=" * 70)
    print("Running hab_dead_selection(niche_gaussian) / del_individual checks...")
    print("=" * 70)

    for t in tests:
        ok = run_test(t)
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 70)
    print(f"Finished. PASSED={passed}, FAILED={failed}")
    print("=" * 70)

    if failed > 0:
        raise SystemExit(1)
