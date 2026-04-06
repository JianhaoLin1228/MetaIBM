#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regression checks for:
1) non-square habitat grid shape [length][width]
2) add/delete/disturbance on non-square habitats
3) patch/metacommunity getters on non-square habitats
4) meta_offset_environmental_values default delta_var handling

Run from test/:
    python test_regression_non_square.py
"""

import traceback

import bootstrap_metaibm as _bootstrap  # same import style as experiments/model.py

from metaibm.habitat import habitat
from metaibm.individual import individual
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity


# ----------------------------
# Helpers
# ----------------------------
def make_individual(species_id="sp1", gender="female"):
    indi = individual(
        species_id=species_id,
        traits_num=1,
        pheno_names_ls=["phenotype"],
        gender=gender,
    )
    indi.random_init_indi(
        mean_pheno_val_ls=[0.2],
        pheno_var_ls=[0.01],
        geno_len_ls=[10],
    )
    return indi


def make_habitat(length=2, width=3, mean=0.2, var=0.01):
    return habitat(
        hab_name="h1",
        hab_index=0,
        hab_location=(0, 0),
        num_env_types=1,
        env_types_name=["environment"],
        mean_env_ls=[mean],
        var_env_ls=[var],
        length=length,
        width=width,
        dormancy_pool_max_size=0,
    )


def make_patch_with_one_hab(length=2, width=3, mean=0.2, var=0.01):
    p = patch(patch_name="patch1", patch_index=0, location=(0, 0))
    p.add_habitat(
        hab_name="h1",
        hab_index=0,
        hab_location=(0, 0),
        num_env_types=1,
        env_types_name=["environment"],
        mean_env_ls=[mean],
        var_env_ls=[var],
        length=length,
        width=width,
        dormancy_pool_max_size=0,
    )
    return p


def make_meta_with_one_patch(length=2, width=3, mean=0.2, var=0.01):
    meta = metacommunity(metacommunity_name="meta")
    p = make_patch_with_one_hab(length=length, width=width, mean=mean, var=var)
    meta.add_patch(patch_name="patch1", patch_object=p)
    return meta


def assert_grid_shape(h):
    grid = h.set["microsite_individuals"]
    assert len(grid) == h.length, f"outer len(grid)={len(grid)} != h.length={h.length}"
    assert all(len(row) == h.width for row in grid), (
        f"row widths={[len(row) for row in grid]} != expected width={h.width}"
    )


def run_test(fn):
    try:
        fn()
        print(f"[PASS] {fn.__name__}")
    except Exception as e:
        print(f"[FAIL] {fn.__name__}: {e}")
        traceback.print_exc()
        return False
    return True


# ----------------------------
# Tests for length/width/row/col
# ----------------------------
def test_empty_microsite_grid_shape_matches_length_width():
    h = make_habitat(length=2, width=3)
    assert_grid_shape(h)


def test_add_and_delete_individual_on_non_square_grid():
    h = make_habitat(length=2, width=3)
    indi = make_individual()

    h.add_individual(indi_object=indi, len_id=1, wid_id=2)
    assert h.set["microsite_individuals"][1][2] is indi
    assert h.indi_num == 1
    assert (1, 2) in h.occupied_site_pos_ls
    assert (1, 2) not in h.empty_site_pos_ls

    h.del_individual(len_id=1, wid_id=2)
    assert h.set["microsite_individuals"][1][2] is None
    assert h.indi_num == 0
    assert (1, 2) not in h.occupied_site_pos_ls
    assert (1, 2) in h.empty_site_pos_ls


def test_hab_initialize_fills_non_square_grid():
    h = make_habitat(length=2, width=3)

    h.hab_initialize(
        traits_num=1,
        pheno_names_ls=["phenotype"],
        pheno_var_ls=[0.01],
        geno_len_ls=[10],
        reproduce_mode="asexual",
        species_2_phenotype_ls=[[0.2]],
    )

    assert_grid_shape(h)
    assert h.indi_num == 6
    assert len(h.empty_site_pos_ls) == 0
    assert len(h.occupied_site_pos_ls) == 6

    for row in range(h.length):
        for col in range(h.width):
            assert h.set["microsite_individuals"][row][col] is not None


def test_hab_dead_selection_neutral_uniform_runs_on_non_square_grid():
    h = make_habitat(length=2, width=3)

    h.hab_initialize(
        traits_num=1,
        pheno_names_ls=["phenotype"],
        pheno_var_ls=[0.01],
        geno_len_ls=[10],
        reproduce_mode="asexual",
        species_2_phenotype_ls=[[0.2]],
    )

    dead_num = h.hab_dead_selection(
        base_dead_rate=0.0,
        fitness_wid=0.5,
        method="neutral_uniform",
    )

    # neutral_uniform + base_dead_rate=0 => survival_rate = 1
    assert dead_num == 0
    assert h.indi_num == 6
    assert len(h.asexual_parent_pos_ls) == 6


def test_habitat_disturbance_resets_grid_shape_and_state():
    h = make_habitat(length=2, width=3)
    indi = make_individual()

    h.add_individual(indi_object=indi, len_id=1, wid_id=2)
    assert h.indi_num == 1

    h.habitat_disturbance_process()

    assert_grid_shape(h)
    assert h.indi_num == 0
    assert len(h.occupied_site_pos_ls) == 0
    assert len(h.empty_site_pos_ls) == 6

    for row in range(h.length):
        for col in range(h.width):
            assert h.set["microsite_individuals"][row][col] is None

    # verify add still works after reset
    indi2 = make_individual()
    h.add_individual(indi_object=indi2, len_id=1, wid_id=2)
    assert h.set["microsite_individuals"][1][2] is indi2


# ----------------------------
# Patch / metacommunity getter smoke tests
# ----------------------------
def test_patch_species_distribution_getter_runs_on_non_square_grid():
    p = make_patch_with_one_hab(length=2, width=3)

    p.patch_initialize(
        traits_num=1,
        pheno_names_ls=["phenotype"],
        pheno_var_ls=[0.01],
        geno_len_ls=[10],
        reproduce_mode="asexual",
        species_2_phenotype_ls=[[0.2]],
    )

    values = p.get_patch_microsites_individals_sp_id_values()
    assert values.shape == (1, 6)


def test_patch_phenotype_distribution_getter_runs_on_non_square_grid():
    p = make_patch_with_one_hab(length=2, width=3)

    p.patch_initialize(
        traits_num=1,
        pheno_names_ls=["phenotype"],
        pheno_var_ls=[0.01],
        geno_len_ls=[10],
        reproduce_mode="asexual",
        species_2_phenotype_ls=[[0.2]],
    )

    values = p.get_patch_microsites_individals_phenotype_values("phenotype")
    assert values.shape == (1, 6)


def test_metacommunity_species_distribution_getter_runs_on_non_square_grid():
    meta = make_meta_with_one_patch(length=2, width=3)

    meta.meta_initialize(
        traits_num=1,
        pheno_names_ls=["phenotype"],
        pheno_var_ls=[0.01],
        geno_len_ls=[10],
        reproduce_mode="asexual",
        species_2_phenotype_ls=[[0.2]],
    )

    values = meta.get_meta_microsites_individuals_sp_id_values()
    assert values.shape == (1, 6)


# ----------------------------
# Tests for meta_offset_environmental_values
# ----------------------------
def test_meta_offset_environment_default_delta_var_keeps_variance():
    """
    This test assumes you have already fixed:
        if delta_var_ls == None:
            delta_var_ls = [0 for i in range(len(env_name_ls))]
    in metacommunity.meta_offset_environmental_values().
    """
    meta = make_meta_with_one_patch(length=2, width=3, mean=0.2, var=0.01)

    h = meta.set["patch1"].set["h1"]
    old_mean = h.mean_env_ls[0]
    old_var = h.var_env_ls[0]

    meta.meta_offset_environmental_values(
        env_name_ls=["environment"],
        delta_mean_ls=[0.1],
    )

    assert h.mean_env_ls[0] == old_mean + 0.1
    assert h.var_env_ls[0] == old_var


def test_meta_offset_environment_updates_mean_and_variance_when_delta_var_given():
    meta = make_meta_with_one_patch(length=2, width=3, mean=0.2, var=0.01)

    h = meta.set["patch1"].set["h1"]
    old_mean = h.mean_env_ls[0]
    old_var = h.var_env_ls[0]

    meta.meta_offset_environmental_values(
        env_name_ls=["environment"],
        delta_mean_ls=[0.1],
        delta_var_ls=[0.02],
    )

    assert h.mean_env_ls[0] == old_mean + 0.1
    assert h.var_env_ls[0] == old_var + 0.02


# ----------------------------
# Main runner
# ----------------------------
if __name__ == "__main__":
    tests = [
        test_empty_microsite_grid_shape_matches_length_width,
        test_add_and_delete_individual_on_non_square_grid,
        test_hab_initialize_fills_non_square_grid,
        test_hab_dead_selection_neutral_uniform_runs_on_non_square_grid,
        test_habitat_disturbance_resets_grid_shape_and_state,
        test_patch_species_distribution_getter_runs_on_non_square_grid,
        test_patch_phenotype_distribution_getter_runs_on_non_square_grid,
        test_metacommunity_species_distribution_getter_runs_on_non_square_grid,
        test_meta_offset_environment_default_delta_var_keeps_variance,
        test_meta_offset_environment_updates_mean_and_variance_when_delta_var_given,
    ]

    passed = 0
    failed = 0

    print("=" * 70)
    print("Running regression checks for non-square habitat and env offset...")
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
