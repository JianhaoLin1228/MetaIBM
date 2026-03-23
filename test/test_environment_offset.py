#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regression checks for metacommunity.meta_offset_environmental_values().

This script verifies:
1) when delta_var_ls is omitted, only mean_env_ls is updated
2) when delta_var_ls is provided, both mean_env_ls and var_env_ls are updated
3) the environment offset propagates from metacommunity -> patch -> habitat

Run from experiments/:
    python test_environment_offset.py
"""

import traceback

import bootstrap_metaibm as _bootstrap  # same import style as experiments/model.py

from metaibm.patch import patch
from metaibm.metacommunity import metacommunity


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def make_meta_with_one_patch(length=2, width=3, mean=0.2, var=0.01):
    meta = metacommunity(metacommunity_name="meta")
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
    meta.add_patch(patch_name="patch1", patch_object=p)
    return meta


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
def test_meta_offset_environment_default_delta_var_keeps_variance():
    """
    Verify that when delta_var_ls is not provided,
    mean_env_ls changes but var_env_ls stays unchanged.
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
    """
    Verify that when delta_var_ls is explicitly provided,
    both mean_env_ls and var_env_ls are updated.
    """
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


def test_meta_offset_environment_propagates_to_all_habitats():
    """
    Verify the offset propagates from metacommunity to all habitats
    across multiple patches.
    """
    meta = metacommunity(metacommunity_name="meta")

    for i in range(2):
        p = patch(patch_name=f"patch{i+1}", patch_index=i, location=(i, 0))
        p.add_habitat(
            hab_name="h1",
            hab_index=0,
            hab_location=(0, 0),
            num_env_types=1,
            env_types_name=["environment"],
            mean_env_ls=[0.2 + i * 0.1],
            var_env_ls=[0.01],
            length=2,
            width=3,
            dormancy_pool_max_size=0,
        )
        meta.add_patch(patch_name=f"patch{i+1}", patch_object=p)

    old_vals = []
    for patch_id, patch_obj in meta.set.items():
        h = patch_obj.set["h1"]
        old_vals.append((h.mean_env_ls[0], h.var_env_ls[0]))

    meta.meta_offset_environmental_values(
        env_name_ls=["environment"],
        delta_mean_ls=[0.05],
    )

    for idx, (patch_id, patch_obj) in enumerate(meta.set.items()):
        h = patch_obj.set["h1"]
        old_mean, old_var = old_vals[idx]
        assert h.mean_env_ls[0] == old_mean + 0.05
        assert h.var_env_ls[0] == old_var


# -------------------------------------------------
# Main runner
# -------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_meta_offset_environment_default_delta_var_keeps_variance,
        test_meta_offset_environment_updates_mean_and_variance_when_delta_var_given,
        test_meta_offset_environment_propagates_to_all_habitats,
    ]

    passed = 0
    failed = 0

    print("=" * 70)
    print("Running environment offset regression checks...")
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
