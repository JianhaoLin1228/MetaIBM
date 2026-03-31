# -*- coding: utf-8 -*-
"""
IDE-runnable tests for global_habitat_network + dispersal_among_patch.

Usage:
1. Put this file under project_root/test/test_global_habitat_network.py
2. Run directly in IDE, or:
   python test/test_global_habitat_network.py
3. Or with pytest:
   pytest -q test/test_global_habitat_network.py
"""

import sys
import random
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import bootstrap_metaibm as _bootstrap # Make sure project root is on sys.path
from metaibm.metacommunity import metacommunity
from metaibm.patch import patch


def _make_patch(patch_name, patch_index, patch_location, habitats):
    """
    habitats: list of dict, each dict like:
        {
            "hab_name": "h0",
            "hab_index": 0,
            "hab_location": (x, y),
            "length": 1,
            "width": 1,
        }
    """
    p = patch(patch_name, patch_index, patch_location)
    for h in habitats:
        p.add_habitat(
            hab_name=h["hab_name"],
            hab_index=h["hab_index"],
            hab_location=h["hab_location"],
            num_env_types=1,
            env_types_name=["env"],
            mean_env_ls=[0.5],
            var_env_ls=[0.0],
            length=h.get("length", 1),
            width=h.get("width", 1),
            dormancy_pool_max_size=h.get("dormancy_pool_max_size", 0),
        )
    return p


@pytest.fixture
def meta_two_patches_one_hab_each():
    """
    Minimal global habitat network:
    patch1 -> h0 at (0, 0)
    patch2 -> h0 at (3, 4)
    Distance = 5
    """
    meta = metacommunity("meta_test")

    p1 = _make_patch(
        "patch1", 0, (0, 0),
        [{"hab_name": "h0", "hab_index": 0, "hab_location": (0, 0), "length": 1, "width": 1}]
    )
    p2 = _make_patch(
        "patch2", 1, (1, 0),
        [{"hab_name": "h0", "hab_index": 0, "hab_location": (3, 4), "length": 1, "width": 1}]
    )

    meta.add_patch("patch1", p1)
    meta.add_patch("patch2", p2)
    return meta


@pytest.fixture
def meta_two_patches_two_habs_each():
    """
    For testing within-patch block = 0 and cross-patch block = kernel.
    patch1: h0(0,0), h1(0,1)
    patch2: h0(3,4), h1(3,5)
    """
    meta = metacommunity("meta_test_2x2")

    p1 = _make_patch(
        "patch1", 0, (0, 0),
        [
            {"hab_name": "h0", "hab_index": 0, "hab_location": (0, 0), "length": 1, "width": 1},
            {"hab_name": "h1", "hab_index": 1, "hab_location": (0, 1), "length": 1, "width": 1},
        ]
    )
    p2 = _make_patch(
        "patch2", 1, (1, 0),
        [
            {"hab_name": "h0", "hab_index": 0, "hab_location": (3, 4), "length": 1, "width": 1},
            {"hab_name": "h1", "hab_index": 1, "hab_location": (3, 5), "length": 1, "width": 1},
        ]
    )

    meta.add_patch("patch1", p1)
    meta.add_patch("patch2", p2)
    return meta


# -----------------------------------------------------------------------------
# Registry and distance matrix
# -----------------------------------------------------------------------------

def test_global_habitat_registry(meta_two_patches_one_hab_each):
    meta = meta_two_patches_one_hab_each

    assert meta.global_habitat_id_ls == [("patch1", "h0"), ("patch2", "h0")]
    assert meta.global_habitat_id_2_index_dir[("patch1", "h0")] == 0
    assert meta.global_habitat_id_2_index_dir[("patch2", "h0")] == 1



def test_global_habitat_distance_matrix(meta_two_patches_one_hab_each):
    meta = meta_two_patches_one_hab_each
    D = np.asarray(meta.global_habitat_distance_matrix, dtype=float)

    expected = np.array([
        [0.0, 5.0],
        [5.0, 0.0],
    ])
    assert_allclose(D, expected, atol=1e-12)


# -----------------------------------------------------------------------------
# Kernel / normalized kernel / rate matrix
# -----------------------------------------------------------------------------

def test_global_habitat_kernel_uniform(meta_two_patches_two_habs_each):
    meta = meta_two_patches_two_habs_each
    K = np.asarray(
        meta.calculate_global_habitat_network_dispersal_kernel_strength_matrix(method="uniform"),
        dtype=float
    )

    expected = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ])
    assert_allclose(K, expected, atol=1e-12)



def test_global_habitat_kernel_exponential(meta_two_patches_one_hab_each):
    meta = meta_two_patches_one_hab_each
    rho = 2.0
    K = np.asarray(
        meta.calculate_global_habitat_network_dispersal_kernel_strength_matrix(
            method="exponential", rho=rho
        ),
        dtype=float
    )

    expected_val = rho * np.exp(-rho * 5.0)
    expected = np.array([
        [0.0, expected_val],
        [expected_val, 0.0],
    ])
    assert_allclose(K, expected, atol=1e-12)



def test_normalized_global_habitat_kernel_uniform(meta_two_patches_two_habs_each):
    meta = meta_two_patches_two_habs_each
    K_norm = np.asarray(
        meta.normalized_calculate_global_habitat_network_dispersal_kernel_strength_matrix(
            method="uniform"
        ),
        dtype=float
    )

    expected = np.array([
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ])
    assert_allclose(K_norm, expected, atol=1e-12)



def test_global_habitat_rate_matrix_uniform(meta_two_patches_two_habs_each):
    meta = meta_two_patches_two_habs_each
    total_disp_among_rate = 0.2

    M = np.asarray(
        meta.global_habitat_dispersal_among_rate_matrix(
            total_disp_among_rate, method="uniform"
        ),
        dtype=float
    )

    expected = np.array([
        [0.8, 0.0, 0.1, 0.1],
        [0.0, 0.8, 0.1, 0.1],
        [0.1, 0.1, 0.8, 0.0],
        [0.1, 0.1, 0.0, 0.8],
    ])
    assert_allclose(M, expected, atol=1e-12)


# -----------------------------------------------------------------------------
# Pool number matrices
# -----------------------------------------------------------------------------

def test_global_habitat_pool_num_matrices(meta_two_patches_one_hab_each):
    meta = meta_two_patches_one_hab_each

    h10 = meta.set["patch1"].set["h0"]
    h20 = meta.set["patch2"].set["h0"]

    h10.offspring_pool = ["o1", "o2", "o3"]
    h20.offspring_pool = ["o4"]

    h10.offspring_marker_pool = [("p1", "h0", "a")] * 2
    h20.offspring_marker_pool = [("p2", "h0", "a")] * 5

    h10.dormancy_pool = ["d1"]
    h20.dormancy_pool = ["d2", "d3", "d4"]

    O_off = np.asarray(meta.get_global_habitat_network_offspring_pool_num_matrix(), dtype=float)
    O_marker = np.asarray(meta.get_global_habitat_network_offspring_marker_pool_num_matrix(), dtype=float)
    O_dorm = np.asarray(meta.get_global_habitat_network_dormancy_pool_num_matrix(), dtype=float)

    assert_allclose(O_off, np.diag([3, 1]), atol=1e-12)
    assert_allclose(O_marker, np.diag([2, 5]), atol=1e-12)
    assert_allclose(O_dorm, np.diag([1, 3]), atol=1e-12)


# -----------------------------------------------------------------------------
# Emigrants / immigrants / disp-among-num matrices
# -----------------------------------------------------------------------------

def test_global_habitat_emigrants_matrix(meta_two_patches_one_hab_each):
    meta = meta_two_patches_one_hab_each

    h10 = meta.set["patch1"].set["h0"]
    h20 = meta.set["patch2"].set["h0"]

    h10.offspring_pool = ["o1", "o2", "o3"]
    h20.offspring_pool = ["o4"]

    h10.dormancy_pool = ["d1"]
    h20.dormancy_pool = ["d2", "d3"]

    EM = np.asarray(
        meta.get_global_habitat_network_emigrants_matrix(
            total_disp_among_rate=0.25, method="uniform"
        ),
        dtype=float
    )

    expected = np.array([
        [3.0, 1.0],
        [0.75, 2.25],
    ])
    assert_allclose(EM, expected, atol=1e-12)



def test_global_habitat_immigrants_matrix(meta_two_patches_one_hab_each):
    meta = meta_two_patches_one_hab_each

    h10 = meta.set["patch1"].set["h0"]
    h20 = meta.set["patch2"].set["h0"]

    h10.offspring_pool = ["o1", "o2", "o3"]
    h20.offspring_pool = ["o4"]

    h10.dormancy_pool = ["d1"]
    h20.dormancy_pool = ["d2", "d3"]

    IM = np.asarray(
        meta.get_global_habitat_network_immigrants_matrix(
            total_disp_among_rate=0.25, method="uniform"
        ),
        dtype=float
    )

    expected = np.array([
        [0.8, 1.0 / 3.25],
        [0.2, 2.25 / 3.25],
    ])
    assert_allclose(IM, expected, atol=1e-12)



def test_global_habitat_disp_among_num_matrix(meta_two_patches_one_hab_each):
    meta = meta_two_patches_one_hab_each

    h10 = meta.set["patch1"].set["h0"]
    h20 = meta.set["patch2"].set["h0"]

    h10.offspring_pool = ["a1", "a2", "a3", "a4"]
    h20.offspring_pool = ["b1", "b2", "b3", "b4"]

    h10.dormancy_pool = []
    h20.dormancy_pool = []

    DA = np.asarray(
        meta.get_global_habitat_network_disp_among_num_matrix(
            total_disp_among_rate=1.0, method="uniform"
        ),
        dtype=float
    )

    expected = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    assert_allclose(DA, expected, atol=1e-12)


# -----------------------------------------------------------------------------
# Execution layer
# -----------------------------------------------------------------------------

def test_dispersal_from_offspring_pool_to_immigrant_pool(meta_two_patches_one_hab_each):
    meta = meta_two_patches_one_hab_each

    h10 = meta.set["patch1"].set["h0"]
    h20 = meta.set["patch2"].set["h0"]

    random.seed(123)
    np.random.seed(123)

    h10.offspring_pool = ["a", "b", "c", "d"]
    h20.offspring_pool = []

    h10.immigrant_pool = []
    h20.immigrant_pool = []

    log_info = meta.dispersal_among_patches_in_global_habitat_network_from_offspring_pool_to_immigrant_pool(
        total_disp_among_rate=1.0, method="uniform"
    )

    assert len(h10.immigrant_pool) == 0
    assert len(h20.immigrant_pool) == 4
    assert set(h20.immigrant_pool) == {"a", "b", "c", "d"}
    assert "there are 4 individuals disperse into immigrant_pool" in log_info



def test_dispersal_from_offspring_marker_pool_to_immigrant_marker_pool(meta_two_patches_one_hab_each):
    meta = meta_two_patches_one_hab_each

    h10 = meta.set["patch1"].set["h0"]
    h20 = meta.set["patch2"].set["h0"]

    random.seed(123)
    np.random.seed(123)

    h10.offspring_marker_pool = [("patch1", "h0", "asexual")] * 3
    h20.offspring_marker_pool = []

    h10.immigrant_marker_pool = []
    h20.immigrant_marker_pool = []

    log_info = meta.dispersal_among_patches_in_global_habitat_network_from_offspring_marker_pool_to_immigrant_marker_pool(
        total_disp_among_rate=1.0, method="uniform"
    )

    assert len(h10.immigrant_marker_pool) == 0
    assert len(h20.immigrant_marker_pool) == 3
    assert "there are 3 offspring markers disperse into immigrant_marker_pool" in log_info



def test_patch_num_one_offspring_dispersal_not_execute():
    meta = metacommunity("single_patch_meta")
    p1 = _make_patch(
        "patch1", 0, (0, 0),
        [{"hab_name": "h0", "hab_index": 0, "hab_location": (0, 0), "length": 1, "width": 1}]
    )
    meta.add_patch("patch1", p1)

    h10 = meta.set["patch1"].set["h0"]
    h10.offspring_pool = ["a", "b"]
    h10.immigrant_pool = []

    log_info = meta.dispersal_among_patches_in_global_habitat_network_from_offspring_pool_to_immigrant_pool(
        total_disp_among_rate=1.0, method="uniform"
    )

    assert len(h10.immigrant_pool) == 0
    assert "patch_num < 2" in log_info


@pytest.mark.xfail(reason="当前版本 normalized kernel / rate matrix 对 patch_num=1 仍可能产生 NaN，等你补 zero-row guard 后再打开")
def test_expected_single_patch_rate_matrix_should_be_zero_not_nan():
    meta = metacommunity("single_patch_meta")
    p1 = _make_patch(
        "patch1", 0, (0, 0),
        [{"hab_name": "h0", "hab_index": 0, "hab_location": (0, 0), "length": 1, "width": 1}]
    )
    meta.add_patch("patch1", p1)

    M = np.asarray(
        meta.global_habitat_dispersal_among_rate_matrix(total_disp_among_rate=0.2, method="uniform"),
        dtype=float
    )

    expected = np.zeros((1, 1), dtype=float)
    assert_allclose(M, expected, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
