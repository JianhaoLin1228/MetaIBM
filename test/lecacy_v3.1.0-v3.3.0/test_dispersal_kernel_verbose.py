#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dispersal-kernel focused regression tests for MetaIBM/metacommunity.py

Usage (from project root):
    python test/test_dispersal_kernel.py

This test file is designed to live under MetaIBM/test/ and import the local package
without requiring installation.

特点：
- 显示每个测试的“测试内容”与“测试结果”
- 对关键测试打印 actual / expected，便于人工核对
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Make project root importable when this file is run from MetaIBM/test/
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metaibm.patch import patch
from metaibm.metacommunity import metacommunity


class VerboseTextTestResult(unittest.TextTestResult):
    """Custom test result that prints test content and outcome more explicitly."""

    def getDescription(self, test):
        doc = test.shortDescription()
        if doc:
            return doc.strip()
        return str(test)

    def startTest(self, test):
        self.stream.write("\n" + "=" * 88 + "\n")
        self.stream.write(f"【测试内容】{self.getDescription(test)}\n")
        self.stream.flush()
        super().startTest(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.stream.write("【测试结果】PASS\n")
        self.stream.flush()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.write("【测试结果】FAIL\n")
        self.stream.flush()

    def addError(self, test, err):
        super().addError(test, err)
        self.stream.write("【测试结果】ERROR\n")
        self.stream.flush()


class VerboseTextTestRunner(unittest.TextTestRunner):
    resultclass = VerboseTextTestResult


class TestDispersalKernel(unittest.TestCase):
    """Focused tests for the dispersal-kernel related methods in metacommunity.py."""

    def setUp(self):
        # Build a tiny, fully controlled metacommunity with 3 patches.
        # Each patch contains 1 habitat of size 1x1, initially empty.
        self.meta = metacommunity('test_meta')

        coords = [(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)]
        for idx, loc in enumerate(coords):
            p = patch(patch_name=f'patch{idx+1}', patch_index=idx, location=loc)
            p.add_habitat(
                hab_name='h1',
                hab_index=0,
                hab_location=loc,
                num_env_types=1,
                env_types_name=['environment'],
                mean_env_ls=[0.0],
                var_env_ls=[0.0],
                length=1,
                width=1,
                dormancy_pool_max_size=0,
            )
            self.meta.add_patch(f'patch{idx+1}', p)

        # Controlled offspring / dormancy / marker pool sizes.
        # Patches are empty, so each patch has exactly 1 empty site.
        counts = [10, 20, 30]
        marker_counts = [3, 4, 5]
        for idx, patch_id in enumerate(self.meta.patch_id_ls):
            h = self.meta.set[patch_id].set['h1']
            h.offspring_pool = [f'off_{patch_id}_{i}' for i in range(counts[idx])]
            h.dormancy_pool = []
            h.offspring_marker_pool = [(patch_id, 'h1', 'asexual') for _ in range(marker_counts[idx])]
            h.immigrant_pool = []
            h.immigrant_marker_pool = []

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @staticmethod
    def _assert_matrix_close(testcase, actual, expected, rtol=1e-7, atol=1e-8):
        testcase.assertTrue(
            np.allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol),
            msg=f"\nACTUAL:\n{np.asarray(actual)}\nEXPECTED:\n{np.asarray(expected)}",
        )

    @staticmethod
    def _print_matrix(title, matrix):
        print(f"{title}:\n{np.asarray(matrix)}")

    @staticmethod
    def _print_scalar(title, value):
        print(f"{title}: {value}")

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------
    def test_pairwise_patch_distance_matrix(self):
        """距离矩阵：检查 add_patch() 自动更新后的 pairwise_patch_distance_matrix 是否正确。"""
        expected = np.array(
            [
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 2.0],
                [3.0, 2.0, 0.0],
            ],
            dtype=float,
        )
        actual = self.meta.pairwise_patch_distance_matrix
        self._print_matrix("Expected distance matrix", expected)
        self._print_matrix("Actual distance matrix", actual)
        self._assert_matrix_close(self, actual, expected)

    # ------------------------------------------------------------------
    # Single-value kernel strength tests
    # ------------------------------------------------------------------
    def test_dispersal_kernel_strength_uniform(self):
        """单值 kernel：uniform 方法应始终返回 1。"""
        actual = self.meta.dispersal_kernel_strength(2.5, method='uniform')
        expected = 1.0
        self._print_scalar("Expected D_ij", expected)
        self._print_scalar("Actual D_ij", actual)
        self.assertAlmostEqual(actual, expected)

    def test_dispersal_kernel_strength_exponential(self):
        """单值 kernel：exponential 方法 D_ij = rho * exp(-rho * d_ij)。"""
        d = 2.0
        rho = 0.5
        expected = rho * np.exp(-rho * d)
        actual = self.meta.dispersal_kernel_strength(d, method='exponential', rho=rho)
        self._print_scalar("Expected D_ij", expected)
        self._print_scalar("Actual D_ij", actual)
        self.assertAlmostEqual(actual, expected)

    def test_dispersal_kernel_strength_gaussian(self):
        """单值 kernel：gaussian 方法 D_ij = exp(-(d^2)/(2*sigma^2))。"""
        d = 2.0
        sigma = 1.5
        expected = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        actual = self.meta.dispersal_kernel_strength(d, method='gaussian', sigma=sigma)
        self._print_scalar("Expected D_ij", expected)
        self._print_scalar("Actual D_ij", actual)
        self.assertAlmostEqual(actual, expected)

    def test_dispersal_kernel_strength_cauchy(self):
        """单值 kernel：cauchy 方法 D_ij = 1 / (1 + (d/gamma)^2)。"""
        d = 2.0
        gamma = 1.5
        expected = 1.0 / (1.0 + (d / gamma) ** 2)
        actual = self.meta.dispersal_kernel_strength(d, method='cauchy', gamma=gamma)
        self._print_scalar("Expected D_ij", expected)
        self._print_scalar("Actual D_ij", actual)
        self.assertAlmostEqual(actual, expected)

    def test_dispersal_kernel_strength_power_law(self):
        """单值 kernel：power_law 方法 D_ij = (1 + d/r0)^(-alpha)。"""
        d = 2.0
        alpha = 2.0
        r0 = 1.5
        expected = (1.0 + d / r0) ** (-alpha)
        actual = self.meta.dispersal_kernel_strength(d, method='power_law', alpha=alpha, r0=r0)
        self._print_scalar("Expected D_ij", expected)
        self._print_scalar("Actual D_ij", actual)
        self.assertAlmostEqual(actual, expected)

    # ------------------------------------------------------------------
    # Kernel strength matrix D and normalized D
    # ------------------------------------------------------------------
    def test_calculate_dispersal_kernel_strength_matrix_uniform(self):
        """矩阵 D：uniform 方法下，非对角线应全为 1，对角线应为 0。"""
        D = self.meta.calculate_dispersal_kernel_strength_matrix(method='uniform')
        expected = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        self._print_matrix("Expected D", expected)
        self._print_matrix("Actual D", D)
        self._assert_matrix_close(self, D, expected)

    def test_normalized_calculate_dispersal_kernel_strength_matrix_uniform(self):
        """归一化矩阵 normalized_D：uniform 方法下每行应均分到其余 patch。"""
        normalized_D = self.meta.normalized_calculate_dispersal_kernel_strength_matrix(method='uniform')
        expected = np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        )
        self._print_matrix("Expected normalized_D", expected)
        self._print_matrix("Actual normalized_D", normalized_D)
        self._assert_matrix_close(self, normalized_D, expected)
        row_sums = np.asarray(normalized_D.sum(axis=1)).reshape(-1)
        self._print_scalar("Row sums of normalized_D", row_sums)
        self.assertTrue(np.allclose(row_sums, np.ones(3)))

    def test_normalized_calculate_dispersal_kernel_strength_matrix_exponential(self):
        """归一化矩阵 normalized_D：exponential 方法下应与手工构造结果一致，且每行和为 1。"""
        rho = 1.0
        D = np.array(
            [
                [0.0, rho * np.exp(-rho * 1.0), rho * np.exp(-rho * 3.0)],
                [rho * np.exp(-rho * 1.0), 0.0, rho * np.exp(-rho * 2.0)],
                [rho * np.exp(-rho * 3.0), rho * np.exp(-rho * 2.0), 0.0],
            ],
            dtype=float,
        )
        expected = D / D.sum(axis=1, keepdims=True)
        actual = self.meta.normalized_calculate_dispersal_kernel_strength_matrix(method='exponential', rho=rho)
        self._print_matrix("Expected normalized_D (exponential)", expected)
        self._print_matrix("Actual normalized_D (exponential)", actual)
        self._assert_matrix_close(self, actual, expected)
        row_sums = np.asarray(actual.sum(axis=1)).reshape(-1)
        self._print_scalar("Row sums of normalized_D", row_sums)
        self.assertTrue(np.allclose(row_sums, np.ones(3)))

    # ------------------------------------------------------------------
    # Final among-patch dispersal rate matrix M
    # ------------------------------------------------------------------
    def test_get_disp_among_rate_matrix_uniform(self):
        """扩散率矩阵 M：uniform 方法下应退化为原始 all-to-all 均匀分配矩阵。"""
        m = 0.2
        M = self.meta.get_disp_among_rate_matrix(m, method='uniform')
        expected = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
            dtype=float,
        )
        self._print_matrix("Expected M (uniform)", expected)
        self._print_matrix("Actual M (uniform)", M)
        self._assert_matrix_close(self, M, expected)
        row_sums = np.asarray(M.sum(axis=1)).reshape(-1)
        self._print_scalar("Row sums of M", row_sums)
        self.assertTrue(np.allclose(row_sums, np.ones(3)))

    def test_get_disp_among_rate_matrix_exponential(self):
        """扩散率矩阵 M：exponential 方法下应与手工构造结果一致，且每行和为 1。"""
        rho = 1.0
        m = 0.2

        D = np.array(
            [
                [0.0, rho * np.exp(-rho * 1.0), rho * np.exp(-rho * 3.0)],
                [rho * np.exp(-rho * 1.0), 0.0, rho * np.exp(-rho * 2.0)],
                [rho * np.exp(-rho * 3.0), rho * np.exp(-rho * 2.0), 0.0],
            ],
            dtype=float,
        )
        normalized_D = D / D.sum(axis=1, keepdims=True)
        expected = m * normalized_D
        np.fill_diagonal(expected, 1.0 - m)

        actual = self.meta.get_disp_among_rate_matrix(m, method='exponential', rho=rho)
        self._print_matrix("Expected M (exponential)", expected)
        self._print_matrix("Actual M (exponential)", actual)
        self._assert_matrix_close(self, actual, expected)
        row_sums = np.asarray(actual.sum(axis=1)).reshape(-1)
        self._print_scalar("Row sums of M", row_sums)
        self.assertTrue(np.allclose(row_sums, np.ones(3)))

    # ------------------------------------------------------------------
    # Downstream matrices
    # ------------------------------------------------------------------
    def test_get_emigrants_matrix_matches_diag_counts_times_M(self):
        """下游矩阵：get_emigrants_matrix() 应等于 diag(pool_counts) @ M。"""
        m = 0.2
        counts = np.diag([10, 20, 30]).astype(float)
        M = np.asarray(self.meta.get_disp_among_rate_matrix(m, method='uniform'))
        expected = counts @ M
        actual = self.meta.get_emigrants_matrix(m, method='uniform')
        self._print_matrix("Expected emigrants_matrix", expected)
        self._print_matrix("Actual emigrants_matrix", actual)
        self._assert_matrix_close(self, actual, expected)

    def test_get_immigrants_matrix_column_sums_equal_empty_site_quotas(self):
        """下游矩阵：get_immigrants_matrix() 的每列和应等于目标 patch 的空位配额（本例均为 1）。"""
        m = 0.2
        immigrants = np.asarray(self.meta.get_immigrants_matrix(m, method='uniform'))
        self._print_matrix("Actual immigrants_matrix", immigrants)
        col_sums = immigrants.sum(axis=0)
        self._print_scalar("Column sums of immigrants_matrix", col_sums)
        # each patch has exactly 1 empty site, so each column sum should be 1
        self.assertTrue(np.allclose(col_sums, np.ones(3)))

    # ------------------------------------------------------------------
    # Marker dispersal smoke test under zero among-patch rate
    # ------------------------------------------------------------------
    def test_marker_dispersal_zero_rate_produces_no_immigrants(self):
        """执行层 smoke test：当 among-patch rate = 0 时，不应产生 immigrant markers。"""
        log_info = self.meta.dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool(
            total_disp_among_rate=0.0,
            method='uniform'
        )
        self._print_scalar("Returned log_info", log_info.strip())
        self._print_scalar("Immigrant marker count", self.meta.meta_immigrant_marker_pool_marker_num())
        self.assertIn('there are 0 individuals disperse', log_info)
        self.assertEqual(self.meta.meta_immigrant_marker_pool_marker_num(), 0)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDispersalKernel)
    runner = VerboseTextTestRunner(verbosity=2)
    runner.run(suite)
