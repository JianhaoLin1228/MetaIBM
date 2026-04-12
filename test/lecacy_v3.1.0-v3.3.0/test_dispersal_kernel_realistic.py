#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MetaIBM dispersal-kernel regression tests (realistic occupancy version)

目标测试场景：
- 6 个 patch
- 每个 patch 4 个 habitat
- 每个 habitat = 100 x 100 microsites
- 每个 habitat 放入 80% 的真实 individual 对象
- 补充生成一个“斑块位置 + 网状连边 + 距离标注”的示意图

运行方式（在 MetaIBM 项目根目录下）：
    python test/test_dispersal_kernel_realistic.py

输出文件：
- test/outputs/dispersal_kernel_patch_network.png

说明：
- 这是一个“较重”的测试文件，因为会构建 24 个 habitat、总共 240,000 个 microsites，
  并向其中填充 80%（192,000 个）真实 individual 对象。
- 测试重点是 dispersal kernel 相关模块是否正确，而不是速度基准。
- 为了控制内存，individual 对象使用最简 phenotype_set / genotype_set 结构，
  但它们仍然是真实的 MetaIBM individual 实例。
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Make project root importable when this file is run from MetaIBM/test/
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT_DIR = HERE / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metaibm.individual import individual
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity


class VerboseTextTestResult(unittest.TextTestResult):
    def getDescription(self, test):
        doc = test.shortDescription()
        if doc:
            return doc.strip()
        return str(test)

    def startTest(self, test):
        self.stream.write("\n" + "=" * 100 + "\n")
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


class TestDispersalKernelRealistic(unittest.TestCase):
    """Dispersal-kernel tests under a more realistic landscape/occupancy setting."""

    @classmethod
    def setUpClass(cls):
        # Landscape specification requested by the user
        cls.patch_num = 6
        cls.habitat_num_per_patch = 4
        cls.hab_length = 100
        cls.hab_width = 100
        cls.hab_size = cls.hab_length * cls.hab_width
        cls.occupancy_rate = 0.8
        cls.occupied_per_habitat = int(cls.hab_size * cls.occupancy_rate)
        cls.empty_per_habitat = cls.hab_size - cls.occupied_per_habitat
        cls.network_fig_path = OUTPUT_DIR / 'dispersal_kernel_patch_network.png'

        cls.meta = metacommunity('realistic_test_meta')

        # Arrange 6 patches on a 2 x 3 grid for deterministic distances.
        # patch indices: 0..5
        cls.patch_coords = [
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
            (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
        ]

        # Within each patch, place 4 habitats on a 2 x 2 local grid.
        local_hab_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for p_idx, p_loc in enumerate(cls.patch_coords):
            p = patch(patch_name=f'patch{p_idx+1}', patch_index=p_idx, location=p_loc)

            for h_idx, h_local in enumerate(local_hab_coords):
                p.add_habitat(
                    hab_name=f'h{h_idx+1}',
                    hab_index=h_idx,
                    hab_location=(p_loc[0] + h_local[0], p_loc[1] + h_local[1]),
                    num_env_types=1,
                    env_types_name=['environment'],
                    mean_env_ls=[0.0],
                    var_env_ls=[0.0],
                    length=cls.hab_length,
                    width=cls.hab_width,
                    dormancy_pool_max_size=0,
                )

            cls.meta.add_patch(f'patch{p_idx+1}', p)

        # ------------------------------------------------------------------
        # Fill each habitat with 80% real individual objects.
        # ------------------------------------------------------------------
        pheno_names_ls = ['phenotype']
        traits_num = 1

        for patch_id in cls.meta.patch_id_ls:
            p = cls.meta.set[patch_id]
            for h_id, h_obj in p.set.items():
                counter = 0
                for row in range(cls.hab_length):
                    for col in range(cls.hab_width):
                        if counter >= cls.occupied_per_habitat:
                            break

                        species_id = 'sp1' if (counter % 2 == 0) else 'sp2'
                        gender = 'female' if (counter % 2 == 0) else 'male'
                        phenotype_value = 0.0 if (counter % 2 == 0) else 1.0

                        indi = individual(
                            species_id=species_id,
                            traits_num=traits_num,
                            pheno_names_ls=pheno_names_ls,
                            gender=gender,
                            genotype_set={'phenotype': [np.array([0], dtype=int), np.array([0], dtype=int)]},
                            phenotype_set={'phenotype': phenotype_value},
                        )
                        h_obj.add_individual(indi, row, col)
                        counter += 1

                    if counter >= cls.occupied_per_habitat:
                        break

        # Controlled offspring / marker pools for downstream matrix tests.
        offspring_counts = [100, 200, 300, 400, 500, 600]
        marker_counts = [10, 20, 30, 40, 50, 60]
        for idx, patch_id in enumerate(cls.meta.patch_id_ls):
            h = cls.meta.set[patch_id].set['h1']
            h.offspring_pool = [f'off_{patch_id}_{i}' for i in range(offspring_counts[idx])]
            h.dormancy_pool = []
            h.offspring_marker_pool = [(patch_id, 'h1', 'asexual') for _ in range(marker_counts[idx])]
            h.immigrant_pool = []
            h.immigrant_marker_pool = []

    # ------------------------------------------------------------------
    # Helper printing / comparisons
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
    # Visualization helper
    # ------------------------------------------------------------------
    def _draw_patch_network_distance_figure(self, save_path):
        coords = []
        labels = []
        for patch_id in self.meta.patch_id_ls:
            x, y = self.meta.set[patch_id].location
            coords.append((x, y))
            labels.append(patch_id)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title('Patch network with pairwise distances')

        # draw complete graph edges with distance labels
        for i, (xi, yi) in enumerate(coords):
            for j, (xj, yj) in enumerate(coords):
                if j <= i:
                    continue
                dist = float(self.meta.pairwise_patch_distance_matrix[i, j])
                ax.plot([xi, xj], [yi, yj], color='gray', alpha=0.45, linewidth=1.0, zorder=1)
                mx, my = (xi + xj) / 2.0, (yi + yj) / 2.0
                ax.text(mx, my, f'{dist:.2f}', fontsize=8, color='dimgray', ha='center', va='center')

        # draw nodes
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax.scatter(xs, ys, s=300, color='tab:blue', edgecolors='black', zorder=3)

        for label, (x, y) in zip(labels, coords):
            ax.text(x, y + 0.08, label, fontsize=10, ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('patch x')
        ax.set_ylabel('patch y')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.margins(0.15)
        fig.tight_layout()
        fig.savefig(save_path, dpi=180)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Landscape / occupancy sanity checks
    # ------------------------------------------------------------------
    def test_landscape_and_real_individual_occupancy(self):
        """真实场景检查：6 patch × 4 habitat × 100x100 microsites，并在每个 habitat 放入 80% 真实个体。"""
        self._print_scalar('patch_num', self.meta.patch_num)
        self._print_scalar('habitat_num_per_patch', self.habitat_num_per_patch)
        self._print_scalar('habitat_size', self.hab_size)
        self._print_scalar('occupied_per_habitat', self.occupied_per_habitat)
        self._print_scalar('empty_per_habitat', self.empty_per_habitat)

        self.assertEqual(self.meta.patch_num, 6)

        for patch_id in self.meta.patch_id_ls:
            p = self.meta.set[patch_id]
            self.assertEqual(p.hab_num, 4)
            for h_id, h_obj in p.set.items():
                self.assertEqual(h_obj.size, 10000)
                self.assertEqual(h_obj.indi_num, self.occupied_per_habitat)
                self.assertEqual(len(h_obj.empty_site_pos_ls), self.empty_per_habitat)
                self.assertEqual(len(h_obj.occupied_site_pos_ls), self.occupied_per_habitat)

    def test_generate_patch_network_distance_figure(self):
        """可视化：生成 6 个 patch 的位置图、网状连边和 pairwise 距离标注图。"""
        self._draw_patch_network_distance_figure(self.network_fig_path)
        self._print_scalar('Generated figure path', self.network_fig_path)
        self.assertTrue(self.network_fig_path.exists())
        self.assertGreater(self.network_fig_path.stat().st_size, 0)

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------
    def test_pairwise_patch_distance_matrix_shape_and_spot_values(self):
        """距离矩阵：检查 6 个 patch 的 pairwise_patch_distance_matrix 形状与关键 spot values 是否正确。"""
        D = np.asarray(self.meta.pairwise_patch_distance_matrix)
        self._print_scalar('Distance matrix shape', D.shape)
        self._print_matrix('Distance matrix', D)

        self.assertEqual(D.shape, (6, 6))
        self.assertTrue(np.allclose(np.diag(D), np.zeros(6)))

        self.assertAlmostEqual(D[0, 1], 1.0)
        self.assertAlmostEqual(D[0, 2], 2.0)
        self.assertAlmostEqual(D[0, 3], 1.0)
        self.assertAlmostEqual(D[0, 4], np.sqrt(2.0))
        self.assertAlmostEqual(D[0, 5], np.sqrt(5.0))

    # ------------------------------------------------------------------
    # Kernel single-value tests
    # ------------------------------------------------------------------
    def test_dispersal_kernel_strength_methods(self):
        """单值 kernel：检查 uniform / gaussian / exponential / cauchy / power_law 的公式输出是否正确。"""
        d = 2.0

        actual_uniform = self.meta.dispersal_kernel_strength(d, method='uniform')
        expected_uniform = 1.0
        self._print_scalar('uniform expected', expected_uniform)
        self._print_scalar('uniform actual', actual_uniform)
        self.assertAlmostEqual(actual_uniform, expected_uniform)

        sigma = 1.5
        actual_gaussian = self.meta.dispersal_kernel_strength(d, method='gaussian', sigma=sigma)
        expected_gaussian = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        self._print_scalar('gaussian expected', expected_gaussian)
        self._print_scalar('gaussian actual', actual_gaussian)
        self.assertAlmostEqual(actual_gaussian, expected_gaussian)

        rho = 0.5
        actual_exp = self.meta.dispersal_kernel_strength(d, method='exponential', rho=rho)
        expected_exp = rho * np.exp(-rho * d)
        self._print_scalar('exponential expected', expected_exp)
        self._print_scalar('exponential actual', actual_exp)
        self.assertAlmostEqual(actual_exp, expected_exp)

        gamma = 1.5
        actual_cauchy = self.meta.dispersal_kernel_strength(d, method='cauchy', gamma=gamma)
        expected_cauchy = 1.0 / (1.0 + (d / gamma) ** 2)
        self._print_scalar('cauchy expected', expected_cauchy)
        self._print_scalar('cauchy actual', actual_cauchy)
        self.assertAlmostEqual(actual_cauchy, expected_cauchy)

        alpha = 2.0
        r0 = 1.5
        actual_power = self.meta.dispersal_kernel_strength(d, method='power_law', alpha=alpha, r0=r0)
        expected_power = (1.0 + d / r0) ** (-alpha)
        self._print_scalar('power_law expected', expected_power)
        self._print_scalar('power_law actual', actual_power)
        self.assertAlmostEqual(actual_power, expected_power)

    # ------------------------------------------------------------------
    # D, normalized_D, M
    # ------------------------------------------------------------------
    def test_uniform_D_normalizedD_and_M_properties(self):
        """矩阵链：uniform 方法下检查 D、normalized_D、M 的结构、行和与对角线语义。"""
        D = np.asarray(self.meta.calculate_dispersal_kernel_strength_matrix(method='uniform'))
        normalized_D = np.asarray(self.meta.normalized_calculate_dispersal_kernel_strength_matrix(method='uniform'))
        M = np.asarray(self.meta.get_disp_among_rate_matrix(0.2, method='uniform'))

        self._print_matrix('D (uniform)', D)
        self._print_matrix('normalized_D (uniform)', normalized_D)
        self._print_matrix('M (uniform)', M)

        self.assertTrue(np.allclose(np.diag(D), np.zeros(6)))
        offdiag = D[~np.eye(6, dtype=bool)]
        self.assertTrue(np.allclose(offdiag, np.ones_like(offdiag)))

        self.assertTrue(np.allclose(np.diag(normalized_D), np.zeros(6)))
        row_sums = normalized_D.sum(axis=1)
        self._print_scalar('Row sums of normalized_D', row_sums)
        self.assertTrue(np.allclose(row_sums, np.ones(6)))
        expected_offdiag_val = 1.0 / 5.0
        self.assertTrue(np.allclose(normalized_D[~np.eye(6, dtype=bool)], expected_offdiag_val))

        self.assertTrue(np.allclose(np.diag(M), np.full(6, 0.8)))
        self.assertTrue(np.allclose(M[~np.eye(6, dtype=bool)], 0.2 * expected_offdiag_val))
        M_row_sums = M.sum(axis=1)
        self._print_scalar('Row sums of M', M_row_sums)
        self.assertTrue(np.allclose(M_row_sums, np.ones(6)))

    def test_exponential_normalizedD_and_M_properties(self):
        """矩阵链：exponential 方法下检查 normalized_D 与 M 的行和，并 spot-check 最近邻权重大于远邻。"""
        rho = 1.0
        normalized_D = np.asarray(
            self.meta.normalized_calculate_dispersal_kernel_strength_matrix(method='exponential', rho=rho)
        )
        M = np.asarray(self.meta.get_disp_among_rate_matrix(0.2, method='exponential', rho=rho))

        self._print_matrix('normalized_D (exponential)', normalized_D)
        self._print_matrix('M (exponential)', M)

        self.assertTrue(np.allclose(normalized_D.sum(axis=1), np.ones(6)))
        self.assertTrue(np.allclose(M.sum(axis=1), np.ones(6)))
        self.assertTrue(np.allclose(np.diag(M), np.full(6, 0.8)))

        self.assertGreater(normalized_D[0, 1], normalized_D[0, 2])
        self.assertGreater(normalized_D[0, 1], normalized_D[0, 5])
        self.assertGreater(normalized_D[0, 3], normalized_D[0, 5])

    # ------------------------------------------------------------------
    # Downstream matrices under realistic occupancy
    # ------------------------------------------------------------------
    def test_get_patch_empty_sites_num_matrix_under_realistic_occupancy(self):
        """空位矩阵：6 个 patch 在 80% occupancy、每 patch 4 habitats × 100x100 下，应各自保留 8000 个空位。"""
        empty_matrix = np.asarray(self.meta.get_patch_empty_sites_num_matrix())
        self._print_matrix('patch_empty_sites_num_matrix', empty_matrix)

        expected_diag = np.full(6, 4 * self.empty_per_habitat)
        self.assertTrue(np.allclose(np.diag(empty_matrix), expected_diag))
        self.assertTrue(np.allclose(empty_matrix - np.diag(np.diag(empty_matrix)), np.zeros((6, 6))))

    def test_get_emigrants_matrix_matches_diag_pool_counts_times_M(self):
        """下游矩阵：get_emigrants_matrix() 应等于 diag(pool_counts) @ M（使用 h1 的 offspring_pool 计数）。"""
        m = 0.2
        counts = np.diag([100, 200, 300, 400, 500, 600]).astype(float)
        M = np.asarray(self.meta.get_disp_among_rate_matrix(m, method='uniform'))
        expected = counts @ M
        actual = self.meta.get_emigrants_matrix(m, method='uniform')

        self._print_matrix('Expected emigrants_matrix', expected)
        self._print_matrix('Actual emigrants_matrix', actual)
        self._assert_matrix_close(self, actual, expected)

    def test_get_immigrants_matrix_column_sums_equal_patch_empty_site_quotas(self):
        """下游矩阵：get_immigrants_matrix() 的每列和应等于对应 patch 的空位配额（本例均为 8000）。"""
        m = 0.2
        immigrants = np.asarray(self.meta.get_immigrants_matrix(m, method='uniform'))
        self._print_matrix('Actual immigrants_matrix', immigrants)
        col_sums = immigrants.sum(axis=0)
        self._print_scalar('Column sums of immigrants_matrix', col_sums)

        expected = np.full(6, 4 * self.empty_per_habitat, dtype=float)
        self.assertTrue(np.allclose(col_sums, expected))

    # ------------------------------------------------------------------
    # Execution-layer smoke test
    # ------------------------------------------------------------------
    def test_marker_dispersal_zero_rate_produces_no_immigrants(self):
        """执行层 smoke test：among-patch rate = 0 时，不应产生 immigrant markers。"""
        log_info = self.meta.dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool(
            total_disp_among_rate=0.0,
            method='uniform'
        )
        self._print_scalar('Returned log_info', log_info.strip())
        self._print_scalar('Immigrant marker count', self.meta.meta_immigrant_marker_pool_marker_num())
        self.assertIn('there are 0 individuals disperse', log_info)
        self.assertEqual(self.meta.meta_immigrant_marker_pool_marker_num(), 0)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDispersalKernelRealistic)
    runner = VerboseTextTestRunner(verbosity=2)
    runner.run(suite)
