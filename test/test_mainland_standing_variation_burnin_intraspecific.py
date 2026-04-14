#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 mainland 需要空跑多少个 time-step 才能累积足够的 standing genetic variation。

本版本相对旧版的两个关键修改
--------------------------------
1. 只统计“同一 habitat 内、同一 species”的 intra-specific standing genetic variation，
   然后对 16 个 habitat（初始化对应 16 个 species source blocks）取均值。
2. 把 t = 0（初始化完成后、任何更新之前）也纳入统计与输出。

核心思路
--------
1. 构造一个与 model.py / model-sloss-GRFE.py 风格一致的 mainland：
   - 1 个 patch
   - 16 个 habitat
   - 每个 habitat 20x20 microsites
   - 2 个 trait, 每个 trait 的 genotype length = 20
2. 按你正式模型的更新方式空跑 mainland：
   - dead selection
   - birth / mutate / germination
3. 每隔若干 step 统计 standing genetic variation 指标：
   - mean heterozygosity (按位点平均 2p(1-p))
   - polymorphic-site fraction (0 < p < 1 的位点比例)
   - phenotype variance
   但统计口径改为：
   - 先在每个 habitat 内、对该 habitat 当前 species 的个体单独统计
   - 再对 16 个 habitat 的结果取平均
4. 通过“最近两个窗口的相对变化是否都小于阈值”来判断是否达到平台期，
   从而给出一个推荐的 burn-in time-step。

运行方式（建议在项目根目录执行）
--------------------------------
python test/test_mainland_standing_variation_burnin_intraspecific.py

如果你想修改参数，可直接在本文件底部的 CONFIG 区域调整。


结果解析：
1. mean_heterozygosity_over_traits
    het = 2.0 * p * (1.0 - p)
    其中 p 是某个位点上等位基因“1”的频率；然后先对一个 trait 的所有位点取平均，再对所有 trait 取平均。

2. polymorphic_fraction_over_traits
    poly_mask = (p > 0.0) & (p < 1.0)
    一个位点只要不是全 0 或全 1，就被视为多态位点。然后统计多态位点占总位点的比例。

3. phenotype_variance_over_traits
    pheno_var = np.var(phenotypes)
    然后对 trait 再求平均。
"""

from __future__ import annotations

import csv
import random
import traceback
from pathlib import Path

import numpy as np

from bootstrap_metaibm import ensure_metaibm_on_path
ensure_metaibm_on_path()

from metaibm.patch import patch
from metaibm.metacommunity import metacommunity


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _safe_rel_change(old: float, new: float, eps: float = 1e-12) -> float:
    denom = max(abs(old), eps)
    return abs(new - old) / denom


def build_mainland(
    meta_name='mainland',
    patch_num=1,
    patch_location_ls=[(0, 0)],
    hab_num=16,
    hab_length=20,
    hab_width=20,
    dormancy_pool_max_size=0,
    micro_environment_values_ls=[0.2, 0.4, 0.6, 0.8],
    macro_environment_values_ls=[0.2, 0.4, 0.6, 0.8],
    environment_types_num=2,
    environment_types_name=('x_axis_environment', 'y_axis_environment'),
    environment_variation_ls=[0.025, 0.025],
    traits_num=2,
    pheno_names_ls=('x_axis_phenotype', 'y_axis_phenotype'),
    pheno_var_ls=(0.025, 0.025),
    geno_len_ls=(20, 20),
    reproduce_mode='sexual',
    species_2_phenotype_ls=None,
):
    if species_2_phenotype_ls is None:
        species_2_phenotype_ls = [[i / 10, j / 10] for i in range(2, 10, 2) for j in range(2, 10, 2)]

    meta_object = metacommunity(metacommunity_name=meta_name)

    for i in range(patch_num):
        patch_name = f'patch{i}'
        patch_index = i
        location = patch_location_ls[i]
        patch_x_loc, patch_y_loc = location[0], location[1]
        p = patch(patch_name, patch_index, location)

        hab_num_length, hab_num_width = int(np.sqrt(hab_num)), int(np.sqrt(hab_num))

        for j in range(hab_num):
            habitat_name = f'h{j}'
            hab_index = j

            hab_x_loc = patch_x_loc * hab_num_length + j // hab_num_width
            hab_y_loc = patch_y_loc * hab_num_width + j % hab_num_width
            hab_location = (hab_x_loc, hab_y_loc)

            micro_environment_mean_value = micro_environment_values_ls[j // hab_num_width]
            macro_environment_mean_value = macro_environment_values_ls[j % hab_num_width]

            p.add_habitat(
                hab_name=habitat_name,
                hab_index=hab_index,
                hab_location=hab_location,
                num_env_types=environment_types_num,
                env_types_name=environment_types_name,
                mean_env_ls=[micro_environment_mean_value, macro_environment_mean_value],
                var_env_ls=environment_variation_ls,
                length=hab_length,
                width=hab_width,
                dormancy_pool_max_size=dormancy_pool_max_size,
            )

        meta_object.add_patch(patch_name=patch_name, patch_object=p)

    meta_object.meta_initialize(
        traits_num=traits_num,
        pheno_names_ls=pheno_names_ls,
        pheno_var_ls=pheno_var_ls,
        geno_len_ls=geno_len_ls,
        reproduce_mode=reproduce_mode,
        species_2_phenotype_ls=species_2_phenotype_ls,
    )
    return meta_object


# ============================================================
# intra-specific / intra-habitat standing variation 统计
# ============================================================

def _collect_habitat_individuals(meta_obj: metacommunity):
    """
    返回一个 dict:
        {(patch_id, h_id): [indi1, indi2, ...]}
    """
    hab_dict = {}
    for patch_id, patch_object in meta_obj.set.items():
        for h_id, h_object in patch_object.set.items():
            individuals = []
            microsites = h_object.set['microsite_individuals']
            for row in range(h_object.length):
                for col in range(h_object.width):
                    indi = microsites[row][col]
                    if indi is not None:
                        individuals.append(indi)
            hab_dict[(patch_id, h_id)] = individuals
    return hab_dict


def _summarize_single_group(individuals, pheno_names_ls):
    """
    对一个“同一 habitat 内、同一 species”的个体组统计 standing variation。
    """
    _assert(len(individuals) > 0, 'individual group 为空，无法统计')

    # 防御性检查：确认同一组内 species_id 一致
    species_ids = {indi.species_id for indi in individuals}
    _assert(len(species_ids) == 1, f'同一 habitat 内检测到多个 species: {species_ids}')

    result = {
        'individual_num': len(individuals),
        'species_id': next(iter(species_ids)),
    }

    all_mean_het = []
    all_poly_frac = []
    all_pheno_var = []

    for trait_name in pheno_names_ls:
        haplotypes = []
        phenotypes = []
        for indi in individuals:
            hap1, hap2 = indi.genotype_set[trait_name]
            haplotypes.append(np.asarray(hap1, dtype=float))
            haplotypes.append(np.asarray(hap2, dtype=float))
            phenotypes.append(float(indi.phenotype_set[trait_name]))

        hap_matrix = np.vstack(haplotypes)       # shape = (2N, L)
        p = hap_matrix.mean(axis=0)
        het = 2.0 * p * (1.0 - p)
        poly_mask = (p > 0.0) & (p < 1.0)

        mean_het = float(np.mean(het))
        poly_frac = float(np.mean(poly_mask))
        pheno_var = float(np.var(phenotypes))

        result[f'{trait_name}_mean_heterozygosity'] = mean_het
        result[f'{trait_name}_polymorphic_fraction'] = poly_frac
        result[f'{trait_name}_phenotype_variance'] = pheno_var

        all_mean_het.append(mean_het)
        all_poly_frac.append(poly_frac)
        all_pheno_var.append(pheno_var)

    result['mean_heterozygosity_over_traits'] = float(np.mean(all_mean_het))
    result['polymorphic_fraction_over_traits'] = float(np.mean(all_poly_frac))
    result['phenotype_variance_over_traits'] = float(np.mean(all_pheno_var))
    return result


def summarize_standing_variation(meta_obj: metacommunity, pheno_names_ls):
    """
    新统计口径：
    1) 先在每个 habitat 内统计“该 habitat 当前 species”的 intra-specific variation
    2) 再对 16 个 habitat 的指标取均值
    """
    habitat_groups = _collect_habitat_individuals(meta_obj)
    group_metrics = []

    for (patch_id, h_id), individuals in habitat_groups.items():
        _assert(len(individuals) > 0, f'{patch_id}-{h_id} 为空，无法统计')
        m = _summarize_single_group(individuals, pheno_names_ls)
        m['patch_id'] = patch_id
        m['habitat_id'] = h_id
        group_metrics.append(m)

    _assert(len(group_metrics) > 0, '没有 habitat metrics，无法统计')

    summary = {
        'habitat_count': len(group_metrics),
        'individual_num_total': int(sum(m['individual_num'] for m in group_metrics)),
        'individual_num_mean_per_habitat': float(np.mean([m['individual_num'] for m in group_metrics])),
    }

    keys_to_average = [
        'mean_heterozygosity_over_traits',
        'polymorphic_fraction_over_traits',
        'phenotype_variance_over_traits',
    ]
    for key in keys_to_average:
        summary[key] = float(np.mean([m[key] for m in group_metrics]))

    return summary, group_metrics


# ============================================================
# mainland burn-in 更新（尽量贴近你的 model.py 主逻辑）
# ============================================================

def step_mainland(
    mainland: metacommunity,
    reproduce_mode: str,
    base_dead_rate: float,
    fitness_wid: float,
    asexual_birth_rate: float,
    sexual_birth_rate: float,
    mutation_rate: float,
    pheno_var_ls,
    recomb_method='multi_crossover',
    recomb_rate=0.0,
):
    mainland.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')

    if reproduce_mode == 'asexual':
        mainland.meta_mainland_asexual_birth_mutate_germinate(
            asexual_birth_rate,
            mutation_rate,
            pheno_var_ls,
        )
    elif reproduce_mode == 'sexual':
        mainland.meta_mainland_mixed_birth_mutate_germinate(
            asexual_birth_rate,
            sexual_birth_rate,
            mutation_rate,
            pheno_var_ls,
            recomb_method=recomb_method,
            recomb_rate=recomb_rate,
        )
    else:
        raise ValueError(f'Unsupported reproduce_mode: {reproduce_mode}')


# ============================================================
# burn-in 判定
# ============================================================

def judge_plateau(history, keys, stabilize_window=4, rel_tol=0.05):
    need = stabilize_window * 2
    if len(history) < need:
        return False, {}

    recent = history[-stabilize_window:]
    prev = history[-need:-stabilize_window]

    rel_changes = {}
    all_ok = True
    for key in keys:
        recent_mean = float(np.mean([x[key] for x in recent]))
        prev_mean = float(np.mean([x[key] for x in prev]))
        rc = _safe_rel_change(prev_mean, recent_mean)
        rel_changes[key] = rc
        if rc >= rel_tol:
            all_ok = False
    return all_ok, rel_changes


# ============================================================
# 主测试逻辑
# ============================================================

def run_burnin_test(
    max_steps=2000,
    check_every=1,
    min_burnin=200,
    stabilize_window=4,
    rel_tol=0.05,
    csv_out='test_mainland_standing_variation_burnin_history_intraspecific.csv',
    reproduce_mode='sexual',
    recomb_method='multi_crossover',
    recomb_rate=0.0,
):
    # ---------------------------
    # 与你 model.py 接近的默认参数
    # ---------------------------
    base_dead_rate = 0.1
    fitness_wid = 0.5
    asexual_birth_rate = 0.5
    sexual_birth_rate = 1.0
    mutation_rate = 0.0001

    traits_num = 2
    pheno_names_ls = ('x_axis_phenotype', 'y_axis_phenotype')
    pheno_var_ls = (0.025, 0.025)
    geno_len_ls = (20, 20)
    species_2_phenotype_ls = [[i / 10, j / 10] for i in range(2, 10, 2) for j in range(2, 10, 2)]

    mainland = build_mainland(
        meta_name='mainland',
        patch_num=1,
        patch_location_ls=[(0, 0)],
        hab_num=16,
        hab_length=20,
        hab_width=20,
        dormancy_pool_max_size=0,
        micro_environment_values_ls=[0.2, 0.4, 0.6, 0.8],
        macro_environment_values_ls=[0.2, 0.4, 0.6, 0.8],
        environment_types_num=2,
        environment_types_name=('x_axis_environment', 'y_axis_environment'),
        environment_variation_ls=[0.025, 0.025],
        traits_num=traits_num,
        pheno_names_ls=pheno_names_ls,
        pheno_var_ls=pheno_var_ls,
        geno_len_ls=geno_len_ls,
        reproduce_mode=reproduce_mode,
        species_2_phenotype_ls=species_2_phenotype_ls,
    )

    history = []
    plateau_keys = [
        'mean_heterozygosity_over_traits',
        'polymorphic_fraction_over_traits',
        'phenotype_variance_over_traits',
    ]

    print('=' * 88)
    print('Testing mainland burn-in for standing genetic variation (intra-specific within habitat)')
    print(f'reproduce_mode   = {reproduce_mode}')
    print(f'recomb_method    = {recomb_method}')
    print(f'recomb_rate      = {recomb_rate}')
    print(f'max_steps        = {max_steps}')
    print(f'check_every      = {check_every}')
    print(f'min_burnin       = {min_burnin}')
    print(f'stabilize_window = {stabilize_window} checks')
    print(f'rel_tol          = {rel_tol}')
    print('=' * 88)

    recommended_burnin = None

    # --------------------------------------------------------
    # 先统计 t = 0（初始化完成后、任何更新之前）
    # --------------------------------------------------------
    metrics0, group_metrics0 = summarize_standing_variation(mainland, pheno_names_ls)
    metrics0['time_step'] = 0
    history.append(metrics0)
    print(
        f"step={0:4d} | habitats={metrics0['habitat_count']:2d} | "
        f"N_total={metrics0['individual_num_total']:5d} | "
        f"mean_het={metrics0['mean_heterozygosity_over_traits']:.6f} | "
        f"poly_frac={metrics0['polymorphic_fraction_over_traits']:.6f} | "
        f"pheno_var={metrics0['phenotype_variance_over_traits']:.6f}"
    )

    for step in range(1, max_steps + 1):
        step_mainland(
            mainland=mainland,
            reproduce_mode=reproduce_mode,
            base_dead_rate=base_dead_rate,
            fitness_wid=fitness_wid,
            asexual_birth_rate=asexual_birth_rate,
            sexual_birth_rate=sexual_birth_rate,
            mutation_rate=mutation_rate,
            pheno_var_ls=pheno_var_ls,
            recomb_method=recomb_method,
            recomb_rate=recomb_rate,
        )

        if step % check_every == 0:
            metrics, group_metrics = summarize_standing_variation(mainland, pheno_names_ls)
            metrics['time_step'] = step
            history.append(metrics)

            print(
                f"step={step:4d} | habitats={metrics['habitat_count']:2d} | "
                f"N_total={metrics['individual_num_total']:5d} | "
                f"mean_het={metrics['mean_heterozygosity_over_traits']:.6f} | "
                f"poly_frac={metrics['polymorphic_fraction_over_traits']:.6f} | "
                f"pheno_var={metrics['phenotype_variance_over_traits']:.6f}"
            )

            if step >= min_burnin:
                ok, rel_changes = judge_plateau(
                    history,
                    keys=plateau_keys,
                    stabilize_window=stabilize_window,
                    rel_tol=rel_tol,
                )
                if ok:
                    recommended_burnin = step
                    print('-' * 88)
                    print('Plateau criterion reached.')
                    for k, v in rel_changes.items():
                        print(f'  relative_change[{k}] = {v:.6f}')
                    print(f'  ==> recommended mainland burn-in = {recommended_burnin} steps')
                    print('-' * 88)
                    break

    # 写出历史 CSV，方便画图或进一步分析
    if len(history) > 0:
        out_path = Path(csv_out)
        with out_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f'History written to: {out_path.resolve()}')

    if recommended_burnin is None:
        recommended_burnin = max_steps
        print('=' * 88)
        print('No plateau reached within max_steps.')
        print(f'Use a provisional recommendation: mainland burn-in = {recommended_burnin} steps')
        print('You may increase max_steps or relax rel_tol if needed.')
        print('=' * 88)

    return recommended_burnin, history


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    # 主测试控制
    'max_steps': 2000,
    'check_every': 50,
    'min_burnin': 200,
    'stabilize_window': 4,
    'rel_tol': 0.05,
    'csv_out': 'test_mainland_standing_variation_burnin_history_intraspecific.csv',

    # 复制你当前测试脚本的设置
    'reproduce_mode': 'sexual',
    'recomb_method': 'multi_crossover',
    'recomb_rate': 0.05,
}


def main():
    recommended_burnin, history = run_burnin_test(**CONFIG)
    print('\n' + '=' * 88)
    print(f'Final recommendation: mainland burn-in = {recommended_burnin} time-steps')
    print('=' * 88)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('FAIL: mainland standing variation burn-in test failed')
        print(f'异常类型: {type(e).__name__}')
        print(f'异常信息: {e}')
        traceback.print_exc()
        raise SystemExit(1)
