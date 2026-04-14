#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark mainland recombination performance on the SAME mainland landscape used in
`test_mainland_standing_variation_burnin_intraspecific.py`.

Test goals
----------
1. Compare `multi_crossover` vs `segregation` in the mainland workflow.
2. Compare different `recomb_rate` values within `multi_crossover`.

Mainland landscape (kept consistent with the burn-in standing-variation test)
-----------------------------------------------------------------------------
- 1 patch
- 16 habitats
- each habitat = 20 x 20 microsites
- 2 traits
- genotype length per trait = 20
- reproduce_mode = 'sexual' (benchmark uses the SAME mainland mixed-birth workflow
  as the current experiment script)

Suggested run
-------------
Run from the project root:

python test/test_mainland_recombination_performance.py

or

python /path/to/test_mainland_recombination_performance.py
"""

from __future__ import annotations

import copy
import csv
import random
import time
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


# ============================================================
# mainland builder (same landscape style as burn-in test)
# ============================================================
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
        _assert(hab_num_length * hab_num_width == hab_num, 'hab_num must be a perfect square')

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
# one benchmark run on mainland workflow
# ============================================================
def run_one_benchmark(
    mainland: metacommunity,
    steps: int,
    base_dead_rate: float,
    fitness_wid: float,
    asexual_birth_rate: float,
    sexual_birth_rate: float,
    mutation_rate: float,
    pheno_var_ls,
    recomb_method: str,
    recomb_rate: float,
):
    t0 = time.perf_counter()
    for _ in range(steps):
        mainland.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
        mainland.meta_mainland_mixed_birth_mutate_germinate(
            asexual_birth_rate,
            sexual_birth_rate,
            mutation_rate,
            pheno_var_ls,
            recomb_method=recomb_method,
            recomb_rate=recomb_rate,
        )
    t1 = time.perf_counter()
    return t1 - t0


# ============================================================
# repeated benchmark helper
# ============================================================
def benchmark_setting(
    base_mainland: metacommunity,
    repeats: int,
    steps: int,
    base_dead_rate: float,
    fitness_wid: float,
    asexual_birth_rate: float,
    sexual_birth_rate: float,
    mutation_rate: float,
    pheno_var_ls,
    recomb_method: str,
    recomb_rate: float,
    base_seed: int = 123,
):
    elapsed_ls = []
    final_n_ls = []

    for rep in range(repeats):
        random.seed(base_seed + rep)
        np.random.seed(base_seed + rep)

        mainland = copy.deepcopy(base_mainland)
        elapsed = run_one_benchmark(
            mainland=mainland,
            steps=steps,
            base_dead_rate=base_dead_rate,
            fitness_wid=fitness_wid,
            asexual_birth_rate=asexual_birth_rate,
            sexual_birth_rate=sexual_birth_rate,
            mutation_rate=mutation_rate,
            pheno_var_ls=pheno_var_ls,
            recomb_method=recomb_method,
            recomb_rate=recomb_rate,
        )
        elapsed_ls.append(elapsed)
        final_n_ls.append(mainland.get_meta_individual_num())

    mean_elapsed = float(np.mean(elapsed_ls))
    std_elapsed = float(np.std(elapsed_ls))
    mean_final_n = float(np.mean(final_n_ls))
    steps_per_sec = steps / mean_elapsed if mean_elapsed > 0 else np.nan

    return {
        'recomb_method': recomb_method,
        'recomb_rate': recomb_rate,
        'repeats': repeats,
        'steps': steps,
        'mean_elapsed_sec': mean_elapsed,
        'std_elapsed_sec': std_elapsed,
        'steps_per_sec': steps_per_sec,
        'mean_final_N': mean_final_n,
    }


def print_results_table(title: str, rows):
    print('\n' + '=' * 96)
    print(title)
    print('=' * 96)
    print(f"{'method':<18} {'rate':>8} {'repeats':>8} {'steps':>8} {'mean_sec':>14} {'std_sec':>14} {'steps/sec':>14} {'final_N':>10}")
    print('-' * 96)
    for r in rows:
        print(
            f"{r['recomb_method']:<18} "
            f"{r['recomb_rate']:>8.4f} "
            f"{r['repeats']:>8d} "
            f"{r['steps']:>8d} "
            f"{r['mean_elapsed_sec']:>14.6f} "
            f"{r['std_elapsed_sec']:>14.6f} "
            f"{r['steps_per_sec']:>14.3f} "
            f"{r['mean_final_N']:>10.1f}"
        )


def write_csv(csv_path: str | Path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'recomb_method', 'recomb_rate', 'repeats', 'steps',
                'mean_elapsed_sec', 'std_elapsed_sec', 'steps_per_sec', 'mean_final_N'
            ]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    # ------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------
    repeats = 5
    steps = 200
    csv_out = 'test_mainland_recombination_performance.csv'

    # mainland workflow parameters (aligned with current experiment defaults)
    base_dead_rate = 0.1
    fitness_wid = 0.5
    asexual_birth_rate = 0.5
    sexual_birth_rate = 1.0
    mutation_rate = 0.0001
    pheno_var_ls = (0.025, 0.025)

    compare_vs_segregation = [
        ('segregation', 0.0),
        ('multi_crossover', 0.05),  # current default in the experiment script
    ]

    compare_multi_rates = [
        ('multi_crossover', 0.0),
        ('multi_crossover', 0.01),
        ('multi_crossover', 0.05),
        ('multi_crossover', 0.10),
        ('multi_crossover', 0.20),
    ]

    # ------------------------------------------------------------
    # Build ONE mainland baseline, then deepcopy it for each benchmark setting
    # ------------------------------------------------------------
    random.seed(123)
    np.random.seed(123)
    base_mainland = build_mainland()

    # ------------------------------------------------------------
    # Benchmark 1: segregation vs multi_crossover
    # ------------------------------------------------------------
    rows_vs = []
    for method, rate in compare_vs_segregation:
        rows_vs.append(
            benchmark_setting(
                base_mainland=base_mainland,
                repeats=repeats,
                steps=steps,
                base_dead_rate=base_dead_rate,
                fitness_wid=fitness_wid,
                asexual_birth_rate=asexual_birth_rate,
                sexual_birth_rate=sexual_birth_rate,
                mutation_rate=mutation_rate,
                pheno_var_ls=pheno_var_ls,
                recomb_method=method,
                recomb_rate=rate,
            )
        )

    print_results_table('Benchmark 1: segregation vs multi_crossover', rows_vs)
    seg_row = next(r for r in rows_vs if r['recomb_method'] == 'segregation')
    mc_row = next(r for r in rows_vs if r['recomb_method'] == 'multi_crossover')
    slowdown = mc_row['mean_elapsed_sec'] / seg_row['mean_elapsed_sec'] if seg_row['mean_elapsed_sec'] > 0 else np.nan
    print(f"\nslowdown (multi_crossover@0.05 / segregation) = {slowdown:.3f} x")

    # ------------------------------------------------------------
    # Benchmark 2: multi_crossover across recomb_rate values
    # ------------------------------------------------------------
    rows_rates = []
    for method, rate in compare_multi_rates:
        rows_rates.append(
            benchmark_setting(
                base_mainland=base_mainland,
                repeats=repeats,
                steps=steps,
                base_dead_rate=base_dead_rate,
                fitness_wid=fitness_wid,
                asexual_birth_rate=asexual_birth_rate,
                sexual_birth_rate=sexual_birth_rate,
                mutation_rate=mutation_rate,
                pheno_var_ls=pheno_var_ls,
                recomb_method=method,
                recomb_rate=rate,
            )
        )

    print_results_table('Benchmark 2: multi_crossover with different recomb_rate', rows_rates)

    all_rows = []
    for r in rows_vs:
        rr = dict(r)
        rr['benchmark'] = 'segregation_vs_multi_crossover'
        all_rows.append(rr)
    for r in rows_rates:
        rr = dict(r)
        rr['benchmark'] = 'multi_crossover_rate_scan'
        all_rows.append(rr)

    # write CSV with benchmark label included
    csv_path = Path(csv_out)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'benchmark', 'recomb_method', 'recomb_rate', 'repeats', 'steps',
                'mean_elapsed_sec', 'std_elapsed_sec', 'steps_per_sec', 'mean_final_N'
            ]
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print('\n' + '-' * 96)
    print(f'CSV written to: {csv_path.resolve()}')
    print('-' * 96)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print('FAIL: mainland recombination performance benchmark failed')
        print(f'Exception type: {type(e).__name__}')
        print(f'Exception message: {e}')
        traceback.print_exc()
        raise SystemExit(1)
