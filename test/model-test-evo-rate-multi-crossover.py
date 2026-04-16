
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model-test-evo-rate-multi-crossover.py

Purpose
-------
Use MetaIBM v3.4.0 source code to compare the evolutionary rate of a single
sexual population under two inheritance modes:
1) segregation
2) multi_crossover

Scenario
--------
- one population of 6400 individuals (80 x 80 microsites)
- initial phenotype_axis1 = 0.2
- initial phenotype_axis2 = 0.2
- habitat environment: e1 = 0.4, e2 = 0.4
- pure sexual reproduction on a mainland-like metacommunity (1 patch, 1 habitat)
- simulate for 5000 time steps
- record population mean phenotype through time
- compare segregation vs multi_crossover
- save a figure and CSV files into the same folder as this script

Outputs
-------
1) model-test-evo-rate-multi-crossover_mean_phenotype_curves.png
2) model-test-evo-rate-multi-crossover_mean_phenotype_history.csv
3) model-test-evo-rate-multi-crossover_summary.csv

How to run
----------
Recommended to run from the project root:
    python test/model-test-evo-rate-multi-crossover.py
"""

from __future__ import annotations

import copy
import csv
import random
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bootstrap_metaibm import ensure_metaibm_on_path
ensure_metaibm_on_path()

from metaibm.individual import individual
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity


# ============================================================
# CONFIG
# ============================================================
TIME_STEPS = 5000
REPEATS = 10

# ecology / evolution parameters (aligned with MetaIBM v3.4.0 defaults where possible)
BASE_DEAD_RATE = 0.1
FITNESS_WID = 0.5
SEXUAL_BIRTH_RATE = 1.0
MUTATION_RATE = 0.0001
PHENO_VAR_LS = (0.025, 0.025)
GENO_LEN_LS = (20, 20)
PHENO_NAMES_LS = ('phenotype_axis1', 'phenotype_axis2')
TRAITS_NUM = 2

# population / environment setting requested by the user
INIT_PHENO_LS = [0.2, 0.2]
ENV_MEAN_LS = [0.4, 0.4]
POP_LENGTH = 80
POP_WIDTH = 80
POP_SIZE = POP_LENGTH * POP_WIDTH  # 6400

# recombination comparison
METHODS = [
    ('segregation', 0.0),
    ('multi_crossover', 0.05),
]

# plotting / output
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PREFIX = SCRIPT_DIR / 'model-test-evo-rate-multi-crossover'
FIG_PATH = OUT_PREFIX.with_name(OUT_PREFIX.name + '_mean_phenotype_curves.png')
HISTORY_CSV = OUT_PREFIX.with_name(OUT_PREFIX.name + '_mean_phenotype_history.csv')
SUMMARY_CSV = OUT_PREFIX.with_name(OUT_PREFIX.name + '_summary.csv')


# ============================================================
# Helpers
# ============================================================
def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def build_single_population_mainland(init_pheno_ls, env_mean_ls):
    """
    Build a metacommunity with 1 patch, 1 habitat, 80x80 microsites (= 6400 individuals).
    Environment is fixed at env_mean_ls, while initial phenotype/genotype mean is init_pheno_ls.

    This bypasses hab_initialize() because we want initial phenotype != environment.
    """
    meta = metacommunity(metacommunity_name='mainland_test')
    p = patch('patch0', 0, (0, 0))
    p.add_habitat(
        hab_name='h0',
        hab_index=0,
        hab_location=(0, 0),
        num_env_types=2,
        env_types_name=('env_axis1', 'env_axis2'),
        mean_env_ls=list(env_mean_ls),
        var_env_ls=[0.0, 0.0],
        length=POP_LENGTH,
        width=POP_WIDTH,
        dormancy_pool_max_size=0,
    )
    meta.add_patch('patch0', p)

    hab = meta.set['patch0'].set['h0']
    species_id = 'sp1'

    for row in range(POP_LENGTH):
        for col in range(POP_WIDTH):
            gender = random.sample(('male', 'female'), 1)[0]
            indi = individual(
                species_id=species_id,
                traits_num=TRAITS_NUM,
                pheno_names_ls=PHENO_NAMES_LS,
                gender=gender,
            )
            indi.random_init_indi(
                mean_pheno_val_ls=init_pheno_ls,
                pheno_var_ls=PHENO_VAR_LS,
                geno_len_ls=GENO_LEN_LS,
            )
            hab.add_individual(indi, row, col)

    _assert(meta.get_meta_individual_num() == POP_SIZE, 'Population initialization failed: size != 6400')
    return meta


def get_population_mean_phenotype(meta_obj):
    """Return mean phenotype for phenotype_axis1 and phenotype_axis2 across all individuals."""
    vals1 = []
    vals2 = []
    for patch_id, patch_object in meta_obj.set.items():
        for h_id, h_object in patch_object.set.items():
            microsites = h_object.set['microsite_individuals']
            for row in range(h_object.length):
                for col in range(h_object.width):
                    indi = microsites[row][col]
                    if indi is not None:
                        vals1.append(float(indi.phenotype_set['phenotype_axis1']))
                        vals2.append(float(indi.phenotype_set['phenotype_axis2']))
    return float(np.mean(vals1)), float(np.mean(vals2))


def run_one_simulation(base_mainland, recomb_method, recomb_rate):
    """
    Run one replicate starting from the same initialized mainland.
    Pure sexual reproduction is used so that the comparison isolates recombination mode.
    """
    mainland = copy.deepcopy(base_mainland)
    mean1_hist = np.zeros(TIME_STEPS + 1, dtype=float)
    mean2_hist = np.zeros(TIME_STEPS + 1, dtype=float)

    # t = 0
    m1, m2 = get_population_mean_phenotype(mainland)
    mean1_hist[0] = m1
    mean2_hist[0] = m2

    for t in range(1, TIME_STEPS + 1):
        mainland.meta_dead_selection(BASE_DEAD_RATE, FITNESS_WID, method='niche_gaussian')
        mainland.meta_mainland_sexual_birth_mutate_germinate(
            SEXUAL_BIRTH_RATE,
            MUTATION_RATE,
            PHENO_VAR_LS,
            recomb_method=recomb_method,
            recomb_rate=recomb_rate,
        )
        m1, m2 = get_population_mean_phenotype(mainland)
        mean1_hist[t] = m1
        mean2_hist[t] = m2

    return mean1_hist, mean2_hist


def first_passage_time(series, threshold):
    """Return the first time index when series >= threshold; NaN if never reached."""
    idx = np.where(series >= threshold)[0]
    if len(idx) == 0:
        return np.nan
    return float(idx[0])


# ============================================================
# Main benchmark / simulation
# ============================================================
def main():
    print('=' * 96)
    print('MetaIBM v3.4.0 test: evolutionary rate comparison')
    print(f'Population size = {POP_SIZE} (80 x 80)')
    print(f'Initial phenotype = {INIT_PHENO_LS}')
    print(f'Environment = {ENV_MEAN_LS}')
    print(f'Time steps = {TIME_STEPS}')
    print(f'Repeats = {REPEATS}')
    print('=' * 96)

    time_axis = np.arange(TIME_STEPS + 1)
    target_halfway = 0.3  # halfway from 0.2 to 0.4

    all_results = {}
    summary_rows = []

    for method, rate in METHODS:
        print(f'Running method={method}, recomb_rate={rate} ...', flush=True)
        rep_mean1 = []
        rep_mean2 = []

        for rep in range(REPEATS):
            seed = 20260416 + rep
            random.seed(seed)
            np.random.seed(seed)
            base_mainland = build_single_population_mainland(INIT_PHENO_LS, ENV_MEAN_LS)

            # identical initial population for both methods within each replicate seed
            mean1_hist, mean2_hist = run_one_simulation(base_mainland, method, rate)
            rep_mean1.append(mean1_hist)
            rep_mean2.append(mean2_hist)

        rep_mean1 = np.vstack(rep_mean1)
        rep_mean2 = np.vstack(rep_mean2)

        mean_curve1 = rep_mean1.mean(axis=0)
        mean_curve2 = rep_mean2.mean(axis=0)
        sd_curve1 = rep_mean1.std(axis=0)
        sd_curve2 = rep_mean2.std(axis=0)

        t_half_1 = first_passage_time(mean_curve1, target_halfway)
        t_half_2 = first_passage_time(mean_curve2, target_halfway)

        summary_rows.append({
            'method': method,
            'recomb_rate': rate,
            'final_mean_phenotype_axis1': float(mean_curve1[-1]),
            'final_mean_phenotype_axis2': float(mean_curve2[-1]),
            'time_to_mean_0.3_axis1': t_half_1,
            'time_to_mean_0.3_axis2': t_half_2,
        })

        all_results[(method, rate)] = {
            'mean_curve1': mean_curve1,
            'mean_curve2': mean_curve2,
            'sd_curve1': sd_curve1,
            'sd_curve2': sd_curve2,
        }

    # ------------------------------------------------------------
    # Save history CSV
    # ------------------------------------------------------------
    with open(HISTORY_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['time_step']
        for method, rate in METHODS:
            key = f'{method}_r{rate}'
            header += [
                f'{key}_mean_axis1', f'{key}_sd_axis1',
                f'{key}_mean_axis2', f'{key}_sd_axis2',
            ]
        writer.writerow(header)

        for t in range(TIME_STEPS + 1):
            row = [t]
            for method, rate in METHODS:
                d = all_results[(method, rate)]
                row += [
                    float(d['mean_curve1'][t]), float(d['sd_curve1'][t]),
                    float(d['mean_curve2'][t]), float(d['sd_curve2'][t]),
                ]
            writer.writerow(row)

    # ------------------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------------------
    with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'method', 'recomb_rate',
                'final_mean_phenotype_axis1', 'final_mean_phenotype_axis2',
                'time_to_mean_0.3_axis1', 'time_to_mean_0.3_axis2',
            ]
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    ax1, ax2 = axes

    color_map = {
        'segregation': '#1f77b4',
        'multi_crossover': '#d62728',
    }

    for method, rate in METHODS:
        d = all_results[(method, rate)]
        c = color_map[method]
        label = f'{method} (r={rate})'

        ax1.plot(time_axis, d['mean_curve1'], color=c, lw=2, label=label)
        ax1.fill_between(time_axis, d['mean_curve1'] - d['sd_curve1'], d['mean_curve1'] + d['sd_curve1'], color=c, alpha=0.15)

        ax2.plot(time_axis, d['mean_curve2'], color=c, lw=2, label=label)
        ax2.fill_between(time_axis, d['mean_curve2'] - d['sd_curve2'], d['mean_curve2'] + d['sd_curve2'], color=c, alpha=0.15)

    for ax, env_val, title in [
        (ax1, ENV_MEAN_LS[0], 'phenotype_axis1'),
        (ax2, ENV_MEAN_LS[1], 'phenotype_axis2'),
    ]:
        ax.axhline(INIT_PHENO_LS[0], color='gray', ls='--', lw=1, alpha=0.7, label='initial phenotype' if ax is ax1 else None)
        ax.axhline(env_val, color='black', ls=':', lw=1.5, alpha=0.8, label='environment optimum' if ax is ax1 else None)
        ax.set_title(title)
        ax.set_xlabel('time step')
        ax.set_ylabel('population mean phenotype')
        ax.set_xlim(0, TIME_STEPS)
        ax.grid(alpha=0.2)

    handles, labels = ax1.get_legend_handles_labels()
    # de-duplicate legend labels
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    fig.legend([x[0] for x in uniq], [x[1] for x in uniq], loc='upper center', ncol=4, frameon=False)
    fig.suptitle('Evolutionary rate comparison in MetaIBM v3.4.0\ninitial phenotype = 0.2, environment = 0.4, population size = 6400')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------
    print('\nSummary')
    print('-' * 96)
    for row in summary_rows:
        print(
            f"method={row['method']:<16} recomb_rate={row['recomb_rate']:<5} "
            f"final_mean_axis1={row['final_mean_phenotype_axis1']:.4f} "
            f"final_mean_axis2={row['final_mean_phenotype_axis2']:.4f} "
            f"t_to_0.3_axis1={row['time_to_mean_0.3_axis1']} "
            f"t_to_0.3_axis2={row['time_to_mean_0.3_axis2']}"
        )

    print('\nFiles written:')
    print(f'  figure  : {FIG_PATH}')
    print(f'  history : {HISTORY_CSV}')
    print(f'  summary : {SUMMARY_CSV}')
    print('=' * 96)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print('FAIL: evolutionary-rate comparison test failed')
        print(f'Exception type: {type(e).__name__}')
        print(f'Exception message: {e}')
        traceback.print_exc()
        raise SystemExit(1)
