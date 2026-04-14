# -*- coding: utf-8 -*-
"""
测试 habitat.py 中本次新增/修改的 6 个方法：
- hab_sex_reproduce_mutate_with_num
- hab_mix_sex_reproduce_mutate_with_num
- hab_sexual_reprodece_germinate
- hab_mixed_reproduce_germinate
- hab_sex_reproduce_mutate_into_offspring_pool
- hab_mix_reproduce_mutate_into_offspring_pool

额外功能：
- 在 multi_crossover 模式下，显式打印父母个体产生 gamete 时对应的
  switches / start / source / gamete，方便人工检查结果是否正确。

运行方式（建议在项目根目录执行）：
    python test/test_habitat_recombination_smoke.py
"""

from __future__ import annotations

import random
import traceback
import numpy as np

from bootstrap_metaibm import ensure_metaibm_on_path
ensure_metaibm_on_path()

from metaibm.individual import individual
from metaibm.habitat import habitat


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _fmt(arr: np.ndarray) -> str:
    return np.array2string(arr, separator=', ')


def _reconstruct_multi_crossover_trace(bi_genotype, recomb_rate, state_before):
    rs = np.random.RandomState()
    rs.set_state(state_before)

    hap1, hap2 = bi_genotype[0], bi_genotype[1]
    L = hap1.shape[0]
    switches = rs.random_sample(max(L - 1, 0)) < recomb_rate
    start = rs.randint(0, 2)

    source = np.empty(L, dtype=np.int8)
    source[0] = start
    if L > 1:
        source[1:] = (source[0] + np.cumsum(switches, dtype=np.int32)) & 1

    return switches, int(start), source


# ----------------------------------------------------------------------
# monkeypatch: 包装 make_gamete，在 multi_crossover 模式下记录 trace
# ----------------------------------------------------------------------
_original_make_gamete = individual.make_gamete


def _patched_make_gamete(self, trait_name, recomb_method='segregation', recomb_rate=0.0001):
    if not hasattr(self, '_recomb_trace_log'):
        self._recomb_trace_log = []

    if recomb_method == 'multi_crossover':
        state_before = np.random.get_state()
        gamete = _original_make_gamete(self, trait_name, recomb_method=recomb_method, recomb_rate=recomb_rate)
        switches, start, source = _reconstruct_multi_crossover_trace(
            self.genotype_set[trait_name],
            recomb_rate,
            state_before,
        )
        self._recomb_trace_log.append({
            'trait_name': trait_name,
            'switches': switches.copy(),
            'start': start,
            'source': source.copy(),
            'gamete': gamete.copy(),
        })
        return gamete
    else:
        return _original_make_gamete(self, trait_name, recomb_method=recomb_method, recomb_rate=recomb_rate)


individual.make_gamete = _patched_make_gamete


def _clear_trace(indi_obj):
    indi_obj._recomb_trace_log = []


def _print_trace(label: str, indi_obj):
    # print original parent genotype before gamete formation
    print('\n' + '-' * 72)
    if hasattr(indi_obj, 'genotype_set'):
        print('  parent original genotype:')
        for trait, (hap1, hap2) in indi_obj.genotype_set.items():
            print(f'    {trait}:')
            print(f'      hap1 = {_fmt(hap1)}')
            print(f'      hap2 = {_fmt(hap2)}')
    
    print(label)
    logs = getattr(indi_obj, '_recomb_trace_log', [])
    if len(logs) == 0:
        print('  no multi_crossover trace recorded')
        return
    for idx, item in enumerate(logs, start=1):
        print(f"  trace {idx} / trait = {item['trait_name']}")
        print(f"    switches = {item['switches'].astype(int).tolist()}")
        print(f"    start    = {item['start']}")
        print(f"    source   = {item['source'].tolist()}")
        print(f"    gamete   = {_fmt(item['gamete'])}")


def build_habitat() -> tuple[habitat, individual, individual]:
    hab = habitat(
        hab_name='h0',
        hab_index=0,
        hab_location=(0, 0),
        num_env_types=2,
        env_types_name=['env1', 'env2'],
        mean_env_ls=[0.5, 0.5],
        var_env_ls=[0.0, 0.0],
        length=2,
        width=2,
        dormancy_pool_max_size=10,
    )

    female = individual(
        species_id='sp_test',
        traits_num=2,
        pheno_names_ls=['trait_1', 'trait_2'],
        gender='female',
        genotype_set={
            'trait_1': [
                np.array([1, 1, 0, 0, 0, 1], dtype=float),
                np.array([0, 1, 1, 1, 0, 0], dtype=float),
            ],
            'trait_2': [
                np.array([0.2, 0.4, 0.6, 0.8], dtype=float),
                np.array([0.1, 0.3, 0.5, 0.7], dtype=float),
            ],
        },
        phenotype_set={'trait_1': 0.0, 'trait_2': 0.0},
    )

    male = individual(
        species_id='sp_test',
        traits_num=2,
        pheno_names_ls=['trait_1', 'trait_2'],
        gender='male',
        genotype_set={
            'trait_1': [
                np.array([0, 0, 1, 1, 1, 0], dtype=float),
                np.array([1, 0, 0, 0, 1, 1], dtype=float),
            ],
            'trait_2': [
                np.array([0.9, 0.8, 0.7, 0.6], dtype=float),
                np.array([0.5, 0.4, 0.3, 0.2], dtype=float),
            ],
        },
        phenotype_set={'trait_1': 0.0, 'trait_2': 0.0},
    )

    hab.add_individual(female, 0, 0)
    hab.add_individual(male, 0, 1)

    hab.asexual_parent_pos_ls = [(0, 0)]
    hab.species_category_for_sexual_parents_pos = {
        'sp_test': {'female': [(0, 0)], 'male': [(0, 1)]}
    }
    return hab, female, male


def _check_offspring_list(label: str, offsprings):
    print('\n' + '=' * 72)
    print(label)
    _assert(isinstance(offsprings, list), '返回值不是 list')
    _assert(len(offsprings) >= 1, 'offsprings 为空')
    child = offsprings[0]
    print(f'  offspring count = {len(offsprings)}')
    for trait in child.pheno_names_ls:
        hap1, hap2 = child.genotype_set[trait]
        print(f'  {trait}:')
        print(f'    hap1 = {_fmt(hap1)}')
        print(f'    hap2 = {_fmt(hap2)}')
        print(f'    phenotype = {child.phenotype_set[trait]}')


def main() -> int:
    np.random.seed(123)
    random.seed(123)

    # 1) hab_sex_reproduce_mutate_with_num
    hab1, f1, m1 = build_habitat()
    _clear_trace(f1); _clear_trace(m1)
    off1 = hab1.hab_sex_reproduce_mutate_with_num(
        mutation_rate=0.0,
        pheno_var_ls=[0.0, 0.0],
        num=1,
        recomb_method='multi_crossover',
        recomb_rate=0.3,
    )
    _check_offspring_list('Case 1: hab_sex_reproduce_mutate_with_num', off1)
    _print_trace('  female parent traces', f1)
    _print_trace('  male parent traces', m1)

    # 2) hab_mix_sex_reproduce_mutate_with_num
    hab2, f2, m2 = build_habitat()
    _clear_trace(f2); _clear_trace(m2)
    off2 = hab2.hab_mix_sex_reproduce_mutate_with_num(
        mutation_rate=0.0,
        pheno_var_ls=[0.0, 0.0],
        num=1,
        recomb_method='multi_crossover',
        recomb_rate=0.3,
    )
    _check_offspring_list('Case 2: hab_mix_sex_reproduce_mutate_with_num', off2)
    _print_trace('  female parent traces', f2)
    _print_trace('  male parent traces', m2)

    # 3) hab_sexual_reprodece_germinate
    hab3, f3, m3 = build_habitat()
    _clear_trace(f3); _clear_trace(m3)
    c3 = hab3.hab_sexual_reprodece_germinate(
        sexual_birth_rate=1.0,
        mutation_rate=0.0,
        pheno_var_ls=[0.0, 0.0],
        recomb_method='multi_crossover',
        recomb_rate=0.3,
    )
    print('\n' + '=' * 72)
    print('Case 3: hab_sexual_reprodece_germinate')
    print(f'  germinated count = {c3}')
    print(f'  indi_num after germination = {hab3.indi_num}')
    _print_trace('  female parent traces', f3)
    _print_trace('  male parent traces', m3)

    # 4) hab_mixed_reproduce_germinate
    hab4, f4, m4 = build_habitat()
    _clear_trace(f4); _clear_trace(m4)
    c4 = hab4.hab_mixed_reproduce_germinate(
        asexual_birth_rate=1.0,
        sexual_birth_rate=1.0,
        mutation_rate=0.0,
        pheno_var_ls=[0.0, 0.0],
        recomb_method='multi_crossover',
        recomb_rate=0.3,
    )
    print('\n' + '=' * 72)
    print('Case 4: hab_mixed_reproduce_germinate')
    print(f'  germinated count = {c4}')
    print(f'  indi_num after germination = {hab4.indi_num}')
    _print_trace('  female parent traces', f4)
    _print_trace('  male parent traces', m4)

    # 5) hab_sex_reproduce_mutate_into_offspring_pool
    hab5, f5, m5 = build_habitat()
    _clear_trace(f5); _clear_trace(m5)
    n5 = hab5.hab_sex_reproduce_mutate_into_offspring_pool(
        sexual_birth_rate=1.0,
        mutation_rate=0.0,
        pheno_var_ls=[0.0, 0.0],
        recomb_method='multi_crossover',
        recomb_rate=0.3,
    )
    _assert(n5 == len(hab5.offspring_pool), 'offspring_pool 数量不一致')
    _check_offspring_list('Case 5: hab_sex_reproduce_mutate_into_offspring_pool', hab5.offspring_pool)
    _print_trace('  female parent traces', f5)
    _print_trace('  male parent traces', m5)

    # 6) hab_mix_reproduce_mutate_into_offspring_pool
    hab6, f6, m6 = build_habitat()
    _clear_trace(f6); _clear_trace(m6)
    n6 = hab6.hab_mix_reproduce_mutate_into_offspring_pool(
        asexual_birth_rate=1.0,
        sexual_birth_rate=1.0,
        mutation_rate=0.0,
        pheno_var_ls=[0.0, 0.0],
        recomb_method='multi_crossover',
        recomb_rate=0.3,
    )
    _assert(n6 == len(hab6.offspring_pool), 'mixed offspring_pool 数量不一致')
    _check_offspring_list('Case 6: hab_mix_reproduce_mutate_into_offspring_pool', hab6.offspring_pool)
    _print_trace('  female parent traces', f6)
    _print_trace('  male parent traces', m6)

    print('\nPASS: habitat.py modified methods test passed')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print('FAIL: habitat.py modified methods test failed')
        print(f'异常类型: {type(e).__name__}')
        print(f'异常信息: {e}')
        traceback.print_exc()
        raise SystemExit(1)
