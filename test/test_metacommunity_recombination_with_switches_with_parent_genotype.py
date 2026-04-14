# -*- coding: utf-8 -*-
"""
测试 metacommunity.py 中本次新增/修改的 5 个方法：
- meta_mainland_sexual_birth_mutate_germinate
- meta_mainland_mixed_birth_mutate_germinate
- meta_sex_reproduce_mutate_into_offspring_pool
- meta_mix_reproduce_mutate_into_offspring_pool
- meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool

额外功能：
- 在 multi_crossover 模式下，显式打印父母个体产生 gamete 时对应的
  switches / start / source / gamete，方便人工检查结果是否正确。

运行方式（建议在项目根目录执行）：
    python test/test_metacommunity_recombination_smoke.py
"""

from __future__ import annotations

import random
import traceback
import numpy as np

from bootstrap_metaibm import ensure_metaibm_on_path
ensure_metaibm_on_path()

from metaibm.individual import individual
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity


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
    if hasattr(indi_obj, 'genotype_set'):
        print('  parent original genotype:')
        for trait, (hap1, hap2) in indi_obj.genotype_set.items():
            print(f'    {trait}:')
            print(f'      hap1 = {_fmt(hap1)}')
            print(f'      hap2 = {_fmt(hap2)}')
    print('\n' + '-' * 72)
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


def build_meta() -> tuple[metacommunity, individual, individual]:
    meta = metacommunity('meta_test')
    p = patch('p0', 0, (0, 0))
    p.add_habitat(
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
    h = p.set['h0']

    female = individual(
        species_id='sp_test', traits_num=2, pheno_names_ls=['trait_1', 'trait_2'], gender='female',
        genotype_set={
            'trait_1': [np.array([1,1,0,0,0,1], dtype=float), np.array([0,1,1,1,0,0], dtype=float)],
            'trait_2': [np.array([0.2,0.4,0.6,0.8], dtype=float), np.array([0.1,0.3,0.5,0.7], dtype=float)],
        },
        phenotype_set={'trait_1': 0.0, 'trait_2': 0.0},
    )
    male = individual(
        species_id='sp_test', traits_num=2, pheno_names_ls=['trait_1', 'trait_2'], gender='male',
        genotype_set={
            'trait_1': [np.array([0,0,1,1,1,0], dtype=float), np.array([1,0,0,0,1,1], dtype=float)],
            'trait_2': [np.array([0.9,0.8,0.7,0.6], dtype=float), np.array([0.5,0.4,0.3,0.2], dtype=float)],
        },
        phenotype_set={'trait_1': 0.0, 'trait_2': 0.0},
    )
    h.add_individual(female, 0, 0)
    h.add_individual(male, 0, 1)
    h.asexual_parent_pos_ls = [(0, 0)]
    h.species_category_for_sexual_parents_pos = {'sp_test': {'female': [(0, 0)], 'male': [(0, 1)]}}

    meta.add_patch('p0', p)
    return meta, female, male


def _print_first_offspring(label: str, meta_obj: metacommunity):
    print('\n' + '=' * 72)
    print(label)
    pool_num = meta_obj.meta_offspring_pool_individual_num()
    print(f'  offspring_pool size = {pool_num}')
    if pool_num > 0:
        child = meta_obj.set['p0'].set['h0'].offspring_pool[0]
        for trait in child.pheno_names_ls:
            hap1, hap2 = child.genotype_set[trait]
            print(f'  {trait}:')
            print(f'    hap1 = {_fmt(hap1)}')
            print(f'    hap2 = {_fmt(hap2)}')
            print(f'    phenotype = {child.phenotype_set[trait]}')


def main() -> int:
    np.random.seed(789)
    random.seed(789)

    # 1) meta_mainland_sexual_birth_mutate_germinate
    m1, f1, m1p = build_meta()
    _clear_trace(f1); _clear_trace(m1p)
    log1 = m1.meta_mainland_sexual_birth_mutate_germinate(
        1.0, 0.0, [0.0, 0.0],
        recomb_method='multi_crossover',
        recomb_rate=0.3
    )
    print('\n' + '=' * 72)
    print('Case 1: meta_mainland_sexual_birth_mutate_germinate')
    print(log1.strip())
    _print_trace('  female parent traces', f1)
    _print_trace('  male parent traces', m1p)

    # 2) meta_mainland_mixed_birth_mutate_germinate
    m2, f2, m2p = build_meta()
    _clear_trace(f2); _clear_trace(m2p)
    log2 = m2.meta_mainland_mixed_birth_mutate_germinate(
        1.0, 1.0, 0.0, [0.0, 0.0],
        recomb_method='multi_crossover',
        recomb_rate=0.3
    )
    print('\n' + '=' * 72)
    print('Case 2: meta_mainland_mixed_birth_mutate_germinate')
    print(log2.strip())
    _print_trace('  female parent traces', f2)
    _print_trace('  male parent traces', m2p)

    # 3) meta_sex_reproduce_mutate_into_offspring_pool
    m3, f3, m3p = build_meta()
    _clear_trace(f3); _clear_trace(m3p)
    log3 = m3.meta_sex_reproduce_mutate_into_offspring_pool(
        1.0, 0.0, [0.0, 0.0],
        recomb_method='multi_crossover',
        recomb_rate=0.3
    )
    print('\n' + '=' * 72)
    print('Case 3: meta_sex_reproduce_mutate_into_offspring_pool')
    print(log3.strip())
    _print_first_offspring('  first offspring from meta sexual offspring_pool', m3)
    _print_trace('  female parent traces', f3)
    _print_trace('  male parent traces', m3p)

    # 4) meta_mix_reproduce_mutate_into_offspring_pool
    m4, f4, m4p = build_meta()
    _clear_trace(f4); _clear_trace(m4p)
    log4 = m4.meta_mix_reproduce_mutate_into_offspring_pool(
        1.0, 1.0, 0.0, [0.0, 0.0],
        recomb_method='multi_crossover',
        recomb_rate=0.3
    )
    print('\n' + '=' * 72)
    print('Case 4: meta_mix_reproduce_mutate_into_offspring_pool')
    print(log4.strip())
    _print_first_offspring('  first offspring from meta mixed offspring_pool', m4)
    _print_trace('  female parent traces', f4)
    _print_trace('  male parent traces', m4p)

    # 5) meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool
    m5, f5, m5p = build_meta()
    _clear_trace(f5); _clear_trace(m5p)
    h5 = m5.set['p0'].set['h0']
    h5.offspring_marker_pool = [('p0', 'h0', 'sexual'), ('p0', 'h0', 'mix_sexual')]
    log5 = m5.meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool(
        mutation_rate=0.0,
        pheno_var_ls=[0.0, 0.0],
        recomb_method='multi_crossover',
        recomb_rate=0.3,
    )
    print('\n' + '=' * 72)
    print('Case 5: meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool')
    print(log5.strip())
    hab = m5.set['p0'].set['h0']
    print(f'  indi_num after birth = {hab.indi_num}')
    print('  current microsite individuals:')
    for row in range(hab.length):
        for col in range(hab.width):
            obj = hab.set['microsite_individuals'][row][col]
            if obj is not None:
                print(f'    microsite ({row}, {col}): species={obj.species_id}, gender={obj.gender}')
    _print_trace('  female parent traces', f5)
    _print_trace('  male parent traces', m5p)

    print('\nPASS: metacommunity.py modified methods test passed')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print('FAIL: metacommunity.py modified methods test failed')
        print(f'异常类型: {type(e).__name__}')
        print(f'异常信息: {e}')
        traceback.print_exc()
        raise SystemExit(1)
