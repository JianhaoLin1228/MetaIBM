# -*- coding: utf-8 -*-
"""
测试 individual.py 中本次新增/修改的方法：
- _copy_haplotype
- _segregate_one_haplotype
- _multi_crossover
- make_gamete

额外功能：
- 显式打印 _multi_crossover 对应的 switches / start / source，方便人工检查重组结果是否正确。

运行方式（建议在项目根目录执行）：
    python test/test_individual_recombination.py
"""

from __future__ import annotations

import random
import traceback
import numpy as np

from bootstrap_metaibm import ensure_metaibm_on_path
ensure_metaibm_on_path()

from metaibm.individual import individual


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _fmt(arr: np.ndarray) -> str:
    return np.array2string(arr, separator=', ')


def _same_genotype_dict(a, b) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        hap_a1, hap_a2 = a[k]
        hap_b1, hap_b2 = b[k]
        if not (np.array_equal(hap_a1, hap_b1) and np.array_equal(hap_a2, hap_b2)):
            return False
    return True


def _reconstruct_multi_crossover_trace(hap1: np.ndarray, recomb_rate: float):
    """
    根据调用 _multi_crossover() 之前保存的 numpy RNG state，
    复原这一次重组实际使用的：
    - switches
    - start
    - source

    注意：必须在真正调用 _multi_crossover() 之前先保存 np.random.get_state()。
    """
    state = np.random.get_state()
    rs = np.random.RandomState()
    rs.set_state(state)

    L = hap1.shape[0]
    switches = rs.random_sample(max(L - 1, 0)) < recomb_rate
    start = rs.randint(0, 2)

    source = np.empty(L, dtype=np.int8)
    source[0] = start
    if L > 1:
        source[1:] = (source[0] + np.cumsum(switches, dtype=np.int32)) & 1

    return switches, start, source


def build_individual() -> individual:
    genotype_set = {
        'trait_1': [
            np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=float),
            np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=float),
        ],
        'trait_2': [
            np.array([0.2, 0.4, 0.6, 0.8], dtype=float),
            np.array([0.1, 0.3, 0.5, 0.7], dtype=float),
        ],
    }
    phenotype_set = {'trait_1': 0.0, 'trait_2': 0.0}
    return individual(
        species_id='sp_test',
        traits_num=2,
        pheno_names_ls=['trait_1', 'trait_2'],
        gender='female',
        genotype_set=genotype_set,
        phenotype_set=phenotype_set,
    )


def main() -> int:
    np.random.seed(42)
    random.seed(42)

    indi = build_individual()
    original_genotype_set = {k: [v[0].copy(), v[1].copy()] for k, v in indi.genotype_set.items()}

    print('=' * 72)
    print('individual.py smoke test')
    print('Parent genotype_set:')
    for trait_name, (hap1, hap2) in indi.genotype_set.items():
        print(f'  {trait_name}:')
        print(f'    hap1 = {_fmt(hap1)}')
        print(f'    hap2 = {_fmt(hap2)}')

    # 1) _copy_haplotype
    print('\n[Test 1] _copy_haplotype')
    copied = indi._copy_haplotype(indi.genotype_set['trait_1'][0])
    _assert(isinstance(copied, np.ndarray), '_copy_haplotype 返回值不是 np.ndarray')
    _assert(np.array_equal(copied, indi.genotype_set['trait_1'][0]), '_copy_haplotype 内容不一致')
    _assert(copied is not indi.genotype_set['trait_1'][0], '_copy_haplotype 没有返回新对象')
    print(f'  copied hap = {_fmt(copied)}')

    # 2) _segregate_one_haplotype
    print('\n[Test 2] _segregate_one_haplotype')
    seg = indi._segregate_one_haplotype(indi.genotype_set['trait_1'])
    _assert(isinstance(seg, np.ndarray), '_segregate_one_haplotype 返回值不是 np.ndarray')
    _assert(seg.shape == (6,), f'_segregate_one_haplotype shape 错误: {seg.shape}')
    hap1_t1, hap2_t1 = indi.genotype_set['trait_1']
    _assert(np.all((seg == hap1_t1) | (seg == hap2_t1)), '_segregate_one_haplotype 结果存在非法位点来源')
    print(f'  segregated gamete = {_fmt(seg)}')

    # 3) _multi_crossover
    print('\n[Test 3] _multi_crossover (5 explicit samples with switches)')
    samples = []
    for i in range(30):
        # 先保存 RNG 状态，再真正调用 _multi_crossover，之后复原本次 switches
        state_before = np.random.get_state()
        g = indi._multi_crossover(indi.genotype_set['trait_1'], recomb_rate=0.35)

        # 复原 trace
        np.random.set_state(state_before)
        switches, start, source = _reconstruct_multi_crossover_trace(hap1_t1, recomb_rate=0.35)

        # 再把全局 RNG 状态恢复成“调用 _multi_crossover 之后”的状态
        # 这里通过再次真正调用一次 RandomState 推进逻辑并不可靠，
        # 因此直接重新设为 _multi_crossover 调用完成后的新状态：
        # 最稳妥方法是重新用 build_individual 之外的真实状态保存/恢复；
        # 这里我们改成在进入循环时先保存 before，再在真正调用完成后立即保存 after。
        # 为简化，这部分用更直接方案：循环里另外存 after_state。
        
        _assert(isinstance(g, np.ndarray), '_multi_crossover 返回值不是 np.ndarray')
        _assert(g.shape == (6,), f'_multi_crossover shape 错误: {g.shape}')
        _assert(np.all((g == hap1_t1) | (g == hap2_t1)), '_multi_crossover 结果存在非法位点来源')
        if i < 5:
            samples.append((g.copy(), switches.copy(), int(start), source.copy()))

        # 为了保证后续随机序列与真实 _multi_crossover 调用一致，这里重新执行一次真实推进：
        # 使用保存的 before 状态恢复后，再调用一次 _multi_crossover 丢弃结果，使 RNG 前进到正确位置。
        np.random.set_state(state_before)
        _ = indi._multi_crossover(indi.genotype_set['trait_1'], recomb_rate=0.35)

    for idx, (g, switches, start, source) in enumerate(samples, start=1):
        print(f'  sample {idx}:')
        print(f'    switches = {switches.astype(int).tolist()}')
        print(f'    start    = {start}')
        print(f'    source   = {source.tolist()}')
        print(f'    gamete   = {_fmt(g)}')

    # 4) make_gamete
    print('\n[Test 4] make_gamete')
    g_seg = indi.make_gamete('trait_1', recomb_method='segregation')

    state_before = np.random.get_state()
    g_rec = indi.make_gamete('trait_1', recomb_method='multi_crossover', recomb_rate=0.35)
    np.random.set_state(state_before)
    switches, start, source = _reconstruct_multi_crossover_trace(hap1_t1, recomb_rate=0.35)
    np.random.set_state(state_before)
    _ = indi.make_gamete('trait_1', recomb_method='multi_crossover', recomb_rate=0.35)

    _assert(isinstance(g_seg, np.ndarray), 'make_gamete(segregation) 返回值不是 np.ndarray')
    _assert(isinstance(g_rec, np.ndarray), 'make_gamete(multi_crossover) 返回值不是 np.ndarray')
    print(f'  make_gamete(segregation)     = {_fmt(g_seg)}')
    print(f'  make_gamete switches         = {switches.astype(int).tolist()}')
    print(f'  make_gamete start            = {start}')
    print(f'  make_gamete source           = {source.tolist()}')
    print(f'  make_gamete(multi_crossover) = {_fmt(g_rec)}')

    # 5) parent genotype 不应被污染
    print('\n[Test 5] parent genotype_set unchanged')
    _assert(_same_genotype_dict(indi.genotype_set, original_genotype_set), '调用后 parent genotype_set 被修改了')
    print('  parent genotype_set unchanged = True')

    print('\nPASS: individual.py modified methods test passed')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print('FAIL: individual.py modified methods test failed')
        print(f'异常类型: {type(e).__name__}')
        print(f'异常信息: {e}')
        traceback.print_exc()
        raise SystemExit(1)
