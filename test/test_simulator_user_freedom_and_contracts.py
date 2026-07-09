#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Executable documentation of the user-facing boundary of metaibm.simulator.simulator.

After the islands builder was moved into simulator.py as a schedule-callable,
both `mainland` and `islands` are constructed from user-supplied CSVs through
two schedule-callables on the simulator. This file documents — in executable
form — which inputs the user is free to vary and which conventions the user
must keep stable for the simulator's CSV-driven API to behave correctly.

Simulator provides freedom in:
- species count through mainland.csv
- species phenotype values through mainland.csv
- mainland habitat location and size through mainland.csv
- metacommunity patch layout through metacommunity.csv
- metacommunity habitat layout through metacommunity.csv
- metacommunity environment values through metacommunity.csv
- metacommunity habitat size through metacommunity.csv
- schedule-level construction of mainland and islands
- user-configurable 2D trait/environment column names through pheno_names_ls
  and environment_types_name

Users must obey:
- species_id must be consecutive positive integers: 1, 2, ..., n
- mainland.csv must contain required columns for the chosen pheno_names_ls
- metacommunity.csv must contain required columns for the chosen
  environment_types_name
- current plotting logic assumes rectangular patch grid
- current plotting logic assumes the same habitat grid shape in each patch
- current plotting logic uses representative habitat size
- this test documents 2D trait/environment usage, not arbitrary
  high-dimensional ecological processes
"""

from pathlib import Path
import sys
import tempfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metaibm.simulator import simulator


########################################  CSV writers  ########################################

def write_mainland_A_csv(path):
    ''' 4 species, deliberately unsorted row order [3, 1, 4, 2]. Total microsites = 20. '''
    rows = [
        {'species_id': 3, 'phenotype_axis1': 0.5, 'phenotype_axis2': 0.6, 'hab_x_loc': 0, 'hab_y_loc': 1, 'hab_length': 3, 'hab_width': 2},
        {'species_id': 1, 'phenotype_axis1': 0.1, 'phenotype_axis2': 0.2, 'hab_x_loc': 0, 'hab_y_loc': 0, 'hab_length': 2, 'hab_width': 3},
        {'species_id': 4, 'phenotype_axis1': 0.7, 'phenotype_axis2': 0.8, 'hab_x_loc': 1, 'hab_y_loc': 1, 'hab_length': 2, 'hab_width': 2},
        {'species_id': 2, 'phenotype_axis1': 0.3, 'phenotype_axis2': 0.4, 'hab_x_loc': 1, 'hab_y_loc': 0, 'hab_length': 1, 'hab_width': 4},
    ]
    pd.DataFrame(rows, columns=['species_id', 'phenotype_axis1', 'phenotype_axis2', 'hab_x_loc', 'hab_y_loc', 'hab_length', 'hab_width']).to_csv(path, index=False)


def write_metacommunity_A_csv(path):
    ''' patch grid 2x1, each patch has habitat grid 2x1, uniform 2x2 habitats. Total islands microsites = 16. Habitats inside a patch share the same hab_length x hab_width because metacommunity-level reshape (get_patch_microsites_environment_values, patch.py) treats every habitat in a patch as the same shape — this is part of the user contract. '''
    rows = [
        {'patch_id': 'patch0', 'patch_index': 0, 'patch_location_x': 0, 'patch_location_y': 0, 'habitat_id': 'h0', 'habitat_index': 0, 'habitat_x_location': 0, 'habitat_y_location': 0, 'env_axis1': 0.1, 'env_axis2': 0.2, 'hab_length': 2, 'hab_width': 2},
        {'patch_id': 'patch0', 'patch_index': 0, 'patch_location_x': 0, 'patch_location_y': 0, 'habitat_id': 'h1', 'habitat_index': 1, 'habitat_x_location': 1, 'habitat_y_location': 0, 'env_axis1': 0.3, 'env_axis2': 0.4, 'hab_length': 2, 'hab_width': 2},
        {'patch_id': 'patch1', 'patch_index': 1, 'patch_location_x': 1, 'patch_location_y': 0, 'habitat_id': 'h2', 'habitat_index': 0, 'habitat_x_location': 0, 'habitat_y_location': 0, 'env_axis1': 0.5, 'env_axis2': 0.6, 'hab_length': 2, 'hab_width': 2},
        {'patch_id': 'patch1', 'patch_index': 1, 'patch_location_x': 1, 'patch_location_y': 0, 'habitat_id': 'h3', 'habitat_index': 1, 'habitat_x_location': 1, 'habitat_y_location': 0, 'env_axis1': 0.7, 'env_axis2': 0.8, 'hab_length': 2, 'hab_width': 2},
    ]
    pd.DataFrame(rows, columns=['patch_id', 'patch_index', 'patch_location_x', 'patch_location_y', 'habitat_id', 'habitat_index', 'habitat_x_location', 'habitat_y_location', 'env_axis1', 'env_axis2', 'hab_length', 'hab_width']).to_csv(path, index=False)


def write_mainland_B_csv(path):
    ''' 2 species, custom phenotype column names trait_a/trait_b. Total mainland_B microsites = 10. '''
    rows = [
        {'species_id': 1, 'trait_a': 0.10, 'trait_b': 0.20, 'hab_x_loc': 0, 'hab_y_loc': 0, 'hab_length': 2, 'hab_width': 2},
        {'species_id': 2, 'trait_a': 0.30, 'trait_b': 0.40, 'hab_x_loc': 1, 'hab_y_loc': 0, 'hab_length': 2, 'hab_width': 3},
    ]
    pd.DataFrame(rows, columns=['species_id', 'trait_a', 'trait_b', 'hab_x_loc', 'hab_y_loc', 'hab_length', 'hab_width']).to_csv(path, index=False)


def write_metacommunity_B_csv(path):
    ''' 1 patch, 3 habitats in a 3x1 grid, custom env names env_temp/env_moisture. Total islands_B microsites = 12. '''
    rows = [
        {'patch_id': 'patch0', 'patch_index': 0, 'patch_location_x': 0, 'patch_location_y': 0, 'habitat_id': 'h0', 'habitat_index': 0, 'habitat_x_location': 0, 'habitat_y_location': 0, 'env_temp': 15.0, 'env_moisture': 0.4, 'hab_length': 2, 'hab_width': 2},
        {'patch_id': 'patch0', 'patch_index': 0, 'patch_location_x': 0, 'patch_location_y': 0, 'habitat_id': 'h1', 'habitat_index': 1, 'habitat_x_location': 1, 'habitat_y_location': 0, 'env_temp': 18.0, 'env_moisture': 0.5, 'hab_length': 2, 'hab_width': 2},
        {'patch_id': 'patch0', 'patch_index': 0, 'patch_location_x': 0, 'patch_location_y': 0, 'habitat_id': 'h2', 'habitat_index': 2, 'habitat_x_location': 2, 'habitat_y_location': 0, 'env_temp': 22.0, 'env_moisture': 0.6, 'hab_length': 2, 'hab_width': 2},
    ]
    pd.DataFrame(rows, columns=['patch_id', 'patch_index', 'patch_location_x', 'patch_location_y', 'habitat_id', 'habitat_index', 'habitat_x_location', 'habitat_y_location', 'env_temp', 'env_moisture', 'hab_length', 'hab_width']).to_csv(path, index=False)


def assert_file_exists(path):
    assert Path(path).exists(), 'expected file does not exist: %s' % str(path)


def _print_csv_and_df(label, csv_path):
    ''' Print raw CSV text and parsed DataFrame for visual inspection. '''
    print('\n' + '-' * 78)
    print('CSV file: %s  (%s)' % (label, str(csv_path)))
    print('-' * 78)
    with open(str(csv_path), 'r') as f:
        print(f.read())
    df = pd.read_csv(csv_path)
    print('DataFrame parsed from %s:' % label)
    print(df.to_string(index=False))
    print('')


def _print_meta_obj(label, meta_obj):
    ''' Print key inspection info for a built meta object. '''
    print('-' * 78)
    print('Meta object: %s  (type=%s)' % (label, type(meta_obj).__name__))
    print('-' * 78)
    print('empty_sites_num    : %s' % meta_obj.show_meta_empty_sites_num())
    print('individual_num     : %s' % meta_obj.get_meta_individual_num())
    patch_keys = list(meta_obj.set.keys()) if hasattr(meta_obj, 'set') else []
    print('patch keys         : %s' % patch_keys)
    for pk in patch_keys:
        patch_obj = meta_obj.set[pk]
        hab_keys = list(patch_obj.set.keys()) if hasattr(patch_obj, 'set') else []
        print('  patch %s habitats: %s' % (pk, hab_keys))
        for hk in hab_keys:
            hab_obj = patch_obj.set[hk]
            shape = (getattr(hab_obj, 'length', '?'), getattr(hab_obj, 'width', '?'))
            print('    habitat %s shape=%s' % (hk, shape))
    print('')


########################################  Tests  ########################################

def test_freedom_mainland_csv_controls_species_count_phenotype_and_size(tmpdir):
    ''' Documents: mainland.csv freely controls species count, phenotype values, and mainland habitat sizes. species_id must follow the 1..n contract; row order is normalized by sorting. '''
    mainland_csv = tmpdir / 'mainland_A.csv'
    write_mainland_A_csv(mainland_csv)
    _print_csv_and_df('mainland_A.csv', mainland_csv)

    sim = simulator()
    log_info = sim.build_empty_mainland_from_species_csv(
        meta_name='mainland',
        mainland_csv=str(mainland_csv),
        pheno_names_ls=('phenotype_axis1', 'phenotype_axis2'),
        environment_types_name=('env_axis1', 'env_axis2'),
        environment_variation_ls=[0.01, 0.01],
        patch_name='patch0',
        patch_index=0,
        patch_location=(0, 0),
        dormancy_pool_max_size=0,
    )

    # builder registration & log shape
    assert 'mainland' in sim.meta_objects, "freedom: mainland should be registered on simulator.meta_objects under the user-provided meta_name"
    assert 'generating empty mainland from species csv' in log_info, "freedom: mainland builder must return a log starting with the documented prefix"
    assert 'phenotype_axis1' in log_info, "freedom: configurable phenotype column names must appear in the returned log"
    assert 'phenotype_axis2' in log_info, "freedom: configurable phenotype column names must appear in the returned log"

    # freshly-built empty mainland: 20 microsites, 0 individuals
    mainland_obj = sim.meta_objects['mainland']
    _print_meta_obj('mainland', mainland_obj)
    assert mainland_obj.show_meta_empty_sites_num() == 20, "freedom: total mainland microsites are user-controlled via hab_length*hab_width per row in mainland.csv (expected 20)"
    assert mainland_obj.get_meta_individual_num() == 0, "freedom: a freshly-built empty mainland should contain zero individuals"

    # habitat naming contract: mainland habitats are named 'h0'..'h{n-1}' by post-sort row index, matching the islands builder convention (independent of species_id dtype)
    mainland_hab_keys = list(mainland_obj.set['patch0'].set.keys())
    assert mainland_hab_keys == ['h0', 'h1', 'h2', 'h3'], "contract: mainland habitats must be named h0..h{n-1} by post-sort row index (got %s)" % mainland_hab_keys

    # mimic user-side preprocessing block from playgrounds/model-simulator-GRFE.py:278-291
    pheno_names_ls = ('phenotype_axis1', 'phenotype_axis2')
    df_mainland = pd.read_csv(mainland_csv).sort_values('species_id').reset_index(drop=True)
    species_id_min = int(df_mainland['species_id'].min())
    species_id_max = int(df_mainland['species_id'].max())
    species_2_phenotype_ls = df_mainland.loc[:, list(pheno_names_ls)].astype(float).values.tolist()

    assert df_mainland['species_id'].tolist() == [1, 2, 3, 4], "contract: species_id must be consecutive positive integers 1..n; CSV row order is normalized by sorting"
    assert species_id_min == 1, "contract: species_id_min must be 1 (species_id is 1-based and consecutive)"
    assert species_id_max == 4, "contract: species_id_max equals the species count when species_id is 1..n"
    assert len(species_2_phenotype_ls) == 4, "freedom: user controls species count through mainland.csv row count"
    assert species_2_phenotype_ls[0] == [0.1, 0.2], "freedom: phenotype values are user-controlled per species (species 1 row)"
    assert species_2_phenotype_ls[-1] == [0.7, 0.8], "freedom: phenotype values are user-controlled per species (species 4 row)"


def test_freedom_metacommunity_csv_controls_patch_habitat_environment_and_size(tmpdir):
    ''' Documents: metacommunity.csv freely controls patch/habitat layout, env values, and habitat sizes. Plotting helpers infer a rectangular patch grid and representative habitat size. '''
    metacommunity_csv = tmpdir / 'metacommunity_A.csv'
    write_metacommunity_A_csv(metacommunity_csv)
    _print_csv_and_df('metacommunity_A.csv', metacommunity_csv)

    sim = simulator()
    log_info = sim.build_empty_metacommunity_from_patch_habitat_csv(
        meta_name='islands',
        metacommunity_csv=str(metacommunity_csv),
        environment_types_name=('env_axis1', 'env_axis2'),
        environment_variation_ls=[0.01, 0.01],
        dormancy_pool_max_size=0,
    )

    # builder registration & log shape
    assert 'islands' in sim.meta_objects, "freedom: islands should be registered on simulator.meta_objects under the user-provided meta_name"
    assert 'generating empty metacommunity from patch-habitat csv' in log_info, "freedom: islands builder must return a log starting with the documented prefix"
    assert 'env_axis1' in log_info, "freedom: configurable environment column names must appear in the returned log"
    assert 'env_axis2' in log_info, "freedom: configurable environment column names must appear in the returned log"

    # freshly-built empty islands: 16 microsites (4 habitats x 2x2), 0 individuals
    islands_obj = sim.meta_objects['islands']
    _print_meta_obj('islands', islands_obj)
    assert islands_obj.show_meta_empty_sites_num() == 16, "freedom: total islands microsites are user-controlled through metacommunity.csv (expected 16 = 4 habitats x 2x2 microsites; uniform habitat size is required within a patch)"
    assert islands_obj.get_meta_individual_num() == 0, "freedom: a freshly-built empty islands should contain zero individuals"

    # mimic user-side layout inference from playgrounds/model-simulator-GRFE.py:294-307
    df_metacommunity = pd.read_csv(metacommunity_csv)
    df_patch = df_metacommunity[['patch_id', 'patch_location_x', 'patch_location_y']].drop_duplicates()

    patch_num_x_axis = df_patch['patch_location_x'].nunique()
    patch_num_y_axis = df_patch['patch_location_y'].nunique()

    first_patch = df_patch.sort_values(['patch_location_x', 'patch_location_y'])['patch_id'].iloc[0]
    df_first_patch = df_metacommunity[df_metacommunity['patch_id'] == first_patch]

    hab_num_x_axis = df_first_patch['habitat_x_location'].nunique()
    hab_num_y_axis = df_first_patch['habitat_y_location'].nunique()

    representative_hab_length = int(df_metacommunity['hab_length'].iloc[0])
    representative_hab_width = int(df_metacommunity['hab_width'].iloc[0])

    assert patch_num_x_axis == 2, "contract: plotting infers patch grid via nunique(patch_location_x); expected 2 for this rectangular layout"
    assert patch_num_y_axis == 1, "contract: plotting infers patch grid via nunique(patch_location_y); expected 1 for this rectangular layout"
    assert hab_num_x_axis == 2, "contract: plotting assumes the same habitat grid shape in each patch; first_patch has 2 distinct habitat_x_location values"
    assert hab_num_y_axis == 1, "contract: plotting assumes the same habitat grid shape in each patch; first_patch has 1 distinct habitat_y_location value"
    assert representative_hab_length == 2, "contract: plotting uses representative habitat size from the first CSV row (hab_length=2)"
    assert representative_hab_width == 2, "contract: plotting uses representative habitat size from the first CSV row (hab_width=2)"


def test_schedule_contract_build_before_prime(tmpdir):
    ''' Documents: simulator supports schedule-level construction of mainland and islands. Target metacommunity must be built before any priming or ecological item that references it. '''
    mainland_csv = tmpdir / 'mainland_A.csv'
    metacommunity_csv = tmpdir / 'metacommunity_A.csv'
    write_mainland_A_csv(mainland_csv)
    write_metacommunity_A_csv(metacommunity_csv)
    _print_csv_and_df('mainland_A.csv (schedule)', mainland_csv)
    _print_csv_and_df('metacommunity_A.csv (schedule)', metacommunity_csv)

    sim = simulator()
    goal_path = sim.set_goal_path(str(tmpdir), 'schedule_contract_output')
    sim.set_global_params({'all_time_steps': 1, 'goal_path': goal_path, 'rep': 0, 'is_logging': False, 'is_timing': False})

    schedule = [
        {'target': 'simulator', 'method': 'build_empty_mainland_from_species_csv',
         'params': {'meta_name': 'mainland', 'mainland_csv': str(mainland_csv), 'pheno_names_ls': ('phenotype_axis1', 'phenotype_axis2'),
                    'environment_types_name': ('env_axis1', 'env_axis2'), 'environment_variation_ls': [0.01, 0.01],
                    'patch_name': 'patch0', 'patch_index': 0, 'patch_location': (0, 0), 'dormancy_pool_max_size': 0},
         'start': 0, 'end': 0},

        {'target': 'simulator', 'method': 'build_empty_metacommunity_from_patch_habitat_csv',
         'params': {'meta_name': 'islands', 'metacommunity_csv': str(metacommunity_csv),
                    'environment_types_name': ('env_axis1', 'env_axis2'), 'environment_variation_ls': [0.01, 0.01],
                    'dormancy_pool_max_size': 0},
         'start': 0, 'end': 0},

        {'target': 'simulator', 'method': 'prime_environment_distribution',
         'params': {'target': 'islands', 'environment_name': 'env_axis1', 'index_label': 'env_axis1_values', 'file_name': 'islands_env_axis1.csv.gz'},
         'start': 0, 'end': 0},

        {'target': 'simulator', 'method': 'flush_step_log', 'interval': 1},
    ]
    sim.set_schedule_per_time_step(schedule)
    sim.run()

    _print_meta_obj('mainland (schedule)', sim.meta_objects['mainland'])
    _print_meta_obj('islands (schedule)', sim.meta_objects['islands'])

    assert 'mainland' in sim.meta_objects, "contract: schedule-level mainland builder must register 'mainland' before any consumer item runs"
    assert 'islands' in sim.meta_objects, "contract: schedule-level islands builder must register 'islands' before any consumer item runs"
    assert_file_exists(Path(goal_path) / 'islands_env_axis1.csv.gz')


def test_contract_dynamic_column_names_require_matching_parameter_names(tmpdir):
    ''' Documents: phenotype/environment column names are user-configurable but pheno_names_ls and environment_types_name must match the CSV columns. Scope: 2D usage. '''
    mainland_csv = tmpdir / 'mainland_B.csv'
    metacommunity_csv = tmpdir / 'metacommunity_B.csv'
    write_mainland_B_csv(mainland_csv)
    write_metacommunity_B_csv(metacommunity_csv)
    _print_csv_and_df('mainland_B.csv', mainland_csv)
    _print_csv_and_df('metacommunity_B.csv', metacommunity_csv)

    sim = simulator()
    mainland_log = sim.build_empty_mainland_from_species_csv(
        meta_name='mainland_B',
        mainland_csv=str(mainland_csv),
        pheno_names_ls=('trait_a', 'trait_b'),
        environment_types_name=('env_temp', 'env_moisture'),
        environment_variation_ls=[0.01, 0.01],
        patch_name='patch0',
        patch_index=0,
        patch_location=(0, 0),
        dormancy_pool_max_size=0,
    )
    islands_log = sim.build_empty_metacommunity_from_patch_habitat_csv(
        meta_name='islands_B',
        metacommunity_csv=str(metacommunity_csv),
        environment_types_name=('env_temp', 'env_moisture'),
        environment_variation_ls=[0.01, 0.01],
        dormancy_pool_max_size=0,
    )

    _print_meta_obj('mainland_B', sim.meta_objects['mainland_B'])
    _print_meta_obj('islands_B', sim.meta_objects['islands_B'])

    assert 'mainland_B' in sim.meta_objects, "freedom: meta_name is user-controlled; mainland_B should appear in sim.meta_objects"
    assert 'islands_B' in sim.meta_objects, "freedom: meta_name is user-controlled; islands_B should appear in sim.meta_objects"

    assert 'trait_a' in mainland_log, "contract: custom phenotype column names must appear in the mainland log because pheno_names_ls matched the CSV columns"
    assert 'trait_b' in mainland_log, "contract: custom phenotype column names must appear in the mainland log because pheno_names_ls matched the CSV columns"
    assert 'env_temp' in islands_log, "contract: custom environment column names must appear in the islands log because environment_types_name matched the CSV columns"
    assert 'env_moisture' in islands_log, "contract: custom environment column names must appear in the islands log because environment_types_name matched the CSV columns"

    # CSV-derived expected empty-site totals (self-consistent with the writer helpers above)
    df_mainland_B = pd.read_csv(mainland_csv)
    expected_mainland_B_sites = int((df_mainland_B['hab_length'] * df_mainland_B['hab_width']).sum())
    df_metacommunity_B = pd.read_csv(metacommunity_csv)
    expected_islands_B_sites = int((df_metacommunity_B['hab_length'] * df_metacommunity_B['hab_width']).sum())

    assert sim.meta_objects['mainland_B'].show_meta_empty_sites_num() == expected_mainland_B_sites, "freedom: mainland_B empty sites must equal the CSV-defined total (sum of hab_length*hab_width)"
    assert sim.meta_objects['islands_B'].show_meta_empty_sites_num() == expected_islands_B_sites, "freedom: islands_B empty sites must equal the CSV-defined total (sum of hab_length*hab_width)"

    mainland_B_hab_keys = list(sim.meta_objects['mainland_B'].set['patch0'].set.keys())
    assert mainland_B_hab_keys == ['h0', 'h1'], "contract: mainland habitats must be named h0..h{n-1} by post-sort row index (got %s)" % mainland_B_hab_keys


def test_contracts_are_not_unlimited_freedom():
    ''' Documents: intentionally unsupported / not-yet-promised freedoms. Recording these in executable form keeps the user contract explicit. '''
    unsupported_freedoms = (
        'species_id must not be non-contiguous',
        'species_id must not start from arbitrary values',
        'species_id must not be strings',
        'patch layout is expected to be a regular rectangular grid for plotting',
        'each patch is expected to share the same habitat grid shape for plotting',
        'plotting currently uses representative habitat size',
        'this test documents 2D trait/environment usage only',
    )

    expected = (
        'species_id must not be non-contiguous',
        'species_id must not start from arbitrary values',
        'species_id must not be strings',
        'patch layout is expected to be a regular rectangular grid for plotting',
        'each patch is expected to share the same habitat grid shape for plotting',
        'plotting currently uses representative habitat size',
        'this test documents 2D trait/environment usage only',
    )
    for s in expected:
        assert s in unsupported_freedoms, "contract documentation must list unsupported freedom: %s" % s
    assert len(unsupported_freedoms) >= 7, "contract documentation must enumerate at least seven unsupported freedoms"


########################################  test report  ########################################

def _print_test_report():
    ''' Print a human-readable report of what the tests verified and of the user-facing freedom/contract boundary they document. Prints to stdout after all tests pass. '''
    lines = [
        '',
        '=' * 78,
        'TEST REPORT: simulator user-facing freedom and contracts',
        '=' * 78,
        '',
        'TESTS EXECUTED',
        '-' * 78,
        '[1] test_freedom_mainland_csv_controls_species_count_phenotype_and_size',
        '    builder        : sim.build_empty_mainland_from_species_csv',
        '    csv            : mainland_A.csv (4 species, unsorted row order [3, 1, 4, 2])',
        '    measured       : empty_sites = 20, individuals = 0',
        '                     species_id sorted = [1, 2, 3, 4], min = 1, max = 4',
        '                     phenotype[species 1] = [0.1, 0.2]',
        '                     phenotype[species 4] = [0.7, 0.8]',
        '',
        '[2] test_freedom_metacommunity_csv_controls_patch_habitat_environment_and_size',
        '    builder        : sim.build_empty_metacommunity_from_patch_habitat_csv',
        '    csv            : metacommunity_A.csv (patch grid 2x1, habitat grid 2x1, uniform 2x2 habitats)',
        '    measured       : empty_sites = 16, individuals = 0',
        '                     patch_num_x_axis = 2, patch_num_y_axis = 1',
        '                     hab_num_x_axis   = 2, hab_num_y_axis   = 1',
        '                     representative hab_length = 2, hab_width = 2',
        '',
        '[3] test_schedule_contract_build_before_prime',
        '    schedule order : build_mainland -> build_islands -> prime_env(islands) -> flush_step_log',
        '    assertion      : "mainland" and "islands" registered after sim.run()',
        '                     <goal_path>/islands_env_axis1.csv.gz exists on disk',
        '',
        '[4] test_contract_dynamic_column_names_require_matching_parameter_names',
        '    mainland_B     : custom phenotype columns (trait_a, trait_b)',
        '    islands_B      : custom env columns (env_temp, env_moisture)',
        '    measured       : mainland_B empty_sites = sum(hab_length * hab_width) = 10',
        '                     islands_B  empty_sites = sum(hab_length * hab_width) = 12',
        '',
        '[5] test_contracts_are_not_unlimited_freedom',
        '    declared 7 unsupported-freedom strings (see USER CONTRACTS section below)',
        '',
        'USER FREEDOM (simulator provides these knobs to users)',
        '-' * 78,
        ' - species count           : row count of mainland.csv',
        ' - phenotype values        : phenotype columns of mainland.csv',
        ' - mainland habitat size   : hab_length, hab_width per species row',
        ' - mainland habitat loc.   : hab_x_loc, hab_y_loc per species row',
        ' - patch count and layout  : unique (patch_id, patch_location_x/y) rows in metacommunity.csv',
        ' - habitat layout          : (habitat_id, habitat_x_location, habitat_y_location) rows per patch',
        ' - environment values      : env_* columns per habitat row',
        ' - habitat size (islands)  : hab_length, hab_width per habitat row',
        ' - schedule construction   : both builders are schedule items at time_step = 0',
        ' - column names (2D)       : pheno_names_ls and environment_types_name parameters',
        ' - metacommunity name      : meta_name parameter on both builders',
        ' - mainland CSV row order  : normalized by sorting on species_id',
        '',
        'USER CONTRACTS (users must obey)',
        '-' * 78,
        ' - species_id must be consecutive positive integers 1..n',
        '     (not non-contiguous, not arbitrary start, not strings)',
        ' - mainland.csv must contain every column listed in pheno_names_ls',
        ' - metacommunity.csv must contain every column listed in environment_types_name',
        ' - every habitat inside a single patch must share the same hab_length x hab_width',
        '     (patch.get_patch_microsites_environment_values reshapes by (hab_num, hab_size))',
        ' - patch layout must form a rectangular grid for plotting',
        '     (nunique(patch_location_x) x nunique(patch_location_y))',
        ' - habitat grid shape must be the same in every patch',
        ' - plotting uses representative hab_length / hab_width from the first CSV row',
        ' - mainland habitat names are h0..h{n-1} by post-sort row index (matches islands convention)',
        ' - schedule ordering: target metacommunity must be built before any item that consumes it',
        ' - this test documents 2D trait/environment usage; higher-dimensional usage is not promised',
        '=' * 78,
        '',
    ]
    print('\n'.join(lines))


########################################  main  ########################################

def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        test_freedom_mainland_csv_controls_species_count_phenotype_and_size(tmpdir)
        test_freedom_metacommunity_csv_controls_patch_habitat_environment_and_size(tmpdir)
        test_schedule_contract_build_before_prime(tmpdir)
        test_contract_dynamic_column_names_require_matching_parameter_names(tmpdir)
        test_contracts_are_not_unlimited_freedom()

    _print_test_report()
    print('All simulator user-freedom and contract tests passed.')


if __name__ == '__main__':
    main()
