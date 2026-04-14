#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialize a MetaIBM metacommunity landscape from:
1) patch_habitat_layouts.csv (patch/habitat spatial structure; controlled by patch_num)
2) two 32x32 environment CSV files (environment axis 1 and axis 2)

This script ONLY builds the empty metacommunity landscape (patches + habitats + environment means).
It does NOT run the subsequent ecological/evolutionary simulation.

How to use:
- edit the GLOBAL VARIABLES section below
- run the file directly
- no argparse / parser.add_argument is used
"""


from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import bootstrap_metaibm as _bootstrap
_bootstrap.ensure_metaibm_on_path()

from metaibm.patch import patch
from metaibm.metacommunity import metacommunity


# =========================================================
# GLOBAL VARIABLES: edit these directly
# =========================================================
PATCH_NUM = 256                      # choose from: 1 / 4 / 16 / 64 / 256
IS_SAME_HETEROGENEITY = True       # True = same; False = reduce
META_NAME = 'metacommunity'

LAYOUT_CSV = 'patch_habitat_layouts.csv'
ENV_AXIS1_CSV = '32x32_habitats_env1.csv'
ENV_AXIS2_CSV = '32x32_habitats_env2.csv'

# Optional: set your data directory here. If None, the script searches around itself.
# Example:
# DATA_DIR = r'C:/Users/Jianhao Lin/Desktop/MetaIBM-read-environmental-gradient/experiments'
DATA_DIR = None

HAB_LENGTH = 10
HAB_WIDTH = 10
DORMANCY_POOL_MAX_SIZE = 0
ENVIRONMENT_TYPES_NAME = ('x_axis_environment', 'y_axis_environment')
ENVIRONMENT_VARIATION_LS = (0.025, 0.025)

# Optional: export merged initialization summary to CSV
SUMMARY_CSV = None
# SUMMARY_CSV = 'init_summary_patch16_same.csv'


def _candidate_roots(explicit_data_dir: Optional[Path] = None) -> list[Path]:
    here = Path(__file__).resolve().parent
    roots = [here]
    if explicit_data_dir is not None:
        roots.insert(0, Path(explicit_data_dir).resolve())
    roots.extend([here.parent, Path.cwd(), Path.cwd().resolve()])

    seen = set()
    ordered = []
    for p in roots:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered


def _resolve_first_existing(candidates: Sequence[str], roots: Sequence[Path]) -> Path:
    for root in roots:
        for name in candidates:
            path = root / name
            if path.exists():
                return path

    msg = (
        'Cannot find any of the candidate files:\n'
        + '\n'.join(f'  - {name}' for name in candidates)
        + '\nSearched under:\n'
        + '\n'.join(f'  - {str(root)}' for root in roots)
    )
    raise FileNotFoundError(msg)


def load_patch_habitat_layout(layout_csv: Path, patch_num: int) -> pd.DataFrame:
    df = pd.read_csv(layout_csv)
    required_cols = [
        'patch_num', 'patch_size', 'patch_id', 'patch_index',
        'patch_location_x', 'patch_location_y',
        'global_habitat_num', 'habitat_id', 'habitat_index',
        'habitat_x_location', 'habitat_y_location',
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Layout CSV is missing required columns: {missing}')

    sub = df[df['patch_num'] == patch_num].copy()
    if sub.empty:
        raise ValueError(f'No rows found in layout CSV for patch_num={patch_num}')

    unique_coords = sub[['habitat_x_location', 'habitat_y_location']].drop_duplicates()
    if len(unique_coords) != 1024:
        raise ValueError(
            f'patch_num={patch_num} should contain 1024 unique habitat coordinates, got {len(unique_coords)}'
        )
    return sub


def _read_env_multiindex_csv(env_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(env_csv, header=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def extract_environment_grid(env_df: pd.DataFrame, patch_num: int, is_same_heterogeneity: bool) -> pd.DataFrame:
    key = (f'patch_num={patch_num}', str(bool(is_same_heterogeneity)))
    needed = [('meta', 'x_index'), ('meta', 'y_index'), key]
    try:
        sub = env_df.loc[:, needed].copy()
    except KeyError as e:
        raise KeyError(
            f'Cannot find columns for patch_num={patch_num}, '
            f'is_same_heterogeneity={is_same_heterogeneity}. Expected key: {key}'
        ) from e

    sub.columns = ['x_index', 'y_index', 'value']
    sub['x_index'] = sub['x_index'].astype(int)
    sub['y_index'] = sub['y_index'].astype(int)

    unique_coords = sub[['x_index', 'y_index']].drop_duplicates()
    if len(unique_coords) != 1024:
        raise ValueError(
            f'Environment CSV slice for patch_num={patch_num}, '
            f'is_same_heterogeneity={is_same_heterogeneity} should have 1024 unique coordinates.'
        )
    return sub


def initialize_metacommunity_from_layout(
    patch_num: int,
    is_same_heterogeneity: bool,
    meta_name: str = 'metacommunity',
    layout_csv: Optional[str | Path] = None,
    env_axis1_csv: Optional[str | Path] = None,
    env_axis2_csv: Optional[str | Path] = None,
    data_dir: Optional[str | Path] = None,
    hab_length: int = 10,
    hab_width: int = 10,
    dormancy_pool_max_size: int = 0,
    environment_types_name: Tuple[str, str] = ('x_axis_environment', 'y_axis_environment'),
    environment_variation_ls: Tuple[float, float] = (0.025, 0.025),
) -> tuple[metacommunity, pd.DataFrame]:

    if patch_num not in {1, 4, 16, 64, 256}:
        raise ValueError('patch_num must be one of {1, 4, 16, 64, 256}')

    roots = _candidate_roots(Path(data_dir).resolve() if data_dir is not None else None)
    layout_candidates = [str(layout_csv)] if layout_csv is not None else [LAYOUT_CSV]
    env_axis1_candidates = [str(env_axis1_csv)] if env_axis1_csv is not None else [ENV_AXIS1_CSV]
    env_axis2_candidates = [str(env_axis2_csv)] if env_axis2_csv is not None else [ENV_AXIS2_CSV]

    layout_path = _resolve_first_existing(layout_candidates, roots)
    env_axis1_path = _resolve_first_existing(env_axis1_candidates, roots)
    env_axis2_path = _resolve_first_existing(env_axis2_candidates, roots)

    layout_df = load_patch_habitat_layout(layout_path, patch_num)
    env1_df = _read_env_multiindex_csv(env_axis1_path)
    env2_df = _read_env_multiindex_csv(env_axis2_path)

    env1_grid = extract_environment_grid(env1_df, patch_num, is_same_heterogeneity).rename(columns={'value': 'env_axis1'})
    env2_grid = extract_environment_grid(env2_df, patch_num, is_same_heterogeneity).rename(columns={'value': 'env_axis2'})
    env_grid = env1_grid.merge(env2_grid, on=['x_index', 'y_index'], how='inner')

    merged = layout_df.merge(
        env_grid,
        left_on=['habitat_x_location', 'habitat_y_location'],
        right_on=['x_index', 'y_index'],
        how='left',
    )

    if merged[['env_axis1', 'env_axis2']].isna().any().any():
        raise ValueError(
            'Found unmatched habitat coordinates when merging layout with environment grids. '
            'Please check patch_habitat_layouts.csv and the two environment CSVs.'
        )

    meta_object = metacommunity(metacommunity_name=meta_name)
    patch_rows = (
        merged[['patch_id', 'patch_index', 'patch_location_x', 'patch_location_y']]
        .drop_duplicates()
        .sort_values(['patch_index', 'patch_id'])
    )

    for _, p_row in patch_rows.iterrows():
        patch_name = p_row['patch_id']
        patch_index = int(p_row['patch_index'])
        patch_location = (float(p_row['patch_location_x']), float(p_row['patch_location_y']))
        patch_object = patch(patch_name, patch_index, patch_location)

        hab_rows = (
            merged[merged['patch_id'] == patch_name]
            .copy()
            .sort_values(['habitat_index', 'habitat_id'])
        )

        for _, h_row in hab_rows.iterrows():
            patch_object.add_habitat(
                hab_name=str(h_row['habitat_id']),
                hab_index=int(h_row['habitat_index']),
                hab_location=(int(h_row['habitat_x_location']), int(h_row['habitat_y_location'])),
                num_env_types=2,
                env_types_name=environment_types_name,
                mean_env_ls=[float(h_row['env_axis1']), float(h_row['env_axis2'])],
                var_env_ls=environment_variation_ls,
                length=hab_length,
                width=hab_width,
                dormancy_pool_max_size=dormancy_pool_max_size,
            )

        meta_object.add_patch(patch_name=patch_name, patch_object=patch_object)

    return meta_object, merged


def summarize_meta_object(meta_object: metacommunity, merged_df: pd.DataFrame) -> str:
    patch_count = merged_df['patch_id'].nunique()
    habitats_per_patch = merged_df.groupby('patch_id')['habitat_id'].count().iloc[0]
    env1_min, env1_max = merged_df['env_axis1'].min(), merged_df['env_axis1'].max()
    env2_min, env2_max = merged_df['env_axis2'].min(), merged_df['env_axis2'].max()
    return (
        f'metacommunity_name={meta_object.metacommunity_name}\n'
        f'patch_count={patch_count}\n'
        f'total_habitats={len(merged_df)}\n'
        f'habitats_per_patch≈{habitats_per_patch}\n'
        f'env_axis1_range=({env1_min:.6f}, {env1_max:.6f})\n'
        f'env_axis2_range=({env2_min:.6f}, {env2_max:.6f})\n'
    )


if __name__ == '__main__':
    meta_object, summary_df = initialize_metacommunity_from_layout(
        patch_num=PATCH_NUM,
        is_same_heterogeneity=IS_SAME_HETEROGENEITY,
        meta_name=META_NAME,
        layout_csv=LAYOUT_CSV,
        env_axis1_csv=ENV_AXIS1_CSV,
        env_axis2_csv=ENV_AXIS2_CSV,
        data_dir=DATA_DIR,
        hab_length=HAB_LENGTH,
        hab_width=HAB_WIDTH,
        dormancy_pool_max_size=DORMANCY_POOL_MAX_SIZE,
        environment_types_name=ENVIRONMENT_TYPES_NAME,
        environment_variation_ls=ENVIRONMENT_VARIATION_LS,
    )

    print('Landscape initialization completed successfully.')
    print(summarize_meta_object(meta_object, summary_df))

    if SUMMARY_CSV is not None:
        out_path = Path(SUMMARY_CSV)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'Saved merged summary CSV to: {out_path.resolve()}')
    
    goal_path = os.getcwd()
    rep = 0
    hab_length, hab_width = 10, 10
    patch_num_x_axis, patch_num_y_axis = int(np.sqrt(PATCH_NUM)), int(np.sqrt(PATCH_NUM))
    hab_num_x_axis, hab_num_y_axis = int(32/patch_num_x_axis), int(32/patch_num_y_axis)
    
    meta_object.meta_show_environment_distribution(environment_name='x_axis_environment', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, mask_loc=None, cmap=plt.get_cmap('turbo'), file_name=goal_path+'/'+'rep=%d-metacommunity_env1.jpg'%(rep))
    meta_object.meta_show_environment_distribution(environment_name='y_axis_environment', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, mask_loc=None, cmap=plt.get_cmap('turbo'), file_name=goal_path+'/'+'rep=%d-metacommunity_env2.jpg'%(rep))
