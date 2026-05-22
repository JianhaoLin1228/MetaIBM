#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:18:58 2026

@author: jianhaolin
"""

import os

import numpy as np
import pandas as pd

import bootstrap_metaibm as _bootstrap
import metaibm
from metaibm.individual import individual
from metaibm.habitat import habitat
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity
from metaibm.simulator import simulator

################################################## schedule builders ####################################################################################
def build_schedule_asexual(asexual_birth_rate, mutation_rate, pheno_var_ls, base_dead_rate, fitness_wid, species_2_phenotype_ls,
                           propagules_rain_num, disp_within_rate, total_disp_among_rate, patch_dist_rate, goal_path, rep, recorder_interval,
                           rank, job_num, task_idx, all_time_step, patch_num_x_axis, patch_num_y_axis, hab_num_x_axis, hab_num_y_axis,
                           hab_length, hab_width, species_id_min, species_id_max, traits_num, pheno_names_ls, geno_len_ls,
                           metacommunity_csv, mainland_csv, environment_types_name, environment_variation_ls, dormancy_pool_max_size):
    return [
        # --- one-shot simulator-level mainland builder (must precede mainland meta_initialize) ---
        {'target': 'simulator', 'method': 'build_empty_mainland_from_species_csv',
         'params': {'meta_name': 'mainland', 'mainland_csv': mainland_csv, 'pheno_names_ls': pheno_names_ls,
                    'environment_types_name': environment_types_name, 'environment_variation_ls': environment_variation_ls,
                    'patch_name': 'patch0', 'patch_index': 0, 'patch_location': (0, 0), 'dormancy_pool_max_size': dormancy_pool_max_size},
         'start': 0, 'end': 0},

        # --- one-shot mainland initialization at time_step=0 (must precede mainland eco-evo items) ---
        {'target': 'mainland', 'method': 'meta_initialize',
         'params': {'traits_num': traits_num, 'pheno_names_ls': pheno_names_ls, 'pheno_var_ls': pheno_var_ls, 'geno_len_ls': geno_len_ls, 'reproduce_mode': 'asexual', 'species_2_phenotype_ls': species_2_phenotype_ls},
         'start': 0, 'end': 0},

        # --- one-shot simulator-level islands builder (must precede islands priming and ecological items) ---
        {'target': 'simulator', 'method': 'build_empty_metacommunity_from_patch_habitat_csv',
         'params': {'meta_name': 'islands', 'metacommunity_csv': metacommunity_csv, 'environment_types_name': environment_types_name, 'environment_variation_ls': environment_variation_ls, 'dormancy_pool_max_size': dormancy_pool_max_size},
         'start': 0, 'end': 0},

        # --- one-shot priming (mode='w') at time_step=0 ---
        {'target': 'simulator', 'method': 'prime_optimum_sp_distribution',
         'params': {'target': 'islands', 'base_dead_rate': base_dead_rate, 'fitness_wid': fitness_wid, 'species_2_phenotype_ls': species_2_phenotype_ls, 'file_name': 'meta_species_distribution_all_time.csv.gz'},
         'start': 0, 'end': 0},

        {'target': 'simulator', 'method': 'prime_environment_distribution',
         'params': {'target': 'islands', 'environment_name': 'env_axis1', 'index_label': 'x_axis_environment_values', 'file_name': 'meta_x_axis_phenotype_all_time.csv.gz'},
         'start': 0, 'end': 0},

        {'target': 'simulator', 'method': 'prime_environment_distribution',
         'params': {'target': 'islands', 'environment_name': 'env_axis2', 'index_label': 'y_axis_environment_values', 'file_name': 'meta_y_axis_phenotype_all_time.csv.gz'},
         'start': 0, 'end': 0},

        # --- mainland: dead selection then mainland reproduction ---
        {'target': 'mainland', 'method': 'meta_dead_selection',
         'params': {'base_dead_rate': base_dead_rate, 'fitness_wid': fitness_wid, 'method': 'niche_gaussian'}},
        
        {'target': 'mainland', 'method': 'meta_mainland_asexual_birth_mutate_germinate',
         'params': {'asexual_birth_rate': asexual_birth_rate, 'mutation_rate': mutation_rate, 'pheno_var_ls': pheno_var_ls}},

        # --- islands: dead selection ---
        {'target': 'islands', 'method': 'meta_dead_selection',
         'params': {'base_dead_rate': base_dead_rate, 'fitness_wid': fitness_wid, 'method': 'niche_gaussian'}},

        # --- islands: reproduction into offspring-marker pool ---
        {'target': 'islands', 'method': 'meta_asex_reproduce_calculation_into_offspring_marker_pool',
         'params': {'asexual_birth_rate': asexual_birth_rate}},

        # --- islands: dispersal ---
        {'target': 'islands', 'method': 'meta_colonize_from_propagules_rains',
         'params': {'mainland_obj': '@mainland', 'propagules_rain_num': propagules_rain_num}},

        {'target': 'islands', 'method': 'meta_dispersal_within_patch_from_offspring_marker_to_immigrant_marker_pool',
         'params': {'disp_within_rate': disp_within_rate}},

        {'target': 'islands', 'method': 'dispersal_among_patches_in_global_habitat_network_from_offspring_marker_pool_to_immigrant_marker_pool',
         'params': {'total_disp_among_rate': total_disp_among_rate, 'method': 'exponential', 'rho': 0.2}},

        # --- islands: germination + birth ---
        {'target': 'islands', 'method': 'meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool',
         'params': {'mutation_rate': mutation_rate, 'pheno_var_ls': pheno_var_ls}},

        # --- islands: disturbance ---
        {'target': 'islands', 'method': 'meta_disturbance_process_in_patches',
         'params': {'patch_dist_rate': patch_dist_rate}},

        # --- islands: cleanup ---
        {'target': 'islands', 'method': 'meta_clear_up_offspring_marker_and_immigrant_marker_pool'},

        # --- simulator-level: log, progress, recorders ---
        {'target': 'simulator', 'method': 'flush_step_log', 
        'interval': 1},

        {'target': 'simulator', 'method': 'print_progress',
         'params': {'target': 'islands', 'rank': rank, 'job_num': job_num, 'task_idx': task_idx},
         'interval': 1000},

        {'target': 'simulator', 'method': 'record_species_distribution',
         'params': {'target': 'islands', 'file_name': 'meta_species_distribution_all_time.csv.gz', 'mode': 'a'},
         'interval': recorder_interval},

        {'target': 'simulator', 'method': 'record_phenotype_distribution',
         'params': {'target': 'islands', 'trait_name': 'phenotype_axis1','file_name': 'meta_x_axis_phenotype_all_time.csv.gz', 'mode': 'a'},
         'interval': recorder_interval},

        {'target': 'simulator', 'method': 'record_phenotype_distribution',
         'params': {'target': 'islands', 'trait_name': 'phenotype_axis2', 'file_name': 'meta_y_axis_phenotype_all_time.csv.gz', 'mode': 'a'},
         'interval': recorder_interval},

        # --- end-of-run plots (single fire at the last step) ---
        {'target': 'simulator', 'method': 'plot_species_distribution',
         'params': {'target': 'islands', 'sub_row': patch_num_y_axis, 'sub_col': patch_num_x_axis, 'hab_num_x_axis_in_patch': hab_num_x_axis, 'hab_num_y_axis_in_patch': hab_num_y_axis, 'hab_y_len': hab_length, 'hab_x_len': hab_width, 'vmin': species_id_min, 'vmax': species_id_max, 'cmap': 'tab20', 'file_name': 'metacommunity_sp_dis.jpg'},
         'start': all_time_step - 1, 'end': all_time_step - 1},

        {'target': 'simulator', 'method': 'plot_species_phenotype_distribution',
         'params': {'target': 'islands', 'trait_name': 'phenotype_axis1', 'sub_row': patch_num_y_axis, 'sub_col': patch_num_x_axis, 'hab_num_x_axis_in_patch': hab_num_x_axis, 'hab_num_y_axis_in_patch': hab_num_y_axis, 'hab_y_len': hab_length, 'hab_x_len': hab_width, 'cmap': 'turbo', 'file_name': 'metacommunity_sp_phenotype_axis1_dis.jpg'},
         'start': all_time_step - 1, 'end': all_time_step - 1},

        {'target': 'simulator', 'method': 'plot_species_phenotype_distribution',
         'params': {'target': 'islands', 'trait_name': 'phenotype_axis2', 'sub_row': patch_num_y_axis, 'sub_col': patch_num_x_axis, 'hab_num_x_axis_in_patch': hab_num_x_axis, 'hab_num_y_axis_in_patch': hab_num_y_axis, 'hab_y_len': hab_length, 'hab_x_len': hab_width, 'cmap': 'turbo', 'file_name': 'metacommunity_sp_phenotype_axis2_dis.jpg'},
         'start': all_time_step - 1, 'end': all_time_step - 1},

        {'target': 'simulator', 'method': 'plot_environment_distribution',
         'params': {'target': 'islands', 'environment_name': 'env_axis1', 'sub_row': patch_num_y_axis, 'sub_col': patch_num_x_axis, 'hab_num_x_axis_in_patch': hab_num_x_axis, 'hab_num_y_axis_in_patch': hab_num_y_axis, 'hab_y_len': hab_length, 'hab_x_len': hab_width, 'mask_loc': None, 'cmap': 'turbo', 'file_name': 'metacommunity_env_axis1_environment.jpg'},
         'start': all_time_step - 1, 'end': all_time_step - 1},

        {'target': 'simulator', 'method': 'plot_environment_distribution',
         'params': {'target': 'islands', 'environment_name': 'env_axis2', 'sub_row': patch_num_y_axis, 'sub_col': patch_num_x_axis, 'hab_num_x_axis_in_patch': hab_num_x_axis, 'hab_num_y_axis_in_patch': hab_num_y_axis, 'hab_y_len': hab_length, 'hab_x_len': hab_width, 'mask_loc': None, 'cmap': 'turbo', 'file_name': 'metacommunity_env_axis2_environment.jpg'},
         'start': all_time_step - 1, 'end': all_time_step - 1},
    ]

def build_schedule_sexual(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls, base_dead_rate, fitness_wid, species_2_phenotype_ls,
                          propagules_rain_num, disp_within_rate, total_disp_among_rate, patch_dist_rate, goal_path, rep, recorder_interval,
                          rank, job_num, task_idx, all_time_step, patch_num_x_axis, patch_num_y_axis, hab_num_x_axis, hab_num_y_axis, hab_length, hab_width,
                          species_id_min, species_id_max, traits_num, pheno_names_ls, geno_len_ls,
                          metacommunity_csv, mainland_csv, environment_types_name, environment_variation_ls, dormancy_pool_max_size):
    return [
        # --- one-shot simulator-level mainland builder (must precede mainland meta_initialize) ---
        {'target': 'simulator', 'method': 'build_empty_mainland_from_species_csv',
         'params': {'meta_name': 'mainland', 'mainland_csv': mainland_csv, 'pheno_names_ls': pheno_names_ls, 'environment_types_name': environment_types_name, 'environment_variation_ls': environment_variation_ls, 'patch_name': 'patch0', 'patch_index': 0, 'patch_location': (0, 0), 'dormancy_pool_max_size': dormancy_pool_max_size},
         'start': 0, 'end': 0},

        # --- one-shot mainland initialization at time_step=0 (must precede mainland eco-evo items) ---
        {'target': 'mainland', 'method': 'meta_initialize',
         'params': {'traits_num': traits_num, 'pheno_names_ls': pheno_names_ls, 'pheno_var_ls': pheno_var_ls, 'geno_len_ls': geno_len_ls, 'reproduce_mode': 'sexual', 'species_2_phenotype_ls': species_2_phenotype_ls},
         'start': 0, 'end': 0},

        # --- one-shot simulator-level islands builder (must precede islands priming and ecological items) ---
        {'target': 'simulator', 'method': 'build_empty_metacommunity_from_patch_habitat_csv',
         'params': {'meta_name': 'islands', 'metacommunity_csv': metacommunity_csv, 'environment_types_name': environment_types_name, 'environment_variation_ls': environment_variation_ls, 'dormancy_pool_max_size': dormancy_pool_max_size},
         'start': 0, 'end': 0},

        # --- one-shot priming (mode='w') at time_step=0 ---
        {'target': 'simulator', 'method': 'prime_optimum_sp_distribution',
         'params': {'target': 'islands', 'base_dead_rate': base_dead_rate, 'fitness_wid': fitness_wid, 'species_2_phenotype_ls': species_2_phenotype_ls, 'file_name': 'meta_species_distribution_all_time.csv.gz'},
         'start': 0, 'end': 0},

        {'target': 'simulator', 'method': 'prime_environment_distribution',
         'params': {'target': 'islands', 'environment_name': 'env_axis1', 'index_label': 'x_axis_environment_values', 'file_name': 'meta_x_axis_phenotype_all_time.csv.gz'},
         'start': 0, 'end': 0},

        {'target': 'simulator', 'method': 'prime_environment_distribution',
         'params': {'target': 'islands', 'environment_name': 'env_axis2', 'index_label': 'y_axis_environment_values', 'file_name': 'meta_y_axis_phenotype_all_time.csv.gz'},
         'start': 0, 'end': 0},

        # --- mainland ---
        {'target': 'mainland', 'method': 'meta_dead_selection',
         'params': {'base_dead_rate': base_dead_rate, 'fitness_wid': fitness_wid, 'method': 'niche_gaussian'}},

        {'target': 'mainland', 'method': 'meta_mainland_mixed_birth_mutate_germinate',
         'params': {'asexual_birth_rate': asexual_birth_rate, 'sexual_birth_rate': sexual_birth_rate, 'mutation_rate': mutation_rate, 'pheno_var_ls': pheno_var_ls}},

        # --- islands: dead selection ---
        {'target': 'islands', 'method': 'meta_dead_selection',
         'params': {'base_dead_rate': base_dead_rate, 'fitness_wid': fitness_wid, 'method': 'niche_gaussian'}},

        # --- islands: mixed reproduction into offspring-marker pool ---
        {'target': 'islands', 'method': 'meta_mix_reproduce_calculation_with_offspring_marker_pool',
         'params': {'asexual_birth_rate': asexual_birth_rate, 'sexual_birth_rate': sexual_birth_rate}},

        # --- islands: dispersal ---
        {'target': 'islands', 'method': 'pairwise_sexual_colonization_from_prpagules_rains',
         'params': {'mainland_obj': '@mainland', 'propagules_rain_num': propagules_rain_num}},

        {'target': 'islands', 'method': 'meta_dispersal_within_patch_from_offspring_marker_to_immigrant_marker_pool',
         'params': {'disp_within_rate': disp_within_rate}},

        {'target': 'islands', 'method': 'dispersal_among_patches_in_global_habitat_network_from_offspring_marker_pool_to_immigrant_marker_pool',
         'params': {'total_disp_among_rate': total_disp_among_rate, 'method': 'exponential', 'rho': 0.2}},

        # --- islands: germination + birth ---
        {'target': 'islands', 'method': 'meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool',
         'params': {'mutation_rate': mutation_rate, 'pheno_var_ls': pheno_var_ls}},

        # --- islands: disturbance ---
        {'target': 'islands', 'method': 'meta_disturbance_process_in_patches',
         'params': {'patch_dist_rate': patch_dist_rate}},

        # --- islands: cleanup ---
        {'target': 'islands', 'method': 'meta_clear_up_offspring_marker_and_immigrant_marker_pool'},

        # --- simulator-level: log, progress, recorders ---
        {'target': 'simulator', 'method': 'flush_step_log', 'interval': 1},
        {'target': 'simulator', 'method': 'print_progress', 'params': {'target': 'islands', 'rank': rank, 'job_num': job_num, 'task_idx': task_idx},
         'interval': 1000},

        {'target': 'simulator', 'method': 'record_species_distribution',
         'params': {'target': 'islands', 'file_name': 'meta_species_distribution_all_time.csv.gz', 'mode': 'a'},
         'interval': recorder_interval},

        {'target': 'simulator', 'method': 'record_phenotype_distribution',
         'params': {'target': 'islands', 'trait_name': 'phenotype_axis1', 'file_name': 'meta_x_axis_phenotype_all_time.csv.gz', 'mode': 'a'},
         'interval': recorder_interval},

        {'target': 'simulator', 'method': 'record_phenotype_distribution',
         'params': {'target': 'islands', 'trait_name': 'phenotype_axis2', 'file_name': 'meta_y_axis_phenotype_all_time.csv.gz', 'mode': 'a'},
         'interval': recorder_interval},

        # --- end-of-run plots (single fire at the last step) ---
        {'target': 'simulator', 'method': 'plot_species_distribution',
         'params': {'target': 'islands', 'sub_row': patch_num_y_axis, 'sub_col': patch_num_x_axis, 'hab_num_x_axis_in_patch': hab_num_x_axis, 'hab_num_y_axis_in_patch': hab_num_y_axis, 'hab_y_len': hab_length, 'hab_x_len': hab_width, 'vmin': species_id_min, 'vmax': species_id_max, 'cmap': 'tab20', 'file_name': 'metacommunity_sp_dis.jpg'},
         'start': all_time_step - 1, 'end': all_time_step - 1},

        {'target': 'simulator', 'method': 'plot_species_phenotype_distribution',
         'params': {'target': 'islands', 'trait_name': 'phenotype_axis1', 'sub_row': patch_num_y_axis, 'sub_col': patch_num_x_axis, 'hab_num_x_axis_in_patch': hab_num_x_axis, 'hab_num_y_axis_in_patch': hab_num_y_axis, 'hab_y_len': hab_length, 'hab_x_len': hab_width, 'cmap': 'turbo', 'file_name': 'metacommunity_sp_phenotype_axis1_dis.jpg'},
         'start': all_time_step - 1, 'end': all_time_step - 1},

        {'target': 'simulator', 'method': 'plot_species_phenotype_distribution',
         'params': {'target': 'islands', 'trait_name': 'phenotype_axis2', 'sub_row': patch_num_y_axis, 'sub_col': patch_num_x_axis, 'hab_num_x_axis_in_patch': hab_num_x_axis, 'hab_num_y_axis_in_patch': hab_num_y_axis, 'hab_y_len': hab_length, 'hab_x_len': hab_width, 'cmap': 'turbo', 'file_name': 'metacommunity_sp_phenotype_axis2_dis.jpg'},
         'start': all_time_step - 1, 'end': all_time_step - 1},

        {'target': 'simulator', 'method': 'plot_environment_distribution',
         'params': {'target': 'islands', 'environment_name': 'env_axis1', 'sub_row': patch_num_y_axis, 'sub_col': patch_num_x_axis, 'hab_num_x_axis_in_patch': hab_num_x_axis, 'hab_num_y_axis_in_patch': hab_num_y_axis, 'hab_y_len': hab_length, 'hab_x_len': hab_width, 'mask_loc': None, 'cmap': 'turbo', 'file_name': 'metacommunity_env_axis1_environment.jpg'},
         'start': all_time_step - 1, 'end': all_time_step - 1},

        {'target': 'simulator', 'method': 'plot_environment_distribution',
         'params': {'target': 'islands', 'environment_name': 'env_axis2', 'sub_row': patch_num_y_axis, 'sub_col': patch_num_x_axis, 'hab_num_x_axis_in_patch': hab_num_x_axis, 'hab_num_y_axis_in_patch': hab_num_y_axis, 'hab_y_len': hab_length, 'hab_x_len': hab_width, 'mask_loc': None, 'cmap': 'turbo', 'file_name': 'metacommunity_env_axis2_environment.jpg'},
         'start': all_time_step - 1, 'end': all_time_step - 1},
    ]

################################################## def main() ###########################################################################################
def main(rep, patch_num, is_same_heterogeneity, reproduce_mode, total_disp_among_rate, disp_within_rate, patch_dist_rate, root_path, rank=0, job_num=1, task_idx=1):

    ''' time-step scales parameters '''
    all_time_step = 5000

    ''' land config. files '''
    mainland_csv_path = 'mainland.csv'
    metacommunity_csv_path = 'metacommunity_N=%d_is_same_heterogeneity=%s.csv'% (patch_num, str(is_same_heterogeneity))

    ''' mainland parameters '''
    df_mainland = pd.read_csv(mainland_csv_path)
    df_mainland = df_mainland.sort_values('species_id').reset_index(drop=True)
    species_id_min = int(df_mainland['species_id'].min())
    species_id_max = int(df_mainland['species_id'].max())
    colonize_rate = 0.001
    mainland_total_microsite_num = int((df_mainland['hab_length'] * df_mainland['hab_width']).sum())
    propagules_rain_num = mainland_total_microsite_num * colonize_rate

    ''' species parameters '''
    pheno_names_ls = ('phenotype_axis1', 'phenotype_axis2')
    traits_num = len(pheno_names_ls)
    pheno_var_ls = (0.025, 0.025)
    geno_len_ls = (20, 20)
    species_2_phenotype_ls = df_mainland.loc[:, list(pheno_names_ls)].astype(float).values.tolist()

    ''' islands configuration '''
    df_metacommunity_config = pd.read_csv(metacommunity_csv_path)
    df_patch = df_metacommunity_config[['patch_id', 'patch_location_x', 'patch_location_y']].drop_duplicates()

    patch_num_x_axis = df_patch['patch_location_x'].nunique()
    patch_num_y_axis = df_patch['patch_location_y'].nunique()

    first_patch = df_patch.sort_values(['patch_location_x', 'patch_location_y'])['patch_id'].iloc[0]
    df_first_patch = df_metacommunity_config[df_metacommunity_config['patch_id'] == first_patch]
    
    hab_num_x_axis = df_first_patch['habitat_x_location'].nunique() # habitat_num per patch along x-axis
    hab_num_y_axis = df_first_patch['habitat_y_location'].nunique() # habitat_num per patch along y-axis

    hab_length = int(df_metacommunity_config['hab_length'].iloc[0])
    hab_width = int(df_metacommunity_config['hab_width'].iloc[0])

    ''' environmental parameters '''
    environment_types_name = ('env_axis1', 'env_axis2')   # environment_types_num = len(environment_types_name)
    environment_variation_ls = [0.025, 0.025]
    dormancy_pool_max_size = 0

    ''' demography parameters '''
    base_dead_rate = 0.1
    fitness_wid = 0.5
    asexual_birth_rate = 0.5
    sexual_birth_rate = 1
    mutation_rate = 0.0001

    ''' logging & recording & timing '''
    is_logging = True
    recorder_interval = 1
    is_timing=True

    ''' build and run the simulator '''
    sim = simulator()

    if root_path is None: root_path = os.getcwd()
    goal_path = sim.set_goal_path(root_path, 'rep=%d'%rep, reproduce_mode, 'patch_num=%03d'%patch_num, 'is_heterogeneity=%s'%str(is_same_heterogeneity), 'disp_among=%f-disp_within=%f'%(total_disp_among_rate, disp_within_rate), 'patch_dist_rate=%f'%patch_dist_rate)
    sim.set_global_params({'all_time_steps': all_time_step, 'goal_path': goal_path, 'rep': rep, 'is_logging': is_logging, 'is_timing': is_timing})

    if reproduce_mode == 'asexual':
        schedule = build_schedule_asexual(asexual_birth_rate=asexual_birth_rate, mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls,
                                          base_dead_rate=base_dead_rate, fitness_wid=fitness_wid, species_2_phenotype_ls=species_2_phenotype_ls,
                                          propagules_rain_num=propagules_rain_num, disp_within_rate=disp_within_rate, total_disp_among_rate=total_disp_among_rate, patch_dist_rate=patch_dist_rate,
                                          goal_path=goal_path, rep=rep, recorder_interval=recorder_interval, rank=rank, job_num=job_num, task_idx=task_idx,
                                          all_time_step=all_time_step, patch_num_x_axis=patch_num_x_axis, patch_num_y_axis=patch_num_y_axis, hab_num_x_axis=hab_num_x_axis, hab_num_y_axis=hab_num_y_axis,
                                          hab_length=hab_length, hab_width=hab_width, species_id_min=species_id_min, species_id_max=species_id_max, traits_num=traits_num, pheno_names_ls=pheno_names_ls, geno_len_ls=geno_len_ls,
                                          metacommunity_csv=metacommunity_csv_path, mainland_csv=mainland_csv_path, environment_types_name=environment_types_name, environment_variation_ls=environment_variation_ls, dormancy_pool_max_size=dormancy_pool_max_size)

    elif reproduce_mode == 'sexual':
        schedule = build_schedule_sexual(asexual_birth_rate=asexual_birth_rate, sexual_birth_rate=sexual_birth_rate, mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls,
                                         base_dead_rate=base_dead_rate, fitness_wid=fitness_wid, species_2_phenotype_ls=species_2_phenotype_ls,
                                         propagules_rain_num=propagules_rain_num, disp_within_rate=disp_within_rate, total_disp_among_rate=total_disp_among_rate, patch_dist_rate=patch_dist_rate,
                                         goal_path=goal_path, rep=rep, recorder_interval=recorder_interval, rank=rank, job_num=job_num, task_idx=task_idx,
                                         all_time_step=all_time_step, patch_num_x_axis=patch_num_x_axis, patch_num_y_axis=patch_num_y_axis, hab_num_x_axis=hab_num_x_axis, hab_num_y_axis=hab_num_y_axis,
                                         hab_length=hab_length, hab_width=hab_width, species_id_min=species_id_min, species_id_max=species_id_max, traits_num=traits_num, pheno_names_ls=pheno_names_ls, geno_len_ls=geno_len_ls,
                                         metacommunity_csv=metacommunity_csv_path, mainland_csv=mainland_csv_path, environment_types_name=environment_types_name, environment_variation_ls=environment_variation_ls, dormancy_pool_max_size=dormancy_pool_max_size)

    sim.set_schedule_per_time_step(schedule)

    sim.run()

##############################################################################################################################################################################
if __name__ == '__main__':
    main(rep=0, patch_num=16, is_same_heterogeneity=True, reproduce_mode='sexual', total_disp_among_rate=0.01, disp_within_rate=0.1, patch_dist_rate=0.0001, root_path=None)
