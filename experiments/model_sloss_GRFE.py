#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 23:36:53 2024

@author: jianhaolin
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import bootstrap_metaibm as _bootstrap
import metaibm
from metaibm.individual import individual
from metaibm.habitat import habitat
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity
#################################################
def new_round(_float, _len = 0):
    """
    Parameters
    ----------
    _float: float
    _len: int, 指定四舍五入需要保留的小数点后几位数为_len

    Returns
    -------
    type ==> float, 返回四舍五入后的值
    """
    if isinstance(_float, float):
        if str(_float)[::-1].find('.') <= _len:
            return (_float)
        if str(_float)[-1] == '5':
            return (round(float(str(_float)[:-1] + '6'), _len))
        else:
            return (round(_float, _len))
    else:
        return (round(_float, _len))

def _load_patch_habitat_layout(layout_csv_path, patch_num):
    df = pd.read_csv(layout_csv_path)
    sub = df[df['patch_num'] == patch_num].copy()
    return sub

def _read_env_multiindex_csv(env_csv):
    df = pd.read_csv(env_csv, header=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df

def _extract_environment_grid(env_df, patch_num, is_same_heterogeneity):
    key = (f'patch_num={patch_num}', str(bool(is_same_heterogeneity)))
    needed = [('meta', 'x_index'), ('meta', 'y_index'), key] 
    sub = env_df.loc[:, needed].copy()

    sub.columns = ['x_index', 'y_index', 'value']
    sub['x_index'], sub['y_index'] = sub['x_index'].astype(int), sub['y_index'].astype(int)
    return sub

def read_landscape_configuration(layout_csv, env1_csv, env2_csv, patch_num, is_same_heterogeneity):
    df_patch_habitat_layout = _load_patch_habitat_layout(layout_csv_path=layout_csv, patch_num=patch_num)
    df_env1 = _read_env_multiindex_csv(env1_csv)
    df_env2 = _read_env_multiindex_csv(env2_csv)

    env1_habitats = _extract_environment_grid(env_df=df_env1, patch_num=patch_num, is_same_heterogeneity=is_same_heterogeneity)
    env2_habitats = _extract_environment_grid(env_df=df_env2, patch_num=patch_num, is_same_heterogeneity=is_same_heterogeneity)
    env_habitats = env1_habitats.merge(env2_habitats, on=['x_index', 'y_index'], how='inner')
    env_habitats.columns = ['x_index', 'y_index', 'env_axis1', 'env_axis2']

    patch_habitat_env_layout = df_patch_habitat_layout.merge(env_habitats, left_on=['habitat_x_location', 'habitat_y_location'], right_on=['x_index', 'y_index'], how='left')
    return patch_habitat_env_layout

def generate_empty_metacommunity(meta_name, layout_csv, env1_csv, env2_csv, patch_num, is_same_heterogeneity, num_env_types, env_types_name, var_env_ls, hab_length, hab_width, dormancy_pool_max_size):
    ''' '''
    log_info = 'generating empty metacommunity ... \n'
    df_patch_habitat_env_layout = read_landscape_configuration(layout_csv=layout_csv, env1_csv=env1_csv, env2_csv=env2_csv, patch_num=patch_num, is_same_heterogeneity=is_same_heterogeneity)
    meta_obj = metacommunity(metacommunity_name=meta_name)

    df_patch_register_info = df_patch_habitat_env_layout[['patch_id', 'patch_index', 'patch_location_x', 'patch_location_y']].drop_duplicates().sort_values(['patch_index', 'patch_id'])
    for _, patch_reg_info in df_patch_register_info.iterrows():
        patch_name = patch_reg_info['patch_id']
        patch_index = int(patch_reg_info['patch_index'])
        patch_location = (float(patch_reg_info['patch_location_x']), float(patch_reg_info['patch_location_y']))
        patch_object = patch(patch_name, patch_index, patch_location)
        #print(patch_name, patch_index, patch_location)

        df_hab_register_infos = df_patch_habitat_env_layout[df_patch_habitat_env_layout['patch_id'] == patch_name].copy().sort_values(['habitat_index', 'habitat_id'])
        for _, hab_reg_info in df_hab_register_infos.iterrows():
            hab_name = str(hab_reg_info['habitat_id'])
            hab_index = int(hab_reg_info['habitat_index'])
            hab_location = (int(hab_reg_info['habitat_x_location']), int(hab_reg_info['habitat_y_location']))
            mean_env_ls = [float(hab_reg_info['env_axis1']), float(hab_reg_info['env_axis2'])]

            patch_object.add_habitat(hab_name=hab_name, hab_index=hab_index, hab_location=hab_location, 
                                     num_env_types=num_env_types, env_types_name=env_types_name, mean_env_ls=mean_env_ls, var_env_ls=var_env_ls, 
                                     length=hab_length, width=hab_width, dormancy_pool_max_size=dormancy_pool_max_size)

            info = '%s: %s, %s, %s, %s: env_axis1=%s, env_axis2=%s'%(meta_obj.metacommunity_name, patch_name, str(patch_location), hab_name, str(hab_location), str(mean_env_ls[0]), str(mean_env_ls[1]))
            log_info = log_info + info + '\n'
        meta_obj.add_patch(patch_name=patch_name, patch_object=patch_object)
    #print(log_info)
    return meta_obj, log_info

def generate_empty_mainland(meta_name, patch_num, patch_location_ls, hab_num, hab_length, hab_width, dormancy_pool_max_size, 
                            micro_environment_values_ls, macro_environment_values_ls, environment_types_num, environment_types_name, environment_variation_ls):
    log_info = 'generating empty mainland ... \n'
    meta_object = metacommunity(metacommunity_name=meta_name)
    for i in range(0, patch_num):
        patch_name = 'patch%d'%(i)
        patch_index = i
        location = patch_location_ls[i]
        patch_x_loc, patch_y_loc = location[0], location[1]
        p = patch(patch_name, patch_index, location)
        
        hab_num_length, hab_num_width = int(np.sqrt(hab_num)), int(np.sqrt(hab_num))
        
        for j in range(hab_num):
            habitat_name = 'h%s'%str(j)
            hab_index = j
            
            hab_x_loc, hab_y_loc = patch_x_loc*hab_num_length+j//hab_num_width, patch_y_loc*hab_num_width+j%hab_num_width
            hab_location = (hab_x_loc, hab_y_loc)
            
            micro_environment_mean_value = micro_environment_values_ls[j//hab_num_width]
            macro_environment_mean_value = macro_environment_values_ls[j%hab_num_width]
            
            p.add_habitat(hab_name=habitat_name, hab_index=hab_index, hab_location=hab_location, num_env_types=environment_types_num, env_types_name=environment_types_name, 
                          mean_env_ls=[micro_environment_mean_value, macro_environment_mean_value], var_env_ls=environment_variation_ls, length=hab_length, width=hab_width, dormancy_pool_max_size=dormancy_pool_max_size)
            
            info = '%s: %s, %s, %s, %s: micro_environment_mean_value=%s, macro_environment_mean_value=%s \n'%(meta_object.metacommunity_name, patch_name, str(location), habitat_name, str(hab_location), str(micro_environment_mean_value), str(macro_environment_mean_value))
            log_info = log_info + info
        meta_object.add_patch(patch_name=patch_name, patch_object=p)
    #print(log_info)
    return meta_object, log_info
########################################################################################################################################################
def mkdir_if_not_exist(reproduce_mode, patch_num, is_heterogeneity, disp_among_within_rate, patch_dist_rate):
    root_path = os.getcwd()
    
    reproduce_mode_dir = {'asexual':'/asexual', 'sexual':'/sexual'}
    patch_num_files_name = '/patch_num=%03d'%patch_num
    is_heterogeneity_files_name = '/is_heterogeneity=%s'%str(is_heterogeneity)
    disp_amomg_within_rate_files_name = '/disp_among=%f-disp_within=%f'%(disp_among_within_rate[0], disp_among_within_rate[1])
    patch_dist_rate_files_name = '/patch_dist_rate=%f'%patch_dist_rate
    
    goal_path = root_path + reproduce_mode_dir[reproduce_mode] + patch_num_files_name + is_heterogeneity_files_name + disp_amomg_within_rate_files_name + patch_dist_rate_files_name
    if os.path.exists(goal_path) == False:
        os.makedirs(goal_path)
    else:
        pass
    return goal_path
################################################## logging module ########################################################################################
def write_logger(log_info, is_logging=False, logger_file=None):
    if is_logging == True:
        print(log_info, file=logger_file)
    elif is_logging == False:
        print(log_info)
################################################## def main() ######################################################################################################
def main(rep, patch_num, is_same_heterogeneity, reproduce_mode, total_disp_among_rate, disp_within_rate, patch_dist_rate, goal_path=None, rank=0, job_num=1, task_idx=1):
    if goal_path==None: goal_path = os.getcwd()
    
    ''' timer '''
    all_time_start = time.time()
    
    ''' replication index (not running) '''
    #rep = 0
    
    ''' time-step scales parameters '''
    all_time_step = 5000
    
    ''' map size parameters '''
    #patch_num = 16  # 1, 4, 16, 64, 256, 1024
    patch_num_x_axis, patch_num_y_axis = int(np.sqrt(patch_num)), int(np.sqrt(patch_num))
    hab_num_in_patch = int(1024 / patch_num)
    hab_num_x_axis, hab_num_y_axis = int(np.sqrt(hab_num_in_patch)), int(np.sqrt(hab_num_in_patch)) # hab_num_x_axis in a patch, hab_num_y_axis in a patch
    hab_length, hab_width = 10, 10

    ''' landscape configuration '''
    patch_hab_layout_path = 'patch_habitat_layouts.csv'
    env_axis1_path = '32x32_habitats_env1.csv'
    env_axis2_path = '32x32_habitats_env2.csv'

    ''' environmental parameters '''
    environment_types_num = 2
    environment_types_name=('env_axis1', 'env_axis2')
    environment_variation_ls = [0.025, 0.025]
    dormancy_pool_max_size = 0
    
    ''' demography parameters '''
    base_dead_rate=0.1
    fitness_wid=0.5
    #reproduce_mode = 'sexual'
    asexual_birth_rate = 0.5
    sexual_birth_rate = 1
    mutation_rate=0.0001
    recomb_method='multi_crossover'
    recomb_rate=0.05
    
    ''' landscape parameters '''
    colonize_rate = 0.001
    #total_disp_among_rate = 0.001
    #disp_within_rate =0.1
    propagules_rain_num = 6400 * colonize_rate
    #patch_dist_rate = 0.00001
    
    ''' species parameters '''
    #species_num = 4
    traits_num = 2
    pheno_names_ls = ('phenotype_axis1', 'phenotype_axis2')
    pheno_var_ls=(0.025, 0.025)
    geno_len_ls=(20, 20)
    species_2_phenotype_ls = [[i/10, j/10] for i in range(2,10,2) for j in range(2,10,2)]
    #[[0.2,0.2], [0.4,0.4], [0.6,0.6], [0.8,0.8]] # (index+1) indicates species_id
    
    ''' logging or not logging '''
    is_logging = True
    
    ''' logging module '''
    if is_logging == True: 
        logger_file_name = goal_path+'/'+'rep=%d-logger.log'%(rep)
        logger_file = open(logger_file_name, "a")
    
    ''' initailization processes '''
    write_in_logger_info = ''
    
    mainland, log_info = generate_empty_mainland(meta_name='mainland', patch_num=1, patch_location_ls=[(0,0)], hab_num=16, hab_length=20, hab_width=20, dormancy_pool_max_size=0, 
                                                 micro_environment_values_ls=[0.2, 0.4, 0.6, 0.8], macro_environment_values_ls=[0.2, 0.4, 0.6, 0.8], 
                                                 environment_types_num=environment_types_num, environment_types_name=environment_types_name, environment_variation_ls=environment_variation_ls)
    write_in_logger_info += log_info
    
    meta_obj, log_info = generate_empty_metacommunity(meta_name='metacommunity', layout_csv=patch_hab_layout_path, env1_csv=env_axis1_path, env2_csv=env_axis2_path, patch_num=patch_num, is_same_heterogeneity=is_same_heterogeneity, 
                                                      num_env_types=environment_types_num, env_types_name=environment_types_name, var_env_ls=environment_variation_ls, hab_length=hab_length, hab_width=hab_width, dormancy_pool_max_size=dormancy_pool_max_size)
    write_in_logger_info += log_info

    write_in_logger_info += mainland.meta_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
    write_logger(write_in_logger_info, is_logging, logger_file)
    
    ''' data saving and files controling '''
    columns_patch_id, columns_habitat_id, columns_mocrosite_id = meta_obj.columns_patch_habitat_microsites_id()
    columns = [columns_patch_id, columns_habitat_id, columns_mocrosite_id]
    #mode='w'
    #meta_sp_dis_all_time = meta_obj.get_meta_microsites_optimum_sp_id_val(base_dead_rate, fitness_wid, species_2_phenotype_ls)
    #meta_x_axis_phenotype_all_time = meta_obj.get_meta_microsite_environment_values(environment_name='x_axis_environment')
    #meta_y_axis_phenotype_all_time = meta_obj.get_meta_microsite_environment_values(environment_name='y_axis_environment')
    #mode='a'
    meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_optimum_sp_id_val(base_dead_rate, fitness_wid, species_2_phenotype_ls), 
                                                       file_name=goal_path+'/'+'rep=%d-meta_species_distribution_all_time.csv.gz'%(rep), 
                                                       index=['optimun_sp_id_values'], columns=columns, mode='w')
    meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsite_environment_values(environment_name='env_axis1'), 
                                                       file_name=goal_path+'/'+'rep=%d-meta_x_axis_phenotype_all_time.csv.gz'%(rep), 
                                                       index=['x_axis_environment_values'], columns=columns, mode='w')
    meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsite_environment_values(environment_name='env_axis2'), 
                                                       file_name=goal_path+'/'+'rep=%d-meta_y_axis_phenotype_all_time.csv.gz'%(rep), 
                                                       index=['y_axis_environment_values'], columns=columns, mode='w')
    
    all_burn_in_time_steps = 100
    write_in_logger_info = 'burn in simulation for the mainland for 100 time-steps ...... '
    write_logger(write_in_logger_info, is_logging, logger_file)
    for _ in range(all_burn_in_time_steps):
        mainland.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
        if reproduce_mode == 'asexual': mainland.meta_mainland_asexual_birth_mutate_germinate(asexual_birth_rate, mutation_rate, pheno_var_ls)
        elif reproduce_mode == 'sexual': mainland.meta_mainland_mixed_birth_mutate_germinate(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method=recomb_method, recomb_rate=recomb_rate)

    for time_step in range(all_time_step): 

        if time_step==0 or (time_step+1)%1000==0:
            print('process%d, '%(rank), 'task_idx/job_num=%d/%d, '%(task_idx, job_num), 'time_step/all_time_step=%d/%d: '%(time_step, all_time_step), 'indi_num=%d, '%meta_obj.get_meta_individual_num(), 'empty_sites_num=%d'%meta_obj.show_meta_empty_sites_num(), flush=True)
        
        write_in_logger_info = ''
        write_in_logger_info += 'time_step=%d \n'%time_step
        #print('time_step=%d'%time_step)
        if reproduce_mode == 'asexual':
            ''' dead selection process in mainland and metacommunity '''
            write_in_logger_info += mainland.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            write_in_logger_info += mainland.meta_mainland_asexual_birth_mutate_germinate(asexual_birth_rate, mutation_rate, pheno_var_ls)
            write_in_logger_info += meta_obj.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            ''' reproduction process '''
            write_in_logger_info += meta_obj.meta_asex_reproduce_calculation_into_offspring_marker_pool(asexual_birth_rate)
            #meta_obj.meta_asex_reproduce_mutate_into_offspring_pool(asexual_birth_rate, mutation_rate, pheno_var_ls)
            ''' dispersal processes '''
            write_in_logger_info += meta_obj.meta_colonize_from_propagules_rains(mainland, propagules_rain_num)
            write_in_logger_info += meta_obj.meta_dispersal_within_patch_from_offspring_marker_to_immigrant_marker_pool(disp_within_rate)
            #meta_obj.meta_dispersal_within_patch_from_offspring_to_immigrant_pool(disp_within_rate)
            #write_in_logger_info += meta_obj.dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool(total_disp_among_rate, method='exponential', rho=0.2)
            write_in_logger_info += meta_obj.dispersal_among_patches_in_global_habitat_network_from_offspring_marker_pool_to_immigrant_marker_pool(total_disp_among_rate, method='exponential', rho=0.2)
            #meta_obj.dispersal_aomng_patches_from_offspring_pool_to_immigrant_pool(total_disp_among_rate)
            ''' germination processes '''
            write_in_logger_info += meta_obj.meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool(mutation_rate, pheno_var_ls)
            #meta_obj.meta_local_germinate_from_offspring_and_immigrant_pool()
            #meta_obj.meta_local_germinate_from_offspring_immigrant_and_dormancy_pool()
            ''' dormancy process (not running) '''
            #meta_obj.meta_dormancy_process_from_offspring_pool_and_immigrant_pool()
            ''' disturbance process '''
            write_in_logger_info += meta_obj.meta_disturbance_process_in_patches(patch_dist_rate)
            ''' eliminating offspring and immigrant (marker) pool '''
            meta_obj.meta_clear_up_offspring_marker_and_immigrant_marker_pool()
            #meta_obj.meta_clear_up_offspring_and_immigrant_pool()
            
        elif reproduce_mode == 'sexual':
            ''' dead selection process in mainland and metacommunity '''
            write_in_logger_info += mainland.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            write_in_logger_info += mainland.meta_mainland_mixed_birth_mutate_germinate(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method=recomb_method, recomb_rate=recomb_rate)
            write_in_logger_info += meta_obj.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            ''' reproduction process '''
            write_in_logger_info += meta_obj.meta_mix_reproduce_calculation_with_offspring_marker_pool(asexual_birth_rate, sexual_birth_rate)
            #meta_obj.meta_mix_reproduce_mutate_into_offspring_pool(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method=recomb_method, recomb_rate=recomb_rate)
            ''' dispersal processes '''
            write_in_logger_info += meta_obj.pairwise_sexual_colonization_from_prpagules_rains(mainland, propagules_rain_num)
            write_in_logger_info += meta_obj.meta_dispersal_within_patch_from_offspring_marker_to_immigrant_marker_pool(disp_within_rate)
            #meta_obj.meta_dispersal_within_patch_from_offspring_to_immigrant_pool(disp_within_rate)
            #write_in_logger_info += meta_obj.dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool(total_disp_among_rate, method='exponential', rho=0.2)
            write_in_logger_info += meta_obj.dispersal_among_patches_in_global_habitat_network_from_offspring_marker_pool_to_immigrant_marker_pool(total_disp_among_rate, method='exponential', rho=0.2)
            #meta_obj.dispersal_aomng_patches_from_offspring_pool_to_immigrant_pool(total_disp_among_rate)
            ''' germination processes '''
            write_in_logger_info += meta_obj.meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool(mutation_rate, pheno_var_ls, recomb_method=recomb_method, recomb_rate=recomb_rate)
            #meta_obj.meta_local_germinate_from_offspring_and_immigrant_pool()
            #meta_obj.meta_local_germinate_from_offspring_immigrant_and_dormancy_pool()
            ''' dormancy process (not running) '''
            #meta_obj.meta_dormancy_process_from_offspring_pool_and_immigrant_pool()
            ''' disturbance process '''
            write_in_logger_info += meta_obj.meta_disturbance_process_in_patches(patch_dist_rate)
            ''' eliminating offspring and immigrant (marker) pool '''
            meta_obj.meta_clear_up_offspring_marker_and_immigrant_marker_pool()
            #meta_obj.meta_clear_up_offspring_and_immigrant_pool()
        
        ''' logging module '''
        write_logger(write_in_logger_info, is_logging, logger_file)
        
        ''' GUI interface updates at each time-step and saves as a jpg file (not running) '''
        #meta_obj.meta_show_species_distribution(sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, cmap=plt.get_cmap('tab20'), file_name=goal_path+'/'+'rep=%d-time_step=%d-metacommunity_sp_x_axis_phenotype_dis.jpg'%(rep, time_step))

        ''' data saving and files controling '''
        #mode='w
        #meta_sp_dis_all_time = np.vstack((meta_sp_dis_all_time, meta_obj.get_meta_microsites_individuals_sp_id_values()))
        #meta_x_axis_phenotype_all_time = np.vstack((meta_x_axis_phenotype_all_time, meta_obj.get_meta_microsites_individuals_phenotype_values(trait_name='x_axis_phenotype')))
        #meta_y_axis_phenotype_all_time = np.vstack((meta_y_axis_phenotype_all_time, meta_obj.get_meta_microsites_individuals_phenotype_values(trait_name='y_axis_phenotype')))
        #mode='a'
        meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_individuals_sp_id_values(), file_name=goal_path+'/'+'rep=%d-meta_species_distribution_all_time.csv.gz'%(rep), index=['time_step%d'%time_step], columns=columns, mode='a')
        meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_individuals_phenotype_values(trait_name='phenotype_axis1'), file_name=goal_path+'/'+'rep=%d-meta_x_axis_phenotype_all_time.csv.gz'%(rep), index=['time_step%d'%time_step], columns=columns, mode='a')
        meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_individuals_phenotype_values(trait_name='phenotype_axis2'), file_name=goal_path+'/'+'rep=%d-meta_y_axis_phenotype_all_time.csv.gz'%(rep), index=['time_step%d'%time_step], columns=columns, mode='a')
    
    ''' GUI interface updates at the end of a simulation saves as a jpg file '''
    meta_obj.meta_show_species_distribution(sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, vmin=1, vmax=16, cmap=plt.get_cmap('tab20'), file_name=goal_path+'/'+'rep=%d-time_step=%d-metacommunity_sp_dis.jpg'%(rep, time_step))
    meta_obj.meta_show_species_phenotype_distribution(trait_name='phenotype_axis1', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, cmap=plt.get_cmap('turbo'), file_name=goal_path+'/'+'rep=%d-time_step=%d-metacommunity_sp_x_axis_phenotype_dis.jpg'%(rep, time_step))
    meta_obj.meta_show_species_phenotype_distribution(trait_name='phenotype_axis2', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, cmap=plt.get_cmap('turbo'), file_name=goal_path+'/'+'rep=%d-time_step=%d-metacommunity_sp_y_axis_phenotype_dis.jpg'%(rep, time_step))
    meta_obj.meta_show_environment_distribution(environment_name='env_axis1', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, mask_loc=None, cmap=plt.get_cmap('turbo'), file_name=goal_path+'/'+'rep=%d-metacommunity_axis1_environment.jpg'%(rep))
    meta_obj.meta_show_environment_distribution(environment_name='env_axis2', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, mask_loc=None, cmap=plt.get_cmap('turbo'), file_name=goal_path+'/'+'rep=%d-metacommunity_axis2_environment.jpg'%(rep))
    #meta_obj.meta_show_two_environment_distribution(environment1_name='env_axis1', environment2_name='env_axis2', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, mask_loc1='upper', mask_loc2='lower', cmap1=plt.get_cmap('Blues'), cmap2=plt.get_cmap('Greens'), file_name=goal_path+'/'+'rep=%d-metacommunity_environment.jpg'%(rep))

    #''' data saving as a csv.gz files, mode='w' '''
    #mode='w'
    #meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_sp_dis_all_time, file_name=goal_path+'/'+'rep=%d_meta_species_distribution_all_time.csv.gz'%(rep), index=['optimun_sp_id_values']+['time_step%d'%i for i in range(all_time_step)], columns=columns, mode='w')
    #meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_x_axis_phenotype_all_time, file_name=goal_path+'/'+'rep=%d_meta_x_axis_phenotype_all_time.csv.gz'%(rep), index=['x_axis_environment_values']+['time_step%d'%i for i in range(all_time_step)], columns=columns, mode='w')
    #meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_y_axis_phenotype_all_time, file_name=goal_path+'/'+'rep=%d_meta_y_axis_phenotype_all_time.csv.gz'%(rep), index=['y_axis_environment_values']+['time_step%d'%i for i in range(all_time_step)], columns=columns, mode='w')
    
    ''' timer '''
    all_time_end = time.time()
    
    ''' logging module '''
    log_info = "总模拟运行时间：%.8s s \n" % (all_time_end-all_time_start)
    write_logger(log_info, is_logging, logger_file)
    if is_logging == True: logger_file.close()
    #print(log_info)

##############################################################################################################################################################################
if __name__ == '__main__':
    main(rep=0, patch_num=16, is_same_heterogeneity=True, reproduce_mode='sexual', total_disp_among_rate=0.01, disp_within_rate=0.1, patch_dist_rate=0.00001, goal_path=None)