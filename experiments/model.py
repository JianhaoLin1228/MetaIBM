#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 23:36:53 2024

@author: jianhaolin
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import bootstrap_metaibm as _bootstrap
import metaibm
from metaibm.individual import individual
from metaibm.habitat import habitat
from metaibm.patch import patch
from metaibm.metacommunity import metacommunity

######################################### def simulation() #########################################
def generating_empty_metacommunity(meta_name, patch_num, patch_location_ls, hab_num, hab_length, hab_width, dormancy_pool_max_size, 
                                   environment_mean_value, environment_types_num, environment_types_name, environment_variation_ls):
    ''' '''
    log_info = 'generating empty metacommunity ... \n'
    meta_object = metacommunity(metacommunity_name=meta_name)
    patch_num_x_axis, patch_num_y_axis = int(np.sqrt(patch_num)), int(np.sqrt(patch_num))
    hab_num_x_axis, hab_num_y_axis = int(np.sqrt(hab_num)), int(np.sqrt(hab_num))
    
    for i in range(patch_num):
        patch_name = 'patch%d'%(i+1)
        patch_index = i
        location = patch_location_ls[i]
        p = patch(patch_name, patch_index, location)
        
        patch_x_loc, patch_y_loc = location[0], location[1]

        for j in range(hab_num):
            habitat_name = 'h%s'%str(j+1)
            hab_index = j
            
            hab_x_loc, hab_y_loc = patch_x_loc*hab_num_x_axis+j//hab_num_y_axis, patch_y_loc*hab_num_y_axis+j%hab_num_y_axis
            hab_location = (hab_x_loc, hab_y_loc)
                            
            p.add_habitat(hab_name=habitat_name, hab_index=hab_index, hab_location=hab_location, num_env_types=environment_types_num, env_types_name=environment_types_name, 
                          mean_env_ls=[environment_mean_value], var_env_ls=environment_variation_ls, length=hab_length, width=hab_width, dormancy_pool_max_size=dormancy_pool_max_size)
  
            info = '%s: %s, %s, %s, %s: environment_mean_value=%s'%(meta_object.metacommunity_name, patch_name, str(location), habitat_name, str(hab_location), str(environment_mean_value))
            log_info = log_info + info + '\n'
        meta_object.add_patch(patch_name=patch_name, patch_object=p)
    #print(log_info)
    return meta_object, log_info

################################################## logging module ########################################################################################
def write_logger(log_info, is_logging=False, logger_file=None):
    if is_logging == True:
        print(log_info, file=logger_file)
    elif is_logging == False:
        print(log_info)

###################################### def main() ######################################################
def main(reproduce_mode, patch_dist_rate, mutation_rate, environment_mean_value, delta_mean_ls, rep=0, goal_path=None):
    
    ''' timer '''
    all_time_start = time.time()
    
    ''' time_steps controller '''
    time_step = -1
    all_time_step = 1200
    
    ''' landscape map '''
    meta_length, meta_width = 320, 320
    
    patch_num = 100
    patch_num_x_axis, patch_num_y_axis = int(np.sqrt(patch_num)), int(np.sqrt(patch_num))
    patch_location_ls = [(i,j) for i in range(patch_num_x_axis) for j in range(patch_num_y_axis)]
    
    hab_num_in_patch = 1
    hab_num_x_axis, hab_num_y_axis = int(np.sqrt(hab_num_in_patch)), int(np.sqrt(hab_num_in_patch)) # hab_num_x_axis in a patch, hab_num_y_axis in a patch
    hab_length, hab_width = int(meta_length/patch_num_y_axis/hab_num_y_axis), int(meta_width/patch_num_x_axis/hab_num_x_axis) 

    ''' environment '''
    #environment_mean_value = 0.3
    environment_types_num = 1
    environment_types_name = ['environment']
    environment_variation_ls = [0.025]
    
    ''' changing environment '''
    start_change_time, end_change_time = time_step+100, all_time_step-100
    change_time_step = 100     # if (time_step+1)%change_time_step==0: 
    env_name_ls = ['environment']
    #delta_mean_ls = [0.1]
    delta_var_ls = [0]
    
    ''' species parameters '''
    species_2_phenotype_ls = [[0.0], [1.0]]
    
    ''' individuals parameters'''
    traits_num = 1
    pheno_names_ls = ['phenotype']
    pheno_var_ls = [0.025]
    geno_len_ls = [20]
    #reproduce_mode = 'sexual' # 'asexual'

    ''' eco-evo processes parameters '''
    ''' dispersal processes '''
    dispersal_amomg_rate = 0.0001
    propagules_rain_num = 10

    ''' demography parameters '''
    base_dead_rate = 0.1
    fitness_wid = 0.5
    asexual_birth_rate = 0.5
    sexual_birth_rate = 1
    #mutation_rate = 0.0001
    
    ''' dormancy processes '''
    #dormancy_pool_max_size=0
    
    ''' disturbance processes '''
    #patch_dist_rate = 0.0001
    
    ''' files systems path parameters '''
    if goal_path==None: goal_path = os.getcwd()
    
    ''' logging parameters '''
    is_logging = True
    
    ############################################################################################################################
    ''' logging module '''
    if is_logging == True: 
        logger_file_name = goal_path+'/'+'rep=%d-logger.log'%(rep)
        logger_file = open(logger_file_name, "a")
    
    ''' create landscape '''
    write_in_logger_info = ''
    
    meta_object, log_info = generating_empty_metacommunity(meta_name='metacommunity', 
                                patch_num=patch_num, patch_location_ls=patch_location_ls, 
                                hab_num=hab_num_in_patch, hab_length=hab_length, hab_width=hab_width, dormancy_pool_max_size=0, 
                                environment_mean_value=environment_mean_value, environment_types_num=environment_types_num, 
                                environment_types_name=environment_types_name, environment_variation_ls=environment_variation_ls)
    
    write_in_logger_info += log_info
    
    mainland1, log_info = generating_empty_metacommunity(meta_name='mainland1', 
                              patch_num=1, patch_location_ls=[(0,0)], 
                              hab_num=1, hab_length=100, hab_width=100, dormancy_pool_max_size=0, 
                              environment_mean_value=0.0, environment_types_num=1, 
                              environment_types_name=['environment'], environment_variation_ls=[0.025])
    
    write_in_logger_info += log_info
    
    mainland2, log_info = generating_empty_metacommunity(meta_name='mainland2', 
                              patch_num=1, patch_location_ls=[(0,0)], 
                              hab_num=1, hab_length=100, hab_width=100, dormancy_pool_max_size=0, 
                              environment_mean_value=1.0, environment_types_num=1, 
                              environment_types_name=['environment'], environment_variation_ls=[0.025])
    
    write_in_logger_info += log_info
    
    ''' initializing species pool in the mainlands '''
    write_in_logger_info += mainland1.meta_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
    write_in_logger_info += mainland2.meta_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
    
    write_logger(write_in_logger_info, is_logging, logger_file)
    
    ''' data saving and files controlling systems module '''
    '''Create files and Write appendingly '''
    columns_patch_id, columns_habitat_id, columns_mocrosite_id = meta_object.columns_patch_habitat_microsites_id()
    columns = [columns_patch_id, columns_habitat_id, columns_mocrosite_id]
    # Create files, mode = 'a'
    meta_object.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_object.get_meta_microsites_optimum_sp_id_val(base_dead_rate, fitness_wid, species_2_phenotype_ls), 
                                                       file_name=goal_path+'/'+'rep=%d-meta_species_distribution_all_time.csv.gz'%(rep), 
                                                       index=['optimun_sp_id_values'], columns=columns, mode='w')
    meta_object.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_object.get_meta_microsite_environment_values(environment_name='environment'), 
                                                       file_name=goal_path+'/'+'rep=%d-meta_phenotype_all_time.csv.gz'%(rep), 
                                                       index=['environment'], columns=columns, mode='w')

    #meta_object.meta_show_environment_distribution(environment_name='environment', sub_row=10, sub_col=10, hab_num_x_axis_in_patch=1, hab_num_y_axis_in_patch=1, hab_y_len=hab_width, hab_x_len=hab_length, mask_loc=None, cmap=plt.get_cmap('Blues'), file_name='time_step=%d-metacommunity_environment.jpg'%(time_step))
    #mainland1.meta_show_environment_distribution(environment_name='environment', sub_row=1, sub_col=1, hab_num_x_axis_in_patch=1, hab_num_y_axis_in_patch=1, hab_y_len=100, hab_x_len=100, mask_loc=None, cmap=plt.get_cmap('Blues'), file_name='time_step=%d-mainland1_environment.jpg'%(time_step))
    #mainland2.meta_show_environment_distribution(environment_name='environment', sub_row=1, sub_col=1, hab_num_x_axis_in_patch=1, hab_num_y_axis_in_patch=1, hab_y_len=100, hab_x_len=100, mask_loc=None, cmap=plt.get_cmap('Blues'), file_name='time_step=%d-mainland2_environment.jpg'%(time_step))
    
    #meta_object.meta_show_species_distribution(sub_row=10, sub_col=10, hab_num_x_axis_in_patch=1, hab_num_y_axis_in_patch=1, hab_y_len=hab_width, hab_x_len=hab_length, vmin=1, vmax=2, cmap=plt.get_cmap('tab20'), file_name='time_step=%d-metacommunity_sp_dis.jpg'%(time_step))
    #mainland1.meta_show_species_distribution(sub_row=1, sub_col=1, hab_num_x_axis_in_patch=1, hab_num_y_axis_in_patch=1, hab_y_len=100, hab_x_len=100, vmin=1, vmax=2, cmap=plt.get_cmap('tab20'), file_name='time_step=%d-mainland1_sp_dis.jpg'%(time_step))
    #mainland2.meta_show_species_distribution(sub_row=1, sub_col=1, hab_num_x_axis_in_patch=1, hab_num_y_axis_in_patch=1, hab_y_len=100, hab_x_len=100, vmin=1, vmax=2, cmap=plt.get_cmap('tab20'), file_name='time_step=%d-mainland2_sp_dis.jpg'%(time_step))  

    for time_step in range(all_time_step): 
        write_in_logger_info = ''
        write_in_logger_info += 'time_step=%d \n'%time_step
        #print('time_step=%d'%time_step)
        if reproduce_mode == 'asexual':
            
            write_in_logger_info += mainland1.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            write_in_logger_info += mainland1.meta_mainland_asexual_birth_mutate_germinate(asexual_birth_rate, mutation_rate, pheno_var_ls)
            
            write_in_logger_info += mainland2.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            write_in_logger_info += mainland2.meta_mainland_asexual_birth_mutate_germinate(asexual_birth_rate, mutation_rate, pheno_var_ls)
            
            write_in_logger_info += meta_object.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            write_in_logger_info += meta_object.meta_asex_reproduce_calculation_into_offspring_marker_pool(asexual_birth_rate)

            if (time_step+1) % change_time_step == 0 and start_change_time<=time_step<=end_change_time: write_in_logger_info += meta_object.meta_offset_environmental_values(env_name_ls, delta_mean_ls, delta_var_ls)
            
            write_in_logger_info += meta_object.meta_colonize_from_propagules_rains(mainland1, propagules_rain_num) 
            write_in_logger_info += meta_object.meta_colonize_from_propagules_rains(mainland2, propagules_rain_num)
            
            write_in_logger_info += meta_object.dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool(dispersal_amomg_rate)
            write_in_logger_info += meta_object.meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool(mutation_rate, pheno_var_ls)
            write_in_logger_info += meta_object.meta_disturbance_process_in_patches(patch_dist_rate)
            
            meta_object.meta_clear_up_offspring_marker_and_immigrant_marker_pool()
        
        elif reproduce_mode == 'sexual':
            write_in_logger_info += mainland1.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            write_in_logger_info += mainland1.meta_mainland_mixed_birth_mutate_germinate(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)

            write_in_logger_info += mainland2.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            write_in_logger_info += mainland2.meta_mainland_mixed_birth_mutate_germinate(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)

        
            write_in_logger_info += meta_object.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            write_in_logger_info += meta_object.meta_mix_reproduce_calculation_with_offspring_marker_pool(asexual_birth_rate, sexual_birth_rate)

            if (time_step+1) % change_time_step == 0 and start_change_time<=time_step<=end_change_time: write_in_logger_info += meta_object.meta_offset_environmental_values(env_name_ls, delta_mean_ls, delta_var_ls)
            
            write_in_logger_info += meta_object.meta_colonize_from_propagules_rains(mainland1, propagules_rain_num) 
            write_in_logger_info += meta_object.meta_colonize_from_propagules_rains(mainland2, propagules_rain_num)

            write_in_logger_info += meta_object.dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool(dispersal_amomg_rate)
            write_in_logger_info += meta_object.meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool(mutation_rate, pheno_var_ls)
            write_in_logger_info += meta_object.meta_disturbance_process_in_patches(patch_dist_rate)

            meta_object.meta_clear_up_offspring_marker_and_immigrant_marker_pool()
            
        ''' logging module '''
        write_logger(write_in_logger_info, is_logging, logger_file) 
        
        #mode='a'
        meta_object.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_object.get_meta_microsites_individuals_sp_id_values(), file_name=goal_path+'/'+'rep=%d-meta_species_distribution_all_time.csv.gz'%(rep), index=['time_step%d'%time_step], columns=columns, mode='a')
        meta_object.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_object.get_meta_microsites_individuals_phenotype_values(trait_name='phenotype'), file_name=goal_path+'/'+'rep=%d-meta_phenotype_all_time.csv.gz'%(rep), index=['time_step%d'%time_step], columns=columns, mode='a')
    
    ''' timer '''
    all_time_end = time.time()
    
    ''' logging module '''
    log_info = "all the simulation time：%.8s s \n" % (all_time_end-all_time_start)
    write_logger(log_info, is_logging, logger_file)
    if is_logging == True: logger_file.close()
    
    meta_object.meta_show_species_distribution(sub_row=10, sub_col=10, hab_num_x_axis_in_patch=1, hab_num_y_axis_in_patch=1, hab_y_len=hab_width, hab_x_len=hab_length, vmin=1, vmax=2, cmap=plt.get_cmap('Paired'), file_name=goal_path+'/'+'rep=%d-time_step=%d-metacommunity_sp_dis.jpg'%(rep, time_step))
    meta_object.meta_show_species_phenotype_distribution(trait_name='phenotype', sub_row=10, sub_col=10, hab_num_x_axis_in_patch=1, hab_num_y_axis_in_patch=1, hab_y_len=hab_width, hab_x_len=hab_length, cmap=plt.get_cmap('Paired'), file_name=goal_path+'/'+'rep=%d-time_step=%d-metacommunity_sp_x_axis_phenotype_dis.jpg'%(rep, time_step))
    return locals()

##################################################################################################################################
if __name__ == '__main__':
    local = main(reproduce_mode='sexual', patch_dist_rate=0.00001, mutation_rate=0.0001, environment_mean_value=0.0, delta_mean_ls=[0.1], rep=0, goal_path=None)






















