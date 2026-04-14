# -*- coding: utf-8 -*-
"""
Split from metacommunity_IBM.py.
Compatibility-preserving class module.
"""

import numpy as np
import random
import copy
import re

from .habitat import habitat


class patch():
    def __init__(self, patch_name, patch_index, location):
        self.name = patch_name
        self.index = patch_index
        self.set = {}            # self.data_set={} # to be improved
        self.hab_num = 0
        self.hab_id_ls = []
        self.location = location
        
    def get_data(self):
        output = {}
        for key, value in self.set.items():
            output[key]=value.set
        return output

    def add_habitat(self, hab_name, hab_index, hab_location, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width, dormancy_pool_max_size):
        h_object = habitat(hab_name, hab_index, hab_location, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width, dormancy_pool_max_size)
        self.set[hab_name] = h_object
        self.hab_id_ls.append(hab_name)
        self.hab_num += 1
        
    def patch_initialize(self, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls):
        for h_id, h_object in self.set.items():
            h_object.hab_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
        return 0    

    def patch_dead_selection(self, base_dead_rate, fitness_wid, method):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_dead_selection(base_dead_rate, fitness_wid, method)
        return counter

    def get_patch_individual_num(self):
        num = 0
        for key, value in self.set.items():
            num += value.indi_num
        return num

    def get_patch_empty_sites_ls(self):
        ''' return patch_empty_pos_ls as [(h_id, len_id, wid_id)] '''
        patch_empty_pos_ls = []
        for h_id, h_object in self.set.items():
            empty_site_pos_ls = h_object.empty_site_pos_ls
            for site_pos in empty_site_pos_ls:
                site_pos = (h_id, ) + site_pos
                patch_empty_pos_ls.append(site_pos)
        return patch_empty_pos_ls

    def patch_empty_sites_num(self):
        ''' return the number of empty microsite in the patches '''
        return len(self.get_patch_empty_sites_ls())

    def get_patch_pairwise_empty_sites_ls(self):
        ''' return patch_empty_pos_ls as [((h_id, len_id, wid_id), (h_id, len_id, wid_id))...] '''
        patch_pairwise_empty_pos_ls = []
        for h_id, h_object in self.set.items():
            pairwise_empty_sites_pos_ls = h_object.get_hab_pairwise_empty_site_pos_ls()
            for (empty_site_1_pos, empty_site_2_pos) in pairwise_empty_sites_pos_ls:
                empty_site_1_pos = (h_id, ) + empty_site_1_pos
                empty_site_2_pos = (h_id, ) + empty_site_2_pos
                patch_pairwise_empty_pos_ls.append((empty_site_1_pos, empty_site_2_pos))
        return patch_pairwise_empty_pos_ls
    
    def get_patch_pairwise_occupied_sites_ls(self):
        ''' return patch_empty_pos_ls as [((h_id, len_id, wid_id), (h_id, len_id, wid_id))...] '''
        patch_pairwise_occupied_pos_ls = []
        for h_id, h_object in self.set.items():
            pairwise_occupied_sites_pos_ls = h_object.get_hab_pairwise_occupied_site_pos_ls()
            for (occupied_site_1_pos, occupied_site_2_pos) in pairwise_occupied_sites_pos_ls:
                occupied_site_1_pos = (h_id, ) + occupied_site_1_pos
                occupied_site_2_pos = (h_id, ) + occupied_site_2_pos
                patch_pairwise_occupied_pos_ls.append((occupied_site_1_pos, occupied_site_2_pos))
        return patch_pairwise_occupied_pos_ls
    
    def get_patch_occupied_sites_ls(self):
        ''' return patch_occupied_pos_ls as [(h_id, len_id, wid_id)] '''
        patch_occupied_pos_ls = []
        for h_id, h_object in self.set.items():
            occupied_site_pos_ls = h_object.occupied_site_pos_ls
            for site_pos in occupied_site_pos_ls:
                site_pos = (h_id, ) + site_pos
                patch_occupied_pos_ls.append(site_pos)
        return patch_occupied_pos_ls
    
    def get_patch_offspring_marker_pool(self):
        ''' get the combination of all the offspring marker pool in all habitat in the patch 
        return as a list of offspring marker. '''
        patch_offsprings_marker_pool = []
        for h_id, h_object in self.set.items():
            patch_offsprings_marker_pool += h_object.offspring_marker_pool
        return patch_offsprings_marker_pool
    
    def get_patch_offspring_pool(self):
        ''' get the combination of all the offspring pool in all habitat in the patch 
        return as a list of offspring individual object. '''
        patch_offsprings_pool = []
        for h_id, h_object in self.set.items():
            patch_offsprings_pool += h_object.offspring_pool
        return patch_offsprings_pool
    
    def get_patch_dormancy_pool(self):
        ''' get the combination of all the dormancy pool in all habitat in the patch 
        return as a list of dormancy individual object. '''
        patch_dormancy_pool = []
        for h_id, h_object in self.set.items():
            patch_dormancy_pool += h_object.dormancy_pool
        return patch_dormancy_pool
    
    def get_patch_offspring_and_dormancy_pool(self):
        ''' get the combination of all the offspring_pool_and_dormancy_pool in all habitat in the patch 
        return as a list of offspring_and_dormancy individual object. '''
        patch_offspring_and_dormancy_pool = []
        for h_id, h_object in self.set.items():
            patch_offspring_and_dormancy_pool += h_object.offspring_pool
            patch_offspring_and_dormancy_pool += h_object.dormancy_pool
        return patch_offspring_and_dormancy_pool
    
    def get_patch_microsites_individals_sp_id_values(self):
        ''' get species_id, phenotypes distribution in the patch as values set '''
        values_set = np.array([], dtype=int)
        for h_id, h_object in self.set.items():
            hab_len = h_object.length
            hab_wid = h_object.width
            for row in range(hab_len):
                for col in range(hab_wid):
                    individual_object = h_object.set['microsite_individuals'][row][col]
                    if individual_object ==None:
                        values_set = np.append(values_set, np.nan)
                    else:
                        species_id = individual_object.species_id
                        species_id_value = int(re.findall(r"\d+",species_id)[0])
                        values_set = np.append(values_set, species_id_value)
        values_set = values_set.reshape(self.hab_num, h_object.size)
        return values_set
    
    def get_patch_microsites_individals_phenotype_values(self, trait_name):
        ''' get species_id, phenotypes distribution in the patch as values set '''
        values_set = []
        for h_id, h_object in self.set.items():
            hab_len = h_object.length
            hab_wid = h_object.width
            for row in range(hab_len):
                for col in range(hab_wid):
                    individual_object = h_object.set['microsite_individuals'][row][col]
                    if individual_object ==None:
                        values_set.append(np.nan)
                    else:
                        phenotype = individual_object.phenotype_set[trait_name]
                        values_set.append(phenotype)
        values_set = np.array(values_set).reshape(self.hab_num, h_object.size)
        return values_set
    
    def get_patch_microsites_environment_values(self, environment_name):
        ''' get microsite environment values distribution in the patch as values set '''
        values_array = np.array([])
        for h_id, h_object in self.set.items():
            hab_environment_values_array = h_object.set[environment_name] 
            values_array = np.append(values_array, hab_environment_values_array) # dimension of the return of np.append() is always in dim=1
        values_array = values_array.reshape(self.hab_num, h_object.size)
        return values_array

# ********************* changing environmental values (offset) ********************************
    def patch_offset_environmental_values(self, env_name_ls, delta_mean_ls, delta_var_ls=None):
        ''' multi mean (var) values of env_name_ls in habitats + or - delta_mean_ls (or delta_val_ls) across the patches '''
        if delta_var_ls == None: delta_var_ls = [0 for i in range(len(env_name_ls))]
        for h_id, h_object in self.set.items():
            h_object.hab_offset_environment_values(env_name_ls, delta_mean_ls, delta_var_ls)
        return 0
  
##********************* for patch in mainland only and do not contains dormant bank *******************************####
    def patch_asexual_birth_germinate(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        ''' birth into empty site directly without considering the competition between local offspring and immigrant offspring '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_asexual_reprodece_germinate(asexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
    
    def patch_sexual_birth_germinate(self, sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method='segregation', recomb_rate=0.0):
        ''' birth into empty site directly without considering the competition between local offspring and immigrant offspring '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_sexual_reprodece_germinate(sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method=recomb_method, recomb_rate=recomb_rate)
        return counter
    
    def patch_mixed_birth_germinate(self, asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method='segregation', recomb_rate=0.0):
        ''' birth into empty site directly without considering the competition between local offspring and immigrant offspring '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_mixed_reproduce_germinate(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method=recomb_method, recomb_rate=recomb_rate)
        return counter
    
#** calculating the offspring num and generating offspring markers but do not reproduce actually until local germination process afterward after dispersal process.*
    def patch_asex_reproduce_calculation_into_offspring_marker_pool(self, asexual_birth_rate):
        ''' used only when dormancy process do not occur '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_asex_reproduce_calculation_into_offspring_marker_pool(self.name, asexual_birth_rate)
        return counter
    
    def patch_sex_reproduce_calculation_into_offspring_marker_pool(self, sexual_birth_rate):
        ''' used only when dormancy process do not occur '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_sex_reproduce_calculation_into_offspring_marker_pool(self.name, sexual_birth_rate)
        return counter
    
    def patch_mix_reproduce_calculation_into_offspring_marker_pool(self, asexual_birth_rate, sexual_birth_rate):
        ''' used only when dormancy process do not occur '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_mix_reproduce_calculation_into_offspring_marker_pool(self.name, asexual_birth_rate, sexual_birth_rate)
        return counter
        
#********************************************************************************************************************#
#************************ all hab in patch reproduce into offspring pool ********************************************#
    def patch_asex_reproduce_mutate_into_offspring_pool(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_asex_reproduce_mutate_into_offspring_pool(asexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
    
    def patch_sex_reproduce_mutate_into_offspring_pool(self, sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method='segregation', recomb_rate=0.0):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_sex_reproduce_mutate_into_offspring_pool(sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method=recomb_method, recomb_rate=recomb_rate)
        return counter
    
    def patch_mix_reproduce_mutate_into_offspring_pool(self, asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method='segregation', recomb_rate=0.0):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_mix_reproduce_mutate_into_offspring_pool(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls, recomb_method=recomb_method, recomb_rate=recomb_rate)
        return counter
#****************************************** dispersal among patch into habs offspring pool *******************************************************
    def split(self, ls, n):
        ''' split a ls into n pieces '''
        k, m = divmod(len(ls), n)
        divided_ls = list(ls[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
        random.shuffle(divided_ls)
        return divided_ls

    def migrants_to_patch_into_habs_immigrant_pool(self, migrants_indi_obj_to_patch_ls):
        divide_into_habs_indi_obj_ls = self.split(migrants_indi_obj_to_patch_ls, self.hab_num)
        for h_id, h_object in self.set.items():
            h_object.immigrant_pool += divide_into_habs_indi_obj_ls[h_object.index]
        return 0
    
    def migrants_marker_to_patch_into_habs_immigrant_marker_pool(self, migrants_to_j_offspring_marker_ls):
        divide_into_habs_offs_marker_ls = self.split(migrants_to_j_offspring_marker_ls, self.hab_num)
        for h_id, h_object in self.set.items():
            h_object.immigrant_marker_pool += divide_into_habs_offs_marker_ls[h_object.index]
        return 0
#****************************************** dispersal within patch ******************************************************
    def patch_matrix_around(self, matrix):   
        ''' 元素的小数部分，按照概率四舍五入 '''
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isnan(matrix[i, j]) == False:
                    integer = int(matrix[i,j])
                    demacial = matrix[i,j] - integer
                    if np.random.uniform(0,1,1)[0] <= demacial:
                        integer += 1
                    else:
                        pass
                else:
                    integer = 0
                matrix[i, j] = integer
        return matrix
    
    def get_patch_dispersal_within_rate_matrix(self, disp_within_rate):
        dispersal_within_rate_matrix = np.matrix(np.ones((self.hab_num, self.hab_num)))*disp_within_rate/(self.hab_num-1)
        dispersal_within_rate_matrix[np.diag_indices_from(dispersal_within_rate_matrix)] = 1-disp_within_rate  # elements in diagonal_line = 1-m
        return dispersal_within_rate_matrix

    def get_patch_habs_offspring_num_matrix(self):
        habs_offspring_num_matrix = np.matrix(np.zeros((self.hab_num, self.hab_num)))
        for h_id, h_object in self.set.items():
            offspring_num = len(h_object.offspring_pool)
            habs_offspring_num_matrix[h_object.index, h_object.index] = offspring_num
        return habs_offspring_num_matrix
    
    def get_patch_habs_offspring_marker_num_matrix(self):
        habs_offspring_marker_num_matrix = np.matrix(np.zeros((self.hab_num, self.hab_num)))
        for h_id, h_object in self.set.items():
            offspring_marker_num = len(h_object.offspring_marker_pool)
            habs_offspring_marker_num_matrix[h_object.index, h_object.index] = offspring_marker_num
        return habs_offspring_marker_num_matrix
    
    def get_patch_habs_dormancy_num_matrix(self):
        habs_dormancy_num_matrix = np.matrix(np.zeros((self.hab_num, self.hab_num)))
        for h_id, h_object in self.set.items():
            dormancy_num = len(h_object.dormancy_pool)
            habs_dormancy_num_matrix[h_object.index, h_object.index] = dormancy_num
        return habs_dormancy_num_matrix

    def get_patch_habs_empty_sites_num_matrix(self):
        habs_empty_sites_num_matrix = np.matrix(np.zeros((self.hab_num, self.hab_num)))
        for h_id, h_object in self.set.items():
            empty_site_num = len(h_object.empty_site_pos_ls)
            habs_empty_sites_num_matrix[h_object.index, h_object.index] = empty_site_num
        return habs_empty_sites_num_matrix
    
    def get_habs_emigrants_matrix(self, disp_within_rate):
        habs_offspring_num_matrix = self.get_patch_habs_offspring_num_matrix()
        habs_dormancy_num_matrix = self.get_patch_habs_dormancy_num_matrix()
        dispersal_within_rate_matrix = self.get_patch_dispersal_within_rate_matrix(disp_within_rate)
        return (habs_offspring_num_matrix + habs_dormancy_num_matrix) * dispersal_within_rate_matrix
    
    def get_habs_immigrants_matrix(self, disp_within_rate):
        habs_emigrants_matrix = self.get_habs_emigrants_matrix(disp_within_rate)
        habs_empty_sites_num_matrix = self.get_patch_habs_empty_sites_num_matrix()
        return (habs_emigrants_matrix/habs_emigrants_matrix.sum(axis=0))*habs_empty_sites_num_matrix
    
    def get_dispersal_within_num_matrix(self, disp_within_rate):
        habs_emigrants_matrix = self.get_habs_emigrants_matrix(disp_within_rate)
        habs_immigrants_matrix = self.get_habs_immigrants_matrix(disp_within_rate)
        return self.patch_matrix_around(np.minimum(habs_emigrants_matrix, habs_immigrants_matrix))

    def patch_dipersal_within_from_offspring_and_dormancy_pool(self, disp_within_rate):
        disp_within_num_matrix = self.get_dispersal_within_num_matrix(disp_within_rate)
        counter = 0
        for j in range(self.hab_num):
            h_j_id = self.hab_id_ls[j]
            h_j_object = self.set[h_j_id]
            h_j_empty_site_ls = h_j_object.empty_site_pos_ls
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.hab_num):
                h_i_id = self.hab_id_ls[i]
                h_i_object = self.set[h_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(disp_within_num_matrix[i, j])
                    hab_i_offspring_and_dormancy_pool = h_i_object.offspring_pool + h_i_object.dormancy_pool
                    migrants_to_j_indi_object_ls += random.sample(hab_i_offspring_and_dormancy_pool, migrants_i_j_num)
                    
            random.shuffle(h_j_empty_site_ls)
            random.shuffle(migrants_to_j_indi_object_ls)
            for (len_id, wid_id), migrants_object in list(zip(h_j_empty_site_ls, migrants_to_j_indi_object_ls)):
                self.set[h_j_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id)
                counter += 1
        return counter
    
    def patch_dispersal_within_from_offspring_marker_pool_to_immigrant_marker_pool(self, disp_within_rate):
        offspring_marker_dispersal_matrix =  self.patch_matrix_around(self.get_patch_habs_offspring_marker_num_matrix()*self.get_patch_dispersal_within_rate_matrix(disp_within_rate))
        counter = 0
        for j in range(self.hab_num):
            h_j_id = self.hab_id_ls[j]
            h_j_object = self.set[h_j_id]
            migrants_to_j_offspring_marker_ls = []
            
            for i in range(self.hab_num):
                h_i_id = self.hab_id_ls[i]
                h_i_object = self.set[h_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(offspring_marker_dispersal_matrix[i, j])
                    hab_i_offspring_marker_pool = h_i_object.offspring_marker_pool
                    migrants_to_j_offspring_marker_ls += random.sample(hab_i_offspring_marker_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_offspring_marker_ls)
            h_j_object.immigrant_marker_pool += migrants_to_j_offspring_marker_ls
            counter += len(migrants_to_j_offspring_marker_ls)
        return counter
    
    def patch_dispersal_within_from_offspring_pool_to_immigrant_pool(self, disp_within_rate):
        offspring_dispersal_matrix = self.patch_matrix_around(self.get_patch_habs_offspring_num_matrix()*self.get_patch_dispersal_within_rate_matrix(disp_within_rate))
        counter = 0
        for j in range(self.hab_num):
            h_j_id = self.hab_id_ls[j]
            h_j_object = self.set[h_j_id]
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.hab_num):
                h_i_id = self.hab_id_ls[i]
                h_i_object = self.set[h_i_id]
                if i==j: 
                    continue
                else:
                    migrants_i_j_num = int(offspring_dispersal_matrix[i, j])
                    hab_i_offspring_pool = h_i_object.offspring_pool
                    migrants_to_j_indi_object_ls += random.sample(hab_i_offspring_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_indi_object_ls)
            h_j_object.immigrant_pool += migrants_to_j_indi_object_ls
            counter += len(migrants_to_j_indi_object_ls)
        return counter
    
    def patch_dispersal_within_from_offspring_pool_and_dormancy_pool_to_immigrant_pool(self, disp_within_rate):
        offspring_and_dormancy_dispersal_matrix =  self.patch_matrix_around(self.get_habs_emigrants_matrix(disp_within_rate))
        counter = 0
        for j in range(self.hab_num):
            h_j_id = self.hab_id_ls[j]
            h_j_object = self.set[h_j_id]
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.hab_num):
                h_i_id = self.hab_id_ls[i]
                h_i_object = self.set[h_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(offspring_and_dormancy_dispersal_matrix[i, j])
                    hab_i_offspring_and_dormancy_pool = h_i_object.offspring_pool + h_i_object.dormancy_pool
                    migrants_to_j_indi_object_ls += random.sample(hab_i_offspring_and_dormancy_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_indi_object_ls)
            h_j_object.immigrant_pool += migrants_to_j_indi_object_ls
            counter += len(migrants_to_j_indi_object_ls)
        return counter        
        
#******************** local germination in all habitat in the patch *****************************#
    def patch_local_germinate_from_offspring_and_dormancy_pool(self):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_local_germinate_from_offspring_and_dormancy_pool()
        return counter
    
    def patch_local_germinate_from_offspring_and_immigrant_pool(self):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_local_germinate_from_offspring_and_immigrant_pool()
        return counter
            
    def patch_local_germinate_from_offspring_immigrant_and_dormancy_pool(self):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_local_germinate_from_offspring_immigrant_and_dormancy_pool()
        return counter

#********************************** dormancy process in the patch **********************************
    def patch_dormancy_process_from_offspring_pool_to_dormancy_pool(self):
        survival_counter = 0
        eliminate_counter = 0
        new_dormancy_counter = 0
        all_dormancy_num = 0
        for h_id, h_object in self.set.items():
            survival_num, eliminate_num, new_dormancy_num, dormancy_num = h_object.hab_dormancy_process_from_offspring_pool_to_dormancy_pool()
            survival_counter += survival_num
            eliminate_counter += eliminate_num
            new_dormancy_counter += new_dormancy_num
            all_dormancy_num += dormancy_num
        return survival_counter, eliminate_counter, new_dormancy_counter, all_dormancy_num
    
    def patch_dormancy_process_from_offspring_pool_and_immigrant_pool(self):
        survival_counter = 0
        eliminate_counter = 0
        new_dormancy_counter = 0
        all_dormancy_num = 0
        for h_id, h_object in self.set.items():
            survival_num, eliminate_num, new_dormancy_num, dormancy_num = h_object.hab_dormancy_process_from_offspring_pool_and_immigrant_pool()
            survival_counter += survival_num
            eliminate_counter += eliminate_num
            new_dormancy_counter += new_dormancy_num
            all_dormancy_num += dormancy_num
        return survival_counter, eliminate_counter, new_dormancy_counter, all_dormancy_num    
    
# ******************** clearing up offspring_and_immigrant_pool when dormancy pool do not run *******************************************************************#
    def patch_clear_up_offspring_and_immigrant_pool(self):
        for h_id, h_object in self.set.items():
            h_object.hab_clear_up_offspring_and_immigrant_pool()
    
    def patch_clear_up_offspring_marker_and_immigrant_marker_pool(self):
        for h_id, h_object in self.set.items():   
            h_object.hab_clear_up_offspring_marker_and_immigrant_marker_pool()
    
#********************************** disturbance process in the patch ************************************
    def patch_disturbance_process(self):
        for h_id, h_object in self.set.items():
            h_object.habitat_disturbance_process()
            
#********************************** data generating for saving ******************************************
    def get_patch_microsites_optimum_sp_id_value_array(self, d, w, species_2_phenotype_ls):
        ''''''
        values_array = np.array([], dtype=int)
        for h_id, h_object in self.set.items():
            hab_len = h_object.length
            hab_wid = h_object.width
            mean_env_tuple = h_object.mean_env_ls
            
            hab_opt_survival_rate = 0
            hab_opt_sp_id_val = np.nan
            for phenotype_ls in species_2_phenotype_ls:
                survival_rate = h_object.survival_rate(d=d, phenotype_ls=phenotype_ls, env_val_ls=mean_env_tuple, w=w)
                if survival_rate > hab_opt_survival_rate:
                    hab_opt_survival_rate = survival_rate
                    hab_opt_sp_id_val = species_2_phenotype_ls.index(phenotype_ls)+1

            hab_sp_id_val_array = np.ones(hab_len*hab_wid, dtype=int)*hab_opt_sp_id_val
            values_array = np.append(values_array, hab_sp_id_val_array)
        
        values_array = values_array.reshape(self.hab_num, h_object.size)
        return values_array
