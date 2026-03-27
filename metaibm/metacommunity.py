# -*- coding: utf-8 -*-
"""
Split from metacommunity_IBM.py.
Compatibility-preserving class module.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .patch import patch


class metacommunity():
    def __init__(self, metacommunity_name):
        self.set = {}                       # self.data_set={} # to be improved
        self.patch_num = 0
        #self.meta_map = nx.Graph()
        self.metacommunity_name = metacommunity_name
        self.patch_id_ls = []
        self.patch_id_2_index_dir = {}
        self.pairwise_patch_distance_matrix = np.matrix([])

    def get_data(self):
        output = {}
        for key, value in self.set.items():
            output[key]=value.get_data()
        return output
    
    def __str__(self):
        return str(self.get_data())
    
    def update_disp_current_matrix(self):
        dist_matrix = np.zeros((self.patch_num, self.patch_num), dtype=float)
        for patch_i_id, patch_i_obj in self.set.items():        
            xi, yi = patch_i_obj.location        
            i = patch_i_obj.index
            for patch_j_id, patch_j_obj in self.set.items():            
                xj, yj = patch_j_obj.location            
                j = patch_j_obj.index            
                dist_matrix[i, j] = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        self.pairwise_patch_distance_matrix = np.matrix(dist_matrix)
        return self.pairwise_patch_distance_matrix

    def add_patch(self, patch_name, patch_object):
        ''' add new patch to the metacommunity. '''
        self.set[patch_name] = patch_object
        self.patch_num += 1
        #self.meta_map.add_node(patch_name)
        self.patch_id_ls.append(patch_name)
        self.patch_id_2_index_dir[patch_name] = patch_object.index
        self.update_disp_current_matrix()
        
    def reshape_habitat_data_in_patch(self, df, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc=None):
        ''' reshape habitat_data in the a coorderation order for plotting'''
        reshape_data_col_row = np.empty((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0))
        mask_data_col_row = np.empty((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0))
        
        for x_loc in range(0, hab_num_x_axis_in_patch): #x坐标从原点开始
            reshape_data_col = np.empty((0, hab_x_len))
            mask_data_col = np.empty((0, hab_x_len))
            
            for y_loc in range(0, hab_num_y_axis_in_patch): #y坐标从原点开始
                h_index = x_loc*hab_num_y_axis_in_patch + y_loc
                hab_data = df.loc[h_index].to_numpy().reshape(hab_y_len, hab_x_len)
                mask_data = np.zeros((hab_y_len, hab_x_len))
                
                if mask_loc=='lower': 
                    mask_data[np.tril_indices_from(mask_data)] = True #下三角遮盖
                    
                elif mask_loc=='upper': 
                    mask_data[np.tril_indices_from(mask_data)] = True #上三角遮盖
                    mask_data = mask_data.T
                
                if y_loc == 0:
                    #hab_gap_for_vstack = np.ones((0, hab_x_len)) * np.nan
                    #hab_gap_for_vstack = np.empty((0, hab_x_len))
                    hab_gap_for_vstack = np.zeros((0, hab_x_len))
                    mask_gap_for_vstack = np.ones((0, hab_x_len))
                    
                else:
                    #hab_gap_for_vstack = np.ones((1, hab_x_len)) * np.nan
                    #hab_gap_for_vstack = np.empty((1, hab_x_len))
                    hab_gap_for_vstack = np.zeros((1, hab_x_len))
                    mask_gap_for_vstack = np.ones((1, hab_x_len))
                    
                reshape_data_col = np.vstack((hab_data, hab_gap_for_vstack, reshape_data_col))
                mask_data_col = np.vstack((mask_data, mask_gap_for_vstack, mask_data_col))
                
            if x_loc ==0:
                #hab_gap_for_hstack = np.ones((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0)) * np.nan
                #hab_gap_for_hstack = np.empty((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0)) 
                hab_gap_for_hstack = np.zeros((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0)) 
                mask_gap_for_hstack = np.ones((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0)) 
            else:
                #hab_gap_for_hstack = np.ones((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 1)) * np.nan
                #hab_gap_for_hstack = np.empty((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 1)) 
                hab_gap_for_hstack = np.zeros((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 1))
                mask_gap_for_hstack = np.ones((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 1)) 
                
            reshape_data_col_row = np.hstack((reshape_data_col_row, hab_gap_for_hstack, reshape_data_col))
            mask_data_col_row = np.hstack((mask_data_col_row, mask_gap_for_hstack, mask_data_col))
        return reshape_data_col_row, mask_data_col_row        
    
    def meta_show_species_distribution(self, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, vmin, vmax, cmap, file_name):
        fig = plt.figure(figsize=(40, 40))

        for patch_id, patch_object in self.set.items():
            #location = patch_object.index + 1
            location = sub_col*(sub_row-1)+1+(patch_object.index//sub_row)-(patch_object.index % sub_row)*sub_col
            ax = fig.add_subplot(sub_row, sub_col, location)
            ax.set_title(patch_id, fontsize = 40)
            plt.tight_layout()

            df = pd.DataFrame(patch_object.get_patch_microsites_individals_sp_id_values())
            
            reshape_numpy_data, mask = self.reshape_habitat_data_in_patch(df, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len)
            sns.heatmap(data=reshape_numpy_data, vmin=vmin, vmax=vmax, cbar=False, mask=mask, cmap=cmap, annot=True)
            
        plt.savefig(file_name)
        plt.clf()
        return 0
    
    def meta_show_species_phenotype_distribution(self, trait_name, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, cmap, file_name):
        fig = plt.figure(figsize=(40, 40))

        for patch_id, patch_object in self.set.items():
            #location = patch_object.index + 1
            location = sub_col*(sub_row-1)+1+(patch_object.index//sub_row)-(patch_object.index % sub_row)*sub_col
            ax = fig.add_subplot(sub_row, sub_col, location)
            ax.set_title(patch_id, fontsize = 40)
            plt.tight_layout()

            df = pd.DataFrame(patch_object.get_patch_microsites_individals_phenotype_values(trait_name))
            
            reshape_numpy_data, mask = self.reshape_habitat_data_in_patch(df, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len)
            sns.heatmap(data=reshape_numpy_data, vmin=0, vmax=0.8, cbar=False, mask=mask, cmap=cmap)
            
        plt.savefig(file_name)
        plt.clf()
        return 0
    
    def meta_show_environment_distribution(self, environment_name, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc, cmap, file_name):
        fig = plt.figure(figsize=(40, 40))

        for patch_id, patch_object in self.set.items():
            #location = patch_object.index + 1
            location = sub_col*(sub_row-1)+1+(patch_object.index//sub_row)-(patch_object.index % sub_row)*sub_col
            ax = fig.add_subplot(sub_row, sub_col, location)
            ax.set_title(patch_id, fontsize = 80/((sub_row)/4))
            plt.tight_layout()

            df = pd.DataFrame(patch_object.get_patch_microsites_environment_values(environment_name))
            
            reshape_numpy_data, mask = self.reshape_habitat_data_in_patch(df, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc)
            sns.heatmap(data=reshape_numpy_data, vmin=0, vmax=0.8, cbar=False, mask=mask, cmap=cmap)
            
        plt.savefig(file_name)
        plt.clf()
        return 0
    
    def meta_show_two_environment_distribution(self, environment1_name, environment2_name, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc1, mask_loc2, cmap1, cmap2, file_name):
        fig = plt.figure(figsize=(40, 40))

        for patch_id, patch_object in self.set.items():
            #location = patch_object.index + 1
            location = sub_col*(sub_row-1)+1+(patch_object.index//sub_row)-(patch_object.index % sub_row)*sub_col
            ax = fig.add_subplot(sub_row, sub_col, location)
            ax.set_title(patch_id, fontsize = 80/((sub_row)/4))
            plt.tight_layout()

            df1 = pd.DataFrame(patch_object.get_patch_microsites_environment_values(environment1_name))
            reshape_numpy_data, mask = self.reshape_habitat_data_in_patch(df1, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc1)
            sns.heatmap(data=reshape_numpy_data, vmin=0, vmax=0.8, cbar=False, mask=mask, cmap=cmap1)
            
            df2 = pd.DataFrame(patch_object.get_patch_microsites_environment_values(environment2_name))
            reshape_numpy_data, mask = self.reshape_habitat_data_in_patch(df2, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc2)
            sns.heatmap(data=reshape_numpy_data, vmin=0, vmax=0.8, cbar=False, mask=mask, cmap=cmap2)
            
            
        plt.savefig(file_name)
        plt.clf()
        return 0
    
    def meta_initialize(self, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
        log_info = '%s initialization done! \n'%(self.metacommunity_name)
        #print(log_info)
        return log_info

    def get_meta_individual_num(self):
        num = 0
        for patch_id, patch_object in self.set.items():
            num += patch_object.get_patch_individual_num()
        return num
    
    def get_meta_empty_sites_ls(self):
        ''' return meta_empty_sites_ls as [(patch_id, h_id, len_id, wid_id)] '''   
        meta_empty_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_empty_pos_ls = patch_object.get_patch_empty_sites_ls()
            for empty_pos in patch_empty_pos_ls:
                empty_pos = (patch_id, ) + empty_pos
                meta_empty_sites_ls.append(empty_pos)
        return meta_empty_sites_ls

    def show_meta_empty_sites_num(self):
        return len(self.get_meta_empty_sites_ls())
    
    def get_meta_pairwise_empty_sites_ls(self):
        ''' return meta_empty_sites_ls as [((patch_id, h_id, len_id, wid_id), (patch_id, h_id, len_id, wid_id))...],
        pairwise means the same habitat '''   
        meta_pairwise_empty_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_pairwise_empty_pos_ls = patch_object.get_patch_pairwise_empty_sites_ls()
            for (empty_site_1_pos, empty_site_2_pos) in patch_pairwise_empty_pos_ls:
                empty_site_1_pos = (patch_id, ) + empty_site_1_pos
                empty_site_2_pos = (patch_id, ) + empty_site_2_pos
                meta_pairwise_empty_sites_ls.append((empty_site_1_pos, empty_site_2_pos))
        return meta_pairwise_empty_sites_ls
    
    def meta_get_occupied_location_ls(self):
        ''' return as [(patch_id, h_id, row_id, col_id)] '''
        meta_occupied_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_empty_pos_ls = patch_object.get_patch_occupied_sites_ls()
            for site_pos in patch_empty_pos_ls:
                site_pos = (patch_id, ) + site_pos
                meta_occupied_sites_ls.append(site_pos)
        return meta_occupied_sites_ls
    
    def get_meta_pairwise_occupied_sites_ls(self):
        ''' return meta_occupied_sites_ls as [((patch_id, h_id, len_id, wid_id), (patch_id, h_id, len_id, wid_id))...],
        pairwise means the same species ((female_pos), (male_pos)) '''   
        meta_pairwise_occupied_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_pairwise_occupied_pos_ls = patch_object.get_patch_pairwise_occupied_sites_ls()
            for (occupied_site_1_pos, occupied_site_2_pos) in patch_pairwise_occupied_pos_ls:
                occupied_site_1_pos = (patch_id, ) + occupied_site_1_pos
                occupied_site_2_pos = (patch_id, ) + occupied_site_2_pos
                meta_pairwise_occupied_sites_ls.append((occupied_site_1_pos, occupied_site_2_pos))
        return meta_pairwise_occupied_sites_ls  
    #****************************************************#
    def meta_offspring_pool_individual_num(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                counter += len(h_object.offspring_pool)
        return counter
    
    def meta_immigrant_pool_individual_num(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                counter += len(h_object.immigrant_pool)
        return counter
    
    def meta_dormancy_pool_individual_num(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                counter += len(h_object.dormancy_pool)
        return counter
    #****************************************************#
    def meta_offspring_marker_pool_marker_num(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                counter += len(h_object.offspring_marker_pool)
        return counter
    
    def meta_immigrant_marker_pool_marker_num(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                counter += len(h_object.immigrant_marker_pool)
        return counter 

#******************************** changing environmental values at metacommunity scales ************************** #    
    def meta_offset_environmental_values(self, env_name_ls, delta_mean_ls, delta_var_ls=None):
        ''' multi mean (var) values of env_name in habitats + or - delta_mean_ls (delta_var_ls) across the metacommunity (Input: list) '''
        if delta_var_ls == None: delta_var_ls = [0 for i in range(len(env_name_ls))]
        for patch_id, patch_object in self.set.items():
            patch_object.patch_offset_environmental_values(env_name_ls, delta_mean_ls, delta_var_ls)
            
        log_info = '[changing environment process] in %s: the delta environmental values of %s is mean=%s, var=%s \n'%(self.metacommunity_name, str(env_name_ls), str(delta_mean_ls), str(delta_var_ls))
        return log_info

# ********************************* dead selection in metacommunity ***********************************************
    def meta_dead_selection(self, base_dead_rate, fitness_wid, method):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dead_selection(base_dead_rate, fitness_wid, method)
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Dead selection] in %s: there are %d individuals dead in selection; there are %d individuals in the %s; there are %d empty sites in the %s \n'%(self.metacommunity_name, counter, indi_num, self.metacommunity_name, empty_sites_num, self.metacommunity_name)
        #print(log_info)
        return log_info
    
#************************** for mainland only and do not contain dormant bank **************************************#
    def meta_mainland_asexual_birth_mutate_germinate(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_asexual_birth_germinate(asexual_birth_rate, mutation_rate, pheno_var_ls)
        
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Birth process] in %s: there are %d individuals germinating from local habitat; there are %d individuals in the %s; there are %d empty sites in the %s \n'%(self.metacommunity_name, counter, indi_num, self.metacommunity_name, empty_sites_num, self.metacommunity_name)
        #print(log_info)
        return log_info
    
    def meta_mainland_sexual_birth_mutate_germinate(self, sexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_sexual_birth_germinate(sexual_birth_rate, mutation_rate, pheno_var_ls)
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Birth process] in %s: there are %d individuals germinating from local habitat; there are %d individuals in the %s; there are %d empty sites in the %s \n'%(self.metacommunity_name, counter, indi_num, self.metacommunity_name, empty_sites_num, self.metacommunity_name)
        #print(log_info)
        return log_info
    
    def meta_mainland_mixed_birth_mutate_germinate(self, asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_mixed_birth_germinate(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)
        
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Birth process] in %s: there are %d individuals germinating from local habitat; there are %d individuals in the %s; there are %d empty sites in the %s \n'%(self.metacommunity_name, counter, indi_num, self.metacommunity_name, empty_sites_num, self.metacommunity_name)
        #print(log_info)
        return log_info    
#*******************************************************************************************************************#  
#****************************************colonize_via_propagules_rains from mainland ********************************#  
    def meta_colonize_from_propagules_rains(self, mainland_obj, propagules_rain_num):
        mainland_occupied_sites_ls = mainland_obj.meta_get_occupied_location_ls()
        random.shuffle(mainland_occupied_sites_ls)
        
        int_propagules_rain_num, dem_propagules_rain_num = int(propagules_rain_num), propagules_rain_num - int(propagules_rain_num)
        if dem_propagules_rain_num >= np.random.uniform(0,1,1)[0]:
            int_propagules_rain_num += 1
            propagules_rain_num = int_propagules_rain_num
        else:
            propagules_rain_num = int_propagules_rain_num
        
        propagules_rain_pos_ls = random.sample(mainland_occupied_sites_ls, propagules_rain_num)
        meta_empty_pos_ls = self.get_meta_empty_sites_ls()
        random.shuffle(meta_empty_pos_ls)
        counter = 0
        for propagules_rain_pos, meta_empty_pos in list(zip(propagules_rain_pos_ls, meta_empty_pos_ls)):
            propagules_patch_id, propagules_h_id, propagules_row_id, propagules_col_id =  propagules_rain_pos[0], propagules_rain_pos[1], propagules_rain_pos[2], propagules_rain_pos[3]
            patch_id, h_id, len_id, wid_id = meta_empty_pos[0], meta_empty_pos[1], meta_empty_pos[2], meta_empty_pos[3]
            indi_object = mainland_obj.set[propagules_patch_id].set[propagules_h_id].set['microsite_individuals'][propagules_row_id][propagules_col_id]
            self.set[patch_id].set[h_id].add_individual(indi_object = indi_object, len_id=len_id, wid_id=wid_id)
            #self.set[patch_id].set[h_id].immigrant_pool.append(indi_object)
            counter += 1
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Colonization process]: there are %d individuals colonizing the metacommunity from mainland; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info            
        
    def pairwise_sexual_colonization_from_prpagules_rains(self, mainland_obj, propagules_rain_num):
        mainland_pairwise_occupied_sites_ls = mainland_obj.get_meta_pairwise_occupied_sites_ls()
        random.shuffle(mainland_pairwise_occupied_sites_ls)

        pairwise_propgules_num = propagules_rain_num/2
        int_pairwise_propgules_num, dem_pairwise_propgules_num = int(pairwise_propgules_num), pairwise_propgules_num-int(pairwise_propgules_num)
        if dem_pairwise_propgules_num >= np.random.uniform(0,1,1)[0]:
            int_pairwise_propgules_num += 1
            pairwise_propgules_num = int_pairwise_propgules_num
        else:
            pairwise_propgules_num = int_pairwise_propgules_num
        
        propagules_rain_pairwise_pos_ls = random.sample(mainland_pairwise_occupied_sites_ls, pairwise_propgules_num)
        meta_pairwise_empty_sites_ls = self.get_meta_pairwise_empty_sites_ls()
        random.shuffle(meta_pairwise_empty_sites_ls)
        counter = 0
        for (female_obj_pos, male_obj_pos), (empty_site_1_pos, empty_site_2_pos) in list(zip(propagules_rain_pairwise_pos_ls, meta_pairwise_empty_sites_ls)):
            
            female_propagules_patch_id, female_propagules_h_id, female_propagules_row_id, female_propagules_col_id =  female_obj_pos[0], female_obj_pos[1], female_obj_pos[2], female_obj_pos[3]
            female_obj = mainland_obj.set[female_propagules_patch_id].set[female_propagules_h_id].set['microsite_individuals'][female_propagules_row_id][female_propagules_col_id]
            site_1_patch_id, site_1_h_id, site_1_len_id, site_1_wid_id = empty_site_1_pos[0], empty_site_1_pos[1], empty_site_1_pos[2], empty_site_1_pos[3]
            self.set[site_1_patch_id].set[site_1_h_id].add_individual(indi_object = female_obj, len_id=site_1_len_id, wid_id=site_1_wid_id)
            #self.set[site_1_patch_id].set[site_1_h_id].immigrant_pool.append(female_obj)
            
            male_propagules_patch_id, male_propagules_h_id, male_propagules_row_id, male_propagules_col_id =  male_obj_pos[0], male_obj_pos[1], male_obj_pos[2], male_obj_pos[3]
            male_obj = mainland_obj.set[male_propagules_patch_id].set[male_propagules_h_id].set['microsite_individuals'][male_propagules_row_id][male_propagules_col_id]
            site_2_patch_id, site_2_h_id, site_2_len_id, site_2_wid_id = empty_site_2_pos[0], empty_site_2_pos[1], empty_site_2_pos[2], empty_site_2_pos[3]
            self.set[site_2_patch_id].set[site_2_h_id].add_individual(indi_object = male_obj, len_id=site_2_len_id, wid_id=site_2_wid_id)
            #self.set[site_2_patch_id].set[site_2_h_id].immigrant_pool.append(male_obj)
            
            counter += 2
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Colonization process] there are %d individuals colonizing the metacommunity from mainland; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
#******************** all hab in all patch in metacommunity, reproduce_mutate_process_into_offspring_pool ****************************************************#       
    def meta_asex_reproduce_mutate_into_offspring_pool(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_asex_reproduce_mutate_into_offspring_pool(asexual_birth_rate, mutation_rate, pheno_var_ls)
            
        log_info = '%s: there are %d individuals born into the offspring_pool; there are %d individuals in the offspring_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_pool_individual_num())
        #print(log_info)
        return log_info
        
    def meta_sex_reproduce_mutate_into_offspring_pool(self, sexual_birth_rate, mutation_rate, pheno_var_ls):  
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_sex_reproduce_mutate_into_offspring_pool(sexual_birth_rate, mutation_rate, pheno_var_ls)
        
        log_info = '%s: there are %d individuals born into the offspring_pool; there are %d individuals in the offspring_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_pool_individual_num())
        #print(log_info)
        return log_info
            
    def meta_mix_reproduce_mutate_into_offspring_pool(self, asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls):       
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_mix_reproduce_mutate_into_offspring_pool(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)
        
        log_info = '[Reproduction into offspring_pool] in %s: there are %d individuals born into the offspring_pool; there are %d individuals in the offspring_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_pool_individual_num())
        #print(log_info)
        return log_info
#******** calculating the offspring num but do not reproduce actually until local germination process afterward after dispersal process. *************
    def meta_asex_reproduce_calculation_into_offspring_marker_pool(self, asexual_birth_rate):
        ''' used only when dormancy process do not occur'''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_asex_reproduce_calculation_into_offspring_marker_pool(asexual_birth_rate)
            
        log_info = '[Reproduction into offspring_marker_pool] in %s: there are %d offspring_marker born into the offspring_marker_pool; there are %d offspring_marker in the offspring_marker_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_marker_pool_marker_num())
        #print(log_info)
        return log_info
    
    def meta_sex_reproduce_calculation_with_offspring_marker_pool(self, sexual_birth_rate):
        ''' used only when dormancy process do not occur'''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_sex_reproduce_calculation_into_offspring_marker_pool(sexual_birth_rate)
            
        log_info = '[Reproduction into offspring_marker_pool] in %s: there are %d offspring_marker born into the offspring_marker_pool; there are %d offspring_marker in the offspring_marker_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_marker_pool_marker_num())
        #print(log_info)
        return log_info
    
    def meta_mix_reproduce_calculation_with_offspring_marker_pool(self, asexual_birth_rate, sexual_birth_rate):
        ''' used only when dormancy process do not occur'''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_mix_reproduce_calculation_into_offspring_marker_pool(asexual_birth_rate, sexual_birth_rate)
            
        log_info = '[Reproduction into offspring_marker_pool] in %s: there are %d offspring_marker born into the offspring_marker_pool; there are %d offspring_marker in the offspring_marker_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_marker_pool_marker_num())
        #print(log_info)
        return log_info
    
# ****************************** disperal among patches process *************************************************************************
    def matrix_around(self, matrix):   
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

    def dispersal_kernel_strength(self, d_ij, method='uniform', **kwargs):
        ''' Input: d_ij (Euclidean distance between patch_i and patch_j); method == 'uniform', 'guassian', 'exponential', 'cauchy', or 'power_law'
        Method-specific parameters: **kwargs = {'sigma': sigma, 'rho': rho, 'gamma': gamma, 'alpha': alpha, 'r0': r0}
            no parameter for 'uniform'; sigma for 'gaussian'; rho for 'exponential'; gamma for 'cauchy'; alpha & r0 for 'power_law' 
        Return: the relative strength of dispersal between patch_i to patch_j (i!=j) '''
        if method == 'uniform': 
            D_ij = 1.0

        elif method == 'gaussian':        # short-tail
            sigma = kwargs.get('sigma')
            D_ij = np.exp(-(d_ij ** 2) / (2.0 * sigma ** 2))

        elif method == 'exponential':     # short-tail
            rho = kwargs.get('rho')
            D_ij = rho * np.exp(-rho * d_ij)

        elif method == 'cauchy':          # long-tail
            gamma = kwargs.get('gamma')
            D_ij =  1.0 / (1.0 + (d_ij / gamma) ** 2)

        elif method == 'power_law':       # long-tail
            alpha = kwargs.get('alpha')        
            r0 = kwargs.get('r0')
            D_ij =  (1.0 + d_ij / r0) ** (-alpha)

        else: raise ValueError(f"Unsupported dispersal kernel method: {method}")
        return D_ij
    
    def calculate_dispersal_kernel_strength_matrix(self, method='uniform', **kwargs):
        ''' Return: the unnormalized dispersal kernel strength matrix D = [D_ij]'''
        if self.patch_num == 1: return np.matrix([[0.0]])

        D = np.matrix(np.zeros((self.patch_num, self.patch_num), dtype=float))

        for i in range(self.patch_num):
            for j in range(self.patch_num):
                if i==j: D[i, j] = 0
                else:
                    d_ij = self.pairwise_patch_distance_matrix[i, j]
                    D[i, j] = self.dispersal_kernel_strength(d_ij, method=method, **kwargs)
        return D
    
    def normalized_calculate_dispersal_kernel_strength_matrix(self, method='uniform', **kwargs):
        D = self.calculate_dispersal_kernel_strength_matrix(method=method, **kwargs)
        normalized_D = D/D.sum(axis=1)
        return normalized_D

    def get_disp_among_rate_matrix(self, total_disp_among_rate, method='uniform', **kwargs):
        ''' the element p_i_j is the propability that emigrants would disperse from patch_i to patch_j '''
        normalized_D = self.normalized_calculate_dispersal_kernel_strength_matrix(method=method, **kwargs)
        disp_amomg_matrix = total_disp_among_rate * normalized_D
        disp_amomg_matrix[np.diag_indices_from(disp_amomg_matrix)] = 1- total_disp_among_rate  # elements in diagonal_line = 1-m
        return disp_amomg_matrix
    
    def get_offspring_marker_pool_num_matrix(self):
        ''' the element o_i_i is the offspring_marker_nums in the offspring_marker_pool of patch_i '''
        offspring_marker_pool_num_matrix = np.matrix(np.zeros((self.patch_num, self.patch_num)))
        for patch_id, patch_object in self.set.items():
            patch_offspring_marker_num = 0
            for h_id, h_object in patch_object.set.items():
                patch_offspring_marker_num += len(h_object.offspring_marker_pool)
            offspring_marker_pool_num_matrix[patch_object.index, patch_object.index] = patch_offspring_marker_num
        return offspring_marker_pool_num_matrix

    def get_offspring_pool_num_matrix(self):
        ''' the element o_i_i is the offspring_muns in the doffspring_pool in patch_i '''
        offspring_pool_num_matrix = np.matrix(np.zeros((self.patch_num, self.patch_num)))
        for patch_id, patch_object in self.set.items():
            patch_offspring_num = 0
            for h_id, h_object in patch_object.set.items():
                patch_offspring_num += len(h_object.offspring_pool)
            offspring_pool_num_matrix[patch_object.index, patch_object.index] = patch_offspring_num
        return offspring_pool_num_matrix
            
    def get_dormance_pool_num_matrix(self):
        ''' the element d_i_i is the dormancy_num in the dormancy_pool in patch_i '''
        dormancy_pool_num_matrix = np.matrix(np.zeros((self.patch_num, self.patch_num)))
        for patch_id, patch_object in self.set.items():
            patch_dormancy_pool_num = 0
            for h_id, h_object in patch_object.set.items():
                patch_dormancy_pool_num += len(h_object.dormancy_pool)
            dormancy_pool_num_matrix[patch_object.index, patch_object.index] = patch_dormancy_pool_num
        return dormancy_pool_num_matrix
    
    def get_patch_empty_sites_num_matrix(self):
        patch_empty_sites_num_matrix = np.matrix(np.zeros((self.patch_num, self.patch_num)))
        for patch_id, patch_object in self.set.items():
            patch_empty_sites_num = patch_object.patch_empty_sites_num()
            patch_empty_sites_num_matrix[patch_object.index, patch_object.index] = patch_empty_sites_num
        return patch_empty_sites_num_matrix
    
    def get_emigrants_matrix(self, total_disp_among_rate, method='uniform', **kwargs):
        ''' the element em_i_j is the expectation of emigrants_num disperse from patch_i to patch_j '''
        return (self.get_offspring_pool_num_matrix() + self.get_dormance_pool_num_matrix()) * self.get_disp_among_rate_matrix(total_disp_among_rate, method=method, **kwargs)

    def get_immigrants_matrix(self, total_disp_among_rate, method='uniform', **kwargs):
        ''' Return: a matrix about how to alloacte empty microsites to each patch (including itself). '''
        emigrants_matrix = self.get_emigrants_matrix(total_disp_among_rate, method=method, **kwargs)
        patch_empty_sites_num_matrix = self.get_patch_empty_sites_num_matrix()
        return (emigrants_matrix/emigrants_matrix.sum(axis=0))*patch_empty_sites_num_matrix
    
    def get_dispersal_among_num_matrix(self, total_disp_among_rate, method='uniform', **kwargs):
        immigrants_matrix = self.get_immigrants_matrix(total_disp_among_rate, method=method, **kwargs)
        emigrants_matrix = self.get_emigrants_matrix(total_disp_among_rate, method=method, **kwargs)
        return self.matrix_around(np.minimum(emigrants_matrix, immigrants_matrix))

    def dispersal_among_patches_from_offspring_pool_and_dormancy_pool(self, total_disp_among_rate, method='uniform', **kwargs):
        ''' dispersal from patch_i to patch_j '''
        if self.patch_num < 2:
            log_info = '[Dispersal among patches] in %s: patch_num < 2, there are 0 individuals disperse among patches \n'
            #print(log_info)
            return log_info
        dispersal_among_num_matrix = self.get_dispersal_among_num_matrix(total_disp_among_rate, method=method, **kwargs)
        counter = 0
        for j in range(self.patch_num):
            patch_j_id = self.patch_id_ls[j]
            patch_j_object = self.set[patch_j_id]
            patch_j_empty_site_ls = patch_j_object.get_patch_empty_sites_ls()
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.patch_num):
                patch_i_id = self.patch_id_ls[i]
                patch_i_object = self.set[patch_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(dispersal_among_num_matrix[i, j])
                    patch_i_offspring_and_dormancy_pool = patch_i_object.get_patch_offspring_and_dormancy_pool()
                    migrants_to_j_indi_object_ls += random.sample(patch_i_offspring_and_dormancy_pool, migrants_i_j_num)
                    
            random.shuffle(patch_j_empty_site_ls)
            random.shuffle(migrants_to_j_indi_object_ls)
            for (h_id, len_id, wid_id), migrants_object in list(zip(patch_j_empty_site_ls, migrants_to_j_indi_object_ls)):
                self.set[patch_j_id].set[h_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id)
                counter += 1
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals disperse among patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def dispersal_aomng_patches_from_offspring_pool_to_immigrant_pool(self, total_disp_among_rate, method='uniform', **kwargs):
        ''' exchange between offspring pools among patches into immigrant pool'''
        if self.patch_num < 2:
            log_info = '[Dispersal among patches] in %s: patch_num < 2, there are 0 individuals disperse into habs_immigrant_pool among patches \n'
            #print(log_info)
            return log_info
        offspring_dispersal_matrix = self.matrix_around(self.get_offspring_pool_num_matrix()*self.get_disp_among_rate_matrix(total_disp_among_rate, method=method, **kwargs))
        counter = 0
        for j in range(self.patch_num):
            patch_j_id = self.patch_id_ls[j]
            patch_j_object = self.set[patch_j_id]
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.patch_num):
                patch_i_id = self.patch_id_ls[i]
                patch_i_object = self.set[patch_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(offspring_dispersal_matrix[i, j])
                    patch_i_offspring_pool = patch_i_object.get_patch_offspring_pool()
                    migrants_to_j_indi_object_ls += random.sample(patch_i_offspring_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_indi_object_ls)
            patch_j_object.migrants_to_patch_into_habs_immigrant_pool(migrants_to_j_indi_object_ls)
            counter += len(migrants_to_j_indi_object_ls)
        log_info = '[Dispersal among patches] in %s: there are %d individuals disperse into habs_immigrant_pool among patches; there are %d individuals in the immigrant pools in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_pool_individual_num())    
        #print(log_info)
        return log_info
    
    def dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool(self, total_disp_among_rate, method='uniform', **kwargs):
        ''' exchange between offspring marker pools among patches into immigrant marker pool '''
        if self.patch_num < 2:
            log_info = '[Dispersal among patches] in %s: patch_num < 2, there are 0 individuals disperse into habs_immigrant_marker_pool among patches \n'
            #print(log_info)
            return log_info
        
        offspring_marker_dispersal_matrix = self.matrix_around(self.get_offspring_marker_pool_num_matrix()*self.get_disp_among_rate_matrix(total_disp_among_rate, method=method, **kwargs))
        counter = 0
        for j in range(self.patch_num):
            patch_j_id = self.patch_id_ls[j]
            patch_j_object = self.set[patch_j_id]
            migrants_to_j_offspring_marker_ls = []
            
            for i in range(self.patch_num):
                patch_i_id = self.patch_id_ls[i]
                patch_i_object = self.set[patch_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(offspring_marker_dispersal_matrix[i, j])
                    patch_i_offspring_marker_pool = patch_i_object.get_patch_offspring_marker_pool()
                    migrants_to_j_offspring_marker_ls += random.sample(patch_i_offspring_marker_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_offspring_marker_ls)
            patch_j_object.migrants_marker_to_patch_into_habs_immigrant_marker_pool(migrants_to_j_offspring_marker_ls)
            counter += len(migrants_to_j_offspring_marker_ls)
        log_info = '[Dispersal among patches] in %s: there are %d individuals disperse into habs_immigrant_marker_pool among patches; there are %d individuals in the immigrant_marker_pools in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_marker_pool_marker_num())     
        #print(log_info)
        return log_info

    def dispersal_among_patches_from_offsrping_pool_and_dormancy_pool_to_immigrant_pool(self, total_disp_among_rate, method='uniform', **kwargs):
        ''' exchange between offspring pools and dormancy among patches into immigrant pool'''
        if self.patch_num < 2:
            log_info = '[Dispersal among patches] in %s: patch_num < 2, there are 0 individuals disperse into habs_immigrant_pool among patches \n'
            #print(log_info)
            return log_info
        offspring_dormancy_dispersal_matrix = self.matrix_around(self.get_emigrants_matrix(total_disp_among_rate, method=method, **kwargs))
        counter = 0
        for j in range(self.patch_num):
            patch_j_id = self.patch_id_ls[j]
            patch_j_object = self.set[patch_j_id]
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.patch_num):
                patch_i_id = self.patch_id_ls[i]
                patch_i_object = self.set[patch_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(offspring_dormancy_dispersal_matrix[i, j])
                    patch_i_offspring_and_dormancy_pool = patch_i_object.get_patch_offspring_and_dormancy_pool()
                    migrants_to_j_indi_object_ls += random.sample(patch_i_offspring_and_dormancy_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_indi_object_ls)
            patch_j_object.migrants_to_patch_into_habs_immigrant_pool(migrants_to_j_indi_object_ls)
            counter += len(migrants_to_j_indi_object_ls)
        log_info = '%s: there are %d individuals disperse into habs_immigrant_pool among patches; there are %d individuals in the immigrant pools in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_pool_individual_num())
        #print(log_info)
        return log_info    
            
#******************************************* dispersal within patch process ************************************************
    def meta_dispersal_within_patch_from_offspring_marker_to_immigrant_marker_pool(self, disp_within_rate):
        ''' random dispersal within patch, offspring marker is the denotion of an dispering offspring to be born '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dispersal_within_from_offspring_marker_pool_to_immigrant_marker_pool(disp_within_rate)
        log_info = '[Dispersal within process] in %s: there are %d offspring marker disperse into habs_immigrant_marker_pool within patches; there are %d offspring marker in the immigrant marker pool in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_marker_pool_marker_num())
        #print(log_info)
        return log_info

    def meta_dispersal_within_patch_from_offspring_to_immigrant_pool(self, disp_within_rate):
        ''' random dispersal within patch '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dispersal_within_from_offspring_pool_to_immigrant_pool(disp_within_rate)
        log_info = '[Dispersal within process] %s: there are %d individuals disperse into habs_immigrant_pool within patches; there are %d individuals in the immigrant pools in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_pool_individual_num())
        #print(log_info)
        return log_info
    
    def meta_dispersal_within_patch_from_offspring_and_dormancy_to_immigrant_pool(self, disp_within_rate):
        ''' random dispersal within patch '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dispersal_within_from_offspring_pool_and_dormancy_pool_to_immigrant_pool(disp_within_rate)
        log_info = '[Dispersal within process] %s: there are %d individuals disperse into habs_immigrant_pool within patches; there are %d individuals in the immigrant pools in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_pool_individual_num())
        #print(log_info)
        return log_info        

    def meta_dispersal_within_patch_from_offspring_and_dormancy_pool(self, disp_within_rate):
        ''' random dispersal within patch to empty sites directly'''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dipersal_within_from_offspring_and_dormancy_pool(disp_within_rate)
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '%s: there are %d individuals disperse within patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n '%(self.metacommunity_name, counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
#************************************************ local germination *******************************************************************
    def meta_local_germinate_from_offspring_and_dormancy_pool(self):
        ''' germination individual randomly chosen from local habitst offspring pool + dormancy_pool '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_local_germinate_from_offspring_and_dormancy_pool()
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '%s: there are %d individuals germinating from local offspring_pool and dormancy_pool in the local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(self.metacommunity_name, counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def meta_local_germinate_from_offspring_and_immigrant_pool(self):
        ''' germination individual randomly chosen from local habitst offspring pool + immigrant_pool '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_local_germinate_from_offspring_and_immigrant_pool()
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Germination process] in %s: there are %d individuals germinating from local offspring_pool and immigrant_pool in the local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(self.metacommunity_name, counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def meta_local_germinate_from_offspring_immigrant_and_dormancy_pool(self):
        ''' germination individual randomly chosen from local habitst offspring pool + immigrant_pool + dormancy_pool '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_local_germinate_from_offspring_immigrant_and_dormancy_pool()
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Germination process] %s: there are %d individuals germinating from local offspring_pool and dormancy_pool and immigrant_pool in the local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(self.metacommunity_name, counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool(self, mutation_rate, pheno_var_ls):
        ''' germination individual marker randomly chosen from local habitst offspring pool + immigrant_pool and then birth process according to the marker information '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                hab_empty_pos_ls = h_object.empty_site_pos_ls
                hab_offspring_marker_and_immigrant_marker_pool = h_object.offspring_marker_pool + h_object.immigrant_marker_pool
                
                random.shuffle(hab_empty_pos_ls)
                random.shuffle(hab_offspring_marker_and_immigrant_marker_pool)
                
                for (row_id, col_id), indi_marker in list(zip(hab_empty_pos_ls, hab_offspring_marker_and_immigrant_marker_pool)):
                    birth_patch_id, birth_h_id, reproduce_mode = indi_marker[0], indi_marker[1], indi_marker[2]
                    birth_h_object = self.set[birth_patch_id].set[birth_h_id]  # birth place h_object
                    
                    if reproduce_mode == 'asexual':
                        indi_object = birth_h_object.hab_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)[0]
                    elif reproduce_mode == 'sexual':
                        indi_object = birth_h_object.hab_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)[0]
                    elif reproduce_mode == 'mix_asexual':
                        indi_object = birth_h_object.hab_mix_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)[0]
                    elif reproduce_mode == 'mix_sexual':
                        indi_object = birth_h_object.hab_mix_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)[0]
                    
                    h_object.add_individual(indi_object=indi_object, len_id=row_id, wid_id=col_id)
                    counter += 1
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Germination & Birth process] %s: there are %d individuals germinating_and_birth from local offspring_marker_pool and immigrant_marker_pool in the local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(self.metacommunity_name, counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
        
#************************************************* dormancy process in the metacommunity *********************************************************************
    def meta_dormancy_process_from_offspring_pool_to_dormancy_pool(self):
        survival_counter = 0
        eliminate_counter = 0
        new_dormancy_counter = 0
        all_dormancy_num = 0
        for patch_id, patch_object in self.set.items():
            survival_num, eliminate_num, new_dormancy_num, dormancy_num = patch_object.patch_dormancy_process_from_offspring_pool_to_dormancy_pool()
            survival_counter += survival_num
            eliminate_counter += eliminate_num
            new_dormancy_counter += new_dormancy_num
            all_dormancy_num += dormancy_num
            
        log_info_1 = '%s: there are %d dormancy survived in the dormancy pool; there are %d dormancy eliminated from the dormancy pool; there are %d new dormancy into the dormancy pool; there are totally %d dormancy in the dormancy pool across the metacommunity \n'%(self.metacommunity_name, survival_counter, eliminate_counter, new_dormancy_counter, all_dormancy_num)
        log_info_2 = '%s: there are %d individuals in the offspring_pool; there are %d individuals in the immigrant_pool; there are %d individuals in the dormancy_pool \n'%(self.metacommunity_name, self.meta_offspring_pool_individual_num(), self.meta_immigrant_pool_individual_num(), self.meta_dormancy_pool_individual_num())
        #print(log_info_1)
        #print(log_info_2)
        return log_info_1 + log_info_2
    
    def meta_dormancy_process_from_offspring_pool_and_immigrant_pool(self):
        survival_counter = 0
        eliminate_counter = 0
        new_dormancy_counter = 0
        all_dormancy_num = 0
        for patch_id, patch_object in self.set.items():
            survival_num, eliminate_num, new_dormancy_num, dormancy_num = patch_object.patch_dormancy_process_from_offspring_pool_and_immigrant_pool()
            survival_counter += survival_num
            eliminate_counter += eliminate_num
            new_dormancy_counter += new_dormancy_num
            all_dormancy_num += dormancy_num
            
        log_info_1 = '[Dormancy process] in %s: there are %d dormancy survived in the dormancy pool; there are %d dormancy eliminated from the dormancy pool; there are %d new dormancy into the dormancy pool; there are totally %d dormancy in the dormancy pool across the metacommunity \n'%(self.metacommunity_name, survival_counter, eliminate_counter, new_dormancy_counter, all_dormancy_num)
        log_info_2 = '%s: there are %d individuals in the offspring_pool; there are %d individuals in the immigrant_pool; there are %d individuals in the dormancy_pool \n'%(self.metacommunity_name, self.meta_offspring_pool_individual_num(), self.meta_immigrant_pool_individual_num(), self.meta_dormancy_pool_individual_num())
        #print(log_info_1)
        #print(log_info_2)
        return log_info_1 + log_info_2
    
# ******************** clearing up offspring_and_immigrant_pool when dormancy pool do not run *******************************************************************#
    def meta_clear_up_offspring_and_immigrant_pool(self):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_clear_up_offspring_and_immigrant_pool()
    
    def meta_clear_up_offspring_marker_and_immigrant_marker_pool(self):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_clear_up_offspring_marker_and_immigrant_marker_pool()
    
#************************************************** disturbance process in the metacommunity **************************************************************************
    def meta_disturbance_process_in_patches(self, patch_dist_rate):
        patch_dist_occur_ls = []
        for patch_id, patch_object in self.set.items():
            if np.random.uniform(0,1,1)[0] < patch_dist_rate:
                patch_object.patch_disturbance_process()
                patch_dist_occur_ls.append(patch_id)
            else:
                continue
        log_info = f'[Disturbance process] occurred in {patch_dist_occur_ls} \n'
        #print(log_info)
        return log_info
 
    def meta_disturbance_process_in_habitat(self, hab_dist_rate):
        habitat_dist_occur_ls = []
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                if np.random.uniform(0,1,1)[0] < hab_dist_rate:
                    h_object.habitat_disturbance_process()
                    habitat_dist_occur_ls.append(patch_id+'_'+h_id)
                else:
                    continue
        log_info = f'[Disturbance process] occurred in {habitat_dist_occur_ls} \n'
        #print(log_info)
        return log_info
                
#************************************************ data generating and saving module *********************************************************************
    def get_meta_microsites_optimum_sp_id_val(self, d, w, species_2_phenotype_ls):
        ''' '''
        meta_optimum_sp_id_val_dis = np.array([])
        for patch_id, patch_object in self.set.items():
            patch_optimum_sp_id_val_dis = patch_object.get_patch_microsites_optimum_sp_id_value_array(d, w, species_2_phenotype_ls)
            meta_optimum_sp_id_val_dis = np.append(meta_optimum_sp_id_val_dis, patch_optimum_sp_id_val_dis.reshape(-1))
        return meta_optimum_sp_id_val_dis.reshape(1,-1)
    
    def get_meta_microsite_environment_values(self, environment_name, digits=3):
        ''''''
        meta_environment_dis = np.array([])
        for patch_id, patch_object in self.set.items():
            patch_environment_dis = patch_object.get_patch_microsites_environment_values(environment_name)
            meta_environment_dis = np.append(meta_environment_dis, patch_environment_dis.reshape(-1))
            meta_environment_dis = np.around(meta_environment_dis, digits) #保留几位小数
        return meta_environment_dis.reshape(1,-1)

    def get_meta_microsites_individuals_sp_id_values(self):
        ''' '''
        meta_sp_dis = np.array([], dtype=int)
        for patch_id, patch_object in self.set.items():
            patch_sp_dis = patch_object.get_patch_microsites_individals_sp_id_values()
            meta_sp_dis = np.append(meta_sp_dis, patch_sp_dis.reshape(-1))
        return meta_sp_dis.reshape(1,-1)
    
    def get_meta_microsites_individuals_phenotype_values(self, trait_name, digits=3):
        ''''''
        meta_phenotype_dis = np.array([])
        for patch_id, patch_object in self.set.items():
            patch_phenotype_dis = patch_object.get_patch_microsites_individals_phenotype_values(trait_name)
            meta_phenotype_dis = np.append(meta_phenotype_dis, patch_phenotype_dis.reshape(-1))
        meta_phenotype_dis = np.around(meta_phenotype_dis, digits) # #保留几位小数 to save the storage
        return meta_phenotype_dis.reshape(1,-1)    

    def columns_patch_habitat_microsites_id(self):
        ''' return 3 lists of patch_id, h_id, microsite_id as the header of meta_sp_dis table '''
        columns_patch_id = np.array([])
        columns_habitat_id = np.array([])
        columns_mocrosite_id = np.array([])
        
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                h_len, h_wid = h_object.length, h_object.width
                columns_patch_id = np.append(columns_patch_id, np.array([patch_id for _ in range(h_len*h_wid)]))
                columns_habitat_id = np.append(columns_habitat_id, np.array([h_id for _ in range(h_len*h_wid)]))
                columns_mocrosite_id = np.append(columns_mocrosite_id, np.array(['r%d, c%d'%(i, j) for i in range(h_len) for j in range(h_wid)]))
        return columns_patch_id, columns_habitat_id, columns_mocrosite_id

    def meta_distribution_data_all_time_to_csv_gz(self, dis_data_all_time, file_name, index, columns, mode='w'):
        ''' '''
        df_species_distribution = pd.DataFrame(dis_data_all_time, index=index, columns=columns)
        if mode=='w': 
            df_species_distribution.to_csv(file_name, mode=mode, compression='gzip')
        elif mode=='a': 
            df_species_distribution.to_csv(file_name, mode=mode, compression='gzip', header=False)
        return df_species_distribution
