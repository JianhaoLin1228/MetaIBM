#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:14:32 2026

@author: jianhaolin
"""

import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from .patch import patch
from .metacommunity import metacommunity

class simulator():
    def __init__(self):
        self.meta_objects = {}            # keys is meta_obj.name; values is meta_obj
        self.global_params = {}           # non-eco-evo global params: keys is params_name; values is params
        self.schedule_one_time_step = []  # list of {} items (the info. of a eco-evo process), executed in orders

        ''' runtime states '''
        self.time_step = 0                # current time_steps
        self.logger_file = None           # object = Open(file_name)
        self.current_step_log = ""        # text of logs
        self.start_time = None            # wall-clock start of sim.run() (set when is_timing=True)
        self.end_time = None              # wall-clock end of sim.run() (set when is_timing=True)

    def add_metacommunity_obj(self, name, obj):
        ''' Register a metacommunity-like object under a user-provided name. '''
        self.meta_objects[name] = obj

    def build_empty_mainland_from_species_csv(self, meta_name, mainland_csv, pheno_names_ls, environment_types_name, environment_variation_ls, patch_name="patch0", patch_index=0, patch_location=(0, 0), dormancy_pool_max_size=0):
        ''' Schedule-callable: build an empty mainland metacommunity from a species CSV. One row represents one species and one mainland habitat. Habitat location is read from hab_x_loc/hab_y_loc, habitat size from hab_length/hab_width, and phenotype/environment mean values from the columns specified by pheno_names_ls. Registers the metacommunity on the simulator and returns log_info. '''
        log_info = 'generating empty mainland from species csv ... \n'

        df = pd.read_csv(mainland_csv)
        df = df.sort_values('species_id').reset_index(drop=True)

        meta_object = metacommunity(metacommunity_name=meta_name)
        patch_object = patch(patch_name, patch_index, patch_location)

        for j, row in df.iterrows():
            species_id = int(row['species_id'])
            hab_index = j
            habitat_name = 'h%d' % hab_index

            hab_x_loc = int(row['hab_x_loc'])
            hab_y_loc = int(row['hab_y_loc'])
            hab_location = (hab_x_loc, hab_y_loc)

            mean_env_ls = [float(row[pheno_name]) for pheno_name in pheno_names_ls]
            hab_length = int(row['hab_length'])
            hab_width = int(row['hab_width'])

            patch_object.add_habitat(hab_name=habitat_name, hab_index=hab_index, hab_location=hab_location, num_env_types=len(environment_types_name), env_types_name=environment_types_name, mean_env_ls=mean_env_ls, var_env_ls=environment_variation_ls, length=hab_length, width=hab_width, dormancy_pool_max_size=dormancy_pool_max_size)

            info = '%s: %s, %s, %s, %s: species_id=%s' % (meta_object.metacommunity_name, patch_name, str(patch_location), habitat_name, str(hab_location), str(species_id))
            for pheno_name, pheno_val in zip(pheno_names_ls, mean_env_ls):
                info += ', %s=%s' % (pheno_name, str(pheno_val))
            info += ', hab_length=%s, hab_width=%s' % (str(hab_length), str(hab_width))
            log_info += info + '\n'

        meta_object.add_patch(patch_name=patch_name, patch_object=patch_object)
        self.add_metacommunity_obj(meta_object.metacommunity_name, meta_object)
        return log_info

    def build_empty_metacommunity_from_patch_habitat_csv(self, meta_name, metacommunity_csv, environment_types_name, environment_variation_ls, dormancy_pool_max_size=0):
        ''' Schedule-callable: build an empty metacommunity (islands) from a patch-habitat CSV. One row represents one habitat; patches are inferred from unique (patch_id, patch_index, patch_location_x, patch_location_y) rows. Registers the metacommunity on the simulator and returns log_info. '''
        log_info = 'generating empty metacommunity from patch-habitat csv ... \n'

        df = pd.read_csv(metacommunity_csv)

        meta_object = metacommunity(metacommunity_name=meta_name)
        num_env_types = len(environment_types_name)

        df_patch_register_info = df[['patch_id', 'patch_index', 'patch_location_x', 'patch_location_y']].drop_duplicates().sort_values(['patch_index', 'patch_id'])
        for _, patch_reg_info in df_patch_register_info.iterrows():
            patch_name = patch_reg_info['patch_id']
            patch_index = int(patch_reg_info['patch_index'])
            patch_location = (float(patch_reg_info['patch_location_x']), float(patch_reg_info['patch_location_y']))
            patch_object = patch(patch_name, patch_index, patch_location)

            df_hab_register_infos = df[df['patch_id'] == patch_name].copy().sort_values(['habitat_index', 'habitat_id'])
            for _, hab_reg_info in df_hab_register_infos.iterrows():
                hab_name = str(hab_reg_info['habitat_id'])
                hab_index = int(hab_reg_info['habitat_index'])
                hab_location = (int(hab_reg_info['habitat_x_location']), int(hab_reg_info['habitat_y_location']))
                hab_length = int(hab_reg_info['hab_length'])
                hab_width = int(hab_reg_info['hab_width'])
                mean_env_ls = [float(hab_reg_info[env_name]) for env_name in environment_types_name]

                patch_object.add_habitat(hab_name=hab_name, hab_index=hab_index, hab_location=hab_location, num_env_types=num_env_types, env_types_name=environment_types_name, mean_env_ls=mean_env_ls, var_env_ls=environment_variation_ls, length=hab_length, width=hab_width, dormancy_pool_max_size=dormancy_pool_max_size)

                env_info = ', '.join('%s=%s' % (environment_types_name[i], str(mean_env_ls[i])) for i in range(num_env_types))
                info = '%s: %s, %s, %s, %s, hab_length=%d, hab_width=%d: %s' % (meta_object.metacommunity_name, patch_name, str(patch_location), hab_name, str(hab_location), hab_length, hab_width, env_info)
                log_info = log_info + info + '\n'

            meta_object.add_patch(patch_name=patch_name, patch_object=patch_object)

        self.add_metacommunity_obj(meta_object.metacommunity_name, meta_object)
        return log_info

    def set_global_params(self, global_params):
        ''' Set runtime parameters used by the Simulator itself. '''
        self.global_params = global_params

    def _mkdir_if_not_exist(self, goal_path):
        ''' Build a folder hierarchy by goal_path '''
        os.makedirs(goal_path, exist_ok=True)
        
    def set_goal_path(self, root_path, *multi_layer_folder):
        ''' set root_path and goal_path as global params; mkdir of goal_path '''
        goal_path = os.path.join(root_path, *multi_layer_folder)
        self.global_params['root_path'] = root_path
        self.global_params['goal_path'] = goal_path
        os.makedirs(goal_path, exist_ok=True)
        return goal_path

    def set_schedule_per_time_step(self, schedule_one_time_step):
        ''' Set the ordered list of schedule items (eco-evo processes) executed each time step. '''
        self.schedule_one_time_step = schedule_one_time_step

    def add_schedule_item_per_time_step(self, item):
        ''' Append one schedule item to self.schedule_one_time_step.'''
        self.schedule_one_time_step.append(item)

################## log module ###################
    def open_logger(self, mode='a'):
        ''' Open the logger file if is_logging==True; mode='a' (default) '''
        if self.global_params['is_logging'] == True:
            goal_path = self.global_params['goal_path']
            logger_file_name = os.path.join(self.global_params['goal_path'], 'logger.log')
            self.logger_file = open(logger_file_name, mode)

    def close_logger(self):
        ''' Close self.logger_file if it was opened. '''
        if self.logger_file is not None:
            self.logger_file.close()

    def append_step_log(self, log_info):
        ''' Append a returned log string to self.current_step_log. '''
        self.current_step_log += log_info

    def write_logger(self, log_info):
        ''' Write log_info if is_logging==True; otherwise print to stdout. '''
        if self.global_params["is_logging"] == True: print(log_info, file=self.logger_file)
        elif self.global_params["is_logging"] == False: print(log_info)

    def flush_step_log(self):
        ''' Schedule-callable: write the accumulated self.current_step_log for this time step. '''
        self.write_logger(self.current_step_log)

############## main loop of eco-evo simulations #####################
    def run(self):
        ''' Run the full simulation and return the finalize() dictionary.
        Times only the schedule loop when global_params["is_timing"]==True (default True). '''
        self.open_logger()

        if self.global_params.get("is_timing", True) == True:
            self.start_time = time.time()

        all_time_steps = self.global_params["all_time_steps"]
        for time_step in range(all_time_steps):
            self.run_one_time_step(time_step)

        if self.global_params.get("is_timing", True) == True:
            self.end_time = time.time()
            final_log = "total simulation time: %.8s s \n" % (self.end_time - self.start_time)
            self.write_logger(final_log)

        self.close_logger()
        return self.finalize()

    def run_one_time_step(self, time_step):
        ''' Run all schedule items (eco-evo processes) for a single time step, in order. '''
        self.time_step = time_step
        self.current_step_log = "time_step=%d \n" % time_step
        for item in self.schedule_one_time_step:
            if self.should_run(item, time_step):
                self.run_one_schedule_item(item)

    def should_run(self, item, time_step):
        ''' Decide whether a schedule item (eco-evo process) should run at the current step.
        Supported optional item fields: {'enabled': True, 'start': 0, 'end': None, interval: 1} (by default) '''
        
        enabled = item.get("enabled", True)
        start = item.get("start", 0)
        end = item.get("end", None)                                   ### all_time_step 还是 None， 有待统一标准
        interval = item.get("interval", 1)                            ### 是否允许用户输入None，有待统一标准

        if not enabled: return False
        if time_step < start: return False
        if end is not None and time_step > end: return False          ### all_time_step 还是 None 需要统一标准
        if interval is None: return True                              ### 是否允许用户输入None，有待统一标准

        is_run = (time_step - start) % interval == 0
        return is_run

    def run_one_schedule_item(self, item):
        ''' Execute one schedule item (one eco-evo process).
        The returned log info. of (eco-evo process) is appended to self.current_step_log.
        Simulator-level methods read self.time_step directly when they need the current step. '''

        target = item["target"]
        method_name = item["method"]
        params = self._resolve_params(item.get("params", {}))

        if target == 'simulator':                       # dispatch simulator-level special methods
            method = getattr(self, method_name)
            result = method(**params)
        else:                                           # dispatch metacommunity-onject-like-level methods
            obj = self.meta_objects[target]
            method = getattr(obj, method_name)
            result = method(**params)

        if isinstance(result, str): self.append_step_log(result)

#######################  Parameters resolution helper #########################
    def _resolve_params(self, params):
        ''' Resolve every value in a params dict via self._resolve_value. '''
        return {k: self._resolve_value(v) for k, v in params.items()}

    def _resolve_value(self, value):
        ''' Resolve one parameter value: '@name' strings become self.meta_objects['name']. All other values are returned unchanged '''
        if isinstance(value, str) and value.startswith('@'):
            return self.meta_objects[value[1:]]
        if isinstance(value, list):
            return [self._resolve_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_value(v) for v in value)
        if isinstance(value, dict):
            return {k: self._resolve_value(v) for k, v in value.items()}
        return value

##########################  Recorders ##################################
    def _build_record_path(self, file_name):
        ''' Compose the record file path. If file_name is absolute, return it as-is; otherwise join it under goal_path. '''
        if os.path.isabs(file_name): return file_name
        else: return os.path.join(self.global_params['goal_path'], file_name)

    def _get_columns(self, target):
        ''' Build the [patch_id, habitat_id, microsite_id] columns list for one registered object. '''
        cols_patch_id, cols_hab_id, cols_microsite_id = self.meta_objects[target].columns_patch_habitat_microsites_id()
        return [cols_patch_id, cols_hab_id, cols_microsite_id]

    def prime_optimum_sp_distribution(self, target, base_dead_rate, fitness_wid, species_2_phenotype_ls, file_name):
        ''' Schedule-callable: write the mode='w' header row of optimum-species-id values for one registered object. '''
        meta_obj = self.meta_objects[target]
        data = meta_obj.get_meta_microsites_optimum_sp_id_val(base_dead_rate, fitness_wid, species_2_phenotype_ls)
        meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=data, file_name=self._build_record_path(file_name), index=['optimun_sp_id_values'], columns=self._get_columns(target), mode='w')

    def prime_environment_distribution(self, target, environment_name, index_label, file_name):
        ''' Schedule-callable: write the mode='w' header row of one environment axis for one registered object. '''
        meta_obj = self.meta_objects[target]
        data = meta_obj.get_meta_microsite_environment_values(environment_name=environment_name)
        meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=data, file_name=self._build_record_path(file_name), index=[index_label], columns=self._get_columns(target), mode='w')

    def record_species_distribution(self, target, file_name, mode='a'):
        ''' Schedule-callable: Record species distribution at self.time_step for one registered object. '''
        meta_obj = self.meta_objects[target]
        sp_dis_data = meta_obj.get_meta_microsites_individuals_sp_id_values()
        meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=sp_dis_data, file_name=self._build_record_path(file_name), index=["time_step%d" % self.time_step], columns=self._get_columns(target), mode=mode)

    def record_phenotype_distribution(self, target, trait_name, file_name, mode='a'):
        ''' Schedule-callable: Record phenotype distribution at self.time_step for one trait of one registered object. '''
        meta_obj = self.meta_objects[target]
        pheno_dis_data = meta_obj.get_meta_microsites_individuals_phenotype_values(trait_name=trait_name)
        meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=pheno_dis_data, file_name=self._build_record_path(file_name), index=["time_step%d" % self.time_step], columns=self._get_columns(target), mode=mode)

##########################  Plotters ##################################
    def _build_plot_path(self, file_name):
        ''' Compose the plot file path. If file_name is absolute, keep its directory and prefix 'time_step=<t>-' to the basename; otherwise prefix 'time_step=<t>-' and join under goal_path. '''
        if os.path.isabs(file_name): return os.path.join(os.path.dirname(file_name), 'time_step=%d-' % self.time_step + os.path.basename(file_name))
        else: return os.path.join(self.global_params['goal_path'], 'time_step=%d-' % self.time_step + file_name)

    def plot_species_distribution(self, target, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, vmin, vmax, cmap, file_name):
        ''' Schedule-callable: write one JPG of species distribution at self.time_step for one registered object. '''
        meta_obj = self.meta_objects[target]
        meta_obj.meta_show_species_distribution(sub_row=sub_row, sub_col=sub_col, hab_num_x_axis_in_patch=hab_num_x_axis_in_patch, hab_num_y_axis_in_patch=hab_num_y_axis_in_patch, hab_y_len=hab_y_len, hab_x_len=hab_x_len, vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap), file_name=self._build_plot_path(file_name))

    def plot_species_phenotype_distribution(self, target, trait_name, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, cmap, file_name):
        ''' Schedule-callable: write one JPG of phenotype distribution at self.time_step for one trait of one registered object. '''
        meta_obj = self.meta_objects[target]
        meta_obj.meta_show_species_phenotype_distribution(trait_name=trait_name, sub_row=sub_row, sub_col=sub_col, hab_num_x_axis_in_patch=hab_num_x_axis_in_patch, hab_num_y_axis_in_patch=hab_num_y_axis_in_patch, hab_y_len=hab_y_len, hab_x_len=hab_x_len, cmap=plt.get_cmap(cmap), file_name=self._build_plot_path(file_name))

    def plot_environment_distribution(self, target, environment_name, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc, cmap, file_name):
        ''' Schedule-callable: write one JPG of one environment axis at self.time_step for one registered object. '''
        meta_obj = self.meta_objects[target]
        meta_obj.meta_show_environment_distribution(environment_name=environment_name, sub_row=sub_row, sub_col=sub_col, hab_num_x_axis_in_patch=hab_num_x_axis_in_patch, hab_num_y_axis_in_patch=hab_num_y_axis_in_patch, hab_y_len=hab_y_len, hab_x_len=hab_x_len, mask_loc=mask_loc, cmap=plt.get_cmap(cmap), file_name=self._build_plot_path(file_name))

########################  Progress / finalize  #########################
    def print_progress(self, target=None, rank=0, job_num=1, task_idx=1):
        ''' Schedule-callable: Print a minimal progress line at self.time_step. '''
        all_time_steps = self.global_params["all_time_steps"]
        if target is not None:
            meta_obj = self.meta_objects[target]
            meta_indi_num = meta_obj.get_meta_individual_num()
            meta_empty_num = meta_obj.show_meta_empty_sites_num()
            print('process%d, task_idx/job_num=%d/%d, time_step/all_time_step=%d/%d: indi_num=%d, empty_sites_num=%d'%(rank,task_idx,job_num,self.time_step,all_time_steps,meta_indi_num,meta_empty_num),flush=True)
        else:
            print('time_step/all_time_step=%d/%d' % (self.time_step, all_time_steps),flush=True)

    def finalize(self):
        ''' Return a lightweight dictionary describing the final state. '''
        return {"meta_objects": self.meta_objects, "global_params": self.global_params, "schedule_one_time_step": self.schedule_one_time_step}
