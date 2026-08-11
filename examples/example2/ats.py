"""Alternative stable states in a metacommunity under rapid environmental change -- simulation only.

The simulation half of example2.ipynb, stripped of everything that only exists to display results:
no matplotlib / seaborn / pandas, no tmp_nb_code2, no figures, no tables, and none of the section-3.3
special cases. What is left is the model (sections 1-2 of the notebook) and the two recorded .csv.gz
tables it writes, which the notebook's section 3 -- or any other analysis -- can read back afterwards.

main() is one run of one parameter combination -- same shape as experiments/model.py, so the same
mpi4py launcher pattern drives it. The parameter grid itself lives in the launcher, not here:
    cd examples/example2 && mpiexec -np 18 python mpi_running.py     # the whole 3 x 3 x 2 design, one run per rank

One combination on its own, from the shell or from python:
    python ats.py
    import ats; ats.main('sexual', 0.001, 0.0001, 0.2, [0.1])
"""
import os, time

import numpy as np

import bootstrap_metaibm
import metaibm


# --------------------------------------------------------------------------- #
# Parameters (notebook section 2.2 -- the values every run shares).
# The swept axes of section 2.1 are arguments of main(), and their levels live in mpi_running.py.
# --------------------------------------------------------------------------- #
# ---- time (section 2.2) ----
all_time_step = 800
change_time_step, start_change_time, end_change_time = 100, 99, 700
burn_in_steps = 100                                              # the source pools evolve for this long, then stay frozen
record_every = 4                                                 # record a landscape snapshot every this many steps

# ---- landscape (section 2.2) ----
patch_num, patch_num_x_axis, patch_num_y_axis = 100, 10, 10
hab_length, hab_width = 10, 10                                     # microsites per patch; x patch_num for the landscape (6x6 = 3600, 10x10 = 10000, 20x20 = 40000, 32x32 = 102400)
mainland_hab_length, mainland_hab_width = 32, 32                 # 1024 individuals in each source pool

# ---- environment (section 2.2) ----
environment_types_name, environment_variation_ls = ('environment', ), (0.025, )
env_name_ls, delta_var_ls = ['environment'], [0]

# ---- species & individuals (section 2.2) ----
species_2_phenotype_ls = [[0.2], [0.8]]                          # sp1 is cold-adapted, sp2 is warm-adapted
traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls = 1, ('phenotype', ), (0.025, ), (20, )

# ---- demography, selection and dispersal (section 2.2) ----
base_dead_rate, fitness_wid = 0.1, 0.5
asexual_birth_rate, sexual_birth_rate = 0.5, 1
propagules_rain_num, dispersal_amomg_rate = 10, 0.0001

# ---- output paths ----
output_root_suffix = ''                                             # set when a variant grid shares the patch size of another one, e.g. '_lowdist' for the same 32x32 run at the old disturbance axis
output_root = 'example2_pyoutput_%dx%d%s' % (hab_length, hab_width, output_root_suffix)   # the size is in the folder name, so the grids of different patch sizes cannot overwrite each other
species_file, phenotype_file = 'species_distribution_over_time.csv.gz', 'phenotype_distribution_over_time.csv.gz'

patch_env_offset = np.zeros((patch_num_x_axis, patch_num_y_axis))   # per-patch offset from the run's starting climate; all-zero = a spatially uniform landscape
# e.g. a west-east gradient instead:  patch_env_offset = np.tile(np.linspace(-0.15, 0.15, patch_num_x_axis).reshape(-1, 1), (1, patch_num_y_axis))


# --------------------------------------------------------------------------- #
# Building the landscape (section 2.4)
# --------------------------------------------------------------------------- #
def build_mainland(meta_name, environment_mean_value, reproduce_mode):
    ''' One patch holding one habitat whose environment is one species' optimum, filled with that species. '''
    mainland = metaibm.metacommunity(metacommunity_name=meta_name)
    mainland_patch = metaibm.patch(patch_name='patch1', patch_index=0, location=(0, 0))
    mainland.add_patch(patch_name='patch1', patch_object=mainland_patch)
    mainland_patch.add_habitat(hab_name='h1', hab_index=0, hab_location=(0, 0), num_env_types=len(environment_types_name), env_types_name=environment_types_name, mean_env_ls=[environment_mean_value], var_env_ls=list(environment_variation_ls), length=mainland_hab_length, width=mainland_hab_width)
    mainland.meta_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
    return mainland


def build_metacommunity(environment_mean_value):
    ''' 100 empty patches on a 10 x 10 grid, one habitat each, mean environment = environment_mean_value + this patch's offset.
        Every habitat gets its own mean_env_ls list, because meta_offset_environmental_values mutates it in place. '''
    env_offset = patch_env_offset
    meta_obj = metaibm.metacommunity(metacommunity_name='metacommunity')

    for x in range(patch_num_x_axis):
        for y in range(patch_num_y_axis):
            patch_index = x * patch_num_y_axis + y
            patch_obj = metaibm.patch(patch_name='patch%d' % (patch_index + 1), patch_index=patch_index, location=(x, y))
            patch_obj.add_habitat(hab_name='h1', hab_index=0, hab_location=(x, y), num_env_types=len(environment_types_name), env_types_name=environment_types_name, mean_env_ls=[float(environment_mean_value + env_offset[x][y])], var_env_ls=list(environment_variation_ls), length=hab_length, width=hab_width)
            meta_obj.add_patch(patch_name=patch_obj.name, patch_object=patch_obj)

    return meta_obj


# --------------------------------------------------------------------------- #
# Output paths and recording (section 2.5)
# --------------------------------------------------------------------------- #
def save_snapshot(meta_obj, goal_path, columns, time_step):
    ''' Append one landscape snapshot -- species ids and phenotypes -- to the two csv.gz files. '''
    meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_individuals_sp_id_values(), file_name=os.path.join(goal_path, species_file), index=['time_step%d' % time_step], columns=columns, mode='a')
    meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_individuals_phenotype_values(trait_name='phenotype'), file_name=os.path.join(goal_path, phenotype_file), index=['time_step%d' % time_step], columns=columns, mode='a')


def goal_path_of(rep, reproduce_mode, mutation_rate, patch_dist_rate, environment_value, root_path=None):
    ''' The output folder of one run; the path encodes its own parameters. Pure -- nothing is created. '''
    if root_path is None: root_path = output_root
    return os.path.join(root_path, 'rep=%d' % rep, str(reproduce_mode), 'mutation_rate=%f' % mutation_rate, 'patch_dist_rate=%f' % patch_dist_rate, 'environment=%2f' % environment_value)


def mkdir_if_not_exist(rep, reproduce_mode, mutation_rate, patch_dist_rate, environment_value, root_path=None):
    ''' The same folder, created. '''
    goal_path = goal_path_of(rep, reproduce_mode, mutation_rate, patch_dist_rate, environment_value, root_path=root_path)
    os.makedirs(goal_path, exist_ok=True)
    return goal_path


# --------------------------------------------------------------------------- #
# One run (section 2.5) -- the entry point mpi_running.py calls, one rank per combination
# --------------------------------------------------------------------------- #
def main(reproduce_mode, patch_dist_rate, mutation_rate, environment_mean_value, delta_mean_ls, rep=0, goal_path=None, root_path=None):
    ''' One replicate of one parameter combination: burn the two source pools in, then run the landscape.
        Nothing is seeded -- the model is stochastic and every run, including two runs of the same rep, is
        an independent draw. goal_path defaults to the folder the parameters name (section 2.5); the
        launcher may pass its own. Returns the output folder -- the run lives on disk, not in memory. '''
    all_time_start = time.time()
    if goal_path is None: goal_path = mkdir_if_not_exist(rep, reproduce_mode, mutation_rate, patch_dist_rate, environment_mean_value, root_path=root_path)
    os.makedirs(goal_path, exist_ok=True)
    mainland1, mainland2 = build_mainland('mainland1', species_2_phenotype_ls[0][0], reproduce_mode), build_mainland('mainland2', species_2_phenotype_ls[1][0], reproduce_mode)

    for _ in range(burn_in_steps):                                                 # burn-in: standing variation, then the pools are never touched again
        for mainland in (mainland1, mainland2):
            mainland.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            if reproduce_mode == 'asexual': mainland.meta_asex_reproduce_mutate_into_offspring_pool(asexual_birth_rate, mutation_rate, pheno_var_ls)
            elif reproduce_mode == 'sexual': mainland.meta_sex_reproduce_mutate_into_offspring_pool(sexual_birth_rate, mutation_rate, pheno_var_ls)
            mainland.meta_local_germinate_from_offspring_and_immigrant_pool()
            mainland.meta_clear_up_offspring_and_immigrant_pool()

    meta_obj = build_metacommunity(environment_mean_value)
    columns = list(meta_obj.columns_patch_habitat_microsites_id())                  # the 3-level patch / habitat / microsite header
    logger_file = open(os.path.join(goal_path, 'logger.log'), 'w')

    # the two reference rows of section 2.5; mode='w' also creates the two files
    meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_optimum_sp_id_val(base_dead_rate, fitness_wid, species_2_phenotype_ls), file_name=os.path.join(goal_path, species_file), index=['optimun_sp_id_values'], columns=columns, mode='w')
    meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsite_environment_values(environment_name='environment'), file_name=os.path.join(goal_path, phenotype_file), index=['environment'], columns=columns, mode='w')

    for time_step in range(all_time_step):
        log_info = 'time_step=%d \n' % time_step

        log_info += meta_obj.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')                                      # 1  selection
        if reproduce_mode == 'asexual': log_info += meta_obj.meta_asex_reproduce_mutate_into_offspring_pool(asexual_birth_rate, mutation_rate, pheno_var_ls)     # 2  reproduction -> offspring_pool
        elif reproduce_mode == 'sexual': log_info += meta_obj.meta_sex_reproduce_mutate_into_offspring_pool(sexual_birth_rate, mutation_rate, pheno_var_ls)      # 2  reproduction -> offspring_pool
        if (time_step + 1) % change_time_step == 0 and start_change_time <= time_step <= end_change_time:
            log_info += meta_obj.meta_offset_environmental_values(env_name_ls, delta_mean_ls, delta_var_ls)                                 # 3  the climate step
        for mainland in (mainland1, mainland2):
            log_info += meta_obj.meta_colonize_from_propagules_rains(mainland, propagules_rain_num)                                         # 4  colonization
        log_info += meta_obj.dispersal_aomng_patches_from_offspring_pool_to_immigrant_pool(dispersal_amomg_rate)                            # 5  dispersal among patches
        log_info += meta_obj.meta_local_germinate_from_offspring_and_immigrant_pool()                                                       # 6  germination
        log_info += meta_obj.meta_disturbance_process_in_patches(patch_dist_rate)                                                           # 7  disturbance
        meta_obj.meta_clear_up_offspring_and_immigrant_pool()                                                                               # 8  clear the pools

        print(log_info, file=logger_file)
        if time_step % record_every == 0: save_snapshot(meta_obj, goal_path, columns, time_step)

    logger_file.close()
    print('%s finished in %.1f min' % (goal_path, (time.time() - all_time_start) / 60), flush=True)
    return goal_path


if __name__ == '__main__':
    local = main(reproduce_mode='sexual', patch_dist_rate=0.001, mutation_rate=0.0001, environment_mean_value=0.2, delta_mean_ls=[0.1], rep=0, goal_path=None)
