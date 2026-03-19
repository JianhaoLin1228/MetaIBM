# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:33:21 2023

@author: JH_Lin
"""

from mpi4py import MPI
import numpy as np
import os
import copy
import time
import model

def mkdir_if_not_exist(rep, reproduce_mode, mutation_rate, patch_dist_rate, environment_value, root_path=None):
    if root_path==None: root_path = os.getcwd()

    rep_files_name = 'rep=%d'%rep
    reproduce_mode_files_name = str(reproduce_mode)
    mutation_rate_files_name = 'mutation_rate=%f'%mutation_rate
    patch_dist_rate_files_name = 'patch_dist_rate=%f'%patch_dist_rate
    environment_files_name = 'environment=%2f'%environment_value
    
    goal_path = root_path+'/'+rep_files_name+'/'+reproduce_mode_files_name+'/'+mutation_rate_files_name+'/'+patch_dist_rate_files_name+'/'+environment_files_name
    if os.path.exists(goal_path) == False:
        os.makedirs(goal_path, exist_ok=True)
    else:
        pass
    return goal_path

def paras_2_time(reproduce_mode):
    ''' '''
    if reproduce_mode=='asexual':
        return 5000
    elif reproduce_mode=='sexual':
        return 5000

def get_minimum_time_ranks(ranks_jobs_time):
    minimum = np.inf
    minimum_time_ranks = []
    for rank, curr_time in ranks_jobs_time.items():
        if curr_time < minimum:
            minimum_time_ranks = [rank]
            minimum = curr_time
        elif curr_time == minimum:
            minimum_time_ranks.append(rank)
        else:
            continue
    return minimum_time_ranks

def allocate_njobs_into_mrank(jobs_parameters, rank_num):
    ranks_jobs_para = {}
    ranks_jobs_time = {}
    all_jobs_parameters = copy.deepcopy(jobs_parameters)
    for rank in range(rank_num):
        ranks_jobs_para[rank], ranks_jobs_time[rank]= [], 0
        
    all_jobs_parameters.sort(key=lambda x: x[1], reverse=False)
    
    while len(all_jobs_parameters) > 0:
        mini_rank_ls = get_minimum_time_ranks(ranks_jobs_time)
        
        for rank, para in list(zip(mini_rank_ls, all_jobs_parameters)):
            ranks_jobs_para[rank].append(para)
            ranks_jobs_time[rank] += paras_2_time(reproduce_mode=para[1])
            all_jobs_parameters.remove(para)
            
    return ranks_jobs_para, ranks_jobs_time


###############################################################################
# mpiexec -np 16 python mpi_running.py                for rep=1
# mpiexec -np 216 python mpi_running.py               for rep=1
# mpiexec -np 2160 python mpi_running.py              for rep=10
###############################################################################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

rep_paras = np.arange(0,10)
reproduce_mode_and_mutation_rate_paras = [('asexual', 0.00001), ('asexual', 0.0001), ('sexual', 0.0001)]
patch_dist_rate_paras = np.array([0.00001, 0.0001, 0.001])
environment_value_paras = np.array([0.0, 1.0])
    
jobs_parameters = [(i,j[0],j[1],k,x) for i in rep_paras for j in reproduce_mode_and_mutation_rate_paras for k in patch_dist_rate_paras for x in environment_value_paras]
# rep, reproduce_mode, mutation_rate, disturbance_rate, environment

if rank==0:
    ranks_jobs_para, ranks_jobs_time = allocate_njobs_into_mrank(jobs_parameters, size)
    
else:
    ranks_jobs_para, ranks_jobs_time = None, None

ranks_jobs_para = comm.bcast(ranks_jobs_para,root=0)
ranks_jobs_time = comm.bcast(ranks_jobs_time,root=0)

this_worker_job_para, this_worker_jobs_time = ranks_jobs_para[rank], ranks_jobs_time[rank]
this_worker_job_num = len(this_worker_job_para)


print(f"Process {rank}, job_num_of_the_prosess={len(this_worker_job_para)}", flush=True)
time.sleep(10)
print(f"Process {rank}, estimated_running_time_of_the_prosess = %d months, %d days, %d:%d:%d"%(this_worker_jobs_time//60//60//24//30, 
this_worker_jobs_time//60//60//24%30, this_worker_jobs_time//60//60%24, this_worker_jobs_time//60%60, this_worker_jobs_time%60), flush=True)
time.sleep(10)

st_time = time.time()
for a_piece_of_work in this_worker_job_para:
    
    rep = a_piece_of_work[0]
    reproduce_mode = a_piece_of_work[1]
    mutation_rate = a_piece_of_work[2]
    patch_dist_rate = a_piece_of_work[3]
    environment_value = a_piece_of_work[4]
    print(f"Process {rank}, model.py is started (reproduce_mode={reproduce_mode}, mutation_rate={mutation_rate}, patch_dist_rate={patch_dist_rate}, environment_value={environment_value}, rep={rep})", flush=True)
    time.sleep(5+rank)
    goal_path = mkdir_if_not_exist(rep, reproduce_mode, mutation_rate, patch_dist_rate, environment_value, root_path='/scratch/project_2018377/ljianhao/ATS_EcoEvo/0.0_1.0_0.1_100')
    if environment_value == 0.0: delta_mean_ls = [0.1]
    if environment_value == 1.0: delta_mean_ls = [-0.1]
    model.main(reproduce_mode, patch_dist_rate, mutation_rate, environment_value, delta_mean_ls, rep, goal_path)
    print(f"Process {rank}, model.py is finished (reproduce_mode={reproduce_mode}, mutation_rate={mutation_rate}, patch_dist_rate={patch_dist_rate}, environment_value={environment_value}, rep={rep})", flush=True)
























