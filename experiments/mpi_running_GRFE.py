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
import model_sloss_GRFE as model

def mkdir_if_not_exist(reproduce_mode, patch_num, is_same_heterogeneity, disp_among_within_rate, patch_dist_rate, root_path=None):
    if root_path==None: root_path = os.getcwd()
    
    reproduce_mode_dir = {'asexual':'asexual', 'sexual':'sexual'}
    patch_num_files_name = 'patch_num=%03d'%patch_num
    is_heterogeneity_files_name = 'is_same_heterogeneity=%s'%str(is_same_heterogeneity)
    disp_amomg_within_rate_files_name = 'disp_among=%f-disp_within=%f'%(disp_among_within_rate[0], disp_among_within_rate[1])
    patch_dist_rate_files_name = 'patch_dist_rate=%f'%patch_dist_rate
    
    goal_path = root_path+'/'+reproduce_mode_dir[reproduce_mode]+'/'+patch_num_files_name+'/'+is_heterogeneity_files_name+'/'+disp_amomg_within_rate_files_name+'/'+patch_dist_rate_files_name
    if os.path.exists(goal_path) == False:
        os.makedirs(goal_path)
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
            ranks_jobs_time[rank] += paras_2_time(reproduce_mode=para[3])
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

rep_paras = np.arange(0,1)
patch_num_paras = np.array([1,4,16,64,256])
is_same_heterogeneity_paras = np.array([True,False])
reproduce_mode_paras=np.array(['asexual','sexual'])
disp_among_within_rate_paras = np.array([(0.001,0.001),(0.001,0.01),(0.001,0.1),(0.01,0.01),(0.01,0.1),(0.1,0.1)]) #among<=within
patch_dist_rate_paras = np.array([0.00001,0.0001,0.001])

jobs_parameters = [(i,j,k,x,y,z) for i in rep_paras for j in patch_num_paras for k in is_same_heterogeneity_paras for x in reproduce_mode_paras for y in disp_among_within_rate_paras for z in patch_dist_rate_paras]
# rep, patch_num, env_heterogeneity, reproduce_mode, (m_among, m_within), disturbance_rate, 

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
task_idx = 0

for a_piece_of_work in this_worker_job_para:
    task_idx += 1
    
    rep = int(a_piece_of_work[0])
    patch_num = int(a_piece_of_work[1])
    is_same_heterogeneity = bool(a_piece_of_work[2])
    reproduce_mode = str(a_piece_of_work[3])
    disp_among_rate, disp_within_rate = float(a_piece_of_work[4][0]), float(a_piece_of_work[4][1])
    patch_dist_rate = float(a_piece_of_work[5])

    print(f"Process {rank}, model.py is started (reproduce_mode={reproduce_mode}, patch_num={patch_num}, is_same_heterogeneity={is_same_heterogeneity}, patch_dist_rate={patch_dist_rate}, (disp_among_rate, disp_within_rate)={(disp_among_rate, disp_within_rate)}, rep={rep})", flush=True)
    time.sleep(5)
    goal_path = mkdir_if_not_exist(reproduce_mode, patch_num, is_same_heterogeneity, (disp_among_rate, disp_within_rate), patch_dist_rate, root_path=None)
    model.main(rep, patch_num, is_same_heterogeneity, reproduce_mode, disp_among_rate, disp_within_rate, patch_dist_rate, goal_path, rank, this_worker_job_num, task_idx)
    
    






















