# -*- coding: utf-8 -*-
"""
MPI launcher for ats.py -- the alternative-stable-states design of example2.ipynb.

Same split as experiments/model.py + experiments/mpi_running.py: ats.main() is one run of one parameter
combination, and this file holds the parameter grid (notebook section 2.1) and hands the runs out to the
ranks. The runs never talk to each other -- each writes its own folder -- so the only communication is the
allocation broadcast at the start and the summary gathered at the end.

Output paths are relative to the working directory, so run it from examples/example2/ if the runs are to
land next to the notebook that reads them.
"""
from mpi4py import MPI
import time

import ats

###############################################################################
# cd examples/example2
# mpiexec -np 18 python mpi_running.py           # rep=1: the 3 x 3 x 2 grid, one run per rank
# mpiexec -np 6 python mpi_running.py            # rep=1 on a laptop: 3 runs per rank
# mpiexec -np 180 python mpi_running.py          # rep=10 (set rep_paras below)
###############################################################################

# ---- the swept axes (notebook section 2.1) ----
rep_paras = list(range(0, 1))                                    # range(0, 10) for the full study; nothing is seeded, so replicates differ by themselves
reproduce_mode_and_mutation_rate_paras = [('asexual', 0.00001), ('asexual', 0.0001), ('sexual', 0.0001)]
patch_dist_rate_paras = [0.001, 0.01, 0.03]                      # brackets the rate at which disturbance breaks the priority effect (~0.01); [0.00001, 0.0001, 0.001] is the old v2.9.13 axis
environment_value_paras = {0.2: [0.1], 0.8: [-0.1]}              # starting environment -> the delta of each climate step, i.e. 0.2 warms and 0.8 cools

# Rough cost of one run, in seconds, as measured on the 32x32 grid -- only the ratios matter, they are what
# the allocation balances on. Disturbance belongs in the key: a patch_dist_rate of 0.001 leaves the landscape
# near full and costs 2-4x one of 0.03, so keying on reproduce_mode alone piles every expensive run onto the
# same two ranks (the 32x32 grid finished in 3:31 with two ranks idle for the last 1:40). The low-disturbance
# levels of the old axis are extrapolated: below ~1e-4 disturbance is effectively absent and occupancy saturates.
job_seconds = {('asexual', 0.00001): 3600, ('asexual', 0.0001): 3400, ('asexual', 0.001): 3100, ('asexual', 0.01): 2900, ('asexual', 0.03): 2400,
               ('sexual', 0.00001): 6000, ('sexual', 0.0001): 5800, ('sexual', 0.001): 5500, ('sexual', 0.01): 2900, ('sexual', 0.03): 1300}
default_job_seconds = {'asexual': 3000, 'sexual': 4500}          # a disturbance rate that is not in the table above still has to be schedulable


def job_cost(para): return job_seconds.get((para[1], para[3]), default_job_seconds[para[1]])


def allocate_njobs_into_mrank(jobs_parameters, rank_num):
    ''' Greedy longest-processing-time-first: hand the most expensive run to the lightest rank, repeat.
        Placing the big jobs first is what keeps a sexual run off the last rank once everyone else has
        finished. Returns {rank: [job, ...]}. '''
    ranks_jobs_para, ranks_jobs_time = {r: [] for r in range(rank_num)}, {r: 0 for r in range(rank_num)}
    for para in sorted(jobs_parameters, key=job_cost, reverse=True):
        r = min(ranks_jobs_time, key=ranks_jobs_time.get)
        ranks_jobs_para[r].append(para)
        ranks_jobs_time[r] += job_cost(para)
    return ranks_jobs_para


def hms(seconds): return '%d:%02d:%02d' % (seconds // 3600, seconds // 60 % 60, seconds % 60)
def job_label(job): return 'rep=%d, %s, mutation_rate=%g, patch_dist_rate=%g, environment_value=%g' % job


comm = MPI.COMM_WORLD
size, rank = comm.Get_size(), comm.Get_rank()

# rep, reproduce_mode, mutation_rate, disturbance_rate, environment
jobs_parameters = [(i, j[0], j[1], k, x) for i in rep_paras for j in reproduce_mode_and_mutation_rate_paras for k in patch_dist_rate_paras for x in environment_value_paras]

if rank == 0: print('%d jobs on %d ranks, into %s/' % (len(jobs_parameters), size, ats.output_root), flush=True)
this_worker_job_para = comm.bcast(allocate_njobs_into_mrank(jobs_parameters, size) if rank == 0 else None, root=0)[rank]
print('Process %d, job_num_of_the_process=%d, estimated_running_time_of_the_process=%s' % (rank, len(this_worker_job_para), hms(sum(job_cost(job) for job in this_worker_job_para))), flush=True)

st_time = time.time()
for job in this_worker_job_para:
    rep, reproduce_mode, mutation_rate, patch_dist_rate, environment_value = job
    print('Process %d, ats.py is started (%s)' % (rank, job_label(job)), flush=True)
    job_start = time.time()
    ats.main(reproduce_mode, patch_dist_rate, mutation_rate, environment_value, environment_value_paras[environment_value], rep)   # the starting environment carries the climate direction with it
    print('Process %d, ats.py is finished in %s (%s)' % (rank, hms(time.time() - job_start), job_label(job)), flush=True)

worker_report = comm.gather((rank, len(this_worker_job_para), time.time() - st_time), root=0)

if rank == 0:
    print('\n%d jobs simulated in %s (slowest rank), %d ranks idle' % (len(jobs_parameters), hms(max(elapsed for _, _, elapsed in worker_report)), sum(1 for _, job_num, _ in worker_report if job_num == 0)), flush=True)
    for r, job_num, elapsed in worker_report:
        if job_num: print('  rank %d: %d jobs in %s' % (r, job_num, hms(elapsed)), flush=True)
