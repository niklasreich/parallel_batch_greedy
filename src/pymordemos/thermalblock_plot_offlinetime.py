import numpy as np
from pymor.core.pickle import load
import matplotlib.pyplot as plt
from os.path import isfile


max_batchsize = 12
max_P = 12

p_opt = 7

t_evaluate = []
t_extend = []
t_greedy = []
t_offline = []
t_postprocess = []
t_reduce = []
t_solve = []
batchsizes = []
batchsizes_str = []

t_evaluate_p = []
t_extend_p= []
t_greedy_p = []
t_offline_p = []
t_postprocess_p = []
t_reduce_p = []
t_solve_p = []
batchsizes_p = []
batchsizes_str_p = []

t_evaluate_po = []
t_extend_po = []
t_greedy_po = []
t_offline_po = []
t_postprocess_po = []
t_reduce_po = []
t_solve_po = []
batchsizes_po = []
batchsizes_str_po = []


for p in range(1, max_P+1):

    for bs in range(1, max_batchsize+1):

        file_string = f'src/pymordemos/thermalblock_2x2_N{p}_BS{bs}.pkl'

        if isfile(file_string):
            with open(file_string, 'rb') as f:
                results = load(f)

            timings = results['timings']

            if p==1:
                t_evaluate.append(timings['evaluate'])
                t_extend.append(timings['extend'])
                t_greedy.append(timings['greedy'])
                t_offline.append(timings['offline'])
                t_postprocess.append(timings['postprocess'])
                t_reduce.append(timings['reduce'])
                t_solve.append(timings['solve'])
                batchsizes_str.append((str(bs)))
                batchsizes.append(bs)
            if p==bs:
                t_evaluate_p.append(timings['evaluate'])
                t_extend_p.append(timings['extend'])
                t_greedy_p.append(timings['greedy'])
                t_offline_p.append(timings['offline'])
                t_postprocess_p.append(timings['postprocess'])
                t_reduce_p.append(timings['reduce'])
                t_solve_p.append(timings['solve'])
                batchsizes_str_p.append((str(bs)))
                batchsizes_p.append(bs)
            if p==p_opt:
                t_evaluate_po.append(timings['evaluate'])
                t_extend_po.append(timings['extend'])
                t_greedy_po.append(timings['greedy'])
                t_offline_po.append(timings['offline'])
                t_postprocess_po.append(timings['postprocess'])
                t_reduce_po.append(timings['reduce'])
                t_solve_po.append(timings['solve'])
                batchsizes_str_po.append((str(bs)))
                batchsizes_po.append(bs)

t_evaluate = np.array(t_evaluate)
t_extend = np.array(t_extend)
t_greedy = np.array(t_greedy)
t_offline = np.array(t_offline)
t_postprocess = np.array(t_postprocess)
t_reduce = np.array(t_reduce)
t_solve = np.array(t_solve)

t_evaluate_p = np.array(t_evaluate_p)
t_extend_p= np.array(t_extend_p)
t_greedy_p = np.array(t_greedy_p)
t_offline_p = np.array(t_offline_p)
t_postprocess_p = np.array(t_postprocess_p)
t_reduce_p = np.array(t_reduce_p)
t_solve_p = np.array(t_solve_p)

t_evaluate_po = np.array(t_evaluate_po)
t_extend_po = np.array(t_extend_po)
t_greedy_po = np.array(t_greedy_po)
t_offline_po = np.array(t_offline_po)
t_postprocess_po = np.array(t_postprocess_po)
t_reduce_po = np.array(t_reduce_po)
t_solve_po = np.array(t_solve_po)

t_other = t_offline - t_evaluate - t_extend - t_postprocess - t_reduce - t_solve
t_other_p = t_offline_p - t_evaluate_p - t_extend_p - t_postprocess_p - t_reduce_p - t_solve_p
t_other_po = t_offline_po - t_evaluate_po - t_extend_po - t_postprocess_po - t_reduce_po - t_solve_po

t_sequential = {'solve': t_solve, 'evaluate': t_evaluate, 'extend': t_extend, 'reduce': t_reduce, 'postprocess': t_postprocess, 'other': t_other}
t_parallel_b = {'solve': t_solve_p,'evaluate': t_evaluate_p, 'extend': t_extend_p, 'reduce': t_reduce_p, 'postprocess': t_postprocess_p, 'other': t_other_p}
t_parallel_opt = {'solve': t_solve_po, 'evaluate': t_evaluate_po, 'extend': t_extend_po, 'reduce': t_reduce_po, 'postprocess': t_postprocess_po, 'other': t_other_po}

fig, ax = plt.subplots(2,2)
bottom = np.zeros(len(batchsizes_str))
width = 0.5
for title, time in t_sequential.items():
    p = ax[0,0].bar(batchsizes_str, time, width, label=title, bottom=bottom)
    bottom += time

bottom = np.zeros(len(batchsizes_str_p))
for title, time in t_parallel_b.items():
    p = ax[0,1].bar(batchsizes_str_p, time, width, label=title, bottom=bottom)
    bottom += time

bottom = np.zeros(len(batchsizes_str_p))
for title, time in t_parallel_opt.items():
    p = ax[1,0].bar(batchsizes_str_po, time, width, label=title, bottom=bottom)
    bottom += time

ax[1,1].plot(batchsizes,t_offline,'--*',label='N_MPI=1')
ax[1,1].plot(batchsizes_p,t_offline_p,'--*',label='N_MPI=b')
ax[1,1].plot(batchsizes_po,t_offline_po,'--*',label=f'N_MPI=N_OPT={p_opt}')

ax[0,0].set_title(f"Sequential: N_MPI=1.")
ax[0,0].legend(loc="upper right")
ax[0,1].set_title(f"Parallel: N_MPI=b.")
ax[0,1].legend(loc="upper right")
ax[1,0].set_title(f"Parallel: N_MPI=N_OPT={p_opt}.")
ax[1,0].legend(loc="upper right")
ax[1,1].set_title("Overall offline times.")
ax[1,1].legend(loc="upper right")

ax[0,0].set_ylabel('time [s]')
ax[0,0].set_xlabel('batchsize b')
ax[0,1].set_ylabel('time [s]')
ax[0,1].set_xlabel('batchsize b')
ax[1,0].set_ylabel('time [s]')
ax[1,0].set_xlabel('batchsize b')
ax[1,1].set_ylabel('time [s]')
ax[1,1].set_xlabel('batchsize b')

_ , ymax0 = ax[0,0].get_ylim()
_ , ymax1 = ax[0,1].get_ylim()
_ , ymax2 = ax[1,0].get_ylim()
ymax = max(ymax0,ymax1,ymax2)
ax[0,0].set_ylim(0,ymax)
ax[0,1].set_ylim(0,ymax)
ax[1,0].set_ylim(0,ymax)
ax[1,1].set_ylim(0,ymax)

plt.show()
