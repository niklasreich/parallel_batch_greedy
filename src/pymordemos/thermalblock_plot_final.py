import numpy as np
from pymor.core.pickle import load
import matplotlib.pyplot as plt
from os.path import isfile


max_batchsize = 30
plot_batch = [1, 2, 4, 8, 16]

procs = 30

t_evaluate = []
t_extend = []
t_greedy = []
t_offline = []
t_postprocess = []
t_reduce = []
t_solve = []
t_online = []

batchsizes = []
batchsizes_str = []

num_ext = []
num_iter = []
num_pp = []

ax00_y_max = 0

fig, ax = plt.subplots(2,2)

for bs in range(1, max_batchsize+1):

    file_string = f'src/pymordemos/thermalblock_2x2_N{procs}_BS{bs}.pkl'

    if isfile(file_string):
        with open(file_string, 'rb') as f:
            results = load(f)

        timings = results['timings']

        t_evaluate.append(timings['evaluate'])
        t_extend.append(timings['extend'])
        t_greedy.append(timings['greedy'])
        t_offline.append(timings['offline'])
        t_postprocess.append(timings['postprocess'])
        t_reduce.append(timings['reduce'])
        t_solve.append(timings['solve'])
        t_online.append(timings['online'])

        batchsizes_str.append((str(bs)))
        batchsizes.append(bs)

        num_ext.append(results['num_extensions'])
        num_iter.append(results['num_iterations'])
        num_pp.append(results['num_extensions'] - len(results['max_errs_pp']))

        ### Subplot 00 (Upper left)
        if bs in plot_batch:
            ax[0,0].semilogy(results['max_rel_errors'][0],label=f'$b={bs}$')
            if len(results['max_rel_errors'][0]) > ax00_y_max:
                ax00_y_max = len(results['max_rel_errors'][0])

t_evaluate = np.array(t_evaluate)
t_extend = np.array(t_extend)
t_greedy = np.array(t_greedy)
t_offline = np.array(t_offline)
t_postprocess = np.array(t_postprocess)
t_reduce = np.array(t_reduce)
t_solve = np.array(t_solve)
t_online = np.array(t_online)

t_other = t_offline - t_evaluate - t_extend - t_postprocess - t_reduce - t_solve

t_sequential = {'solve': t_solve, 'evaluate': t_evaluate, 'extend': t_extend, 'reduce': t_reduce, 'postprocess': t_postprocess, 'other': t_other}

fig.suptitle(f'2x2 Thermalblock')

### Subplot 00 (Upper left)
ax[0,0].legend(loc="upper right")
ax[0,0].set_ylabel('err in $H^1_0$ semi norm')
ax[0,0].set_xlabel('basis size n')
ax[0,0].grid(axis='y')
ax[0,0].set_axisbelow(True)
ax[0,0].plot([0,ax00_y_max],[1e-5,1e-5],'k--')

### Subplot 01 (Upper right)
bottom = np.zeros(len(batchsizes_str))
width = 0.5
for title, time in t_sequential.items():
    p = ax[0,1].bar(batchsizes_str, time, width, label=title, bottom=bottom)
    bottom += time
ax[0,1].legend(loc="upper right")
ax[0,1].set_ylabel('offline time [s]')
ax[0,1].set_xlabel('batch size b')
ax[0,1].grid(axis='y')
ax[0,1].set_axisbelow(True)

### Subplot 10 (Lower left)
ax[1,0].plot([1,np.max(batchsizes)],[num_ext[0],num_ext[0]],'k--')
ax[1,0].plot(batchsizes, num_ext, 'o:', label='Basis size without postprocessing')
ax[1,0].plot(batchsizes, num_pp, 'o:', label='Basis size with postprocessing')
ax[1,0].plot(batchsizes, num_iter, 'o:', label='greedy iterations')
ax[1,0].grid(axis='y')
ax[1,0].set_axisbelow(True)
ax[1,0].set_xlabel('batch size b')
ax[1,0].legend(loc=0)

### Subplot 10 (Lower right)
t_online_n = t_online/t_online[0]
t_offline_n = t_offline/t_offline[0]
ax[1,1].plot([1,np.max(batchsizes)],[1,1],'k--')
ax[1,1].plot(batchsizes, t_online_n, 'o:', label='Norm. online time')
ax[1,1].plot(batchsizes, t_offline_n, 'o:', label='Norm. offline time')
ax[1,1].grid(axis='y')
ax[1,1].set_axisbelow(True)
ax[1,1].set_xlabel('batch size b')
ax[1,1].legend(loc=0)
plt.show()
