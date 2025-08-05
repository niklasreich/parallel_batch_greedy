import numpy as np
from pymor.core.pickle import load
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import listdir

file_string = 'thermalblock_3x3'

# max_batchsize = 30
# plot_batch = [1, 2, 4, 8, 16]

procs = 30

t_evaluate = []
t_extend = []
t_greedy = []
t_offline = []
t_reduce = []
t_solve = []
t_online = []

lambda_tol = []
lambda_str = []

num_ext = []
num_iter = []
num_pp = []

ax00_y_max = 0

files = [f for f in listdir('src/pymordemos') if isfile(join('src/pymordemos', f))]
files = [f for f in files if file_string in f and 'lambda' in f]
files = [f for f in files if f'N{procs}' in f or 'B1' in f]

fig, ax = plt.subplots(2,2)

for f in files:

    with open('src/pymordemos/'+f, 'rb') as f_:
        results = load(f_)

        timings = results['timings']
    if 'B1' in f:
        rev_t_online = timings['online']
        rev_t_offline = timings['offline']
        rev_t_eveluate = timings['evaluate']
        rev_t_extend = timings['extend']
        rev_t_reduce = timings['reduce']
        rev_t_solve = timings['solve']
        rev_t_greedy = timings['greedy']

        rev_num_iter = results['num_iterations']
        rev_num_ext = results['num_extensions']

        ax[0,0].semilogy(results['max_rel_errors'][0],label='classical')
        if len(results['max_rel_errors'][0]) > ax00_y_max:
            ax00_y_max = len(results['max_rel_errors'][0])

    else:

        timings = results['timings']

        t_evaluate.append(timings['evaluate'])
        t_extend.append(timings['extend'])
        t_greedy.append(timings['greedy'])
        t_offline.append(timings['offline'])
        t_reduce.append(timings['reduce'])
        t_solve.append(timings['solve'])
        t_online.append(timings['online'])

        num_ext.append(results['num_extensions'])
        num_iter.append(results['num_iterations'])

        lambda_val = results['settings']['lambda']
        lambda_tol.append(lambda_val)
        lambda_str.append(f"{lambda_val}")

        ### Subplot 00 (Upper left)
        ax[0,0].semilogy(results['max_rel_errors'][0],':',label=f'$\lambda={lambda_val}$')
        if len(results['max_rel_errors'][0]) > ax00_y_max:
            ax00_y_max = len(results['max_rel_errors'][0])


t_evaluate.append(rev_t_eveluate)
t_extend.append(rev_t_extend)
t_offline.append(rev_t_offline)
t_reduce.append(rev_t_reduce)
t_solve.append(rev_t_solve)
t_online.append(rev_t_online)
t_greedy.append(rev_t_greedy)
num_ext.append(rev_num_ext)
num_iter.append(rev_num_iter)
lambda_tol.append(2.) #Put at the end after sorting
lambda_str.append('CWG')


t_evaluate = np.array(t_evaluate)
t_extend = np.array(t_extend)
t_greedy = np.array(t_greedy)
t_offline = np.array(t_offline)
t_reduce = np.array(t_reduce)
t_solve = np.array(t_solve)
t_online = np.array(t_online)
lambda_tol = np.array(lambda_tol)
lambda_str = np.array(lambda_str)
num_ext = np.array(num_ext)
num_iter = np.array(num_iter)

sort_ind = np.argsort(lambda_tol)

t_evaluate = t_evaluate[sort_ind]
t_extend = t_extend[sort_ind]
t_greedy = t_greedy[sort_ind]
t_offline = t_offline[sort_ind]
t_reduce = t_reduce[sort_ind]
t_solve = t_solve[sort_ind]
t_online = t_online[sort_ind]
lambda_tol = lambda_tol[sort_ind]
lambda_str = lambda_str[sort_ind]
num_ext = num_ext[sort_ind]
num_iter = num_iter[sort_ind]

t_other = t_offline - t_evaluate - t_extend - t_reduce - t_solve

t_sequential = {'solve': t_solve, 'evaluate': t_evaluate, 'extend': t_extend, 'reduce': t_reduce, 'other': t_other}

fig.suptitle(file_string)

### Subplot 00 (Upper left)
ax[0,0].legend(loc="lower left")
ax[0,0].set_ylabel('err in $H^1_0$ semi norm')
ax[0,0].set_xlabel('basis size n')
ax[0,0].grid(axis='y')
ax[0,0].set_axisbelow(True)
ax[0,0].plot([0,ax00_y_max],[1e-5,1e-5],'k--')

### Subplot 01 (Upper right)
bottom = np.zeros(len(lambda_tol))
width = 0.5
for title, time in t_sequential.items():
    p = ax[0,1].bar(lambda_str, time, width, label=title, bottom=bottom)
    bottom += time
ax[0,1].legend(loc="upper left")
ax[0,1].set_ylabel('offline time [s]')
ax[0,1].set_xlabel('$\lambda$')
ax[0,1].grid(axis='y')
ax[0,1].set_axisbelow(True)

### Subplot 10 (Lower left)
# ax[1,0].semilogx([np.min(lambda_tol),1],[rev_num_iter,rev_num_iter],'k--')
ax[1,0].semilogx(lambda_tol[:-1], num_ext[:-1]/num_ext[-1], 'o:', label='Basis size')
# ax[1,0].semilogx(lambda_tol[:-1], num_ext[:-1]/num_iter[:-1], 'o:', label='Greedy Iterations')
ax[1,0].grid(axis='y')
ax[1,0].set_axisbelow(True)
ax[1,0].set_xlabel('$\lambda$')
ax[1,0].legend(loc=0)

### Subplot 10 (Lower right)
t_online_n = t_online/rev_t_online
t_offline_n = t_offline/rev_t_offline
ax[1,1].semilogx([np.min(lambda_tol),1],[1,1],'k--')
ax[1,1].semilogx(lambda_tol[:-1], t_online_n[:-1], 'o:', label='Norm. online time')
ax[1,1].semilogx(lambda_tol[:-1], t_offline_n[:-1], 'o:', label='Norm. offline time')
ax[1,1].grid(axis='y')
ax[1,1].set_axisbelow(True)
ax[1,1].set_xlabel('$\lambda$')
ax[1,1].legend(loc=0)
plt.show()
