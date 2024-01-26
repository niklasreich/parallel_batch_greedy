import numpy as np
from pymor.core.pickle import load
import matplotlib.pyplot as plt
from os.path import isfile


max_batchsize = 16
max_P = 16

t_evaluate = []
t_extend = []
t_greedy = []
t_offline = []
t_postprocess = []
t_reduce = []
t_solve = []
batchsizes = []

t_evaluate_p = []
t_extend_p= []
t_greedy_p = []
t_offline_p = []
t_postprocess_p = []
t_reduce_p = []
t_solve_p = []
batchsizes_p = []


for p in range(1, max_P+1):

    for bs in range(1, max_batchsize+1):

        file_string = f'src/pymordemos/thermalblock_3x2_N{p}_BS{bs}.pkl'

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
                batchsizes.append(('b = ' + str(bs)))
            elif p==bs:
                t_evaluate_p.append(timings['evaluate'])
                t_extend_p.append(timings['extend'])
                t_greedy_p.append(timings['greedy'])
                t_offline_p.append(timings['offline'])
                t_postprocess_p.append(timings['postprocess'])
                t_reduce_p.append(timings['reduce'])
                t_solve_p.append(timings['solve'])
                batchsizes_p.append(('b = ' + str(bs)))

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

t_other = t_offline - t_evaluate - t_extend - t_postprocess - t_reduce - t_solve
t_other_p = t_offline_p - t_evaluate_p - t_extend_p - t_postprocess_p - t_reduce_p - t_solve_p

t_sequential = {'evaluate': t_evaluate, 'solve': t_solve, 'extend': t_extend, 'reduce': t_reduce, 'postprocess': t_postprocess, 'other': t_other}
t_parallel = {'evaluate': t_evaluate_p, 'solve': t_solve_p, 'extend': t_extend_p, 'reduce': t_reduce_p, 'postprocess': t_postprocess_p, 'other': t_other_p}

fig, ax = plt.subplots(1,2)
bottom = np.zeros(len(batchsizes))
width = 0.5

for title, time in t_sequential.items():
    p = ax[0].bar(batchsizes, time, width, label=title, bottom=bottom)
    bottom += time

#fig, ax = plt.subplots(1,2)
bottom = np.zeros(len(batchsizes_p))

for title, time in t_parallel.items():
    p = ax[1].bar(batchsizes_p, time, width, label=title, bottom=bottom)
    bottom += time

ax[0].set_title("Sequential offline time decompostiion for different batchsizes.")
ax[0].legend(loc="upper right")
ax[1].set_title("Parallel offline time decompostiion for different batchsizes.")
ax[1].legend(loc="upper right")

_ , ymax0 = ax[0].get_ylim()
_ , ymax1 = ax[1].get_ylim()
ymax = ymax0 if ymax0 > ymax1 else ymax1
ax[0].set_ylim(0,ymax)
ax[1].set_ylim(0,ymax)


plt.show()
