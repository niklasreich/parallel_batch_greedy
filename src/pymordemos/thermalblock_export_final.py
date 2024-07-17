import numpy as np
from pymor.core.pickle import load
from os.path import isfile
from pandas import DataFrame as DF


max_batchsize = 30
# plot_batch = [1, 2, 4, 8, 16]

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

# fig, ax = plt.subplots(2,2)

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

        df = DF()
        df = df.assign(err=results['max_rel_errors'][0])
        df.to_csv('thermalblock_bs' + str(bs) + '.data', sep=',', index_label='n')

t_evaluate = np.array(t_evaluate)
t_extend = np.array(t_extend)
t_greedy = np.array(t_greedy)
t_offline = np.array(t_offline)
t_postprocess = np.array(t_postprocess)
t_reduce = np.array(t_reduce)
t_solve = np.array(t_solve)
t_online = np.array(t_online)

t_offline_n = t_offline/t_offline[0]
t_online_n = t_online/t_online[0]

t_other = t_offline - t_evaluate - t_extend - t_postprocess - t_reduce - t_solve

data = {'batchsizes': batchsizes,
        't_solve': t_solve, 't_evaluate': t_evaluate, 't_extend': t_extend, 't_reduce': t_reduce,
        't_postprocess': t_postprocess, 't_other': t_other,
        't_offline_n': t_offline_n, 't_online_n': t_online_n, 
        'num_ext': num_ext, 'num_iter': num_iter,}

df_seq = DF(data)
df_seq.to_csv('thermalblock_overall.data', index=False)
