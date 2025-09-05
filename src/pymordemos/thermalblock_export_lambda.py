import numpy as np
from pymor.core.pickle import load
from os.path import isfile, join
from os import listdir
from pandas import DataFrame as DF

file_string = 'thermalblock_2x2'

procs = 30

t_evaluate = []
t_extend = []
t_greedy = []
t_offline = []
t_reduce = []
t_solve = []
t_online = []
t_pod = []

lambda_tol = []
lambda_str = []

num_ext = []
num_iter = []
num_pp = []

ax00_y_max = 0

files = [f for f in listdir('src/pymordemos') if isfile(join('src/pymordemos', f))]
files = [f for f in files if file_string in f and 'lambda' in f]
files = [f for f in files if f'N{procs}' in f or 'B1' in f]
files = [f for f in files if 'POD' in f or 'B1' in f]

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

        # ax[0,0].semilogy(results['max_rel_errors'][0],label='classical')
        # if len(results['max_rel_errors'][0]) > ax00_y_max:
        #     ax00_y_max = len(results['max_rel_errors'][0])
        df = DF()
        df = df.assign(err=results['max_rel_errors'][0])
        df.to_csv('thermalblock_lambda_cwg.data', sep=',', index_label='n')

    else:

        timings = results['timings']

        t_evaluate.append(timings['evaluate'])
        t_extend.append(timings['extend'])
        t_greedy.append(timings['greedy'])
        t_offline.append(timings['offline'])
        t_reduce.append(timings['reduce'])
        t_solve.append(timings['solve'])
        t_online.append(timings['online'])
        t_pod.append(timings['pod'])

        num_ext.append(results['num_extensions'])
        num_iter.append(results['num_iterations'])

        lambda_val = results['settings']['lambda']
        lambda_tol.append(lambda_val)
        lambda_str.append(f"{lambda_val}")

        ### Subplot 00 (Upper left)
        # ax[0,0].semilogy(results['max_rel_errors'][0],':',label=f'$\lambda={lambda_val}$')
        # if len(results['max_rel_errors'][0]) > ax00_y_max:
        #     ax00_y_max = len(results['max_rel_errors'][0])
        df = DF()
        df = df.assign(err=results['max_rel_errors'][0])
        df.to_csv('thermalblock_pod_lambda' + str(lambda_val) + '.data', sep=',', index_label='n')


t_evaluate.append(rev_t_eveluate)
t_extend.append(rev_t_extend)
t_offline.append(rev_t_offline)
t_reduce.append(rev_t_reduce)
t_solve.append(rev_t_solve)
t_online.append(rev_t_online)
t_greedy.append(rev_t_greedy)
t_pod.append(0)
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
t_pod = np.array(t_pod)
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

eff_bs = num_ext/num_iter
rel_size = num_ext/num_ext[-1]

t_other = t_offline - t_evaluate - t_extend - t_reduce - t_solve

t_offline_n = t_offline/rev_t_offline
t_online_n = t_online/rev_t_online

data = {'lambda': lambda_tol,
        't_solve': t_solve, 't_evaluate': t_evaluate, 't_extend': t_extend, 't_reduce': t_reduce, 't_pod': t_pod, 't_other': t_other,
        't_offline': t_offline, 't_online': t_online, 't_offline_n': t_offline_n, 't_online_n': t_online_n,
        'num_ext': num_ext, 'num_iter': num_iter, 'eff_bs': eff_bs, 'rel_size': rel_size}

df_seq = DF(data)
df_seq.to_csv('thermalblock_pod_lambda_overall.data', index=False)

