import numpy as np
from pymor.core.pickle import load
from os.path import isfile
from pandas import DataFrame as DF


max_batchsize = 16
max_P = 30

p_opt = 30

t_evaluate = []
t_offline = []
t_offline_n = []
t_solve = []
t_online = []
t_online_n = []
num_ext = []
num_iter = []
num_post = []
batchsizes = []
batchsizes_str = []

t_evaluate_p = []
t_offline_p = []
t_offline_n_p = []
t_solve_p = []
t_online_p = []
t_online_n_p = []
num_ext_p = []
num_iter_p = []
num_post_p = []
batchsizes_p = []
batchsizes_str_p = []

t_evaluate_po = []
t_offline_po = []
t_offline_n_po = []
t_solve_po = []
t_online_po = []
t_online_n_po = []
num_ext_po = []
num_iter_po = []
num_post_po = []
batchsizes_po = []
batchsizes_str_po = []


for p in range(1, max_P+1):

    for bs in range(1, max_batchsize+1):

        file_string = f'src/pymordemos/thermalblock_2x2_N{p}_BS{bs}_nopp.pkl'

        if isfile(file_string):
            with open(file_string, 'rb') as f:
                results = load(f)

            timings = results['timings']

            if p==1:
                t_evaluate.append(timings['evaluate'])
                t_offline.append(timings['offline'])
                t_solve.append(timings['solve'])
                t_online.append(timings['online'])
                num_ext.append(results['num_extensions'])
                num_iter.append(results['num_iterations'])
                num_post.append(results['num_extensions'] - len(results['max_errs_pp']) + 1)
                batchsizes_str.append((str(bs)))
                batchsizes.append(bs)

                # df = DF()
                # df = df.assign(err=results['max_rel_errors'][0])
                # df.to_csv('thermalblock_bs' + str(bs) + '.data', sep=',')
            if p==bs:
                t_evaluate_p.append(timings['evaluate'])
                t_offline_p.append(timings['offline'])
                t_solve_p.append(timings['solve'])
                t_online_p.append(timings['online'])
                num_ext_p.append(results['num_extensions'])
                num_iter_p.append(results['num_iterations'])
                num_post_p.append(results['num_extensions'] - len(results['max_errs_pp']) + 1)
                batchsizes_str_p.append((str(bs)))
                batchsizes_p.append(bs)
            if p==p_opt:
                t_evaluate_po.append(timings['evaluate'])
                t_offline_po.append(timings['offline'])
                t_solve_po.append(timings['solve'])
                t_online_po.append(timings['online'])
                num_ext_po.append(results['num_extensions'])
                num_iter_po.append(results['num_iterations'])
                num_post_po.append(results['num_extensions'] - len(results['max_errs_pp']) + 1)
                batchsizes_str_po.append((str(bs)))
                batchsizes_po.append(bs)

                df = DF()
                df = df.assign(err=results['max_rel_errors'][0])
                df.to_csv('thermalblock_bs' + str(bs) + '_nopp.data', sep=',')

t_evaluate = np.array(t_evaluate)
t_offline = np.array(t_offline)
t_solve = np.array(t_solve)
t_online = np.array(t_online)
num_ext = np.array(num_ext)
num_iter = np.array(num_iter)
num_post = np.array(num_post)

t_evaluate_p = np.array(t_evaluate_p)
t_offline_p = np.array(t_offline_p)
t_solve_p = np.array(t_solve_p)
t_online_p = np.array(t_online_p)
num_ext_p = np.array(num_ext_p)
num_iter_p = np.array(num_iter_p)
num_post_p = np.array(num_post_p)

t_evaluate_po = np.array(t_evaluate_po)
t_offline_po = np.array(t_offline_po)
t_solve_po = np.array(t_solve_po)
t_online_po = np.array(t_online_po)
num_ext_po = np.array(num_ext_po)
num_iter_po = np.array(num_iter_po)
num_post_po = np.array(num_post_po)

t_other = t_offline - t_evaluate - t_solve
t_other_p = t_offline_p - t_evaluate_p - t_solve_p
t_other_po = t_offline_po - t_evaluate_po - t_solve_po

# t_offline_n = t_offline/t_offline[0]
# t_offline_n_p = t_offline_p/t_offline_p[0]
t_offline_n_po = t_offline_po/t_offline_po[0]

# t_online_n = t_online/t_online[0]
# t_online_n_p = t_online_p/t_online_p[0]
t_online_n_po = t_online_po/t_online_po[0]

data_sequential = {'batchsizes': batchsizes, 't_offline': t_offline, 't_offline_n': t_offline_n, 't_solve': t_solve, 't_evaluate': t_evaluate, 't_other': t_other, 't_online': t_online, 't_online_n': t_online_n, 'num_ext': num_ext, 'num_iter': num_iter, 'num_post': num_post}
data_parallel_b = {'batchsizes': batchsizes_p, 't_offline': t_offline_p, 't_offline_n': t_offline_n_p, 't_solve': t_solve_p,'t_evaluate': t_evaluate_p, 't_other': t_other_p, 't_online': t_online_p, 't_online_n': t_online_n_p, 'num_ext': num_ext_p, 'num_iter': num_iter_p, 'num_post': num_post_p}
data_parallel_opt = {'batchsizes': batchsizes_po, 't_offline': t_offline_po, 't_offline_n': t_offline_n_po, 't_solve': t_solve_po, 't_evaluate': t_evaluate_po, 't_other': t_other_po, 't_online': t_online_po, 't_online_n': t_online_n_po, 'num_ext': num_ext_po, 'num_iter': num_iter_po, 'num_post': num_post_po}

df_seq = DF(data_sequential)
df_b = DF(data_parallel_b)
df_opt = DF(data_parallel_opt)

df_seq.to_csv('thermalblock_seq_nopp.data', index=False)
df_b.to_csv('thermalblock_par_b_nopp.data', index=False)
df_opt.to_csv('thermalblock_par_opt_nopp.data', index=False)


