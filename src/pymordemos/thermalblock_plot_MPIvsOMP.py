import numpy as np
from pymor.core.pickle import load
from typer import run
import matplotlib.pyplot as plt
from os.path import isfile

def main():
    
    num_processors = 30

    p_mpi, p_omp = divisors(num_processors)

    t_evaluate = []
    t_extend = []
    t_greedy = []
    t_offline = []
    t_postprocess = []
    t_reduce = []
    t_solve = []
    batchsizes = []
    x_label = []

    iterations = []
    selections = []
    extensions = []
    after_pp = []

    for i in range(len(p_mpi)):

        n_mpi = p_mpi[i]
        n_omp = p_omp[i]

        file_string = f'src/pymordemos/thermalblock_2x2_N{n_mpi}_BS{n_mpi}.pkl'

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
            x_label.append(fr'${n_mpi} \times {n_omp}$')
            batchsizes.append(n_mpi)

            iterations.append(results['num_iterations'])
            selections.append(results['num_iterations'] * n_mpi)
            extensions.append(results['num_extensions'])
            after_pp.append(results['num_extensions'] - len(results['max_errs_pp']) + 1)

    t_evaluate = np.array(t_evaluate)
    t_extend = np.array(t_extend)
    t_greedy = np.array(t_greedy)
    t_offline = np.array(t_offline)
    t_postprocess = np.array(t_postprocess)
    t_reduce = np.array(t_reduce)
    t_solve = np.array(t_solve)
    t_other = t_offline - t_evaluate - t_extend - t_postprocess - t_reduce - t_solve

    iterations = np.array(iterations)
    selections = np.array(selections)
    extensions = np.array(extensions)
    after_pp = np.array(after_pp)

    timings = {'solve': t_solve, 'evaluate': t_evaluate, 'extend': t_extend, 'reduce': t_reduce, 'postprocess': t_postprocess, 'other': t_other}
    stats = {'greedy iterations': iterations, 'selected params': selections, 'prelim. basis size': extensions, 'basis size after postproc.': after_pp}
    
    fig, ax = plt.subplots(2,1)
    bottom = np.zeros(len(x_label))
    width = 0.5
    for title, time in timings.items():
        p = ax[0].bar(x_label, time, width, label=title, bottom=bottom)
        bottom += time
    ax[0].grid(axis='y')
    ax[0].set_axisbelow(True)
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel('offline time in s')
    ax[0].set_xlabel(r'$(n_{mpi}=b) \times (n_{omp})$')

    x = np.arange(len(x_label))
    width = 0.2
    multiplier = 0
    for title, stat in stats.items():
        offset = width*multiplier
        rects = ax[1].bar(x+offset, stat, width, label=title,)
        ax[1].bar_label(rects, padding=3)
        multiplier += 1
    ax[1].set_xticks(x + 1.5*width, x_label)
    ax[1].grid(axis='y')
    ax[1].set_axisbelow(True)
    ax[1].legend(loc="upper left")
    ax[1].set_xlabel(r'$(n_{mpi}=b) \times (n_{omp})$')
    ax[1].set_ylim([0,100])

    plt.show()

def divisors(x):
    """Calculate devisors of (integer) x."""

    x = int(x) 

    left = []
    right = []

    for candidate in range(1,x+1):
        frac, whole = np.modf(x/candidate)
        if frac < np.finfo(float).eps:
            left.append(candidate)
            right.append(int(whole))

    return left, right

if __name__ == '__main__':
    run(main)