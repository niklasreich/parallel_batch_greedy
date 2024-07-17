import numpy as np
from pymor.core.pickle import load
import matplotlib.pyplot as plt
from os.path import isfile

omp_num_threads = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16])
grids = np.array([50, 100, 200, 400, 600, 800, 1000, 1200])

timings_min = np.zeros((len(omp_num_threads),len(grids)))
timings_max = np.zeros((len(omp_num_threads),len(grids)))
timings_mean = np.zeros((len(omp_num_threads),len(grids)))

for i in range(len(omp_num_threads)):
    for j in range(len(grids)):

        n_omp = omp_num_threads[i]
        n_grid = grids[j]

        file_string = f'src/pymordemos/fomsolves_OMP{n_omp}_grid{n_grid}.pkl'

        if isfile(file_string):
            with open(file_string, 'rb') as f:
                results = load(f)
            
            timings_min[i,j] = np.min(results)
            timings_max[i,j] = np.max(results)
            timings_mean[i,j] = np.mean(results)

plt.rcParams['figure.constrained_layout.use'] = True
for j in range(len(grids)):

    # Normalizing
    timings_min[:,j] /= timings_min[0,j]
    timings_mean[:,j] /= timings_mean[0,j]

    n_plot = 240 + j + 1

    plt.subplot(n_plot)
    plt.plot(omp_num_threads,timings_min[:,j],'x:',label=f'min')
    plt.plot(omp_num_threads,timings_mean[:,j],'x:',label=f'mean')
    plt.legend(loc="upper left")
    plt.title(f'grid={grids[j]}')
    # plt.ylabel('time in s')
    plt.xlabel('OMP_NUM_THREADS')
    plt.xticks(omp_num_threads)


plt.show()