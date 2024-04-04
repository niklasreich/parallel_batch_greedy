#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys
import time
import pickle
import numpy as np
import scipy.sparse as sp
from datetime import datetime

from typer import Argument, Option, run

# from pymor.algorithms.error import plot_reduction_error_analysis, reduction_error_analysis, plot_batch_reduction
# from pymor.core.pickle import dump
# from pymor.parallel.default import new_parallel_pool, dummy_pool
# from pymor.tools.typer import Choices
# from pymor.algorithms.batchgreedy import rb_batch_greedy

# python -m thermalblock 3 2 5 100 5 --alg batch_greedy --plot-batch-comparison


def main(
    num_tests: int = Option(20, help='Number of splu tests')
):
    """scipy splu benchmark."""

    n = 10000
    nnz = 50000

    tic  = time.perf_counter()

    rhs = np.random.rand(n)

    for test in range(num_tests):

        data = np.concatenate((np.ones(n),np.random.rand(nnz-n)*1.1-0.1))
        row = np.concatenate((range(n),np.random.randint(0,n-1,nnz-n)))
        col = np.concatenate((range(n),np.random.randint(0,n-1,nnz-n)))
        matrix = sp.coo_matrix((data, (row, col)), shape=(n, n))
        matrix = sp.csc_matrix(matrix)

        tic  = time.perf_counter()

        factorization = sp.linalg.splu(matrix, permc_spec='COLAMD')
        sol = factorization.solve(rhs)

        toc  = time.perf_counter()

        print(f'Elapsed time: {toc-tic}')

        # timings = np.zeros(n_times)

        # fom, fom_summary = discretize_pymor(xblocks, yblocks, grid, use_list_vector_array=False)

        # parameter_space = fom.parameters.space(0.1, 1.)

        # for j in range(n_times):
        #     mu = parameter_space.sample_randomly()
        #     tic  = time.perf_counter()
        #     sol = fom.solve(mu)
        #     toc = time.perf_counter()
        #     timings[j] = toc - tic

        # with open(f'fomsolves_OMP{omp_threads}_grid{grid}.pkl', 'wb') as fp:
        #         pickle.dump(timings, fp)



def discretize_pymor(xblocks, yblocks, grid_num_intervals, use_list_vector_array):
    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.builtin import discretize_stationary_cg
    from pymor.discretizers.builtin.list import convert_to_numpy_list_vector_array

    print('Discretize ...')
    # setup analytical problem
    problem = thermal_block_problem(num_blocks=(xblocks, yblocks))

    # discretize using continuous finite elements
    fom, _ = discretize_stationary_cg(problem, diameter=1. / grid_num_intervals)

    if use_list_vector_array:
        fom = convert_to_numpy_list_vector_array(fom)

    summary = f'''pyMOR model:
   number of blocks: {xblocks}x{yblocks}
   grid intervals:   {grid_num_intervals}
   ListVectorArray:  {use_list_vector_array}
'''

    return fom, summary


def discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order):
    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        fom = mpi_wrap_model(lambda: _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order),
                             use_with=True, pickle_local_spaces=False)
    else:
        fom = _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order)

    summary = f'''FEniCS model:
   number of blocks:      {xblocks}x{yblocks}
   grid intervals:        {grid_num_intervals}
   finite element order:  {element_order}
'''

    return fom, summary


def _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order):

    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.fenics import discretize_stationary_cg

    print('Discretize ...')
    # setup analytical problem
    problem = thermal_block_problem(num_blocks=(xblocks, yblocks))

    # discretize using continuous finite elements
    fom, _ = discretize_stationary_cg(problem, diameter=1. / grid_num_intervals, degree=element_order)

    return fom


def reduce_naive(fom, reductor, parameter_space, basis_size):

    tic = time.perf_counter()

    training_set = parameter_space.sample_randomly(basis_size)

    for mu in training_set:
        reductor.extend_basis(fom.solve(mu), method='trivial')

    rom = reductor.reduce()

    elapsed_time = time.perf_counter() - tic

    summary = f'''Naive basis generation:
   basis size set: {basis_size}
   elapsed time:   {elapsed_time}
'''

    return rom, summary


def reduce_greedy(fom, reductor, parameter_space, snapshots_per_block,
                  extension_alg_name, max_extensions, use_error_estimator, pool):

    from pymor.algorithms.greedy import rb_greedy

    # run greedy
    training_set = parameter_space.sample_uniformly(snapshots_per_block)
    greedy_data = rb_greedy(fom, reductor, training_set,
                            use_error_estimator=use_error_estimator, error_norm=fom.h1_0_semi_norm,
                            extension_params={'method': extension_alg_name}, max_extensions=max_extensions,
                            pool=pool)
    rom = greedy_data['rom']

    # generate summary
    real_rb_size = rom.solution_space.dim
    training_set_size = len(training_set)
    summary = f'''Greedy basis generation:
   size of training set:   {training_set_size}
   error estimator used:   {use_error_estimator}
   extension method:       {extension_alg_name}
   prescribed basis size:  {max_extensions}
   actual basis size:      {real_rb_size}
   elapsed time:           {greedy_data["time"]}
'''

    return rom, summary


def reduce_batch_greedy(fom, reductor, parameter_space, snapshots_per_block,
                        extension_alg_name, max_extensions, use_error_estimator, pool,
                        batchsize, greedy_start, atol, parallel_batch):

    from pymor.algorithms.batchgreedy import rb_batch_greedy

    # run greedy
    training_set = parameter_space.sample_uniformly(snapshots_per_block)
    greedy_data = rb_batch_greedy(fom, reductor, training_set,
                                  use_error_estimator=use_error_estimator, error_norm=fom.h1_0_semi_norm,
                                  extension_params={'method': extension_alg_name}, max_extensions=max_extensions,
                                  pool=pool, batchsize=batchsize, greedy_start=greedy_start, atol=atol, parallel_batch=parallel_batch)
    rom = greedy_data['rom']

    # generate summary
    real_rb_size = rom.solution_space.dim
    training_set_size = len(training_set)
    summary = f'''Greedy basis generation:
   size of training set:   {training_set_size}
   error estimator used:   {use_error_estimator}
   extension method:       {extension_alg_name}
   prescribed basis size:  {max_extensions}
   actual basis size:      {real_rb_size}
   elapsed time:           {greedy_data["time"]}
'''

    return rom, summary, greedy_data


def reduce_adaptive_greedy(fom, reductor, parameter_space, validation_mus,
                           extension_alg_name, max_extensions, use_error_estimator,
                           rho, gamma, theta, pool):

    from pymor.algorithms.adaptivegreedy import rb_adaptive_greedy

    # run greedy
    greedy_data = rb_adaptive_greedy(fom, reductor, parameter_space, validation_mus=-validation_mus,
                                     use_error_estimator=use_error_estimator, error_norm=fom.h1_0_semi_norm,
                                     extension_params={'method': extension_alg_name}, max_extensions=max_extensions,
                                     rho=rho, gamma=gamma, theta=theta, pool=pool)
    rom = greedy_data['rom']

    # generate summary
    real_rb_size = rom.solution_space.dim
    # the validation set consists of `validation_mus` random parameters plus the centers of the
    # adaptive sample set cells
    validation_mus += 1
    summary = f'''Adaptive greedy basis generation:
   initial size of validation set:  {validation_mus}
   error estimator used:            {use_error_estimator}
   extension method:                {extension_alg_name}
   prescribed basis size:           {max_extensions}
   actual basis size:               {real_rb_size}
   elapsed time:                    {greedy_data["time"]}
'''

    return rom, summary


def reduce_pod(fom, reductor, parameter_space, snapshots_per_block, basis_size):
    from pymor.algorithms.pod import pod

    tic = time.perf_counter()

    training_set = parameter_space.sample_uniformly(snapshots_per_block)

    print('Solving on training set ...')
    snapshots = fom.operator.source.empty(reserve=len(training_set))
    for mu in training_set:
        snapshots.append(fom.solve(mu))

    print('Performing POD ...')
    basis, singular_values = pod(snapshots, modes=basis_size, product=reductor.products['RB'])

    print('Reducing ...')
    reductor.extend_basis(basis, method='trivial')
    rom = reductor.reduce()

    elapsed_time = time.perf_counter() - tic

    # generate summary
    real_rb_size = rom.solution_space.dim
    training_set_size = len(training_set)
    summary = f'''POD basis generation:
   size of training set:   {training_set_size}
   prescribed basis size:  {basis_size}
   actual basis size:      {real_rb_size}
   elapsed time:           {elapsed_time}
'''

    return rom, summary


if __name__ == '__main__':
    run(main)
