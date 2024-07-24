#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
# This file was adapted by Niklas Reich.

import time

from typer import Argument, run

from pymor.algorithms.batchgreedy import rb_batch_greedy
from pymor.algorithms.error import reduction_error_analysis
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.discretizers.builtin.list import convert_to_numpy_list_vector_array
from pymor.parallel.default import new_parallel_pool, dummy_pool
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import SimpleCoerciveRBReductor


def main(
    xblocks: int = Argument(..., help='Number of blocks in x direction.'),
    yblocks: int = Argument(..., help='Number of blocks in y direction.'),
    snapshots: int = Argument(
        ...,
        help='Number of training_set parameters per block\n\n'
    ),
    batchsize: int = Argument(..., help='Size of the (parallel) batch in each greedy iteration.')
):
    """Thermalblock script for the parallel batch greedy algorithm."""

    assert batchsize>=1, 'Batch size must be a positiv integer.'

    # Create a worker pool for parallel computing.
    # Without MPI a dummy pool is created which conincides with serial computation.
    pool = new_parallel_pool(allow_mpi=True)
    if pool is not dummy_pool:
        print(f'Using pool of {len(pool)} workers for parallelization.')
    else:
        print('No functional pool. Only dummy_pool is used.')

    # Static Parameters
    # (Remain unchanged)
    grid = 1000 # approx. number of nodes per dim.
    rtol = 1e-5 # rel tolerance for the greedy algorithm.
    rb_size = 500 # max. basis size. Chosen so big that we stop by rtol.
    test_snapshots = 100 # number of random parameters for error analysis
    test_online = 500 # number of random paramters for benchmarking reduced model.

    tic = time.perf_counter()

    fom, _ = discretize_pymor(xblocks, yblocks, grid, use_list_vector_array=False)
    parameter_space = fom.parameters.space(0.1, 1.) # Parameter domain
    fom.enable_caching('memory') # Allow caching.

    print('')
    print('')
    print('RB generation for batch size ' + str(batchsize) + ' ...')

    # define estimator for coercivity constant
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters)

    # inner product for computation of Riesz representatives
    product = fom.h1_0_semi_product

    # choice of error estimator
    reductor = SimpleCoerciveRBReductor(fom,
                                        product=product,
                                        coercivity_estimator=coercivity_estimator,
                                        check_orthonormality=False
                                        )

    training_set = parameter_space.sample_uniformly(snapshots)

    greedy_data = rb_batch_greedy(fom, reductor, training_set,
                                  use_error_estimator=True,
                                  error_norm=fom.h1_0_semi_norm,
                                  max_extensions=rb_size,
                                  pool=pool,
                                  batchsize=batchsize,
                                  rtol=rtol
                                  )

    toc = time.perf_counter()
    offline_time = toc - tic

    rom = greedy_data['rom']

    print('\nA posteriori error analysis:')
    test_sample = parameter_space.sample_randomly(test_snapshots)
    results = reduction_error_analysis(rom,
                                       fom=fom,
                                       reductor=reductor,
                                       error_estimator=True,
                                       error_norms=(fom.h1_0_semi_norm, fom.l2_norm),
                                       error_estimator_norm_index=0,
                                       test_mus=test_sample,
                                       basis_sizes=0,
                                       pool=None
                                       )
    print('')

    # Online time
    tic = time.perf_counter()
    for mu in parameter_space.sample_randomly(test_online):
        reductor.reconstruct(rom.solve(mu))
    toc = time.perf_counter()
    online_time = (toc - tic)/test_online

    # Saving everything necessary in 'results'
    results['num_extensions'] = greedy_data['extensions']
    results['num_iterations'] = greedy_data['iterations']
    results['max_errs_ext'] = greedy_data['max_errs_ext']
    results['times'] = greedy_data['greedytimes']
    results['times']['online'] = online_time
    results.pop('time', None)
    results['times']['offline'] = offline_time
    results['times']['other'] = (offline_time
                                 - results['times']['solve']
                                 - results['times']['evaluate']
                                 - results['times']['extend']
                                 - results['times']['reduce'])
    results['settings'] = {'grid': grid, 'rb_size': rb_size, 'rtol': rtol,
                           'test_snapshots': test_snapshots, 'n_online': test_online}

    # print a summary
    print(
        "\n" + "\033[1m"+"Summary:"+"\033[0m\n"
        "\u2533\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
        "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
        "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
        "\u2501\u2501\u2501\u2501\n"
        "\u2523\u2501 Configuration:\n"
        "\u2503  \u251c\u2500 " # indentation
        "Thermalblock:\n"
        "\u2503  \u2502  \u251c\u2500 " # indentation
        f"Blocks: {xblocks} by {yblocks}\n"
        "\u2503  \u2502  \u251c\u2500 " # indentation
        f"Values per block: {snapshots}\n"
        "\u2503  \u2502  \u251c\u2500 " # indentation
        f"Size of training set: {snapshots**(xblocks*yblocks)}\n"
        "\u2503  \u2502  \u2514\u2500 " # indentation
        f"Degrees of freedom: {fom.order}\n"
        "\u2503  \u251c\u2500 " # indentation
        "Batch greedy algorithm:\n"
        "\u2503  \u2502  \u251c\u2500 " # indentation
        f"Batchsize: {batchsize}\n"
        "\u2503  \u2502  \u2514\u2500 " # indentation
        f"Rel. target tolerance: {rtol}\n"
        "\u2503  \u251c\u2500 " # indentation
        "Size of test sets:\n"
        "\u2503  \u2502  \u251c\u2500 " # indentation
        f"Error analysis: {test_snapshots}\n"
        "\u2503  \u2502  \u2514\u2500 " # indentation
        f"Online benchmark: {test_online}\n"
        "\u2503  \u2514\u2500 " # indentation
        f"(MPI-)Pool: {len(pool)} parallel worker(s)\n"
        "\u2523\u2501 Computation times:\n"
        "\u2503  \u251c\u2500 " # indentation
        f"Offline: {results['times']['offline']:.4e} sec.\n"
        "\u2503  \u2502  \u251c\u2500 " # indentation
        f"Solve:    {results['times']['solve']:.4e} sec. - "
        f"{results['times']['solve']/results['times']['offline']*100:4.1f}%\n"
        "\u2503  \u2502  \u251c\u2500 " # indentation
        f"Evaluate: {results['times']['evaluate']:.4e} sec. - "
        f"{results['times']['evaluate']/results['times']['offline']*100:4.1f}%\n"
        "\u2503  \u2502  \u251c\u2500 " # indentation
        f"Extend:   {results['times']['extend']:.4e} sec. - "
        f"{results['times']['extend']/results['times']['offline']*100:4.1f}%\n"
        "\u2503  \u2502  \u251c\u2500 " # indentation
        f"Reduce:   {results['times']['reduce']:.4e} sec. - "
        f"{results['times']['reduce']/results['times']['offline']*100:4.1f}%\n"
        "\u2503  \u2502  \u2514\u2500 " # indentation
        f"Other:    {results['times']['other']:.4e} sec. - "
        f"{results['times']['other']/results['times']['offline']*100:4.1f}%\n"
        "\u2503  \u2514\u2500 " # indentation
        f"Online:  {results['times']['online']:.4e} sec. on average\n"
        "\u2523\u2501 Reduced basis:\n"
        "\u2503  \u251c\u2500 " # indentation
        f"Final basis size:  {rom.order:4n}\n"
        "\u2503  \u251c\u2500 " # indentation
        f"Greedy iterations: {results['num_iterations']:4n}\n"
        "\u2503  \u251c\u2500 " # indentation
        "Relative error decay:\n"
        "\u2503  \u2502       \u250a       rel. error\n"
        "\u2503  \u2502     n \u250a (h1_0_semi_norm)\n"
        "\u2503  \u2502  \u254c\u254c\u254c\u254c\u254c+"
        "\u254c\u254c\u254c\u254c\u254c\u254c\u254c\u254c\u254c\u254c\u254c\u254c"
        "\u254c\u254c\u254c\u254c\u254c\u254c"
    )
    for i in range(len(results['max_rel_errors'][0])):
        print(f"\u2503  \u2502  {i:4n} \u250a       {results['max_rel_errors'][0][i]:.4e}")
    print(
        "\u253b\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
        "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
        "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
        "\u2501\u2501\u2501\u2501\n\n"
    )


def discretize_pymor(xblocks, yblocks, grid_num_intervals, use_list_vector_array):

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


if __name__ == '__main__':
    run(main)
