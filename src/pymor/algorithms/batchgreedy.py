# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time

import numpy as np
import multiprocessing as mp

from pymor.core.base import BasicObject, abstractmethod
from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.interface import RemoteObject
from pymor.parallel.manager import RemoteObjectManager
from pymor.parallel.mpi import MPIPool
from mpi4py import MPI


def weak_batch_greedy(surrogate, training_set, atol=None, rtol=None, max_extensions=None, pool=None,
                      batchsize=None, greedy_start=None, lambda_tol=0.5):
    """Weak greedy basis generation algorithm :cite:`BCDDPW11`.

    This algorithm generates an approximation basis for a given set of vectors
    associated with a training set of parameters by iteratively evaluating a
    :class:`surrogate <WeakGreedySurrogate>` for the approximation error on
    the training set and adding the worst approximated vector (according to
    the surrogate) to the basis.

    The constructed basis is extracted from the surrogate after termination
    of the algorithm.

    Parameters
    ----------
    surrogate
        An instance of :class:`WeakGreedySurrogate` representing the surrogate
        for the approximation error.
    training_set
        The set of parameter samples on which to perform the greedy search.
    atol
        If not `None`, stop the algorithm if the maximum (estimated) error
        on the training set drops below this value.
    rtol
        If not `None`, stop the algorithm if the maximum (estimated)
        relative error on the training set drops below this value.
    max_extensions
        If not `None`, stop the algorithm after `max_extensions` extension
        steps.
    pool
        If not `None`, a |WorkerPool| to use for parallelization. Parallelization
        needs to be supported by `surrogate`.

    Returns
    -------
    Dict with the following fields:

        :max_errs:               Sequence of maximum estimated errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
        :extensions:             Number of performed basis extensions.
        :time:                   Total runtime of the algorithm.
    """

    if batchsize is None:
        batchsize = 1

    logger = getLogger('pymor.algorithms.greedy.weak_greedy')
    training_set = list(training_set)
    logger.info(f'Started batch greedy search on training set of size {len(training_set)}.')

    tic = time.perf_counter()
    if not training_set:
        logger.info('There is nothing else to do for an empty training set.')
        return {'max_errs': [], 'max_err_mus': [], 'extensions': 0,
                'time': time.perf_counter() - tic}

    # parallel_batch = False
    if pool is None:
        pool = dummy_pool

    # Distribute the training set evenly among the workers.
    training_set_rank = pool.scatter_list(training_set)

    extensions = 0
    iterations = 0
    max_errs_ext = []
    max_err_mus_ext = []
    max_errs_iter = []
    max_err_mus_iter = []
    appended_mus = []

    stopped = False
    while not stopped:
        if extensions==0:
            with logger.block('Estimating errors ...'):
                this_i_errs = surrogate.evaluate(training_set_rank, return_all_values=True)
        with logger.block('Determine batch ...'):
            this_i_mus = []
            this_batch = []
            for i in range(batchsize):
                max_ind = np.argmax(this_i_errs)
                this_batch.append(max_ind)
                if i == 0:  # only once per batch -> once every greedy iteration
                    max_err = this_i_errs[max_ind]
                    max_err_mu = training_set[max_ind]
                    max_errs_iter.append(max_err)
                    max_err_mus_iter.append(max_err_mu)

                # for every mu of the batch -> once every basis extension
                max_errs_ext.append(this_i_errs[max_ind])
                max_err_mus_ext.append(training_set[max_ind])

                this_i_mus.append(training_set[max_ind])
                this_i_errs[max_ind] = 0

                appended_mus.append(training_set[max_ind])
                
                if greedy_start == 'single_zero' and (extensions == 0) and (iterations == 0):
                    break

        logger.info(f'Maximum error after {iterations} iterations ({extensions} extensions): {max_err} (mu = {max_err_mu})')

        if atol is not None and max_err <= atol:
            logger.info(f'Absolute error tolerance ({atol}) reached! Stopping extension loop.')
            stopped = True
            break

        if rtol is not None and max_err / max_errs_iter[0] <= rtol:
            logger.info(f'Relative error tolerance ({rtol}) reached! Stopping extension loop.')
            stopped = True
            break

        # Compute snapshots in parallel
        Us = surrogate.parallel_compute(this_i_mus)

        # Extend with first snapshot of the batch
        with logger.block(f'Extending with the first of the batch...'):
            successful_first = surrogate.extend_U(Us[0])
            
        if successful_first:
            extensions += 1
        else:
            stopped = True
            break

        # Try the rest of the batch
        with logger.block(f'Extending with the rest of the batch...'):
            for batch_extensions in range(1,batchsize):
                with logger.block('Estimating errors ...'):
                    this_i_errs = surrogate.evaluate(training_set_rank, return_all_values=True)
                    batch_errs = this_i_errs[this_batch]
                    max_err = np.max(this_i_errs)
                    max_ind = np.argmax(batch_errs)
                    max_batch_err = batch_errs[max_ind]

                    if atol is not None and max_err <= atol:
                        logger.info(f'Absolute error tolerance ({atol}) reached! Stopping extension loop.')
                        stopped = True
                        break

                    if rtol is not None and max_err / max_errs_iter[0] <= rtol:
                        logger.info(f'Relative error tolerance ({rtol}) reached! Stopping extension loop.')
                        stopped = True
                        break

                    # lambda_criteria
                    if max_batch_err >= lambda_tol*max_err:
                        successful = surrogate.extend_U(Us[max_ind])
                        extensions += successful
                    else:
                        successful = False
                if successful:
                    continue
                break
        
        iterations += 1

        logger.info('')
        if max_extensions is not None and extensions >= max_extensions:
            logger.info(f'Maximum number of {max_extensions} extensions reached.')
            stopped = True
            break

    toc = time.perf_counter()
    logger.info(f'Greedy search took {toc - tic} seconds with an effective batch size of {1.*extensions/iterations}.')
    timings = surrogate.times
    timings['greedy'] = toc - tic
    return {'max_errs_iter': max_errs_iter, 'max_err_mus_iter': max_err_mus_iter,
            'max_errs_ext': max_errs_ext, 'max_err_mus_ext': max_err_mus_ext,
            'extensions': extensions, 'iterations': iterations,
            'timings': timings}


class WeakGreedySurrogate(BasicObject):
    """Surrogate for the approximation error in :func:`weak_greedy`."""

    @abstractmethod
    def evaluate(self, mus, return_all_values=False):
        """Evaluate the surrogate for given parameters.

        Parameters
        ----------
        mus
            List of parameters for which to estimate the approximation
            error. When parallelization is used, `mus` can be a |RemoteObject|.
        return_all_values
            See below.

        Returns
        -------
        If `return_all_values` is `True`, an |array| of the estimated errors.
        If `return_all_values` is `False`, the maximum estimated error as first
        return value and the corresponding parameter as second return value.
        """
        pass

    @abstractmethod
    def extend(self, mu):
        pass


def rb_batch_greedy(fom, reductor, training_set, use_error_estimator=True, error_norm=None,
                    atol=None, rtol=None, max_extensions=None, extension_params=None, pool=None,
                    batchsize=None, greedy_start=None, lambda_tol=0.5):
    """Weak Greedy basis generation using the RB approximation error as surrogate.

    This algorithm generates a reduced basis using the :func:`weak greedy <weak_greedy>`
    algorithm :cite:`BCDDPW11`, where the approximation error is estimated from computing
    solutions of the reduced order model for the current reduced basis and then estimating
    the model reduction error.

    Parameters
    ----------
    fom
        The |Model| to reduce.
    reductor
        Reductor for reducing the given |Model|. This has to be
        an object with a `reduce` method, such that `reductor.reduce()`
        yields the reduced model, and an `exted_basis` method,
        such that `reductor.extend_basis(U, copy_U=False, **extension_params)`
        extends the current reduced basis by the vectors contained in `U`.
        For an example see :class:`~pymor.reductors.coercive.CoerciveRBReductor`.
    training_set
        The training set of |Parameters| on which to perform the greedy search.
    use_error_estimator
        If `False`, exactly compute the model reduction error by also computing
        the solution of `fom` for all |parameter values| of the training set.
        This is mainly useful when no estimator for the model reduction error
        is available.
    error_norm
        If `use_error_estimator` is `False`, use this function to calculate the
        norm of the error. If `None`, the Euclidean norm is used.
    atol
        See :func:`weak_greedy`.
    rtol
        See :func:`weak_greedy`.
    max_extensions
        See :func:`weak_greedy`.
    extension_params
        `dict` of parameters passed to the `reductor.extend_basis` method.
        If `None`, `'gram_schmidt'` basis extension will be used as a default
        for stationary problems (`fom.solve` returns `VectorArrays` of length 1)
        and `'pod'` basis extension (adding a single POD mode) for instationary
        problems.
    pool
        See :func:`weak_greedy`.

    Returns
    -------
    Dict with the following fields:

        :rom:                    The reduced |Model| obtained for the
                                 computed basis.
        :max_errs:               Sequence of maximum errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
        :extensions:             Number of performed basis extensions.
        :time:                   Total runtime of the algorithm.
    """
    surrogate = RBSurrogate(fom, reductor, use_error_estimator, error_norm, extension_params, pool or dummy_pool)

    result = weak_batch_greedy(surrogate, training_set, atol=atol, rtol=rtol, max_extensions=max_extensions, pool=pool,
                               batchsize=batchsize, greedy_start=greedy_start, lambda_tol=lambda_tol)
    result['rom'] = surrogate.rom
    result['greedytimes'] = surrogate.times

    return result


class RBSurrogate(WeakGreedySurrogate):
    """Surrogate for the :func:`weak_greedy` error used in :func:`rb_greedy`.

    Not intended to be used directly.
    """

    def __init__(self, fom, reductor, use_error_estimator, error_norm, extension_params, pool):
        self.__auto_init(locals())
        if use_error_estimator:
            self.remote_fom, self.remote_error_norm, self.remote_reductor = None, None, None
        else:
            self.remote_fom, self.remote_error_norm, self.remote_reductor = \
                pool.push(fom), pool.push(error_norm), pool.push(reductor)
        self.remote_fom = pool.push(fom)
        self.rom = None

        self.times = {'evaluate': 0, 'extend': 0, 'reduce': 0, 'solve': 0}


    def evaluate(self, mus, return_all_values=False):
        tic = time.perf_counter()
        if self.rom is None:
            with self.logger.block('Reducing ...'):
                self.rom = self.reductor.reduce()

        if not isinstance(mus, RemoteObject):
            mus = self.pool.scatter_list(mus)

        result = self.pool.apply(_rb_surrogate_evaluate,
                                 rom=self.rom,
                                 fom=self.remote_fom,
                                 reductor=self.remote_reductor,
                                 mus=mus,
                                 error_norm=self.remote_error_norm,
                                 return_all_values=return_all_values,
                                 use_error_estimator=self.use_error_estimator)
        toc = time.perf_counter()
        self.times['evaluate'] += toc - tic
        if return_all_values:
            return np.hstack(result)
        else:
            errs, max_err_mus = list(zip(*result))
            max_err_ind = np.argmax(errs)
            return errs[max_err_ind], max_err_mus[max_err_ind]

    def extend(self, mu):
        mus = mu if isinstance(mu, list) else [mu]
        if len(mus) == 1:
            msg = f'Computing solution snapshot for mu = {mus[0]} ...'
        else:
            msg = f'Computing solution snapshots for mu = {", ".join(str(mu) for mu in mus)} ...'
        tic = time.perf_counter()
        with self.logger.block(msg):
            Us = self.pool.map(_parallel_solve, mus, fom=self.remote_fom)
        toc = time.perf_counter()
        self.times['solve'] += toc - tic

        tic = time.perf_counter()
        successful_extensions = 0
        for U in Us:
            with self.logger.block('Extending basis with solution snapshot ...'):
                extension_params = self.extension_params
                if len(U) > 1:
                    if extension_params is None:
                        extension_params = {'method': 'pod'}
                    else:
                        extension_params.setdefault('method', 'pod')
                try:
                    self.reductor.extend_basis(U, copy_U=False, **(extension_params or {}))
                    successful_extensions += 1
                except ExtensionError:
                    self.logger.info('Extension failed.')
        toc = time.perf_counter()
        self.times['extend'] += toc -tic

        if not successful_extensions:
            self.logger.info('All extensions failed.')
            raise ExtensionError

        tic = time.perf_counter()
        if not self.use_error_estimator:
            self.remote_reductor = self.pool.push(self.reductor)
        with self.logger.block('Reducing ...'):
            self.rom = self.reductor.reduce()
        toc = time.perf_counter()
        self.times['reduce'] += toc - tic

        return successful_extensions
    
    def extend_U(self, U):
        assert len(U)==1
        tic = time.perf_counter()
        successful = False
        with self.logger.block('Extending basis with solution snapshot ...'):
            extension_params = self.extension_params
            try:
                self.reductor.extend_basis(U, copy_U=False, **(extension_params or {}))
                successful = True
            except ExtensionError:
                self.logger.info('Extension failed.')
        toc = time.perf_counter()
        self.times['extend'] += toc -tic

        if successful:
            tic = time.perf_counter()
            if not self.use_error_estimator:
                self.remote_reductor = self.pool.push(self.reductor)
            with self.logger.block('Reducing ...'):
                self.rom = self.reductor.reduce()
            toc = time.perf_counter()
            self.times['reduce'] += toc - tic

        return successful
    
    def parallel_compute(self, mu):
        mus = mu if isinstance(mu, list) else [mu]
        if len(mus) == 1:
            msg = f'Computing solution snapshot for mu = {mus[0]} ...'
        else:
            msg = f'Computing solution snapshots for mu = {", ".join(str(mu) for mu in mus)} ...'
        tic = time.perf_counter()
        with self.logger.block(msg):
            Us = self.pool.map(_parallel_solve, mus, fom=self.remote_fom)
        toc = time.perf_counter()
        self.times['solve'] += toc - tic
        return Us


def _rb_surrogate_evaluate(rom=None, fom=None, reductor=None, mus=None, error_norm=None,
                           return_all_values=False, use_error_estimator=False):
    if not mus:
        if return_all_values:
            return []
        else:
            return -1., None
        
    if fom is None:
        use_error_estimator = True

    if use_error_estimator:
        errors = [rom.estimate_error(mu) for mu in mus]
    elif error_norm is not None:
        errors = [error_norm(fom.solve(mu) - reductor.reconstruct(rom.solve(mu))) for mu in mus]
    else:
        errors = [(fom.solve(mu) - reductor.reconstruct(rom.solve(mu))).norm() for mu in mus]
    # most error_norms will return an array of length 1 instead of a number,
    # so we extract the numbers if necessary
    errors = [x[0] if hasattr(x, '__len__') else x for x in errors]
    if return_all_values:
        return errors
    else:
        max_err_ind = np.argmax(errors)
        return errors[max_err_ind], mus[max_err_ind]


def _parallel_solve(mu, fom=None):
    return fom.solve(mu)
