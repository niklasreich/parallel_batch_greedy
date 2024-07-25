# Parallel Batch Greedy Algorithm

This repository includes the code to reprodoce the results of the paper

*"A parallel batch greedy algorithm in reduced basis methods: Convergence rates and numerical results"*,  
Niklas Reich, Karsten Urban, Jürgen Vorloeper, 2024,  
arXiv: https://arxiv.org/abs/2407.11631  
doi: https://doi.org/10.48550/arXiv.2407.11631

## Licence

TODO

## Installation
### Necessary Packages

This software has been developed with Python 3.10.
We recommend installation via [pip](https://pip.pypa.io/en/stable/) in a [virtual environment](https://virtualenv.pypa.io/en/latest/).
To install this software clone this repository or download it. When you navigated to the top level of your local copy use

    pip install -e .

to install all necessary packages to run the code.  

### Optional Packages
To achieve similar results to the paper you need to install two additional optional software components/packages.

#### MPI & mpi4py
[MPI](https://www.mpi-forum.org/) is needed to compute the batch in parallel as intended. For more information on how to install MPI see [here](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html). For many Linux distributions it is already installed.  
To use MPI with Python we need an interface from the mpi4py package that can be installed via

    pip install mpi4py

#### SuiteSparse & scikit-umfpack
[SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) is "a suite of sparse matrix algorithms". Among others, it includes *UMFPACK*, a multifrontal LU factorization. This implementation can be used instead of the standard implementation of `splu` by SciPy. How SuiteSparse & scikit-umfpack can be installed is described [here](https://scikit-umfpack.github.io/scikit-umfpack/install.html). If the software is installed correctly, the UMFPACK-implementation is used automatically.

## Start the Benchmark

The benchmark problem that is described in the paper is found within the file `src/batchgreedydemos/thermalblock.py`. After navigating to the directory `src/batchgreedydemos/`, execute the code with

    python thermalblock.py [xblocks] [yblocks] [snapshots] [batchsize]

Here `[xblocks]` is the number of blocks in x direction, `[yblocks]` is the number of blocks in y direction, `[snapshots]` is the number of discrete values per block for the thermal conductivity and `[batchsize]` is the batchsize of the parallel greedy algorithm. By setting `[batchsize]` to `1` we get an classical weak greedy algorithm.

When the code runs successfully it will put out a text-based summary in the end which sums up the used configuration as well as the results.

If you have MPI installed you can leverage a parallel worker pool by executing

    mpiexec -n [numproc] python thermalblock.py [xblocks] [yblocks] [snapshots] [batchsize]

where `[numproc]` is the number of workers.

To give a concrete example the results of the paper were created by executing

    mpiexec -n 30 python thermalblock.py 2 2 25 [batchsize]
<!-- tsk -->
    mpiexec -n 30 python thermalblock.py 3 3 5 [batchsize]
    

and `[batchsize]` was set to `1`, ... , `16`.

### Smaller Test configuration

If you just want to make sure that the code runs, you can use

    python thermalblock.py [xblocks] [yblocks] [snapshots] [batchsize] --test-config

This changes some otherwise static parameters, so that the benchmark finishes much faster.
For example

    python thermalblock.py 2 2 5 3 --test-config

should finish in under a minute.

Of course, when using the test configuration the results are not related to the results 
presented in the paper.
