# Parallel Batch Greedy Algorithm

This repository includes the code to reproduce the results of the paper

*"A parallel batch greedy algorithm in reduced basis methods: Convergence rates and numerical results"*,  
Niklas Reich, Karsten Urban, Jürgen Vorloeper, 2025.  
arXiv: https://arxiv.org/abs/2407.11631  
doi: https://doi.org/10.48550/arXiv.2407.11631

## License

This code is built upon [pyMOR](https://pymor.org/) and therefore includes a full pyMOR distribution.

The authors of this repository created the following files:

* src/batchgreedydemos/thermalblock.py
* src/pymor/algorithms/batchgreedy.py

See these files for more information.

### pyMOR License

Copyright pyMOR developers and contributors. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following
  disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The following files contain source code originating from other open source software projects:

* docs/source/pymordocstring.py  (sphinxcontrib-napoleon)
* src/pymor/algorithms/genericsolvers.py (SciPy)

See these files for more information.

## Desciption of Code

### Changed & Created Files
#### src/batchgreedydemos/thermalblock.py
Implementation of the model problem introduced in the paper. Arguments allow to vary the number of blocks in the domain, as well as the number of discrete values per block for the thermal conductivity. This file is meant to be executed (see below).
#### src/pymor/algorithms/batchgreedy.py
Implementation of the parallel batch greedy algorithm as presented in the paper (see Algorithm 3 an 4).

### Other files
All other code files stem from the used pyMOR distribution. We refer to the [official documentation](https://docs.pymor.org/2024-1-0/index.html).

## Installation
### Necessary Packages

This software has been developed with Python 3.10.
We recommend an installation via [pip](https://pip.pypa.io/en/stable/) in a [virtual environment](https://virtualenv.pypa.io/en/latest/).
To install this software, clone this repository or download it. When you have navigated to the top level of your local copy, use

    pip install -e .

to install all the necessary packages to run the code.  

### Optional Packages
To reproduce the results of the paper, you need to install optional software components/packages.

#### MPI & mpi4py
[MPI](https://www.mpi-forum.org/) is needed to compute the batch in parallel, as intended. For more information on how to install MPI see [here](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html). For many Linux distributions, it is already installed.  
To use MPI with Python we need an interface from the mpi4py package that can be installed via

    pip install mpi4py

## Start the Benchmark

The benchmark problem that is described in the paper is found within the file `src/batchgreedydemos/thermalblock.py`. After navigating to the directory `src/batchgreedydemos/`, execute the code with

    python thermalblock.py [xblocks] [yblocks] [snapshots] [batchsize] [use_pod] [lambda_tol]

Here `[xblocks]` is the number of blocks in x direction, `[yblocks]` is the number of blocks in y direction, `[snapshots]` is the number of discrete values per block for the thermal conductivity, and `[batchsize]` is the batch size of the parallel greedy algorithm. By setting `[batchsize]` to `1` we get a classical weak greedy algorithm. By setting `[use_pod]` to `1` the POD-variant of the batch greedy alorithm is selected; with `0` the bulk version is used. For the POD version `lambda_tol` sets the relative tolerance for POD; otherwise `lambda_tol` is the bulk parameter.

When the code runs successfully, it will output a text-based summary at the end, which sums up the used configuration as well as the results.

If you have MPI installed, you can leverage a parallel worker pool by executing

    mpiexec -n [numproc] python thermalblock.py [xblocks] [yblocks] [snapshots] [batchsize]

where `[numproc]` is the number of workers.

To give a concrete example, the results of the paper were created by executing

    mpiexec -n 30 python thermalblock.py 2 2 25 30 [use_pod] [lambda_tol]
<!-- tsk -->
    mpiexec -n 30 python thermalblock.py 2 3 10 30 [use_pod] [lambda_tol]
    

### Smaller Test configuration

If you just want to make sure that the code runs, you can use

    python thermalblock.py [xblocks] [yblocks] [snapshots] [batchsize] [use_pod] [lambda_tol] --test-config

This changes some otherwise static parameters[^1], so that the benchmark finishes much faster.
For example

    python thermalblock.py 2 2 5 3 [use_pod] [lambda_tol] --test-config

should finish in under a minute.

Of course, when using the test configuration, the results are not related to the results 
presented in the paper.

[^1]: The order of the full model is reduced (coarser spatial discretization), the size of the test set for the error analysis is reduced, and the size of the test set for the benchmarking of the reduced model is lowered.
