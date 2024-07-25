# Parallel Batch Greedy Algorithm

This repository includes the code to reprodoce the results of the paper

*"A parallel batch greedy algorithm in reduced basis methods: Convergence rates and numerical results"*,  
Niklas Reich, Karsten Urban, Jürgen Vorloeper, 2024.  
arXiv: https://arxiv.org/abs/2407.11631  
doi: https://doi.org/10.48550/arXiv.2407.11631

## Licence

This code is build upon [pyMOR](https://pymor.org/) and therefore includes a full pyMOR distribution.

The authors of this repository created/adapted the follwoing files:

* src/batchgreedydemos/thermalblock.py
* src/pymor/algorithms/batchgreedy.py
* src/bindings/scipy.py

See these files for more information.

### pyMOR Licence

Copyright pyMOR developers and contributors. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

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

The benchmark problem that is described in the paper is found within the file `src/batchgreedydemos/thermalblock.py`. Execute the code with

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
