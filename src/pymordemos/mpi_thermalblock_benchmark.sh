#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# mpiexec -n 30 python -m thermalblock_script 3 2 9 1
mpiexec -n 8 python -m thermalblock_script 2 2 15 1 1
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 1
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.8
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.6
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.4
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.2
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.1
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.08
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.06
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.04
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.02
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.01
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.005
mpiexec -n 8 python -m thermalblock_script 2 2 15 8 0.001
