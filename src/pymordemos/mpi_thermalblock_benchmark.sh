#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# mpiexec -n 30 python -m thermalblock_script 3 2 9 1
# mpiexec -n 30 python -m thermalblock_script 2 2 15 1 1
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 1
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 0.3
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 0.1
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 0.03
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 0.01
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 0.003
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 0.001
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 0.0003
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 0.0001
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 0.00003
mpiexec -n 30 python -m thermalblock_script 2 2 25 30 0.00001
# mpiexec -n 30 python -m thermalblock_script 2 2 15 30 0.02
# mpiexec -n 30 python -m thermalblock_script 2 2 15 30 0.01
# mpiexec -n 30 python -m thermalblock_script 2 2 15 30 0.005
# mpiexec -n 30 python -m thermalblock_script 2 2 15 30 0.001
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 1
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 0.3
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 0.1
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 0.03
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 0.01
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 0.003
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 0.001
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 0.0003
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 0.0001
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 0.00003
mpiexec -n 30 python -m thermalblock_script 2 3 10 30 0.00001
