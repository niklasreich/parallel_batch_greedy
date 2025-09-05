#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# mpiexec -n 30 python -m thermalblock_script 3 2 9 1
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0
mpiexec -n 30 python -m thermalblock_script 2 3 10 10 0
mpiexec -n 30 python -m thermalblock_script 2 2 25 10 0
# mpiexec -n 30 python -m thermalblock_script 3 3 5 1 1
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 1
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0.3
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0.1
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0.03
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0.01
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0.003
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0.001
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0.0003
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0.0001
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0.00003
# mpiexec -n 30 python -m thermalblock_script 3 3 5 30 0.00001
