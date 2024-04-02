#!/bin/bash
export OMP_NUM_THREADS = 1
python thermalblock_fomsolve.py 2 2 1
export OMP_NUM_THREADS = 2
python thermalblock_fomsolve.py 2 2 2
export OMP_NUM_THREADS = 4
python thermalblock_fomsolve.py 2 2 4
export OMP_NUM_THREADS = 6
python thermalblock_fomsolve.py 2 2 6
export OMP_NUM_THREADS = 8
python thermalblock_fomsolve.py 2 2 8
export OMP_NUM_THREADS = 10
python thermalblock_fomsolve.py 2 2 10
export OMP_NUM_THREADS = 12
python thermalblock_fomsolve.py 2 2 12
export OMP_NUM_THREADS = 14
python thermalblock_fomsolve.py 2 2 14
export OMP_NUM_THREADS = 16
python thermalblock_fomsolve.py 2 2 16
