from pymor.basic import *
from pymor.discretizers.builtin.cg import discretize_stationary_cg as discretizer
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.parallel.default import new_parallel_pool, dummy_pool
from itertools import product
import numpy as np

set_log_levels({'pymor': 'INFO'})

def parameter_functional_factory(ix, iy, num_blocks):
    return ProjectionParameterFunctional('diffusion',
                                            size=num_blocks[0]*num_blocks[1],
                                            index=ix + iy*num_blocks[0],
                                            name=f'diffusion_{ix}_{iy}')

def diffusion_function_factory(ix, iy, num_blocks):
    if ix + 1 < num_blocks[0]:
        X = '(x[0] >= ix * dx) * (x[0] < (ix + 1) * dx)'
    else:
        X = '(x[0] >= ix * dx)'
    if iy + 1 < num_blocks[1]:
        Y = '(x[1] >= iy * dy) * (x[1] < (iy + 1) * dy)'
    else:
        Y = '(x[1] >= iy * dy)'
    return ExpressionFunction(f'{X} * {Y} * 1.',
                                2, {}, {'ix': ix, 'iy': iy, 'dx': 1. / num_blocks[0], 'dy': 1. / num_blocks[1]},
                                name=f'diffusion_{ix}_{iy}')

diameter = 1/36  # comparable to original paper 
num_snapshots = 5   # same as paper (creates 12x12 grid)
ei_size = 25  # maximum number of bases in EIM
rb_size = 25  # maximum number of bases in RBM
test_snapshots = 15 # same as paper (creates 15x15 grid)
num_blocks = [2, 2]
batchsize = 5

pool = new_parallel_pool(allow_mpi=True)
if pool is not dummy_pool:
    print(f'Using pool of {len(pool)} workers for parallelization.')
else:
    print(f'No functional pool. Only dummy_pool is used.')

domain = RectDomain(([0,0], [1,1]))
rhs = ExpressionFunction('100 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])', dim_domain = 2)
parameters = Parameters({'reaction': 1})
diffusion = LincombFunction([diffusion_function_factory(ix, iy, num_blocks)
                                   for iy, ix in product(range(num_blocks[1]), range(num_blocks[0]))],
                                  [parameter_functional_factory(ix, iy, num_blocks)
                                   for iy, ix in product(range(num_blocks[1]), range(num_blocks[0]))],
                                  name='diffusion')
nonlinear_reaction_coefficient = ConstantFunction(1,2)
test_nonlinearreaction = ExpressionFunction('reaction[0] * (exp(u[0]) - 1)', dim_domain = 1, parameters = parameters, variable = 'u')
test_nonlinearreaction_derivative = ExpressionFunction('reaction[0] * exp(u[0])', dim_domain = 1, parameters = parameters, variable = 'u')
problem = StationaryProblem(domain = domain,
                            rhs = rhs,
                            diffusion = diffusion,
                            nonlinear_reaction_coefficient = nonlinear_reaction_coefficient,
                            nonlinear_reaction = test_nonlinearreaction,
                            nonlinear_reaction_derivative = test_nonlinearreaction_derivative,
                            name=f'NonlinearThermalblock({num_blocks})')
grid, boundary_info = discretize_domain_default(problem.domain, diameter=diameter)
print('Anzahl Element', grid.size(0))
print('Anzahl DoFs', grid.size(2))
fom, data = discretizer(problem, diameter = diameter)

# cache_id = (f'pymordemos.nonlinear_reaction {ei_snapshots} {test_snapshots}')
fom.enable_caching('memory')

parameter_space = fom.parameters.space({'diffusion': (0.1, 10), 'reaction': (0.01, 10)})
parameter_sample = parameter_space.sample_uniformly(num_snapshots)

print('RB generation ...')

reductor = StationaryRBReductor(fom)

greedy_data = rb_greedy(fom, reductor, parameter_sample,
                        use_error_estimator=False,
                        error_norm=fom.h1_0_semi_norm,
                        max_extensions=rb_size,
                        rtol=1e-5,
                        #batchsize=batchsize,
                        #postprocessing=False,
                        pool=pool)

rom = greedy_data['rom']

# print('Testing ROM...')

# max_err = -1
# for mu in test_sample:
#     u_fom = fom.solve(mu)
#     u_rom = rom.solve(mu)
#     this_diff = u_fom - reductor.reconstruct(u_rom)
#     this_err = this_diff.norm(fom.h1_0_semi_product)[0]
#     if this_err > max_err: max_err = this_err

# rel_error = max_err.item()/u_max_norm
print(f'RB size N: {len(reductor.bases["RB"])}')
# print(f'max. rel. error: {rel_error:2.5e}')

# test_sample = parameter_space.sample_uniformly(test_snapshots)
# abs_errs = []
# rel_errs = []
# max_abs_err = -1
# max_rel_err = -1
# max_abs_diff = None
# max_rel_diff = None
# for mu in test_sample:
#     u_fom = fom.solve(mu)
#     u_rom = rom.solve(mu)
#     this_diff = u_fom - reductor.reconstruct(u_rom)
#     this_abs_err = this_diff.norm(fom.h1_0_semi_product)[0]
#     this_rel_err = this_diff.norm(fom.h1_0_semi_product)[0]/u_fom.norm(fom.h1_0_semi_product)[0]
#     abs_errs.append(this_abs_err)
#     rel_errs.append(this_rel_err)
#     if this_abs_err > max_abs_err:
#         max_abs_err = this_abs_err
#         max_abs_diff = this_diff
#     if this_rel_err > max_rel_err:
#         max_rel_err = this_rel_err
#         max_rel_diff = this_diff

# print(f'max. abs. err.: {max_abs_err:2.5e}')
# print(f'max. rel. err.: {max_rel_err:2.5e}')

# fom.visualize((max_abs_diff, max_rel_diff), legend = ('Maximum Absolute Test Error', 'Maximum Relative Test Error'), separate_colorbars=True)