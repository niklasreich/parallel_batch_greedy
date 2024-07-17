import numpy as np
from pymor.core.pickle import load
from os.path import isfile
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

### CONFIG
# 22 for 2x2 thermal block
# 32 for 3x2 thermalblock
# 33 for 3x3 thermalblock
config = 22


if config==22:
    file_string = 'src/pymordemos/thermalblock_2x2_N30_BS1_nopp.pkl'
    file_string_2 = 'src/pymordemos/thermalblock_2x2_N30_BS2_nopp.pkl'
    file_string_4 = 'src/pymordemos/thermalblock_2x2_N30_BS4_nopp.pkl'
    file_string_8 = 'src/pymordemos/thermalblock_2x2_N30_BS8_nopp.pkl'
    file_string_16 = 'src/pymordemos/thermalblock_2x2_N30_BS16_nopp.pkl'
    decay_start = 8
    up_shift = 4e3
    down_shift = 5
    title_string = r'$2 \times 2$ Thermal Block'
elif config==32:
    file_string = 'src/pymordemos/thermalblock_3x2_N30_BS1.pkl'
    file_string_2 = 'src/pymordemos/thermalblock_3x2_N30_BS2.pkl'
    file_string_4 = 'src/pymordemos/thermalblock_3x2_N30_BS4.pkl'
    file_string_8 = 'src/pymordemos/thermalblock_3x2_N30_BS8.pkl'
    file_string_16 = 'src/pymordemos/thermalblock_3x2_N30_BS16.pkl'
    decay_start = 10
    up_shift = 2e2
    down_shift = 4
    title_string = r'$3 \times 2$ Thermal Block'
elif config==33:
    file_string = 'src/pymordemos/thermalblock_3x3_N30_BS1.pkl'
    file_string_2 = 'src/pymordemos/thermalblock_3x3_N30_BS2.pkl'
    file_string_4 = 'src/pymordemos/thermalblock_3x3_N30_BS4.pkl'
    file_string_8 = 'src/pymordemos/thermalblock_3x3_N30_BS8.pkl'
    file_string_16 = 'src/pymordemos/thermalblock_3x3_N30_BS16.pkl'
    decay_start = 45
    up_shift = 3e1
    down_shift = 2
    title_string = r'$3 \times 3$ Thermal Block'


def func_alpha(n, C1, c1, alpha):
    return C1*np.exp(-c1*(n**alpha))

def log_func_alpha(n, C1, c1, alpha):
    return np.log10(C1*np.exp(-c1*(n**alpha)))

def func(n, C1, c1):
    return C1*np.exp(-c1*(n))

def log_func(n, C1, c1):
    return np.log10(C1*np.exp(-c1*(n)))


if isfile(file_string):

    with open(file_string, 'rb') as f:
        results = load(f)
        err = results['max_rel_errors'][0]
    
    plt.semilogy(err,':k',markerfacecolor='none', markeredgecolor='k')

    range = range(decay_start,len(err))
    xData = range
    yData = err[range]
    logyData = np.log10(yData)

    plt.semilogy(xData, yData,'-k', label='classic greedy')
    plt.grid()

    if isfile(file_string_2):
        with open(file_string_2, 'rb') as f:
            results = load(f)
            err = results['max_rel_errors'][0]
            plt.semilogy(err,':',markerfacecolor='none', label='$b=2$')


    if isfile(file_string_4):
        with open(file_string_4, 'rb') as f:
            results = load(f)
            err = results['max_rel_errors'][0]
        plt.semilogy(err,':',markerfacecolor='none', label='$b=4$')

    if isfile(file_string_8):
        with open(file_string_8, 'rb') as f:
            results = load(f)
            err = results['max_rel_errors'][0]
        plt.semilogy(err,':',markerfacecolor='none', label='$b=8$')

    if isfile(file_string_16):
        with open(file_string_16, 'rb') as f:
            results = load(f)
            err = results['max_rel_errors'][0]
        plt.semilogy(err,':',markerfacecolor='none', label='$b=16$')

    fittedParameters, pcov = curve_fit(log_func, xData, logyData, bounds=[0, np.inf], max_nfev=10000)
    fittedParameters_alpha, pcov = curve_fit(log_func_alpha, xData, logyData, bounds=[0, np.inf], max_nfev=10000)

    print('Parameters', fittedParameters)
    print('Parameters with alpha', fittedParameters_alpha)

    fittedParameters_b = fittedParameters.copy()
    fittedParameters_b[0] *= up_shift
    fittedParameters_b[1] *= 2/3
    fittedParameters_alpha_b = fittedParameters_alpha.copy()
    fittedParameters_alpha_b[0] *= up_shift
    fittedParameters_alpha_b[1] *= (2/3)**fittedParameters_alpha_b[2]

    fittedParameters[0] /= down_shift
    fittedParameters_alpha[0] /= down_shift

    xModel = np.linspace(0,len(err))
    yModel = func(xModel, *fittedParameters)
    yModel_alpha = func_alpha(xModel, *fittedParameters_alpha)
    yModel_b = func(xModel, *fittedParameters_b)
    yModel_alpha_b = func_alpha(xModel, *fittedParameters_alpha_b)
    plt.semilogy(xModel, yModel,'r', label=r'shifted classical decay for $\alpha=1$ (fitt.)')
    plt.semilogy(xModel, yModel_alpha,'b', label=r'shifted classical decay (fitt.)')
    plt.semilogy(xModel, yModel_b,'--r', label=r'shifted batch decay for $\alpha=1$ (calc.)')
    plt.semilogy(xModel, yModel_alpha_b,'--b', label=r'shifted batch decay (calc.)')
    
    plt.legend()
    plt.ylabel('relative error')
    plt.xlabel(r'basis size $n$')
    plt.title(title_string)
    plt.show()