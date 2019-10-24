"""
Test functions to use.
"""

from __future__ import division

from argparse import Namespace

import numpy as np

def parabola(x):
    x = float(x)
    return 1 - 4 * (x - 0.5) ** 2

def tough_1d(x):
    x = float(x)
    return np.exp(-100 * (x - 0.5) ** 2) \
           + abs(2 * (x - 0.5) * np.sin(10 * np.pi * (x - 0.5)))

def peak_and_trough(x):
    x = float(x)
    return 0.5 * np.sin(2 * np.pi * x) + 0.5

def two_peaks(x):
    x = float(x)
    if x <= 0.5:
        return 0.5 * np.sin(4 * np.pi * x) + 0.5
    else:
        return 0.4 * np.sin(4 * np.pi * x) + 0.5

def constant(x):
    return 1

oneD_functions = [\
        Namespace(function=parabola, domain=[[0, 1]], max_val=1,
            name='Parabola'),
        Namespace(function=tough_1d, domain=[[0, 1]], max_val=1,
            name='Tough'),
        Namespace(function=peak_and_trough, domain=[[0, 1]], max_val=1,
            name='Peak_Trough'),
        Namespace(function=two_peaks, domain=[[0, 1]], max_val=1,
            name='Two_Peaks'),
        Namespace(function=constant, domain=[[0, 1]], max_val=1,
            name='Constant'),
]
