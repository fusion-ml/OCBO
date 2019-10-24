"""
6D test functions to use.
Many of these functions from http://www.sfu.ca/~ssurjano/optimization.html
"""
from __future__ import division

from argparse import Namespace

import numpy as np

def hartmann6(x):
    x = np.asarray(x)
    A = np.asarray([[10, 3, 17, 3.5, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14]])
    P = np.asarray([[1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381]], dtype=np.float64)
    P *= 10 ** -4
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    exp_terms = np.asarray([-1 * np.sum(A[i] * (x - P[i]) ** 2)
                            for i in range(4)])
    return np.sum(alpha * np.exp(exp_terms))

def parabaloid6(x):
    x = np.asarray(x)
    return 1 - np.sum(x ** 2) / 6

def plane6(x):
    x = np.asarray(x)
    return np.sum(x) / 6

def dprice6(x):
    x = np.asarray(x)
    i = np.arange(2, 7)
    return -1 * ((x[0] - 1) ** 2 + np.sum(i * (2 * x[1:] ** 2 - x[:-1]) ** 2))

def stang6(x):
    x = np.asarray(x)
    return -1 / (2 * 6 * 39.16599) * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)

sixd_functions = [\
        Namespace(function=hartmann6, domain=[[0, 1] for _ in range(6)],
                  max_val=3.32237, name='hartmann6'),
        Namespace(function=parabaloid6, domain=[[0, 1] for _ in range(6)],
                  max_val=1, name='parabaloid6'),
        Namespace(function=plane6, domain=[[0, 1] for _ in range(6)],
                  max_val=1, name='plane6'),
        Namespace(function=dprice6, domain=[[-10, 10] for _ in range(6)],
                  max_val=0, name='dprice6'),
        Namespace(function=stang6, domain=[[-5, 5] for _ in range(6)],
                  max_val=1, name='stang6')
]
