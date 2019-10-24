"""
Functions that are 2D but then we discretize and make several 1D functions
from them.
"""

from __future__ import division
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt

from synth.twod import twod_functions
from synth.sixd import hartmann6

def assemble_chopped_1d_functions(f_name, num_divisions):
    """Chop up a 2d function into many 1D functions.
    Args:
        f_name: Name of the 2D function.
        num_divisions: Number of evenly spaced divisions of the function.
    Returns: List of Namespaces that detail the function.
    """
    full_finfo = None
    for f_info in twod_functions:
        if f_name.lower() == f_info.name.lower():
            full_finfo = f_info
            break
    if full_finfo is None:
        raise ValueError('Could not find function %s.' % f_name)
    f_domain = full_finfo.domain[0]
    divisions = np.linspace(f_domain[0], f_domain[1], num_divisions)
    chopped_fs = []
    def make_chop(z):
        full_f = full_finfo.function
        chop = lambda x: full_f(np.asarray([z, float(x)]))
        return chop
    for div_idx, div in enumerate(divisions):
        chopped_fs.append(Namespace(function=make_chop(div),
                                    domain=[full_finfo.domain[1]],
                                    max_val=None,
                                    f_loc=[div],
                                    name=('%s_%d' % (f_name, div_idx))))
    return chopped_fs

def get_chopped_h42():
    divisions = []
    for i in range(16):
        new_div = [int(b) for b in list('{0:b}'.format(i))]
        while len(new_div) < 4:
            new_div = [0] + new_div
        divisions.append(np.asarray(new_div))
    def make_chop(z):
        full_f = hartmann6
        chop = lambda x: full_f(np.append(z, np.asarray(x)))
        return chop
    chopped_fs = []
    for div_idx, div in enumerate(divisions):
        chopped_fs.append(Namespace(function=make_chop(div),
                                    domain=[[0, 1], [0, 1]],
                                    max_val=None,
                                    f_loc=div,
                                    name=('hartmann_%d' % div_idx)))
    return chopped_fs

def get_chopped_h22():
    ctx1 = np.repeat(np.linspace(0, 1, 3), 3)
    ctx2 = np.tile(np.linspace(0, 1, 3), 3)
    divisions = np.vstack([ctx1, ctx2]).T
    def make_chop(z):
        full_f = hartmann4
        chop = lambda x: full_f(np.append(z, np.asarray(x)))
        return chop
    chopped_fs = []
    for div_idx, div in enumerate(divisions):
        chopped_fs.append(Namespace(function=make_chop(div),
                                    domain=[[0, 1], [0, 1]],
                                    max_val=None,
                                    f_loc=div,
                                    name=('h2-2_%d' % div_idx)))
    return chopped_fs

def get_chopped_h31():
    ctx1 = np.repeat(np.linspace(0, 1, 2), 4)
    ctx2 = np.repeat(np.tile(np.linspace(0, 1, 2), 2), 2)
    ctx3 = np.tile(np.linspace(0, 1, 2), 4)
    divisions = np.vstack([ctx1, ctx2, ctx3]).T
    def make_chop(z):
        full_f = hartmann4
        chop = lambda x: full_f(np.append(z, np.asarray(x)))
        return chop
    chopped_fs = []
    for div_idx, div in enumerate(divisions):
        chopped_fs.append(Namespace(function=make_chop(div),
                                    domain=[[0, 1]],
                                    max_val=None,
                                    f_loc=div,
                                    name=('h3-1_%d' % div_idx)))
    return chopped_fs

def hartmann4(x):
    x = np.asarray(x)
    A = np.asarray([[10, 3, 17, 3.5],
                    [0.05, 10, 17, 0.1],
                    [3, 3.5, 1.7, 10],
                    [17, 8, 0.05, 10]])
    P = np.asarray([[1312, 1696, 5569, 124],
                    [2329, 4135, 8307, 3736],
                    [2348, 1451, 3522, 2883],
                    [4047, 8828, 8732, 5743]], dtype=np.float64)
    P *= 10 ** -4
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    exp_terms = np.asarray([-1 * np.sum(A[i] * (x - P[i]) ** 2)
                            for i in range(4)])
    return (1.1 - np.sum(alpha * np.exp(exp_terms))) / 0.839
