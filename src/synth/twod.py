"""
2D test functions to use.
Many of these functions from http://www.sfu.ca/~ssurjano/optimization.html
"""

from __future__ import division

from argparse import Namespace

import numpy as np

def branin(x):
  """ Computes the Branin function. """
  x1 = x[0] * 15 - 5
  x2 = x[1] * 15
  a = 1
  b = 5.1/(4*np.pi**2)
  c = 5/np.pi
  t = 1/(8*np.pi)
  r = 6
  s = 10
  neg_ret = float(a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)
  return -neg_ret

def rosenbrock(x):
  x1 = x[0] * 15 - 5
  x2 = x[1] * 15 - 5
  result = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
  return -result

def constant2(x):
    return 1

def plane(x):
    return np.sum(x)

def parabaloid(x):
    x1, x2 = x
    return 1 - x1 ** 2 / 2 - x2 ** 2 / 2

def bohachevsky1(x):
    x1, x2 = x
    sq = x1 ** 2 + 2 * x2 ** 2
    coss = 0.3 * np.cos(3 * np.pi * x1) + 0.4 * np.cos(4 * np.pi * x2)
    return coss - sq - 0.7

def camel3(x):
    x1, x2 = x
    camel_f = 2 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6 + x1 * x2 + x2 ** 2
    return -1 * camel_f

def camel6(x):
    x1, x2 = x
    pt1 = (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
    pt2 = x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
    return -pt1 - pt2

def ackley(x):
    x = np.asarray(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    f = a * np.exp(-b * np.linalg.norm(x) / np.sqrt(2))
    f += np.exp(np.sum(np.cos(c * x)) / 2)
    f -= a + np.exp(1)
    return f

def sin_mixture(x):
    x1, x2 = x
    curly_sin = 2 * x1 ** 2 * np.sin(4 * np.pi * x2)
    flat_sin = 0.25 * (1 - x1) * np.sin(np.pi * x2 / 4)
    return curly_sin + flat_sin

def jeff(x):
    x1, x2 = x
    return x1 * (np.sin(x2 * x1) + x2 / 4)

def varying_dif(x):
    x = [abs(x[1] - 0.5), x[0]]
    r = 0
    # Hard problem.
    r += 5 * np.exp(-1 * ((x[0] - 0.1) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -5 * np.exp(-1 * ((x[0] - 0.2) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 6 * np.exp(-1 * ((x[0] - 0.3) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -6 * np.exp(-1 * ((x[0] - 0.4) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 7 * np.exp(-1 * ((x[0] - 0.5) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -7 * np.exp(-1 * ((x[0] - 0.6) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 6 * np.exp(-1 * ((x[0] - 0.7) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -6 * np.exp(-1 * ((x[0] - 0.8) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 5 * np.exp(-1 * ((x[0] - 0.9) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    # Medium problem.
    r += 4 * np.exp(-1 * ((x[0] - 0.1) ** 2 / 0.05 + (x[1] - 0.25) ** 2 / 0.01))
    r += 8 * np.exp(-1 * ((x[0] - 0.6) ** 2 / 0.05 + (x[1] - 0.25) ** 2 / 0.01))
    r += -4 * np.exp(-1 * ((x[0] - 0.8) ** 2 / 1 + (x[1] - 0.25) ** 2 / 0.01))
    r += -4 * np.exp(-1 * ((x[0] - 0.3) ** 2 / 0.075 + (x[1] - 0.25) ** 2 / 0.01))
    # Easy Problem
    r += 4 * np.exp(-1 * ((x[0] - 0.5) ** 2 / 0.1 + (x[1] - 0.15) ** 2 / 0.0025))
    return r

"""Willie's function additions"""

def varying_dif_willie(x):
    yout_list = [varying_dif_single(xi) for xi in x]
    return np.array(yout_list)

def varying_dif_willie_harder(x):
    # Assume: x in [[-5,5],[-5,5]]
    x = np.array(x).reshape(-1,2)
    x = x + 5.
    x = x / 10.
    yout_list = [varying_dif_single_harder_willie(xi) for xi in x]
    return np.array(yout_list)

def varying_dif_single(x):
    # Assume: x in [[0,1],[0,1]]
    x = x.flatten()
    x = [x[1], abs(x[0]-0.5)]
    r = 0
    # Hard problem.
    r += 5 * np.exp(-1 * ((x[0] - 0.1) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -5 * np.exp(-1 * ((x[0] - 0.2) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 6 * np.exp(-1 * ((x[0] - 0.3) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -6 * np.exp(-1 * ((x[0] - 0.4) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 7 * np.exp(-1 * ((x[0] - 0.5) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -7 * np.exp(-1 * ((x[0] - 0.6) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 6 * np.exp(-1 * ((x[0] - 0.7) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -6 * np.exp(-1 * ((x[0] - 0.8) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 5 * np.exp(-1 * ((x[0] - 0.9) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    # Medium problem.
    r += 4 * np.exp(-1 * ((x[0] - 0.1) ** 2 / 0.05 + (x[1] - 0.25) ** 2 / 0.01))
    r += 8 * np.exp(-1 * ((x[0] - 0.6) ** 2 / 0.05 + (x[1] - 0.25) ** 2 / 0.01))
    r += -4 * np.exp(-1 * ((x[0] - 0.8) ** 2 / 1 + (x[1] - 0.25) ** 2 / 0.01))
    r += -4 * np.exp(-1 * ((x[0] - 0.3) ** 2 / 0.075 + (x[1] - 0.25) ** 2 / 0.01))
    # Easy Problem
    r += 4 * np.exp(-1 * ((x[0] - 0.5) ** 2 / 0.1 + (x[1] - 0.15) ** 2 / 0.0025))
    return r

def varying_dif_single_harder_willie(x):
    # Assume: x in [[0,1],[0,1]]
    x = np.array(x).flatten()
    x = [x[1], abs(x[0]-0.5)]
    r = 0
    # Hard problem.
    r += 15 * np.exp(-1 * ((x[0] - 0.1) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -15 * np.exp(-1 * ((x[0] - 0.2) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 16 * np.exp(-1 * ((x[0] - 0.3) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -6 * np.exp(-1 * ((x[0] - 0.4) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 7 * np.exp(-1 * ((x[0] - 0.5) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -7 * np.exp(-1 * ((x[0] - 0.6) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 16 * np.exp(-1 * ((x[0] - 0.7) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += -16 * np.exp(-1 * ((x[0] - 0.8) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    r += 15 * np.exp(-1 * ((x[0] - 0.9) ** 2 / 0.005 + (x[1] - 0.45) ** 2 / 0.01))
    # Medium problem.
    r += 4 * np.exp(-1 * ((x[0] - 0.1) ** 2 / 0.05 + (x[1] - 0.25) ** 2 / 0.01))
    r += 8 * np.exp(-1 * ((x[0] - 0.6) ** 2 / 0.05 + (x[1] - 0.25) ** 2 / 0.01))
    r += -4 * np.exp(-1 * ((x[0] - 0.8) ** 2 / 1 + (x[1] - 0.25) ** 2 / 0.01))
    r += -4 * np.exp(-1 * ((x[0] - 0.3) ** 2 / 0.075 + (x[1] - 0.25) ** 2 / 0.01))
    # Easy Problem
    r += 4 * np.exp(-1 * ((x[0] - 0.5) ** 2 / 0.1 + (x[1] - 0.15) ** 2 / 0.0025))
    return float(r)

# This rest are not 2d but oh well...
def hartmann3(x):
    x = np.asarray(x)
    A = np.asarray([[3, 10, 30],
                    [0.1, 10, 35],
                    [3, 10, 30],
                    [0.1, 10, 35]])
    P = np.asarray([[3689, 1170, 2673],
                    [4699, 4387, 7470],
                    [1091, 8732, 5547],
                    [381, 5743, 8828]], dtype=np.float64)
    P *= 10 ** -4
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    exp_terms = np.asarray([-1 * np.sum(A[i] * (x - P[i]) ** 2)
                            for i in range(4)])
    return np.sum(alpha * np.exp(exp_terms))

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

twod_functions = [\
        Namespace(function=branin, domain=[[0, 1], [0, 1]], max_val=0.397887,
                  name='branin'),
        Namespace(function=rosenbrock, domain=[[0, 1], [0, 1]], max_val=0,
                  name='rosenbrock'),
        Namespace(function=constant2, domain=[[0, 1], [0, 1]], max_val=1,
                  name='constant2'),
        Namespace(function=plane, domain=[[0, 1], [0, 1]], max_val=2,
                  name='plane'),
        Namespace(function=parabaloid, domain=[[-1, 1], [-1, 1]], max_val=1,
                  name='parabaloid'),
        Namespace(function=bohachevsky1, domain=[[-100, 100], [-100,100]],
                  max_val=0, name='bohachevsky1'),
        Namespace(function=camel3, domain=[[-5, 5], [-5, 5]], max_val=0,
                  name='camel3'),
        Namespace(function=camel6, domain=[[-3, 3], [-2, 2]], max_val=1.0316,
                  name='camel6'),
        Namespace(function=ackley, domain=[[-20, 20], [-20, 20]], max_val=0,
                  name='ackley'),
        Namespace(function=sin_mixture, domain=[[0, 1], [0, 1]], max_val=None,
                  name='smix'),
        Namespace(function=jeff, domain=[[-1, 3], [-1, 3]], max_val=None,
                  name='jeff'),
        Namespace(function=varying_dif, domain=[[0, 1], [0, 1]], max_val=None,
                  name='vdif'),
        Namespace(function=varying_dif_willie, domain=[[0, 1], [0, 1]], max_val=None,
                  name='vdif_w'),
        Namespace(function=varying_dif_single_harder_willie, domain=[[0, 1], [0, 1]], max_val=None,
                  name='vdif_w_hard'),
        Namespace(function=hartmann3, domain=[[0, 1], [0, 1], [0,1]],
                  max_val=3.86278, name='hartmann3'),
        Namespace(function=hartmann4, domain=[[0, 1] for _ in range(4)],
                  max_val=None, name='hartmann4'),
]
