#!/usr/bin/env python
# Created by "Thieu" at 10:49, 01/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def griewank_func(x):
    x = np.array(x).ravel()
    t1 = np.sum(x**2) / 4000
    t2 = np.prod([np.cos(x[idx] / np.sqrt(idx+1)) for idx in range(0, len(x))])
    return t1 - t2 + 1


def rosenbrock_func(x):
    x = np.array(x).ravel()
    return np.sum([100*(x[idx]**2 - x[idx+1])**2 + (x[idx] - 1)**2 for idx in range(0, len(x)-1)])


def scaffer_func(x):
    x = np.array(x).ravel()
    return 0.5 + (np.sin(np.sqrt(np.sum(x**2))) - 0.5) / (1 + 0.001 * (np.sum(x**2)))**2


def rastrigin_func(x):
    x = np.array(x).ravel()
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)


def weierstrass_func(x, a=0.5, b=3., k_max=20):
    x = np.array(x).ravel()
    ndim = len(x)
    k = np.arange(0, k_max+1)
    result = 0
    for idx in range(0, ndim):
        result += np.sum(a**k * np.cos(2*np.pi*b**k*(x[idx] + 0.5)))
    return result - ndim * np.sum(a**k * np.cos(np.pi*b**k))


def ackley_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    t1 = np.sum(x**2)
    t2 = np.sum(np.cos(2*np.pi*x))
    return -20*np.exp(-0.2 * np.sqrt(t1 / ndim)) - np.exp(t2 / ndim) + 20 + np.e


def sphere_func(x):
    x = np.array(x).ravel()
    return np.sum(x**2)


def rotated_expanded_scaffer_func(x):
    x = np.array(x).ravel()
    results = [scaffer_func([x[idx], x[idx+1]]) for idx in range(0, len(x)-1)]
    return np.sum(results) + scaffer_func([x[-1], x[0]])


def f8f2_func(x):
    x = np.array(x).ravel()
    results = [griewank_func(rosenbrock_func([x[idx], x[idx+1]])) for idx in range(0, len(x) - 1)]
    return np.sum(results) + griewank_func(rosenbrock_func([x[-1], x[0]]))



























