#!/usr/bin/env python
# Created by "Thieu" at 10:49, 01/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def rounder(x, condition):
    ndim = len(x)
    temp_2x = 2 * x
    for idx in range(0, ndim):
        dec, inter = np.modf(temp_2x[idx])
        if temp_2x[idx] <= 0 and dec >= 0.5:
            temp_2x[idx] = inter - 1
        elif dec < 0.5:
            temp_2x[idx] = inter
        else:
            temp_2x[idx] = inter + 1
    temp_2x = temp_2x / 2
    return np.where(condition < 0.5, x, temp_2x)


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
    return 0.5 + (np.sin(np.sqrt(np.sum(x**2)))**2 - 0.5) / (1 + 0.001 * np.sum(x**2))**2


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


def non_continuous_expanded_scaffer_func(x):
    x = np.array(x).ravel()
    y = rounder(x, np.abs(x))
    results = [scaffer_func([y[idx], y[idx+1]]) for idx in range(0, len(x) - 1)]
    return np.sum(results) + scaffer_func([y[-1], y[0]])


def non_continuous_rastrigin_func(x):
    x = np.array(x).ravel()
    y = rounder(x, np.abs(x))
    results = [rastrigin_func([y[idx], y[idx + 1]]) for idx in range(0, len(x) - 1)]
    return np.sum(results) + rastrigin_func([y[-1], y[0]])


def elliptic_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    idx = np.arange(0, ndim)
    return np.sum((10**6)**(idx/(ndim-1)) * x**2)


def sphere_noise_func(x):
    x = np.array(x).ravel()
    return np.sum(x**2)*(1 + 0.1 * np.abs(np.random.normal(0, 1)))


def twist_func(x):
    # This function in CEC-2008 F7
    return 4 * (x ** 4 - 2 * x ** 3 + x ** 2)


def doubledip(x, c, s):
    # This function in CEC-2008 F7
    if -0.5 < x < 0.5:
        return (-6144 * (x - c) ** 6 + 3088 * (x - c) ** 4 - 392 * (x - c) ** 2 + 1) * s
    else:
        return 0


def fractal_1d_func(x):
    # This function in CEC-2008 F7
    np.random.seed(0)
    result1 = 0.0
    for k in range(1, 4):
        result2 = 0.0
        upper = 2 ** (k-1) + 1
        for t in range(1, upper):
            selected = np.random.choice([0, 1, 2], p=1/3*np.ones(3))
            result2 += np.sum([doubledip(x, np.random.uniform(0, 1), 1.0/(2**(k-1) * (2 - np.random.uniform(0, 1)))) for _ in range(0, selected)])
        result1 += result2
    return result1


def schwefel_12_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    return np.sum([np.sum(x[:idx])**2 for idx in range(0, ndim)])


def tosz_func(x):
    def transform(xi):
        if xi > 0:
            c1, c2, x_sign = 10., 7.9, 1.0
            x_star = np.log(np.abs(xi))
        elif xi == 0:
            c1, c2, x_sign, x_star = 5.5, 3.1, 0., 0.
        else:
            c1, c2, x_sign = 5.5, 3.1, -1.
            x_star = np.log(np.abs(xi))
        return x_sign * np.exp(x_star + 0.049 * (np.sin(c1*x_star) + np.sin(c2*x_star)))

    x = np.array(x).ravel()
    x[0] = transform(x[0])
    x[-1] = transform(x[-1])
    return x


def tasy_func(x, beta=0.5):
    x = np.array(x).ravel()
    ndim = len(x)
    idx = np.arange(0, ndim)
    up = 1 + beta * ((idx - 1) / (ndim-1)) * np.sqrt(np.abs(x))
    x_temp = np.abs(x) ** up
    return np.where(x > 0, x_temp, x)


def bent_cigar_func(x):
    x = np.array(x).ravel()
    return x[0]**2 + 10**6 * np.sum(x[1:]**2)


def discus_func(x):
    x = np.array(x).ravel()
    return 10**6 * x[0]**2 + np.sum(x[1:]**2)


def different_powers_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    idx = np.arange(0, ndim)
    up = 2 + 4*idx/(ndim-1)
    return np.sqrt(np.sum(np.abs(x)**up))


def generate_diagonal_matrix(size, alpha=10):
    idx = np.arange(0, size)
    diagonal = alpha**(idx / (2*(size-1)))
    matrix = np.zeros((size, size), float)
    np.fill_diagonal(matrix, diagonal)
    return matrix


def gz_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    t1 = (500 - np.mod(x, 500)) * np.sin(np.sqrt(np.abs(500 - np.mod(x, 500)))) - (x - 500)**2/(10000*ndim)
    t2 = (np.mod(np.abs(x), 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(np.abs(x), 500) - 500))) - (x+500)**2/(10000*ndim)
    t3 = x*np.sin(np.abs(x)**0.5)
    conditions = [x < -500, (-500 <= x) & (x <= 500), x > 500]
    choices = [t2, t3, t1]
    y = np.select(conditions, choices, default=np.nan)
    # y = x.copy()
    # for idx in range(0, ndim):
    #     if x[idx] > 500:
    #         y[idx] = (500 - np.mod(x[idx], 500)) * np.sin(np.sqrt(np.abs(500 - np.mod(x[idx], 500)))) - (x[idx] - 500)**2/(10000*ndim)
    #     elif x[idx] < -500:
    #         y[idx] = (np.mod(x[idx], 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(np.abs(x[idx]), 500) - 500))) - (x[idx]+500)**2/(10000*ndim)
    #     else:
    #         y[idx] = x[idx]*np.sin(np.abs(x[idx])**0.5)
    return y


def katsuura_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    result = 1.0
    for idx in range(0, ndim):
        temp = np.sum([ np.abs(2**j * x[idx] - np.round(2**j * x[idx])) / 2**j for j in range(1, 33)])
        result *= (1 + (idx+1)*temp) ** (10.0 / ndim**1.2)
    return (result - 1)*10 / ndim**2


def lunacek_bi_rastrigin_func(x, z, miu0=2.5, d=1.):
    x = np.array(x).ravel()
    ndim = len(x)
    s = 1 - 1.0 / (2 * np.sqrt(ndim + 20) - 8.2)
    miu1 = -np.sqrt((miu0**2 - d)/s)
    temp1 = np.sum((x - miu0)**2)
    temp2 = d*ndim + s*np.sum((x - miu1)**2)
    result1 = min(temp1, temp2)
    return result1 + 10*(ndim - np.sum(np.cos(2*np.pi*z)))


def calculate_weight(x, xichma=1.):
    ndim = len(x)
    weight = 1
    temp = np.sum(x ** 2)
    if temp != 0:
        weight = (1.0 / np.sqrt(temp)) * np.exp(-temp / (2 * ndim * xichma ** 2))
    return weight


def modified_schwefel_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    z = x + 4.209687462275036e+002
    return 418.9829 * ndim - np.sum(gz_func(z))


def happy_cat_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    t1 = np.sum(x)
    t2 = np.sum(x**2)
    return np.abs(t2 - ndim)**0.25 + (0.5*t2 + t1) / ndim + 0.5


def hgbat_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    t1 = np.sum(x)
    t2 = np.sum(x**2)
    return np.abs(t2**2 - t1**2)**0.5 + (0.5*t2 + t1) / ndim + 0.5


def zakharov_func(x):
    x = np.array(x).ravel()
    temp = np.sum(0.5*x)
    return np.sum(x**2) + temp**2 + temp**4


def levy_func(x):
    x = np.array(x).ravel()
    w = 1 + (x - 1) / 4
    t1 = np.sin(np.pi*w[0])**2 + (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    t2 = np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
    return t1 + t2


def schaffer_f7_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    result = 0.0
    for idx in range(0, ndim-1):
        t = x[idx]**2 + x[idx+1]**2
        result += np.sqrt(t) * (np.sin(50.*t**0.2) + 1)
    return (result/(ndim-1))**2


def storn_chebyshev_polynomial_fitting_func(x, d=72.661):
    x = np.array(x).ravel()
    ndim = len(x)
    m = 32*ndim
    j1 = np.arange(0, ndim)
    upper = ndim - j1

    u = np.sum(x * 1.2**upper)
    v = np.sum(x * (-1.2) ** upper)
    p1 = 0 if u >= d else (u-d)**2
    p2 = 0 if v >=d else (v-d)**2

    wk = np.array([np.sum(x * (2.*k/m - 1)**upper) for k in range(0, m+1)])
    conditions = [wk < 1, (1 <= wk) & (wk <= 1), wk > 1]
    t1 = (wk + 1)**2
    t2 = np.zeros(len(wk))
    t3 = (wk - 1)**2
    choices = [t1, t2, t3]
    pk = np.select(conditions, choices, default=np.nan)
    p3 = np.sum(pk)
    return p1 + p2 +p3


def inverse_hilbert_matrix_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    n = int(np.sqrt(ndim))
    I = np.identity(n)
    H = np.zeros((n, n))
    Z = np.zeros((n, n))
    for i in range(0, n):
        for k in range(0, n):
            Z[i, k] = x[i + n*k]
            H[i, k] = 1./(i+k+1)
    W = np.matmul(H,Z) - I
    return np.sum(W)


def lennard_jones_minimum_energy_cluster_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    result = 12.7120622568
    n_upper = int(ndim/3)
    for i in range(0, n_upper-1):
        for j in range(i+1, n_upper):
            idx1, idx2 = 3*(i+1), 3*(j+1)
            dij = ((x[idx1-2] - x[idx2-2])**2 + (x[idx1-1] - x[idx2-1])**2 + (x[idx1] - x[idx2])**2)**3
            result += (1./dij**2 - 2./dij)
    return result


expanded_griewank_rosenbrock_func = f8f2_func
expanded_scaffer_f6_func = rotated_expanded_scaffer_func
