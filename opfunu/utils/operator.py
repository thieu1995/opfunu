#!/usr/bin/env python
# Created by "Thieu" at 10:49, 01/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def rounder(x, condition):
    temp_2x = 2 * x
    dec, inter = np.modf(temp_2x)
    temp_2x = np.where(temp_2x <= 0.0, inter - (dec >= 0.5), temp_2x)
    temp_2x = np.where(dec < 0.5, inter, temp_2x)
    temp_2x = np.where(dec >= 0.5, inter + 1, temp_2x)
    return np.where(condition < 0.5, x, temp_2x / 2)


def griewank_func(x):
    x = np.array(x).ravel()
    idx = np.arange(1, len(x) + 1)
    t1 = np.sum(x ** 2) / 4000
    t2 = np.prod(np.cos(x / np.sqrt(idx)))
    return t1 - t2 + 1


def rosenbrock_func(x, shift=0.0):
    x = np.array(x).ravel() + shift
    term1 = 100 * (x[:-1] ** 2 - x[1:]) ** 2
    term2 = (x[:-1] - 1) ** 2
    return np.sum(term1 + term2)

def scaffer_func(x):
    x = np.array(x).ravel()
    return 0.5 + (np.sin(np.sqrt(np.sum(x ** 2))) ** 2 - 0.5) / (1 + 0.001 * np.sum(x ** 2)) ** 2


def rastrigin_func(x):
    x = np.array(x).ravel()
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def weierstrass_func(x, a=0.5, b=3., k_max=20):
    x = np.array(x).ravel()
    ndim = len(x)
    k = np.arange(0, k_max + 1)
    result = 0
    for idx in range(0, ndim):
        result += np.sum(a ** k * np.cos(2 * np.pi * b ** k * (x[idx] + 0.5)))
    return result - ndim * np.sum(a ** k * np.cos(np.pi * b ** k))


def weierstrass_norm_func(x, a=0.5, b=3., k_max=20):
    """
    This function matches CEC2005 description of F11 except for addition of the bias and follows the C implementation
    """
    return weierstrass_func(x, a, b, k_max) - weierstrass_func(np.zeros(len(x)), a, b, k_max)


def ackley_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    t1 = np.sum(x ** 2)
    t2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(t1 / ndim)) - np.exp(t2 / ndim) + 20 + np.e


def sphere_func(x):
    x = np.array(x).ravel()
    return np.sum(x ** 2)


def rotated_expanded_schaffer_func(x):
    x = np.asarray(x).ravel()
    x_pairs = np.column_stack((x, np.roll(x, -1)))
    sum_sq = x_pairs[:, 0] ** 2 + x_pairs[:, 1] ** 2
    # Calculate the Schaffer function for all pairs simultaneously
    schaffer_values = (0.5 + (np.sin(np.sqrt(sum_sq)) ** 2 - 0.5) /
                       (1 + 0.001 * sum_sq) ** 2)
    return np.sum(schaffer_values)


def rotated_expanded_scaffer_func(x):
    x = np.array(x).ravel()
    results = [scaffer_func([x[idx], x[idx + 1]]) for idx in range(0, len(x) - 1)]
    return np.sum(results) + scaffer_func([x[-1], x[0]])


def grie_rosen_cec_func(x):
    """This is based on the CEC version which unrolls the griewank and rosenbrock functions for better performance"""
    z = np.array(x).ravel()
    z += 1.0  # This centers the optimal solution of rosenbrock to 0

    tmp1 = (z[:-1] * z[:-1] - z[1:]) ** 2
    tmp2 = (z[:-1] - 1.0) ** 2
    temp = 100.0 * tmp1 + tmp2
    f = np.sum(temp ** 2 / 4000.0 - np.cos(temp) + 1.0)
    # Last calculation
    tmp1 = (z[-1] * z[-1] - z[0]) ** 2
    tmp2 = (z[-1] - 1.0) ** 2
    temp = 100.0 * tmp1 + tmp2
    f += (temp ** 2) / 4000.0 - np.cos(temp) + 1.0

    return f


def f8f2_func(x):
    x = np.array(x).ravel()
    results = [griewank_func(rosenbrock_func([x[idx], x[idx + 1]])) for idx in range(0, len(x) - 1)]
    return np.sum(results) + griewank_func(rosenbrock_func([x[-1], x[0]]))


def non_continuous_expanded_scaffer_func(x):
    x = np.array(x).ravel()
    y = rounder(x, np.abs(x))
    results = [scaffer_func([y[idx], y[idx + 1]]) for idx in range(0, len(x) - 1)]
    return np.sum(results) + scaffer_func([y[-1], y[0]])


def non_continuous_rastrigin_func(x):
    x = np.array(x).ravel()
    y = rounder(x, np.abs(x))
    shifted_y = np.roll(y, -1)
    results = rastrigin_func(np.column_stack((y, shifted_y)))
    return np.sum(results)


def elliptic_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    idx = np.arange(0, ndim)
    return np.sum(10 ** (6.0 * idx / (ndim - 1)) * x ** 2)



def sphere_noise_func(x):
    x = np.array(x).ravel()
    return np.sum(x ** 2) * (1 + 0.1 * np.abs(np.random.normal(0, 1)))


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
        upper = 2 ** (k - 1) + 1
        for t in range(1, upper):
            selected = np.random.choice([0, 1, 2], p=1 / 3 * np.ones(3))
            result2 += np.sum([doubledip(x, np.random.uniform(0, 1), 1.0 / (2 ** (k - 1) * (2 - np.random.uniform(0, 1)))) for _ in range(0, selected)])
        result1 += result2
    return result1


def schwefel_12_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    return np.sum([np.sum(x[:idx]) ** 2 for idx in range(0, ndim)])


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
        return x_sign * np.exp(x_star + 0.049 * (np.sin(c1 * x_star) + np.sin(c2 * x_star)))

    x = np.array(x).ravel()
    x[0] = transform(x[0])
    x[-1] = transform(x[-1])
    return x


def tasy_func(x, beta=0.5):
    x = np.array(x).ravel()
    ndim = len(x)
    idx = np.arange(0, ndim)
    up = 1 + beta * ((idx - 1) / (ndim - 1)) * np.sqrt(np.abs(x))
    x_temp = np.abs(x) ** up
    return np.where(x > 0, x_temp, x)


def bent_cigar_func(x):
    x = np.array(x).ravel()
    return x[0] ** 2 + 10 ** 6 * np.sum(x[1:] ** 2)


def discus_func(x):
    x = np.array(x).ravel()
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)


def different_powers_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    idx = np.arange(0, ndim)
    up = 2 + 4 * idx / (ndim - 1)
    return np.sqrt(np.sum(np.abs(x) ** up))


def generate_diagonal_matrix(size, alpha=10):
    idx = np.arange(0, size)
    diagonal = alpha ** (idx / (2 * (size - 1)))
    matrix = np.zeros((size, size), float)
    np.fill_diagonal(matrix, diagonal)
    return matrix


def gz_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    t1 = (500 - np.mod(x, 500)) * np.sin(np.sqrt(np.abs(500 - np.mod(x, 500)))) - (x - 500) ** 2 / (10000 * ndim)
    t2 = (np.mod(np.abs(x), 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(np.abs(x), 500) - 500))) - (x + 500) ** 2 / (10000 * ndim)
    t3 = x * np.sin(np.abs(x) ** 0.5)
    conditions = [x < -500, (-500 <= x) & (x <= 500), x > 500]
    choices = [t2, t3, t1]
    y = np.select(conditions, choices, default=np.nan)
    return y


def katsuura_func(x):
    # TODO: New function failed to pass 5 test cases.
    # powers_of_two = 2 ** np.arange(1, 34)
    # reciprocals_of_two = 1 / powers_of_two
    # for idx in range(0, ndim):
    #     temp = np.sum(np.abs(powers_of_two * x[idx] - np.round(powers_of_two * x[idx])) * reciprocals_of_two)
    #     result *= (1 + (idx + 1) * temp) ** (10.0 / ndim ** 1.2)
    # return (result - 1) * 10 / ndim ** 2

    x = np.array(x).ravel()
    ndim = len(x)
    result = 1.0
    for idx in range(0, ndim):
        temp = np.sum([np.abs(2 ** j * x[idx] - np.round(2 ** j * x[idx])) / 2 ** j for j in range(1, 33)])
        result *= (1 + (idx + 1) * temp) ** (10.0 / ndim ** 1.2)
    return (result - 1) * 10 / ndim ** 2


def lunacek_bi_rastrigin_func(x, miu0=2.5, d=1, shift=0.0):
    x = np.array(x).ravel() + shift
    ndim = len(x)
    s = 1.0 - 1.0 / (2 * np.sqrt(ndim + 20) - 8.2)
    miu1 = -np.sqrt((miu0 ** 2 - d) / s)
    delta_x_miu0 = x - miu0
    term1 = np.sum(delta_x_miu0 ** 2)
    term2 = np.sum((x - miu1) ** 2) * s + d * ndim
    result = min(term1, term2) + 10 * (ndim - np.sum(np.cos(2 * np.pi * delta_x_miu0)))
    return result


def calculate_weight(x, delta=1.):
    ndim = len(x)
    temp = np.sum(x ** 2)
    if temp != 0:
        weight = np.sqrt(1.0 / temp) * np.exp(-temp / (2 * ndim * delta ** 2))
    else:
        weight = 1e99  # this is the INF definition in original CEC Calculate logic

    return weight


def modified_schwefel_func(x):
    """
        This is a direct conversion of the CEC2021 C-Code for the Modified Schwefel F11 Function
    """
    z = np.array(x).ravel() + 4.209687462275036e+002
    nx = len(z)

    mask1 = z > 500
    mask2 = z < -500
    mask3 = ~mask1 & ~mask2
    fx = np.zeros(nx)
    fx[mask1] -= ((500.0 - np.fmod(z[mask1], 500)) * np.sin(np.sqrt(500.0 - np.fmod(z[mask1], 500))) -
                 ((z[mask1] - 500.0) / 100.) ** 2 / nx)
    fx[mask2] -= (-500.0 + np.fmod(np.abs(z[mask2]), 500)) * np.sin(np.sqrt(500.0 - np.fmod(np.abs(z[mask2]), 500))) - (
                 (z[mask2] + 500.0) / 100.) ** 2 / nx
    fx[mask3] -= z[mask3] * np.sin(np.sqrt(np.abs(z[mask3])))

    return np.sum(fx) + 4.189828872724338e+002 * nx


def happy_cat_func(x, shift=0.0):
    z = np.array(x).ravel() + shift
    ndim = len(z)
    t1 = np.sum(z)
    t2 = np.sum(z ** 2)
    return np.abs(t2 - ndim) ** 0.25 + (0.5 * t2 + t1) / ndim + 0.5


def hgbat_func(x, shift=0.0):
    x = np.array(x).ravel() + shift
    ndim = len(x)
    t1 = np.sum(x)
    t2 = np.sum(x ** 2)
    return np.abs(t2 ** 2 - t1 ** 2) ** 0.5 + (0.5 * t2 + t1) / ndim + 0.5


def zakharov_func(x):
    x = np.array(x).ravel()
    temp = np.sum(0.5 * x)
    return np.sum(x ** 2) + temp ** 2 + temp ** 4


def levy_func(x, shift=0.0):
    x = np.array(x).ravel() + shift
    w = 1. + (x - 1.) / 4
    t1 = np.sin(np.pi * w[0]) ** 2 + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    t2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    return t1 + t2


def expanded_schaffer_f6_func(x):
    """
    This is a direct conversion of the CEC2021 C-Code for the Expanded Schaffer F6 Function
    """
    z = np.array(x).ravel()

    temp1 = np.sin(np.sqrt(z[:-1] ** 2 + z[1:] ** 2))
    temp1 = temp1 ** 2
    temp2 = 1.0 + 0.001 * (z[:-1] ** 2 + z[1:] ** 2)
    f = np.sum(0.5 + (temp1 - 0.5) / (temp2 ** 2))

    temp1_last = np.sin(np.sqrt(z[-1] ** 2 + z[0] ** 2))
    temp1_last = temp1_last ** 2
    temp2_last = 1.0 + 0.001 * (z[-1] ** 2 + z[0] ** 2)
    f += 0.5 + (temp1_last - 0.5) / (temp2_last ** 2)

    return f


def schaffer_f7_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    result = 0.0
    for idx in range(0, ndim - 1):
        t = x[idx] ** 2 + x[idx + 1] ** 2
        result += np.sqrt(t) * (np.sin(50. * t ** 0.2) + 1)
    return (result / (ndim - 1)) ** 2


def chebyshev_func(x):
    """
    The following was converted from the cec2019 C code
    Storn's Tchebychev - a 2nd ICEO function - generalized version
    """
    x = np.array(x).ravel()
    ndim = len(x)
    sample = 32 * ndim

    dx_arr = np.zeros(ndim)
    dx_arr[:2] = [1.0, 1.2]
    for i in range(2, ndim):
        dx_arr[i] = 2.4 * dx_arr[i-1] - dx_arr[i-2]
    dx = dx_arr[-1]

    dy = 2.0 / sample

    px, y, sum_val = 0, -1, 0
    for i in range(sample + 1):
        px = x[0]
        for j in range(1, ndim):
            px = y * px + x[j]
        if px < -1 or px > 1:
            sum_val += (1.0 - abs(px)) ** 2
        y += dy

    for _ in range(2):
        px = np.sum(1.2 * x[1:]) + x[0]
        mask = px < dx
        sum_val += np.sum(px[mask] ** 2)

    return sum_val


def inverse_hilbert_func(x):
    """
    This is a direct conversion of the cec2019 C code for python optimized to use numpy
    """
    x = np.array(x).ravel()
    ndim = len(x)
    b = int(np.sqrt(ndim))

    # Create the Hilbert matrix
    i, j = np.indices((b, b))
    hilbert = 1.0 / (i + j + 1)

    # Reshape x and compute H*x
    x = x.reshape((b, b))
    y = np.dot(hilbert, x).dot(hilbert.T)

    # Compute the absolute deviations
    result = np.sum(np.abs(y - np.eye(b)))
    return result


def lennard_jones_func(x):
    """
    This version is a direct python conversion from the C-Code of CEC2019 implementation.
    Find the atomic configuration with minimum energy (Lennard-Jones potential)
    Valid for any dimension, D = 3 * k, k = 2, 3, 4, ..., 25.
    k is the number of atoms in 3-D space.
    """
    x = np.array(x).ravel()
    ndim = len(x)
    # Minima values from Cambridge cluster database: http://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html
    minima = np.array([-1., -3., -6., -9.103852, -12.712062, -16.505384, -19.821489, -24.113360,
                       -28.422532, -32.765970, -37.967600, -44.326801, -47.845157, -52.322627, -56.815742,
                       -61.317995, -66.530949, -72.659782, -77.1777043, -81.684571, -86.809782, -02.844472,
                       -97.348815, -102.372663])

    k = ndim // 3
    sum_val = 0

    x_matrix = x.reshape((k, 3))
    for i in range(k-1):
        for j in range(i + 1, k):
            # Use slicing to get the differences between points i and j
            diff = x_matrix[i] - x_matrix[j]
            # Calculate the squared Euclidean distance
            ed = np.sum(diff ** 2)
            # Calculate ud and update sum_val accordingly
            ud = ed ** 3
            if ud > 1.0e-10:
                sum_val += (1.0 / ud - 2.0) / ud
            else:
                sum_val += 1.0e20  # cec2019 version penalizes when ud is <=1e-10
    return sum_val - minima[k - 2]  # Subtract known minima for k


expanded_griewank_rosenbrock_func = grie_rosen_cec_func
expanded_scaffer_f6_func = rotated_expanded_scaffer_func
