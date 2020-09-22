#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:26, 26/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy import ceil, arange, ones
from opfunu.cec_basic.utils import *


def F1(solution=None, shift_num=1, rotate_num=1, f_bias=100):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return elliptic__(z) + f_bias


def F2(solution=None, shift_num=1, rotate_num=1, f_bias=200):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return bent_cigar__(z) + f_bias


def F3(solution=None, shift_num=1, rotate_num=1, f_bias=300):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return discus__(z) + f_bias


def F4(solution=None, shift_num=1, rotate_num=2.048/100, f_bias=400):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num) + 1
    return rosenbrock__(z) + f_bias


def F5(solution=None, shift_num=1, rotate_num=1, f_bias=500):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return ackley__(z) + f_bias


def F6(solution=None, shift_num=1,  rotate_num=0.5 / 100, f_bias=600):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return weierstrass__(z) + f_bias


def F7(solution=None, shift_num=1, rotate_num=600/100, f_bias=700):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return griewank__(z) + f_bias


def F8(solution=None, shift_num=1, rotate_num=5.12 / 100, f_bias=800):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return rastrigin__(z) + f_bias


def F9(solution=None, shift_num=1, rotate_num=5.12 / 100, f_bias=900):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return rastrigin__(z) + f_bias


def F10(solution=None, shift_num=1, rotate_num=1000 / 100, f_bias=1000):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return modified_schwefel__(z) + f_bias


def F11(solution=None, shift_num=1, rotate_num=1000 / 100, f_bias=1100):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return modified_schwefel__(z) + f_bias


def F12(solution=None, shift_num=1, rotate_num=5 / 100, f_bias=1200):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return katsuura__(z) + f_bias


def F13(solution=None, shift_num=1, rotate_num=5 / 100, f_bias=1300):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return katsuura__(z) + f_bias


def F14(solution=None, shift_num=1, rotate_num=5 / 100, f_bias=1400):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return hgbat__(z) + f_bias


def F15(solution=None, shift_num=1, rotate_num=5 / 100, f_bias=1500):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num) + 1
    return expanded_griewank__(z) + f_bias


def F16(solution=None, shift_num=1, rotate_num=1, f_bias=1600):
    x = shift__(solution, shift_num)
    z = rotate__(x, rotate_num)
    return expanded_scaffer__(z) + f_bias


def F17(solution=None, shift_num=1, rotate_num=1, f_bias=1700):
    dim = len(solution)
    n1 = int(ceil(0.3 * dim))
    n2 = int(ceil(0.3 * dim)) + n1
    D = arange(dim)
    x = shift__(solution, shift_num)
    return modified_schwefel__(x[D[:n1]]) + rastrigin__(x[D[n1:n2]]) + elliptic__(x[D[n2:]]) + f_bias


def F18(solution=None, shift_num=1, rotate_num=1, f_bias=1800):
    dim = len(solution)
    n1 = int(ceil(0.3 * dim))
    n2 = int(ceil(0.3 * dim)) + n1
    D = arange(dim)
    x = shift__(solution, shift_num)
    return bent_cigar__(x[D[:n1]]) + hgbat__(x[D[n1:n2]]) + rastrigin__(x[D[n2:]]) + f_bias


def F19(solution=None, shift_num=1, rotate_num=1, f_bias=1900):
    dim = len(solution)
    n1 = int(ceil(0.2 * dim))
    n2 = int(ceil(0.2 * dim)) + n1
    n3 = int(ceil(0.3 * dim)) + n2
    D = arange(dim)
    x = shift__(solution, shift_num)
    return griewank__(x[D[:n1]]) + weierstrass__(x[D[n1:n2]]) + rosenbrock__(x[D[n2:n3]]) + expanded_scaffer__(x[D[n3:]]) + f_bias


def F20(solution=None, shift_num=1, rotate_num=1, f_bias=2000):
    dim = len(solution)
    n1 = int(ceil(0.2 * dim))
    n2 = int(ceil(0.2 * dim)) + n1
    n3 = int(ceil(0.3 * dim)) + n2
    D = arange(dim)
    x = shift__(solution, shift_num)
    return hgbat__(x[D[:n1]]) + discus__(x[D[n1:n2]]) + expanded_griewank__(x[D[n2:n3]]) + rastrigin__(x[D[n3:]]) + f_bias


def F21(solution=None, shift_num=1, rotate_num=1, f_bias=2100):
    dim = len(solution)
    n1 = int(ceil(0.1 * dim))
    n2 = int(ceil(0.2 * dim)) + n1
    n3 = int(ceil(0.2 * dim)) + n2
    n4 = int(ceil(0.2 * dim)) + n3
    D = arange(dim)
    x = shift__(solution, shift_num)
    return expanded_scaffer__(x[D[:n1]]) + hgbat__(x[D[n1:n2]]) + rosenbrock__(x[D[n2:n3]]) + \
           modified_schwefel__(x[D[n3:n4]]) + elliptic__(x[D[n4:]]) + f_bias


def F22(solution=None, shift_num=1, rotate_num=1, f_bias=2200):
    dim = len(solution)
    n1 = int(ceil(0.1 * dim))
    n2 = int(ceil(0.2 * dim)) + n1
    n3 = int(ceil(0.2 * dim)) + n2
    n4 = int(ceil(0.2 * dim)) + n3
    D = arange(dim)
    x = shift__(solution, shift_num)
    return katsuura__(x[D[:n1]]) + happy_cat__(x[D[n1:n2]]) + expanded_griewank__(x[D[n2:n3]]) + \
           modified_schwefel__(x[D[n3:n4]]) + ackley__(x[D[n4:]]) + f_bias


def F23(solution=None, shift_num=1, rotate_num=1, f_bias=2300):
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 20, 30, 40, 50]
    lamda = [1, 1e-6, 1e-26, 1e-6, 1e-6]
    bias = [0, 100, 200, 300, 400]
    fun = [F4, F1, F2, F3, F1]
    dim = len(solution)
    w = ones(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift__(solution, shift_arr[i])
        if sum(x**2) == 0:
            w[i] = 1
        else:
            w[i] = 1 / sqrt(sum(x**2)) * exp(- sum(x**2) / (2 * dim * sigma[i]**2))
    sumw = sum(w)
    result = 0
    for i in range(len(shift_arr)):
        fit = lamda[i] * fun[i](solution, shift_num=shift_num, rotate_num=rotate_num, f_bias=0)
        result += (w[i] / sumw) * (fit + bias[i])
    return result + f_bias


def F24(solution=None, shift_num=1, rotate_num=1, f_bias=2400):
    shift_arr = [1, 2, 3]
    sigma = [20, 20, 20]
    lamda = [1, 1, 1]
    bias = [0, 100, 200]
    fun = [F10, F9, F14]
    dim = len(solution)
    w = ones(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift__(solution, shift_arr[i])
        if sum(x ** 2) == 0:
            w[i] = 1
        else:
            w[i] = 1 / sqrt(sum(x ** 2)) * exp(- sum(x ** 2) / (2 * dim * sigma[i] ** 2))
    sumw = sum(w)
    result = 0
    for i in range(len(shift_arr)):
        result += (w[i] / sumw) * (lamda[i] * fun[i](solution, shift_num=shift_num, rotate_num=rotate_num, f_bias=0) + bias[i])
    return result + f_bias


def F25(solution=None, shift_num=1, rotate_num=1, f_bias=2500):
    shift_arr = [1, 2, 3]
    sigma = [10, 30, 50]
    lamda = [0.25, 1, 1.0e-7]
    bias = [0, 100, 200]
    fun = [F11, F9, F1]
    dim = len(solution)
    w = ones(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift__(solution, shift_arr[i])
        if sum(x ** 2) == 0:
            w[i] = 1
        else:
            w[i] = 1 / sqrt(sum(x ** 2)) * exp(- sum(x ** 2) / (2 * dim * sigma[i] ** 2))
    sumw = sum(w)
    result = 0
    for i in range(len(shift_arr)):
        result += (w[i] / sumw) * (lamda[i] * fun[i](solution, shift_num, rotate_num=rotate_num, f_bias=0) + bias[i])
    return result + f_bias


def F26(solution=None, shift_num=1, rotate_num=1, f_bias=2600):
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 10, 10, 10, 10]
    lamda = [0.25, 1.0, 1.0e-7, 2.5, 10.0]
    bias = [0, 100, 200, 300, 400]
    fun = [F11, F13, F1, F6, F7]
    dim = len(solution)
    w = ones(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift__(solution, shift_arr[i])
        if sum(x ** 2) == 0:
            w[i] = 1
        else:
            w[i] = 1 / sqrt(sum(x ** 2)) * exp(- sum(x ** 2) / (2 * dim * sigma[i] ** 2))
    sumw = sum(w)
    result = 0
    for i in range(len(shift_arr)):
        result += (w[i] / sumw) * (lamda[i] * fun[i](solution, shift_num, rotate_num, f_bias=0) + bias[i])
    return result + f_bias


def F27(solution=None, shift_num=1, rotate_num=1, f_bias=2700):
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 10, 10, 20, 20]
    lamda = [10, 10, 2.5, 25, 1.0e-6]
    bias = [0, 100, 200, 300, 400]
    fun = [F14, F9, F11, F6, F1]
    dim = len(solution)
    w = ones(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift__(solution, shift_arr[i])
        if sum(x ** 2) == 0:
            w[i] = 1
        else:
            w[i] = 1 / sqrt(sum(x ** 2)) * exp(- sum(x ** 2) / (2 * dim * sigma[i] ** 2))
    sumw = sum(w)
    result = 0
    for i in range(len(shift_arr)):
        result += (w[i] / sumw) * (lamda[i] * fun[i](solution, shift_num, rotate_num, f_bias=0) + bias[i])
    return result + f_bias


def F28(solution=None, shift_num=1, rotate_num=1, f_bias=2800):
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 20, 30, 40, 50]
    lamda = [2.5, 10, 2.5, 5.0e-4, 1.0e-6]
    bias =  [0, 100, 200, 300, 400]
    fun = [F15, F13, F11, F16, F1]
    dim = len(solution)
    w = ones(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift__(solution, shift_arr[i])
        if sum(x ** 2) == 0:
            w[i] = 1
        else:
            w[i] = 1 / sqrt(sum(x ** 2)) * exp(- sum(x ** 2) / (2 * dim * sigma[i] ** 2))
    sumw = sum(w)
    result = 0
    for i in range(len(shift_arr)):
        result += (w[i] / sumw) * (lamda[i] * fun[i](solution, shift_num, rotate_num, f_bias=0) + bias[i])
    return result + f_bias


def F29(solution=None, shift_num=1, rotate_num=1, f_bias=2900):
    shift_arr = [1, 2, 3]
    sigma = [10, 30, 50]
    lamda = [1, 1, 1]
    bias = [0, 100, 200]
    fun = [F17, F18, F19]
    dim = len(solution)
    result = 0
    w = ones(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift__(solution, shift_arr[i])
        if sum(x ** 2) == 0:
            w[i] = 1
        else:
            w[i] = 1 / sqrt(sum(x ** 2)) * exp(- sum(x ** 2) / (2 * dim * sigma[i] ** 2))
    sumw = sum(w)
    for i in range(len(shift_arr)):
        result += (w[i] / sumw) * (lamda[i] * fun[i](solution, shift_num, rotate_num, f_bias=0) + bias[i])
    return result + f_bias


def F30(solution=None, shift_num=1, rotate_num=1, f_bias=3000):
    shift_arr = [1, 2, 3]
    sigma = [10, 30, 50]
    lamda = [1, 1, 1]
    bias = [0, 100, 200]
    fun = [F20, F21, F22]
    dim = len(solution)
    w = ones(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift__(solution, shift_arr[i])
        if sum(x ** 2) == 0:
            w[i] = 1
        else:
            w[i] = 1 / sqrt(sum(x ** 2)) * exp(- sum(x ** 2) / (2 * dim * sigma[i] ** 2))
    sumw = sum(w)
    result = 0
    for i in range(len(shift_arr)):
        result += (w[i] / sumw) * (lamda[i] * fun[i](solution, shift_num, rotate_num, f_bias=0) + bias[i])
    return result + f_bias

