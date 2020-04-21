#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 22:16, 21/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%


from numpy.random import seed, permutation
from numpy import dot, ones, array, ceil
from opfunu.cec.cec2014.utils import *

SUPPORT_DIMENSION = [2, 10, 20, 30, 50, 100]
SUPPORT_DIMENSION_2 = [10, 20, 30, 50, 100]

def F1(solution=None, name="Rotated High Conditioned Elliptic Function", shift_data_file="shift_data_1.txt", bias=100):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_1_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix)
    return f1_elliptic__(z) + bias


def F2(solution=None, name="Rotated Bent Cigar Function", shift_data_file="shift_data_2.txt", bias=200):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_2_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix)
    return f2_bent_cigar__(z) + bias


def F3(solution=None, name="Rotated Discus Function", shift_data_file="shift_data_3.txt", bias=300):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_3_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix)
    return f3_discus__(z) + bias


def F4(solution=None, name="Shifted and Rotated Rosenbrock’s Function", shift_data_file="shift_data_4.txt", bias=400):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_4_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 2.048 * (solution - shift_data) / 100
    z = dot(z, matrix) + 1
    return f4_rosenbrock__(z) + bias


def F5(solution=None, name="Shifted and Rotated Ackley’s Function", shift_data_file="shift_data_5.txt", bias=500):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_5_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix)
    return f5_ackley__(z) + bias


def F6(solution=None, name="Shifted and Rotated Weierstrass Function", shift_data_file="shift_data_6.txt", bias=600):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_6_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 0.5 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f6_weierstrass__(z) + bias


def F7(solution=None, name="Shifted and Rotated Griewank’s Function", shift_data_file="shift_data_7.txt", bias=700):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_7_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 600 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f7_griewank__(z) + bias


def F8(solution=None, name="Shifted Rastrigin’s Function", shift_data_file="shift_data_8.txt", bias=800):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_8_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 5.12 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f8_rastrigin__(z) + bias


def F9(solution=None, name="Shifted and Rotated Rastrigin’s Function", shift_data_file="shift_data_9.txt", bias=900):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_9_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 5.12 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f9_modified_schwefel__(z) + bias


def F10(solution=None, name="Shifted Schwefel’s Function", shift_data_file="shift_data_10.txt", bias=1000):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_10_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 1000 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f9_modified_schwefel__(z) + bias


def F11(solution=None, name="Shifted and Rotated Schwefel’s Function", shift_data_file="shift_data_11.txt", bias=1100):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_11_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 1000 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f9_modified_schwefel__(z) + bias


def F12(solution=None, name="Shifted and Rotated Katsuura Function", shift_data_file="shift_data_12.txt", bias=1200):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_12_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f10_katsuura__(z) + bias


def F13(solution=None, name="Shifted and Rotated HappyCat Function", shift_data_file="shift_data_13.txt", bias=1300):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_13_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f11_happy_cat__(z) + bias


def F14(solution=None, name="Shifted and Rotated HGBat Function", shift_data_file="shift_data_14.txt", bias=1400):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_14_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f12_hgbat__(z) + bias


def F15(solution=None, name="Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function", shift_data_file="shift_data_15.txt", bias=1500):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_15_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix) + 1
    return f13_expanded_griewank__(z) + bias


def F16(solution=None, name="Shifted and Rotated Expanded Scaffer’s F6 Function", shift_data_file="shift_data_16.txt", bias=1600):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_16_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix) + 1
    return f14_expanded_scaffer__(z) + bias


### ================== Hybrid function ========================

def F17(solution=None, name="Hybrid Function 1", shift_data_file="shift_data_17.txt", bias=1700, shuffle=None):
    problem_size = len(solution)
    p = array([0.3, 0.3, 0.4])
    n1 = int(p[0] * problem_size) + 1
    n2 = int(p[1] * problem_size) + 1

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_17_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_17_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1+n2)]
    idx3 = shuffle[(n1+n2):]
    mz = dot(solution - shift_data, matrix)
    return f9_modified_schwefel__(mz[idx1]) + f8_rastrigin__(mz[idx2]) + f1_elliptic__(mz[idx3]) + bias


def F18(solution=None, name="Hybrid Function 2", shift_data_file="shift_data_18.txt", bias=1800, shuffle=None):
    problem_size = len(solution)
    p = array([0.3, 0.3, 0.4])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_18_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_18_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):]
    mz = dot(solution - shift_data, matrix)
    return f2_bent_cigar__(mz[idx1]) + f12_hgbat__(mz[idx2]) + f8_rastrigin__(mz[idx3]) + bias


def F19(solution=None, name="Hybrid Function 3", shift_data_file="shift_data_19.txt", bias=1900, shuffle=None):
    problem_size = len(solution)
    p = array([0.2, 0.2, 0.3, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_19_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_19_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1+n2+n3)]
    idx4 = shuffle[n1+n2+n3:]
    mz = dot(solution - shift_data, matrix)
    return f7_griewank__(mz[idx1]) + f6_weierstrass__(mz[idx2]) + f4_rosenbrock__(mz[idx3]) + f14_expanded_scaffer__(mz[idx4])+ bias


def F20(solution=None, name="Hybrid Function 4", shift_data_file="shift_data_20.txt", bias=2000, shuffle=None):
    problem_size = len(solution)
    p = array([0.2, 0.2, 0.3, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_20_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_20_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[n1 + n2 + n3:]
    mz = dot(solution - shift_data, matrix)
    return f12_hgbat__(mz[idx1]) + f3_discus__(mz[idx2]) + f13_expanded_griewank__(mz[idx3]) + f8_rastrigin__(mz[idx4]) + bias


def F21(solution=None, name="Hybrid Function 5", shift_data_file="shift_data_21.txt", bias=2100, shuffle=None):
    problem_size = len(solution)
    p = array([0.1, 0.2, 0.2, 0.2, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))
    n4 = int(ceil(p[3] * problem_size))

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_21_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_21_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[(n1+n2+n3):(n1+n2+n3+n4)]
    idx5 = shuffle[n1+n2+n3+n4:]
    mz = dot(solution - shift_data, matrix)
    return f14_expanded_scaffer__(mz[idx1]) + f12_hgbat__(mz[idx2]) + f4_rosenbrock__(mz[idx3]) + \
           f9_modified_schwefel__(mz[idx4]) + f1_elliptic__(mz[idx5]) + bias


def F22(solution=None, name="Hybrid Function 6", shift_data_file="shift_data_22.txt", bias=2200, shuffle=None):
    problem_size = len(solution)
    p = array([0.1, 0.2, 0.2, 0.2, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))
    n4 = int(ceil(p[3] * problem_size))

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_22_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_21_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[(n1 + n2 + n3):(n1 + n2 + n3 + n4)]
    idx5 = shuffle[n1 + n2 + n3 + n4:]
    mz = dot(solution - shift_data, matrix)
    return f10_katsuura__(mz[idx1]) + f11_happy_cat__(mz[idx2]) + f13_expanded_griewank__(mz[idx3]) + \
           f9_modified_schwefel__(mz[idx4]) + f5_ackley__(mz[idx5]) + bias

### ================== Composition function ========================

def F23(solution=None, name="Composition Function 1", shift_data_file="shift_data_23.txt", f_bias=2300):
    num_funcs = 5
    problem_size = len(solution)
    xichma = array([10, 20, 30, 40, 50])
    lamda = array([1, 1e-6, 1e-26, 1e-6, 1e-6])
    bias = array([0, 100, 200, 300, 400])

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    shift_data = load_matrix_data__(shift_data_file)[:problem_size]
    shift_data = shift_data[:, :problem_size]

    def __fi__(solution=None, idx=None):
        if idx == 0:
            return f4_rosenbrock__(solution)
        elif idx == 1:
            return f1_elliptic__(solution)
        elif idx == 2:
            return f2_bent_cigar__(solution)
        elif idx == 3:
            return f3_discus__(solution)
        else:
            return f1_elliptic__(solution)

    weights = ones(num_funcs)
    fits = ones(num_funcs)
    for i in range(0, num_funcs):
        t1 = lamda[i] * __fi__(solution, i) + bias[i]
        t2 = 1.0 / sqrt(sum((solution - shift_data[i])**2))
        w_i = t2 * exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
        weights[i] = w_i
        fits[i] = t1
    sw = sum(weights)
    result = 0.0
    for i in range(0, num_funcs):
        result += (weights[i] / sw) * fits[i]
    return result + f_bias


def F24(solution=None, name="Composition Function 2", shift_data_file="shift_data_24.txt", f_bias=2400):
    num_funcs = 3
    problem_size = len(solution)
    xichma = array([20, 20, 20])
    lamda = array([1, 1, 1])
    bias = array([0, 100, 200])

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    shift_data = load_matrix_data__(shift_data_file)[:problem_size]
    shift_data = shift_data[:, :problem_size]

    def __fi__(solution=None, idx=None):
        if idx == 0:
            return f10_katsuura__(solution)
        elif idx == 1:
            return f9_modified_schwefel__(solution)
        else:
            return f14_expanded_scaffer__(solution)

    weights = ones(num_funcs)
    fits = ones(num_funcs)
    for i in range(0, num_funcs):
        t1 = lamda[i] * __fi__(solution, i) + bias[i]
        t2 = 1.0 / sqrt(sum((solution - shift_data[i]) ** 2))
        w_i = t2 * exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
        weights[i] = w_i
        fits[i] = t1
    sw = sum(weights)
    result = 0.0
    for i in range(0, num_funcs):
        result += (weights[i] / sw) * fits[i]
    return result + f_bias


def F25(solution=None, name="Composition Function 3", shift_data_file="shift_data_25.txt", f_bias=2500):
    num_funcs = 3
    problem_size = len(solution)
    xichma = array([10, 30, 50])
    lamda = array([0.25, 1, 1e-7])
    bias = array([0, 100, 200])

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    shift_data = load_matrix_data__(shift_data_file)[:problem_size]
    shift_data = shift_data[:, :problem_size]

    def __fi__(solution=None, idx=None):
        if idx == 0:
            return f11_happy_cat__(solution)
        elif idx == 1:
            return f9_modified_schwefel__(solution)
        else:
            return f1_elliptic__(solution)

    weights = ones(num_funcs)
    fits = ones(num_funcs)
    for i in range(0, num_funcs):
        t1 = lamda[i] * __fi__(solution, i) + bias[i]
        t2 = 1.0 / sqrt(sum((solution - shift_data[i]) ** 2))
        w_i = t2 * exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
        weights[i] = w_i
        fits[i] = t1
    sw = sum(weights)
    result = 0.0
    for i in range(0, num_funcs):
        result += (weights[i] / sw) * fits[i]
    return result + f_bias


def F26(solution=None, name="Composition Function 4", shift_data_file="shift_data_26.txt", f_bias=2600):
    num_funcs = 5
    problem_size = len(solution)
    xichma = array([10, 10, 10, 10, 10])
    lamda = array([0.25, 1, 1e-7, 2.5, 10])
    bias = array([0, 100, 200, 300, 400])

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    shift_data = load_matrix_data__(shift_data_file)[:problem_size]
    shift_data = shift_data[:, :problem_size]

    def __fi__(solution=None, idx=None):
        if idx == 0:
            return f11_happy_cat__(solution)
        elif idx == 1:
            return f13_expanded_griewank__(solution)
        elif idx == 2:
            return f1_elliptic__(solution)
        elif idx == 3:
            return f6_weierstrass__(solution)
        else:
            return f7_griewank__(solution)

    weights = ones(num_funcs)
    fits = ones(num_funcs)
    for i in range(0, num_funcs):
        t1 = lamda[i] * __fi__(solution, i) + bias[i]
        t2 = 1.0 / sqrt(sum((solution - shift_data[i]) ** 2))
        w_i = t2 * exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
        weights[i] = w_i
        fits[i] = t1
    sw = sum(weights)
    result = 0.0
    for i in range(0, num_funcs):
        result += (weights[i] / sw) * fits[i]
    return result + f_bias


def F27(solution=None, name="Composition Function 5", shift_data_file="shift_data_27.txt", f_bias=2700):
    num_funcs = 5
    problem_size = len(solution)
    xichma = array([10, 10, 10, 20, 20])
    lamda = array([10, 10, 2.5, 25, 1e-6])
    bias = array([0, 100, 200, 300, 400])

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    shift_data = load_matrix_data__(shift_data_file)[:problem_size]
    shift_data = shift_data[:, :problem_size]

    def __fi__(solution=None, idx=None):
        if idx == 0:
            return f14_expanded_scaffer__(solution)
        elif idx == 1:
            return f9_modified_schwefel__(solution)
        elif idx == 2:
            return f11_happy_cat__(solution)
        elif idx == 3:
            return f6_weierstrass__(solution)
        else:
            return f1_elliptic__(solution)

    weights = ones(num_funcs)
    fits = ones(num_funcs)
    for i in range(0, num_funcs):
        t1 = lamda[i] * __fi__(solution, i) + bias[i]
        t2 = 1.0 / sqrt(sum((solution - shift_data[i]) ** 2))
        w_i = t2 * exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
        weights[i] = w_i
        fits[i] = t1
    sw = sum(weights)
    result = 0.0
    for i in range(0, num_funcs):
        result += (weights[i] / sw) * fits[i]
    return result + f_bias


def F28(solution=None, name="Composition Function 6", shift_data_file="shift_data_28.txt", f_bias=2800):
    num_funcs = 5
    problem_size = len(solution)
    xichma = array([10, 20, 30, 40, 50])
    lamda = array([2.5, 10, 2.5, 5e-4, 1e-6])
    bias = array([0, 100, 200, 300, 400])

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    shift_data = load_matrix_data__(shift_data_file)[:problem_size]
    shift_data = shift_data[:, :problem_size]

    def __fi__(solution=None, idx=None):
        if idx == 0:
            return F15(solution)
        elif idx == 1:
            return f13_expanded_griewank__(solution)
        elif idx == 2:
            return f11_happy_cat__(solution)
        elif idx == 3:
            return F16(solution)
        else:
            return f1_elliptic__(solution)

    weights = ones(num_funcs)
    fits = ones(num_funcs)
    for i in range(0, num_funcs):
        t1 = lamda[i] * __fi__(solution, i) + bias[i]
        t2 = 1.0 / sqrt(sum((solution - shift_data[i]) ** 2))
        w_i = t2 * exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
        weights[i] = w_i
        fits[i] = t1
    sw = sum(weights)
    result = 0.0
    for i in range(0, num_funcs):
        result += (weights[i] / sw) * fits[i]
    return result + f_bias


def F29(solution=None, name="Composition Function 7", shift_data_file="shift_data_29.txt", f_bias=2900):
    num_funcs = 3
    problem_size = len(solution)
    xichma = array([10, 30, 50])
    lamda = array([1, 1, 1])
    bias = array([0, 100, 200])

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    shift_data = load_matrix_data__(shift_data_file)[:problem_size]
    shift_data = shift_data[:, :problem_size]

    def __fi__(solution=None, idx=None):
        if idx == 0:
            return F17(solution, shuffle=29)
        elif idx == 1:
            return F18(solution, shuffle=29)
        else:
            return F19(solution, shuffle=29)

    weights = ones(num_funcs)
    fits = ones(num_funcs)
    for i in range(0, num_funcs):
        t1 = lamda[i] * __fi__(solution, i) + bias[i]
        t2 = 1.0 / sqrt(sum((solution - shift_data[i]) ** 2))
        w_i = t2 * exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
        weights[i] = w_i
        fits[i] = t1
    sw = sum(weights)
    result = 0.0
    for i in range(0, num_funcs):
        result += (weights[i] / sw) * fits[i]
    return result + f_bias


def F30(solution=None, name="Composition Function 8", shift_data_file="shift_data_30.txt", f_bias=3000):
    num_funcs = 3
    problem_size = len(solution)
    xichma = array([10, 30, 50])
    lamda = array([1, 1, 1])
    bias = array([0, 100, 200])

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    shift_data = load_matrix_data__(shift_data_file)[:problem_size]
    shift_data = shift_data[:, :problem_size]

    def __fi__(solution=None, idx=None):
        if idx == 0:
            return F20(solution, shuffle=30)
        elif idx == 1:
            return F21(solution, shuffle=30)
        else:
            return F22(solution, shuffle=30)

    weights = ones(num_funcs)
    fits = ones(num_funcs)
    for i in range(0, num_funcs):
        t1 = lamda[i] * __fi__(solution, i) + bias[i]
        t2 = 1.0 / sqrt(sum((solution - shift_data[i]) ** 2))
        w_i = t2 * exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
        weights[i] = w_i
        fits[i] = t1
    sw = sum(weights)
    result = 0.0
    for i in range(0, num_funcs):
        result += (weights[i] / sw) * fits[i]
    return result + f_bias


