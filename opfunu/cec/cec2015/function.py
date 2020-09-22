#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:19, 24/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import seed, permutation
from numpy import dot, ones, array, ceil
from opfunu.cec.cec2015.utils import *


def F1(solution=None, name="Rotated Bent Cigar Function", f_shift_file="shift_data_1_D", f_matrix_file="M_1_D", bias=100):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix)
    return f1_bent_cigar__(z) + bias


def F2(solution=None, name="Rotated Discus Function", f_shift_file="shift_data_2_D", f_matrix_file="M_2_D", bias=200):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix)
    return f2_discus__(z) + bias


def F3(solution=None, name="Shifted and Rotated Weierstrass Function", f_shift_file="shift_data_3_D", f_matrix_file="M_3_D", bias=300):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(0.5*(solution - shift_data) / 100, matrix)
    return f3_weierstrass__(z) + bias


def F4(solution=None, name="Shifted and Rotated Schwefel’s Function", f_shift_file="shift_data_4_D", f_matrix_file="M_4_D", bias=400):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(1000 * (solution - shift_data) / 100, matrix)
    return f4_modified_schwefel__(z) + bias


def F5(solution=None, name="Shifted and Rotated Katsuura Function", f_shift_file="shift_data_5_D", f_matrix_file="M_5_D", bias=500):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(5 * (solution - shift_data) / 100, matrix)
    return f5_katsuura__(z) + bias


def F6(solution=None, name="Shifted and Rotated HappyCat Function", f_shift_file="shift_data_6_D", f_matrix_file="M_6_D", bias=600):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(5 * (solution - shift_data) / 100, matrix)
    return f6_happy_cat__(z) + bias


def F7(solution=None, name="Shifted and Rotated HGBat Function", f_shift_file="shift_data_7_D", f_matrix_file="M_7_D", bias=700):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(5 * (solution - shift_data) / 100, matrix)
    return f7_hgbat__(z) + bias


def F8(solution=None, name="Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function",
       f_shift_file="shift_data_8_D", f_matrix_file="M_8_D", bias=800):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(5 * (solution - shift_data) / 100, matrix) + 1
    return f8_expanded_griewank__(z) + bias


def F9(solution=None, name="Shifted and Rotated Expanded Scaffer’s F6 Function",
       f_shift_file="shift_data_9_D", f_matrix_file="M_9_D", bias=900):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix) + 1
    return f9_expanded_scaffer__(z) + bias


def F10(solution=None, name="Hybrid Function 1 (N=3)", f_shift_file="shift_data_10_D", f_matrix_file="M_10_D",
        f_shuffle_file="shuffle_data_10_D", bias=1000):
    problem_size = len(solution)
    p = array([0.3, 0.3, 0.4])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    f_shift, f_matrix, f_shuffle = check_problem_size(problem_size, f_shift_file, f_matrix_file, f_shuffle_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle) - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):]
    mz = dot(solution - shift_data, matrix)
    return f4_modified_schwefel__(mz[idx1]) + f12_rastrigin__(mz[idx2]) + f13_elliptic__(mz[idx3]) + bias


def F11(solution=None, name="Hybrid Function 2 (N=4)", f_shift_file="shift_data_11_D", f_matrix_file="M_11_D",
        f_shuffle_file="shuffle_data_11_D", bias=1100):
    problem_size = len(solution)
    p = array([0.2, 0.2, 0.3, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))

    f_shift, f_matrix, f_shuffle = check_problem_size(problem_size, f_shift_file, f_matrix_file, f_shuffle_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle) - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1+n2+n3)]
    idx4 = shuffle[(n1 + n2 + n3):]
    mz = dot(solution - shift_data, matrix)
    return f11_griewank__(mz[idx1]) + f3_weierstrass__(mz[idx2]) + f10_rosenbrock__(mz[idx3]) + f9_expanded_scaffer__(mz[idx4]) + bias


def F12(solution=None, name="Hybrid Function 3 (N=5)", f_shift_file="shift_data_12_D", f_matrix_file="M_12_D",
        f_shuffle_file="shuffle_data_12_D", bias=1200):
    problem_size = len(solution)
    p = array([0.1, 0.2, 0.2, 0.2, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))
    n4 = int(ceil(p[3] * problem_size))

    f_shift, f_matrix, f_shuffle = check_problem_size(problem_size, f_shift_file, f_matrix_file, f_shuffle_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle) - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[(n1 + n2 + n3):(n1+n2+n3+n4)]
    idx5 = shuffle[(n1+n2+n3+n4):]
    mz = dot(solution - shift_data, matrix)
    return f5_katsuura__(mz[idx1]) + f6_happy_cat__(mz[idx2]) + f8_expanded_griewank__(mz[idx3]) + f4_modified_schwefel__(mz[idx4]) + \
           f14_ackley__(mz[idx5]) + bias


def F13(solution=None, name="Composition Function 1 (N=5)", f_shift_file="shift_data_13_D", f_matrix_file="M_13_D", f_bias=1300):
    problem_size = len(solution)
    xichma = array([10, 20, 30, 40, 50])
    lamda = array([1, 1e-6, 1e-26, 1e-6, 1e-6])
    bias = array([0, 100, 200, 300, 400])

    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)

    # 1. Rotated Rosenbrock’s Function F10
    t1 = solution - shift_data[:problem_size]
    g1 = lamda[0] * f10_rosenbrock__(dot(t1, matrix[:problem_size, :])) + bias[0]
    w1 = calculate_weights(problem_size, t1, xichma[0])

    # 2. High Conditioned Elliptic Function
    t2 = solution - shift_data[problem_size:2*problem_size]
    g2 = lamda[1] * f13_elliptic__(solution) + bias[1]
    w2 = calculate_weights(problem_size, t2, xichma[1])

    # 3. Rotated Bent Cigar Function f1
    t3 = solution - shift_data[2*problem_size:3*problem_size]
    g3 = lamda[2] * f1_bent_cigar__(dot(t3, matrix[2*problem_size: 3*problem_size, :])) + bias[2]
    w3 = calculate_weights(problem_size, t3, xichma[2])

    # 4. Rotated Discus Function f2
    t4 = solution - shift_data[3 * problem_size: 4 * problem_size]
    g4 = lamda[3] * f2_discus__(dot(t4, matrix[problem_size*2:3*problem_size, :]) ) + bias[3]
    w4 = calculate_weights(problem_size, t4, xichma[3])

    # 4. High Conditioned Elliptic Function f13
    t5 = solution - shift_data[4 * problem_size:]
    g5 = lamda[4] * f13_elliptic__(solution) + bias[4]
    w5 = calculate_weights(problem_size, t5, xichma[4])

    sw = sum([w1, w2, w3, w4, w5])
    result = (w1*g1 + w2*g2 + w3*g3 + w4*g4 + w5*g5) / sw
    return result + f_bias


def F14(solution=None, name="Composition Function 2 (N=3)", f_shift_file="shift_data_14_D", f_matrix_file="M_14_D", f_bias=1400):
    problem_size = len(solution)
    xichma = array([10, 30, 50])
    lamda = array([0.25, 1, 1e-7])
    bias = array([0, 100, 200])

    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)

    # 1. Rotated Schwefel's Function f4
    t1 = solution - shift_data[:problem_size]
    g1 = lamda[0] * f4_modified_schwefel__(dot(t1, matrix[:problem_size, :])) + bias[0]
    w1 = calculate_weights(problem_size, t1, xichma[0])

    # 2. Rotated Rastrigin’s Function f12
    t2 = solution - shift_data[problem_size:2 * problem_size]
    g2 = lamda[1] * f12_rastrigin__(dot(t2, matrix[problem_size:2*problem_size, :])) + bias[1]
    w2 = calculate_weights(problem_size, t2, xichma[1])

    # 3. Rotated High Conditioned Elliptic Function f13
    t3 = solution - shift_data[2 * problem_size:3 * problem_size]
    g3 = lamda[2] * f13_elliptic__(dot(t3, matrix[2 * problem_size: 3 * problem_size, :])) + bias[2]
    w3 = calculate_weights(problem_size, t3, xichma[2])

    sw = sum([w1, w2, w3])
    result = (w1 * g1 + w2 * g2 + w3 * g3) / sw
    return result + f_bias


def F15(solution=None, name="Composition Function 3 (N=5)", f_shift_file="shift_data_15_D", f_matrix_file="M_15_D", f_bias=1500):
    problem_size = len(solution)
    xichma = array([10, 10, 10, 20, 20])
    lamda = array([10, 10, 2.5, 25, 1e-6])
    bias = array([0, 100, 200, 300, 400])

    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)

    # 1. Rotated HGBat Function f7
    t1 = solution - shift_data[:problem_size]
    g1 = lamda[0] * f7_hgbat__(dot(t1, matrix[:problem_size, :])) + bias[0]
    w1 = calculate_weights(problem_size, t1, xichma[0])

    # 2. Rotated Rastrigin’s Function f12
    t2 = solution - shift_data[problem_size:2 * problem_size]
    g2 = lamda[1] * f12_rastrigin__(dot(t2, matrix[problem_size:2*problem_size, :])) + bias[1]
    w2 = calculate_weights(problem_size, t2, xichma[1])

    # 3. Rotated Schwefel's Function f4
    t3 = solution - shift_data[2 * problem_size:3 * problem_size]
    g3 = lamda[2] * f4_modified_schwefel__(dot(t3, matrix[2 * problem_size: 3 * problem_size, :])) + bias[2]
    w3 = calculate_weights(problem_size, t3, xichma[2])

    # 4. Rotated Weierstrass Function f3
    t4 = solution - shift_data[3 * problem_size: 4 * problem_size]
    g4 = lamda[3] * f3_weierstrass__(dot(t4, matrix[3*problem_size: 4* problem_size, :])) + bias[3]
    w4 = calculate_weights(problem_size, t4, xichma[3])

    # 4. Rotated High Conditioned Elliptic Function f13
    t5 = solution - shift_data[4 * problem_size:]
    g5 = lamda[4] * f13_elliptic__(dot(t5, matrix[4 * problem_size:, :])) + bias[4]
    w5 = calculate_weights(problem_size, t5, xichma[4])

    sw = sum([w1, w2, w3, w4, w5])
    result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
    return result + f_bias

