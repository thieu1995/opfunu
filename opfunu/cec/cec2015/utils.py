#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:11, 24/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

import pkg_resources
from pandas import read_csv
from numpy import sum, exp, cos, sin, sqrt, e, pi, abs, round


CURRENT_PATH = pkg_resources.resource_filename("opfunu", "cec/cec2015/")
SUPPORT_PATH_DATA = pkg_resources.resource_filename("opfunu", "cec/cec2015/support_data/")

SUPPORT_DIMENSION = [10, 30]


def check_problem_size(problem_size, *f_file):
    if problem_size in SUPPORT_DIMENSION:
        return [file + str(problem_size) + ".txt" for file in f_file]
    else:
        print("CEC 2015 function only support problem size 10, 30")
        exit(0)

def load_shift_data__(data_file=None):
    data = read_csv(SUPPORT_PATH_DATA + data_file, delimiter='\s+', index_col=False, header=None)
    return data.values.reshape((-1))


def load_matrix_data__(data_file=None):
    data = read_csv(SUPPORT_PATH_DATA + data_file, delimiter='\s+', index_col=False, header=None)
    return data.values


def f1_bent_cigar__(solution=None):
    return solution[0] ** 2 + 10 ** 6 * sum(solution[1:] ** 2)


def f2_discus__(solution=None):
    return 10 ** 6 * solution[0] ** 2 + sum(solution[1:] ** 2)


def f3_weierstrass__(solution=None, a=0.5, b=3, k_max=20):
    result = 0.0
    for i in range(0, len(solution)):
        t1 = sum([a ** k * cos(2 * pi * b ** k * (solution[i] + 0.5)) for k in range(0, k_max)])
        result += t1
    t2 = len(solution) * sum([a ** k * cos(2 * pi * b ** k * 0.5) for k in range(0, k_max)])
    return result - t2


def f4_modified_schwefel__(solution=None):
    z = solution + 4.209687462275036e+002
    result = 418.9829 * len(solution)
    for i in range(0, len(solution)):
        if z[i] > 500:
            result -= (500 - z[i] % 500) * sin(sqrt(abs(500 - z[i] % 500))) - (z[i] - 500) ** 2 / (10000 * len(solution))
        elif z[i] < -500:
            result -= (z[i] % 500 - 500) * sin(sqrt(abs(z[i] % 500 - 500))) - (z[i] + 500) ** 2 / (10000 * len(solution))
        else:
            result -= z[i] * sin(abs(z[i]) ** 0.5)
    return result


def f5_katsuura__(solution=None):
    result = 1.0
    for i in range(0, len(solution)):
        t1 = sum([abs(2 ** j * solution[i] - round(2 ** j * solution[i])) / 2 ** j for j in range(1, 33)])
        result *= (1 + (i + 1) * t1) ** (10.0 / len(solution) ** 1.2)
    return (result - 1) * 10 / len(solution) ** 2


def f6_happy_cat__(solution=None):
    return (abs(sum(solution ** 2) - len(solution))) ** 0.25 + (0.5 * sum(solution ** 2) + sum(solution)) / len(solution) + 0.5


def f7_hgbat__(solution=None):
    return (abs(sum(solution ** 2) ** 2 - sum(solution) ** 2)) ** 0.5 + (0.5 * sum(solution ** 2) + sum(solution)) / len(solution) + 0.5


def f8_expanded_griewank__(solution=None):
    def __f10__(x=None, y=None):
        return 100 * (x ** 2 - y) ** 2 + (x - 1) ** 2

    def __f11__(z=None):
        return z ** 2 / 4000 - cos(z / sqrt(1)) + 1

    result = __f11__(__f10__(solution[-1], solution[0]))
    for i in range(0, len(solution) - 1):
        result += __f11__(__f10__(solution[i], solution[i + 1]))
    return result


def f9_expanded_scaffer__(solution=None):
    def __xy__(x, y):
        return 0.5 + (sin(sqrt(x ** 2 + y ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2

    result = __xy__(solution[-1], solution[0])
    for i in range(0, len(solution) - 1):
        result += __xy__(solution[i], solution[i + 1])
    return result


def f10_rosenbrock__(solution=None):
    result = 0.0
    for i in range(len(solution) - 1):
        result += 100 * (solution[i] ** 2 - solution[i + 1]) ** 2 + (solution[i] - 1) ** 2
    return result


def f11_griewank__(solution=None):
    result = sum(solution ** 2) / 4000
    temp = 1.0
    for i in range(len(solution)):
        temp *= cos(solution[i] / sqrt(i + 1))
    return result - temp + 1


def f12_rastrigin__(solution=None):
    return sum(solution ** 2 - 10 * cos(2 * pi * solution) + 10)


def f13_elliptic__(solution=None):
    result = 0
    for i in range(len(solution)):
        result += (10 ** 6) ** (i / (len(solution) - 1)) * solution[i] ** 2
    return result


def f14_ackley__(solution=None):
    return -20 * exp(-0.2 * sqrt(sum(solution ** 2) / len(solution))) - exp(sum(cos(2 * pi * solution)) / len(solution)) + 20 + e

def calculate_weights(problem_size, z, xichma):
    weight = 1
    temp = sum(z ** 2)
    if temp != 0:
        weight = (1.0 / sqrt(temp)) * exp(-temp / (2 * problem_size * xichma ** 2))
    return weight