#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:31, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

import pkg_resources
from pandas import read_csv
from numpy import cos, sqrt, pi, e, exp, sum

CURRENT_PATH = pkg_resources.resource_filename("opfunu", "cec/cec2010/")
SUPPORT_PATH_DATA = pkg_resources.resource_filename("opfunu", "cec/cec2010/support_data/")

def load_shift_data__(data_file=None):
    data = read_csv(SUPPORT_PATH_DATA + data_file, delimiter='\s+', index_col=False, header=None)
    return data.values.reshape((-1))

def load_matrix_data__(data_file=None):
    data = read_csv(SUPPORT_PATH_DATA + data_file, delimiter='\s+', index_col=False, header=None)
    return data.values

def f1_sphere__(solution=None):
    return sum(solution**2)

def f2_elliptic__(solution=None):
    result = 0.0
    for i in range(0, len(solution)):
        result += (10**6)**(i/(len(solution)-1)) * solution[i]**2
    return result

def f3_rastrigin__(solution=None):
    return sum(solution ** 2 - 10 * cos(2 * pi * solution) + 10)

def f4_ackley__(solution=None):
    return -20 * exp(-0.2 * sqrt(sum(solution ** 2) / len(solution))) - exp(sum(cos(2 * pi * solution)) / len(solution)) + 20 + e

def f5_schwefel__(solution=None):
    return sum([ sum(solution[:i])**2 for i in range(0, len(solution))])

def f6_rosenbrock__(solution=None):
    result = 0.0
    for i in range(len(solution) - 1):
        result += 100 * (solution[i] ** 2 - solution[i + 1]) ** 2 + (solution[i] - 1) ** 2
    return result

def check_problem_size(problem_size):
    if problem_size > 1000:
        print("CEC 2010 doesn't support for problem size > 1000")
        exit(0)
    return 1

def check_m_group(function_name, problem_size, m_group):
    if problem_size / m_group <= 1:
        print("CEC 2010, {} not support {}. You can change m_group smaller or problem size larger!!!".format(function_name, problem_size))
        exit(0)
    return 1
