#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:29, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#
# Modified by Elliott Pryor 08 March 2021
#-------------------------------------------------------------------------------------------------------%

from numpy.random import seed, permutation
from numpy import dot, ones
from opfunu.cec.cec2010.utils import *


f1_shift_data = None
f1_shift_file = ""


def F1(solution=None, name="Shifted Elliptic Function", shift_data_file="f01_o.txt"):
    problem_size = len(solution)
    check_problem_size(problem_size)
    global f1_shift_data, f1_shift_file
    if f1_shift_file != shift_data_file:
        f1_shift_file = shift_data_file
        f1_shift_data = load_shift_data__(shift_data_file)
    z = solution - f1_shift_data[:problem_size]
    return f2_elliptic__(z)


f2_shift_data = None
f2_shift_file = ""


def F2(solution=None, name="Shifted Rastrigin’s Function", shift_data_file="f02_o.txt"):
    problem_size = len(solution)
    check_problem_size(problem_size)
    global f2_shift_data, f2_shift_file
    if f2_shift_file != shift_data_file:
        f2_shift_file = shift_data_file
        f2_shift_data = load_shift_data__(shift_data_file)
    z = solution - f2_shift_data[:problem_size]
    return f3_rastrigin__(z)


f3_shift_data = None
f3_shift_file = ""


def F3(solution=None, name="Shifted Ackley’s Function", shift_data_file="f03_o.txt"):
    problem_size = len(solution)
    check_problem_size(problem_size)
    global f3_shift_data, f3_shift_file
    if f3_shift_file != shift_data_file:
        f3_shift_file = shift_data_file
        f3_shift_data = load_shift_data__(shift_data_file)
    z = solution - f3_shift_data[:problem_size]
    return f4_ackley__(z)


f4_matrix = None
f4_op_data = None
f4_matrix_file = ""
f4_shift_file = ""


def F4(solution=None, name="Single-group Shifted and m-rotated Elliptic Function", shift_data_file="f04_op.txt", matrix_data_file="f04_m.txt", m_group=50):
    problem_size = len(solution)
    check_problem_size(problem_size)
    global f4_matrix, f4_op_data, f4_matrix_file, f4_shift_file
    if shift_data_file != f4_shift_file and matrix_data_file != f4_matrix_file:
        f4_shift_file = shift_data_file
        f4_matrix_file = matrix_data_file
        f4_matrix = load_matrix_data__(matrix_data_file)
        f4_op_data = load_matrix_data__(shift_data_file)
    
    if problem_size == 1000:
        shift_data = f4_op_data[:1, :].reshape(-1)
        permu_data = (f4_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f4_op_data[:1,:].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    idx1 = permu_data[:m_group]
    idx2 = permu_data[m_group:]
    z_rot_elliptic = dot(z[idx1], f4_matrix[:m_group, :m_group])
    z_elliptic = z[idx2]
    return f2_elliptic__(z_rot_elliptic) * 10**6 + f2_elliptic__(z_elliptic)


f5_matrix = None
f5_op_data = None
f5_matrix_file = ""
f5_shift_file = ""


def F5(solution=None, name="Single-group Shifted and m-rotated Rastrigin’s Function", shift_data_file="f05_op.txt", matrix_data_file="f05_m.txt", m_group=50):
    problem_size = len(solution)
    check_problem_size(problem_size)
    global f5_matrix, f5_op_data, f5_matrix_file, f5_shift_file
    if shift_data_file != f5_shift_file and matrix_data_file != f5_matrix_file:
        f5_shift_file = shift_data_file
        f5_matrix_file = matrix_data_file
        f5_matrix = load_matrix_data__(matrix_data_file)
        f5_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f5_op_data[:1, :].reshape(-1)
        permu_data = (f5_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f5_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    idx1 = permu_data[:m_group]
    idx2 = permu_data[m_group:]
    z_rot_rastrigin = dot(z[idx1], f5_matrix[:m_group, :m_group])
    z_rastrigin = z[idx2]
    return f3_rastrigin__(z_rot_rastrigin) * 10 ** 6 + f3_rastrigin__(z_rastrigin)


f6_matrix = None
f6_op_data = None
f6_matrix_file = ""
f6_shift_file = ""


def F6(solution=None, name="Single-group Shifted and m-rotated Ackley’s Function", shift_data_file="f06_op.txt", matrix_data_file="f06_m.txt", m_group=50):
    problem_size = len(solution)
    check_problem_size(problem_size)
    global f6_matrix, f6_op_data, f6_matrix_file, f6_shift_file
    if shift_data_file != f6_shift_file and matrix_data_file != f6_matrix_file:
        f6_shift_file = shift_data_file
        f6_matrix_file = matrix_data_file
        f6_matrix = load_matrix_data__(matrix_data_file)
        f6_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f6_op_data[:1, :].reshape(-1)
        permu_data = (f6_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f6_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    idx1 = permu_data[:m_group]
    idx2 = permu_data[m_group:]
    z_rot_ackley = dot(z[idx1], f6_matrix[:m_group, :m_group])
    z_ackley = z[idx2]
    return f4_ackley__(z_rot_ackley) * 10 ** 6 + f4_ackley__(z_ackley)


f7_op_data = None
f7_shift_file = ""


def F7(solution=None, name="Single-group Shifted m-dimensional Schwefel’s Problem 1.2", shift_data_file="f07_op.txt", m_group=50):
    problem_size = len(solution)
    check_problem_size(problem_size)
    global f7_op_data, f7_shift_file
    if shift_data_file != f7_shift_file:
        f7_shift_file = shift_data_file
        f7_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f7_op_data[:1, :].reshape(-1)
        permu_data = (f7_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f7_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    idx1 = permu_data[:m_group]
    idx2 = permu_data[m_group:]
    z_schwefel = z[idx1]
    z_shpere = z[idx2]
    return f5_schwefel__(z_schwefel) * 10 ** 6 + f1_sphere__(z_shpere)



f8_op_data = None
f8_shift_file = ""


def F8(solution=None, name=" Single-group Shifted m-dimensional Rosenbrock’s Function", shift_data_file="f08_op.txt", m_group=50):
    problem_size = len(solution)
    check_problem_size(problem_size)
    global f8_op_data, f8_shift_file
    if shift_data_file != f8_shift_file:
        f8_shift_file = shift_data_file
        f8_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f8_op_data[:1, :].reshape(-1)
        permu_data = (f8_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f8_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    idx1 = permu_data[:m_group]
    idx2 = permu_data[m_group:]
    z_rosenbrock = z[idx1]
    z_sphere = z[idx2]
    return f6_rosenbrock__(z_rosenbrock) * 10 ** 6 + f1_sphere__(z_sphere)


f9_matrix = None
f9_op_data = None
f9_matrix_file = ""
f9_shift_file = ""


def F9(solution=None, name="D/2m-group Shifted and m-rotated Elliptic Function", shift_data_file="f09_op.txt", matrix_data_file="f09_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / (2 * m_group))
    check_problem_size(problem_size)
    check_m_group("F9", problem_size, 2*m_group)
    global f9_matrix, f9_op_data, f9_matrix_file, f9_shift_file
    if shift_data_file != f9_shift_file and matrix_data_file != f9_matrix_file:
        f9_shift_file = shift_data_file
        f9_matrix_file = matrix_data_file
        f9_matrix = load_matrix_data__(matrix_data_file)
        f9_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f9_op_data[:1, :].reshape(-1)
        permu_data = (f9_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f9_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i*m_group:(i+1)*m_group]
        z1 = dot(z[idx1], f9_matrix[:len(idx1), :len(idx1)])
        result += f2_elliptic__(z1)
    idx2 = permu_data[int(problem_size/2):problem_size]
    z2 = z[idx2]
    result += f2_elliptic__(z2)
    return result


f10_matrix = None
f10_op_data = None
f10_matrix_file = ""
f10_shift_file = ""


def F10(solution=None, name="D/2m-group Shifted and m-rotated Rastrigin’s Function", shift_data_file="f10_op.txt", matrix_data_file="f10_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / (2 * m_group))
    check_problem_size(problem_size)
    check_m_group("F10", problem_size, 2*m_group)
    global f10_matrix, f10_op_data, f10_matrix_file, f10_shift_file
    if shift_data_file != f10_shift_file and matrix_data_file != f10_matrix_file:
        f10_shift_file = shift_data_file
        f10_matrix_file = matrix_data_file
        f10_matrix = load_matrix_data__(matrix_data_file)
        f10_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f10_op_data[:1, :].reshape(-1)
        permu_data = (f10_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f10_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        z1 = dot(z[idx1], f10_matrix[:len(idx1), :len(idx1)])
        result += f3_rastrigin__(z1)
    idx2 = permu_data[int(problem_size / 2):problem_size]
    z2 = z[idx2]
    result += f3_rastrigin__(z2)
    return result


f11_matrix = None
f11_op_data = None
f11_matrix_file = ""
f11_shift_file = ""


def F11(solution=None, name="D/2m-group Shifted and m-rotated Ackley’s Function", shift_data_file="f11_op.txt", matrix_data_file="f11_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / (2 * m_group))
    check_problem_size(problem_size)
    check_m_group("F11", problem_size, 2*m_group)
    global f11_matrix, f11_op_data, f11_matrix_file, f11_shift_file
    if shift_data_file != f11_shift_file and matrix_data_file != f11_matrix_file:
        f11_shift_file = shift_data_file
        f11_matrix_file = matrix_data_file
        f11_matrix = load_matrix_data__(matrix_data_file)
        f11_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f11_op_data[:1, :].reshape(-1)
        permu_data = (f11_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f11_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        z1 = dot(z[idx1], f11_matrix[:len(idx1), :len(idx1)])
        result += f4_ackley__(z1)
    idx2 = permu_data[int(problem_size / 2):problem_size]
    z2 = z[idx2]
    result += f4_ackley__(z2)
    return result


f12_op_data = None
f12_shift_file = ""


def F12(solution=None, name="D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2", shift_data_file="f12_op.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / (2 * m_group))
    check_problem_size(problem_size)
    check_m_group("F12", problem_size, 2*m_group)
    global f12_op_data, f12_shift_file
    if shift_data_file != f12_shift_file:
        f12_shift_file = shift_data_file
        f12_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f12_op_data[:1, :].reshape(-1)
        permu_data = (f12_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f12_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f5_schwefel__(z[idx1])
    idx2 = permu_data[int(problem_size / 2):problem_size]
    result += f1_sphere__(z[idx2])
    return result


f13_op_data = None
f13_shift_file = ""


def F13(solution=None, name="D/2m-group Shifted m-dimensional Rosenbrock’s Function", shift_data_file="f13_op.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / (2 * m_group))
    check_problem_size(problem_size)
    check_m_group("F13", problem_size, 2*m_group)
    global f13_op_data, f13_shift_file
    if shift_data_file != f13_shift_file:
        f13_shift_file = shift_data_file
        f13_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f13_op_data[:1, :].reshape(-1)
        permu_data = (f13_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f13_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f6_rosenbrock__(z[idx1])
    idx2 = permu_data[int(problem_size / 2):problem_size]
    result += f1_sphere__(z[idx2])
    return result


f14_matrix = None
f14_op_data = None
f14_matrix_file = ""
f14_shift_file = ""


def F14(solution=None, name="D/2m-group Shifted and m-rotated Elliptic Function", shift_data_file="f14_op.txt", matrix_data_file="f14_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / m_group)
    check_problem_size(problem_size)
    check_m_group("F14", problem_size, m_group)
    global f14_matrix, f14_op_data, f14_matrix_file, f14_shift_file
    if shift_data_file != f14_shift_file and matrix_data_file != f14_matrix_file:
        f14_shift_file = shift_data_file
        f14_matrix_file = matrix_data_file
        f14_matrix = load_matrix_data__(matrix_data_file)
        f14_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f14_op_data[:1, :].reshape(-1)
        permu_data = (f14_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f14_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f2_elliptic__(dot(z[idx1], f14_matrix))
    return result


f15_matrix = None
f15_op_data = None
f15_matrix_file = ""
f15_shift_file = ""


def F15(solution=None, name="D/2m-group Shifted and m-rotated Rastrigin’s Function", shift_data_file="f15_op.txt", matrix_data_file="f15_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / m_group)
    check_problem_size(problem_size)
    check_m_group("F15", problem_size, m_group)
    global f15_matrix, f15_op_data, f15_matrix_file, f15_shift_file
    if shift_data_file != f15_shift_file and matrix_data_file != f15_matrix_file:
        f15_shift_file = shift_data_file
        f15_matrix_file = matrix_data_file
        f15_matrix = load_matrix_data__(matrix_data_file)
        f15_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f15_op_data[:1, :].reshape(-1)
        permu_data = (f15_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f15_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f3_rastrigin__(dot(z[idx1], f15_matrix))
    return result


f16_matrix = None
f16_op_data = None
f16_matrix_file = ""
f16_shift_file = ""


def F16(solution=None, name="D/2m-group Shifted and m-rotated Ackley’s Function", shift_data_file="f16_op.txt", matrix_data_file="f16_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / m_group)
    check_problem_size(problem_size)
    check_m_group("F16", problem_size, m_group)
    global f16_matrix, f16_op_data, f16_matrix_file, f16_shift_file
    if shift_data_file != f16_shift_file and matrix_data_file != f16_matrix_file:
        f16_shift_file = shift_data_file
        f16_matrix_file = matrix_data_file
        f16_matrix = load_matrix_data__(matrix_data_file)
        f16_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f16_op_data[:1, :].reshape(-1)
        permu_data = (f16_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f16_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f4_ackley__(dot(z[idx1], f16_matrix))
    return result


f17_op_data = None
f17_shift_file = ""


def F17(solution=None, name="D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2", shift_data_file="f17_op.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / m_group)
    check_problem_size(problem_size)
    check_m_group("F17", problem_size, m_group)
    global f17_op_data, f17_shift_file
    if shift_data_file != f17_shift_file:
        f17_shift_file = shift_data_file
        f17_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f17_op_data[:1, :].reshape(-1)
        permu_data = (f17_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f17_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f5_schwefel__(z[idx1])
    return result


f18_op_data = None
f18_shift_file = ""


def F18(solution=None, name="D/2m-group Shifted m-dimensional Rosenbrock’s Function", shift_data_file="f18_op.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / m_group)
    check_problem_size(problem_size)
    check_m_group("F18", problem_size, m_group)
    global f18_op_data, f18_shift_file
    if shift_data_file != f18_shift_file:
        f18_shift_file = shift_data_file
        f18_op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = f18_op_data[:1, :].reshape(-1)
        permu_data = (f18_op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = f18_op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f6_rosenbrock__(z[idx1])
    return result


f19_shift_data = None
f19_shift_file = ""


def F19(solution=None, name="Shifted Schwefel’s Problem 1.2", shift_data_file="f19_o.txt"):
    problem_size = len(solution)
    check_problem_size(problem_size)
    global f19_shift_data, f19_shift_file
    if f19_shift_file != shift_data_file:
        f19_shift_file = shift_data_file
        f19_shift_data = load_shift_data__(shift_data_file)
    shift_data = f19_shift_data[:problem_size]
    z = solution - shift_data
    return f5_schwefel__(z)


f20_shift_data = None
f20_shift_file = ""


def F20(solution=None, name="Shifted Rosenbrock’s Function", shift_data_file="f20_o.txt"):
    problem_size = len(solution)
    check_problem_size(problem_size)
    global f20_shift_data, f20_shift_file
    if shift_data_file != f20_shift_file:
        f20_shift_file = shift_data_file
        f20_shift_data = load_shift_data__(shift_data_file)
    shift_data = f20_shift_data[:problem_size]
    z = solution - shift_data
    return f6_rosenbrock__(z)
