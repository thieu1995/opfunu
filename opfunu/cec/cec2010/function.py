#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:29, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import seed, permutation
from numpy import dot, ones
from opfunu.cec.cec2010.utils import *


def F1(solution=None, name="Shifted Elliptic Function", shift_data_file="f01_o.txt"):
    problem_size = len(solution)
    check_problem_size(problem_size)
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    z = solution - shift_data
    return f2_elliptic__(z)


def F2(solution=None, name="Shifted Rastrigin’s Function", shift_data_file="f02_o.txt"):
    problem_size = len(solution)
    check_problem_size(problem_size)
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    z = solution - shift_data
    return f3_rastrigin__(z)


def F3(solution=None, name="Shifted Ackley’s Function", shift_data_file="f03_o.txt"):
    problem_size = len(solution)
    check_problem_size(problem_size)
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    z = solution - shift_data
    return f4_ackley__(z)


def F4(solution=None, name="Single-group Shifted and m-rotated Elliptic Function", shift_data_file="f04_op.txt", matrix_data_file="f04_m.txt", m_group=50):
    problem_size = len(solution)
    check_problem_size(problem_size)
    matrix = load_matrix_data__(matrix_data_file)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1,:].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    idx1 = permu_data[:m_group]
    idx2 = permu_data[m_group:]
    z_rot_elliptic = dot(z[idx1], matrix[:m_group, :m_group])
    z_elliptic = z[idx2]
    return f2_elliptic__(z_rot_elliptic) * 10**6 + f2_elliptic__(z_elliptic)


def F5(solution=None, name="Single-group Shifted and m-rotated Rastrigin’s Function", shift_data_file="f05_op.txt", matrix_data_file="f05_m.txt", m_group=50):
    problem_size = len(solution)
    check_problem_size(problem_size)
    matrix = load_matrix_data__(matrix_data_file)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    idx1 = permu_data[:m_group]
    idx2 = permu_data[m_group:]
    z_rot_rastrigin = dot(z[idx1], matrix[:m_group, :m_group])
    z_rastrigin = z[idx2]
    return f3_rastrigin__(z_rot_rastrigin) * 10 ** 6 + f3_rastrigin__(z_rastrigin)


def F6(solution=None, name="Single-group Shifted and m-rotated Ackley’s Function", shift_data_file="f06_op.txt", matrix_data_file="f06_m.txt", m_group=50):
    problem_size = len(solution)
    check_problem_size(problem_size)
    matrix = load_matrix_data__(matrix_data_file)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    idx1 = permu_data[:m_group]
    idx2 = permu_data[m_group:]
    z_rot_ackley = dot(z[idx1], matrix[:m_group, :m_group])
    z_ackley = z[idx2]
    return f4_ackley__(z_rot_ackley) * 10 ** 6 + f4_ackley__(z_ackley)


def F7(solution=None, name="Single-group Shifted m-dimensional Schwefel’s Problem 1.2", shift_data_file="f07_op.txt", m_group=50):
    problem_size = len(solution)
    check_problem_size(problem_size)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    idx1 = permu_data[:m_group]
    idx2 = permu_data[m_group:]
    z_schwefel = z[idx1]
    z_shpere = z[idx2]
    return f5_schwefel__(z_schwefel) * 10 ** 6 + f1_sphere__(z_shpere)


def F8(solution=None, name=" Single-group Shifted m-dimensional Rosenbrock’s Function", shift_data_file="f08_op.txt", m_group=50):
    problem_size = len(solution)
    check_problem_size(problem_size)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    idx1 = permu_data[:m_group]
    idx2 = permu_data[m_group:]
    z_rosenbrock = z[idx1]
    z_sphere = z[idx2]
    return f6_rosenbrock__(z_rosenbrock) * 10 ** 6 + f1_sphere__(z_sphere)


def F9(solution=None, name="D/2m-group Shifted and m-rotated Elliptic Function", shift_data_file="f09_op.txt", matrix_data_file="f09_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / (2 * m_group))
    check_problem_size(problem_size)
    check_m_group("F9", problem_size, 2*m_group)
    matrix = load_matrix_data__(matrix_data_file)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i*m_group:(i+1)*m_group]
        z1 = dot(z[idx1], matrix[:len(idx1), :len(idx1)])
        result += f2_elliptic__(z1)
    idx2 = permu_data[int(problem_size/2):problem_size]
    z2 = z[idx2]
    result += f2_elliptic__(z2)
    return result


def F10(solution=None, name="D/2m-group Shifted and m-rotated Rastrigin’s Function", shift_data_file="f10_op.txt", matrix_data_file="f10_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / (2 * m_group))
    check_problem_size(problem_size)
    check_m_group("F10", problem_size, 2*m_group)
    matrix = load_matrix_data__(matrix_data_file)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        z1 = dot(z[idx1], matrix[:len(idx1), :len(idx1)])
        result += f3_rastrigin__(z1)
    idx2 = permu_data[int(problem_size / 2):problem_size]
    z2 = z[idx2]
    result += f3_rastrigin__(z2)
    return result


def F11(solution=None, name="D/2m-group Shifted and m-rotated Ackley’s Function", shift_data_file="f11_op.txt", matrix_data_file="f11_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / (2 * m_group))
    check_problem_size(problem_size)
    check_m_group("F11", problem_size, 2*m_group)
    matrix = load_matrix_data__(matrix_data_file)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        z1 = dot(z[idx1], matrix[:len(idx1), :len(idx1)])
        result += f4_ackley__(z1)
    idx2 = permu_data[int(problem_size / 2):problem_size]
    z2 = z[idx2]
    result += f4_ackley__(z2)
    return result


def F12(solution=None, name="D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2", shift_data_file="f12_op.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / (2 * m_group))
    check_problem_size(problem_size)
    check_m_group("F12", problem_size, 2*m_group)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f5_schwefel__(z[idx1])
    idx2 = permu_data[int(problem_size / 2):problem_size]
    result += f1_sphere__(z[idx2])
    return result


def F13(solution=None, name="D/2m-group Shifted m-dimensional Rosenbrock’s Function", shift_data_file="f13_op.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / (2 * m_group))
    check_problem_size(problem_size)
    check_m_group("F13", problem_size, 2*m_group)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f6_rosenbrock__(z[idx1])
    idx2 = permu_data[int(problem_size / 2):problem_size]
    result += f1_sphere__(z[idx2])
    return result


def F14(solution=None, name="D/2m-group Shifted and m-rotated Elliptic Function", shift_data_file="f14_op.txt", matrix_data_file="f14_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / m_group)
    check_problem_size(problem_size)
    check_m_group("F14", problem_size, m_group)
    matrix = load_matrix_data__(matrix_data_file)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f2_elliptic__(dot(z[idx1], matrix))
    return result


def F15(solution=None, name="D/2m-group Shifted and m-rotated Rastrigin’s Function", shift_data_file="f15_op.txt", matrix_data_file="f15_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / m_group)
    check_problem_size(problem_size)
    check_m_group("F15", problem_size, m_group)
    matrix = load_matrix_data__(matrix_data_file)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f3_rastrigin__(dot(z[idx1], matrix))
    return result


def F16(solution=None, name="D/2m-group Shifted and m-rotated Ackley’s Function", shift_data_file="f16_op.txt", matrix_data_file="f16_m.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / m_group)
    check_problem_size(problem_size)
    check_m_group("F16", problem_size, m_group)
    matrix = load_matrix_data__(matrix_data_file)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f4_ackley__(dot(z[idx1], matrix))
    return result


def F17(solution=None, name="D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2", shift_data_file="f17_op.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / m_group)
    check_problem_size(problem_size)
    check_m_group("F17", problem_size, m_group)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f5_schwefel__(z[idx1])
    return result


def F18(solution=None, name="D/2m-group Shifted m-dimensional Rosenbrock’s Function", shift_data_file="f18_op.txt", m_group=50):
    problem_size = len(solution)
    epoch = int(problem_size / m_group)
    check_problem_size(problem_size)
    check_m_group("F18", problem_size, m_group)
    op_data = load_matrix_data__(shift_data_file)
    if problem_size == 1000:
        shift_data = op_data[:1, :].reshape(-1)
        permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
    else:
        seed(0)
        shift_data = op_data[:1, :].reshape(-1)[:problem_size]
        permu_data = permutation(problem_size)
    z = solution - shift_data
    result = 0.0
    for i in range(0, epoch):
        idx1 = permu_data[i * m_group:(i + 1) * m_group]
        result += f6_rosenbrock__(z[idx1])
    return result


def F19(solution=None, name="Shifted Schwefel’s Problem 1.2", shift_data_file="f19_o.txt"):
    problem_size = len(solution)
    check_problem_size(problem_size)
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    z = solution - shift_data
    return f5_schwefel__(z)


def F20(solution=None, name="Shifted Rosenbrock’s Function", shift_data_file="f20_o.txt"):
    problem_size = len(solution)
    check_problem_size(problem_size)
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    z = solution - shift_data
    return f6_rosenbrock__(z)
