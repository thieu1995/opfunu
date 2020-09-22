#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:26, 24/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%

import pkg_resources
from pandas import read_csv
from copy import deepcopy
from numpy import sum, exp, cos, sin, sqrt, e, pi, abs, round, log, power, fill_diagonal, zeros, reshape


class BasicFunction:
    def __init__(self, cec_type="cec2013", dimensions=(2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)):
        self.cec_type = cec_type
        self.dimensions = dimensions
        self.CURRENT_PATH = pkg_resources.resource_filename("opfunu", "cec/" + cec_type + "/")
        self.SUPPORT_PATH_DATA = pkg_resources.resource_filename("opfunu", "cec/" + cec_type + "/support_data/")

    def load_shift_data__(self, data_file=None):
        data = read_csv(self.SUPPORT_PATH_DATA + data_file, delimiter='\s+', index_col=False, header=None)
        return data.values.reshape((-1))

    def load_matrix_data__(self, data_file=None):
        data = read_csv(self.SUPPORT_PATH_DATA + data_file, delimiter='\s+', index_col=False, header=None)
        return data.values

    def check_dimensions(self, dimension):
        if dimension not in self.dimensions:
            print("{} only support problem size: {}".format(self.cec_type.upper(), self.dimensions))
            exit(0)

    def ackley__(self, solution=None):
        solution = solution.reshape((-1))
        return -20 * exp(-0.2 * sqrt(sum(solution ** 2) / len(solution))) - exp(sum(cos(2 * pi * solution)) / len(solution)) + 20 + e

    def bent_cigar__(self, solution=None):
        solution = solution.reshape((-1))
        return solution[0] ** 2 + 10 ** 6 * sum(solution[1:] ** 2)

    def discus__(self, solution=None):
        solution = solution.reshape((-1))
        return 10 ** 6 * solution[0] ** 2 + sum(solution[1:] ** 2)

    def weierstrass__(self, solution=None, a=0.5, b=3, k_max=20):
        solution = solution.reshape((-1))
        result = 0.0
        for i in range(0, len(solution)):
            t1 = sum([a ** k * cos(2 * pi * b ** k * (solution[i] + 0.5)) for k in range(0, k_max)])
            result += t1
        t2 = len(solution) * sum([a ** k * cos(2 * pi * b ** k * 0.5) for k in range(0, k_max)])
        return result - t2

    def katsuura__(self, solution=None):
        solution = solution.reshape((-1))
        result = 1.0
        for i in range(0, len(solution)):
            t1 = sum([abs(2 ** j * solution[i] - round(2 ** j * solution[i])) / 2 ** j for j in range(1, 33)])
            result *= (1 + (i + 1) * t1) ** (10.0 / len(solution) ** 1.2)
        return (result - 1) * 10 / len(solution) ** 2

    def happy_cat__(self, solution=None):
        solution = solution.reshape((-1))
        return (abs(sum(solution ** 2) - len(solution))) ** 0.25 + (0.5 * sum(solution ** 2) + sum(solution)) / len(solution) + 0.5

    def hgbat__(self, solution=None):
        solution = solution.reshape((-1))
        return (abs(sum(solution ** 2) ** 2 - sum(solution) ** 2)) ** 0.5 + (0.5 * sum(solution ** 2) + sum(solution)) / len(solution) + 0.5

    def rosenbrock__(self, solution=None):
        solution = solution.reshape((-1))
        result = 0.0
        for i in range(len(solution) - 1):
            result += 100 * (solution[i] ** 2 - solution[i + 1]) ** 2 + (solution[i] - 1) ** 2
        return result

    def rastrigin__(self, solution=None):
        solution = solution.reshape((-1))
        return sum(solution ** 2 - 10 * cos(2 * pi * solution) + 10)

    def elliptic__(self, solution=None):
        solution = solution.reshape((-1))
        result = 0
        for i in range(len(solution)):
            result += (10 ** 6) ** (i / (len(solution) - 1)) * solution[i] ** 2
        return result

    def modified_schwefel__(self, solution=None):
        solution = solution.reshape((-1))
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

    def expanded_scaffer__(self, solution=None):
        solution = solution.reshape((-1))
        def __xy__(x, y):
            return 0.5 + (sin(sqrt(x ** 2 + y ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2

        result = __xy__(solution[-1], solution[0])
        for i in range(0, len(solution) - 1):
            result += __xy__(solution[i], solution[i + 1])
        return result

    def griewank__(self, solution=None):
        solution = solution.reshape((-1))
        result = sum(solution ** 2) / 4000
        temp = 1.0
        for i in range(len(solution)):
            temp *= cos(solution[i] / sqrt(i + 1))
        return result - temp + 1

    def expanded_griewank__(self, solution=None):
        solution = solution.reshape((-1))
        def __rosenbrock__(x=None, y=None):
            return 100 * (x ** 2 - y) ** 2 + (x - 1) ** 2

        def __griewank__(z=None):
            return z ** 2 / 4000 - cos(z / sqrt(1)) + 1

        result = __griewank__(__rosenbrock__(solution[-1], solution[0]))
        for i in range(0, len(solution) - 1):
            result += __griewank__(__rosenbrock__(solution[i], solution[i + 1]))
        return result

    def different_powers__(self, solution=None):
        solution = solution.reshape((-1))
        result = 0.0
        for i in range(0, len(solution)):
            result += power(abs(solution[i]), 2 + 4 * i / (len(solution) - 1))
        return sqrt(result)

    def schaffers_f7__(self, solution=None):
        solution = solution.reshape((-1))
        result = 0.0
        for i in range(0, len(solution) - 1):
            result += power(solution[i] ** 2 + solution[i + 1] ** 2, 0.25) * (1 + sin(50 * sqrt(solution[i] ** 2 + solution[i + 1] ** 2) ** 0.2) ** 2)
        return (result / (len(solution) - 1)) ** 2

    def osz_func__(self, solution=None):
        solution = solution.reshape((-1))
        solution_new = deepcopy(solution)
        for idx in range(0, len(solution)):
            if idx == 0 or idx == len(solution) - 1:
                c1 = 5.5
                c2 = 3.1
                x_sign = 1
                x_star = log(abs(solution[idx]))
                if solution[idx] < 0:
                    x_sign = -1
                elif solution[idx] > 0:
                    c1 = 10
                    c2 = 7.9
                else:
                    x_sign = 0
                    x_star = 0
                solution_new[idx] = x_sign * exp(x_star + 0.049 * (sin(c1 * x_star) + sin(c2 * x_star)))
        return solution_new

    def asy_func__(self, solution=None, beta=0.5):
        solution = solution.reshape((-1))
        solution_new = deepcopy(solution)
        for idx in range(len(solution)):
            if solution[idx] > 0:
                solution_new[idx] = power(solution[idx], 1 + beta * idx * sqrt(solution[idx]) / (len(solution) - 1))
        return solution_new

    def create_diagonal_matrix__(self, size=None, alpha=10):
        matrix = zeros((size, size), float)
        temp = [power(alpha, i / (2 * (size - 1))) for i in range(0, size)]
        fill_diagonal(matrix, temp)
        return matrix
