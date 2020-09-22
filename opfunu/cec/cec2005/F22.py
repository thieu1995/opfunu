#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:09, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2005.root import Root
from numpy import sum, dot, sin, sqrt, array, cos, pi, exp, ones, max


class Model(Root):
    def __init__(self, f_name="Rotated Hybrid Composition Function 3 with High Condition Number Matrix",
                 f_shift_data_file="data_hybrid_func3",
                 f_ext='.txt', f_bias=360, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix

    def __f12__(self, solution=None):
        def __xy__(x, y):
            return 0.5 + (sin(sqrt(x ** 2 + y ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2

        result = __xy__(solution[-1], solution[0])
        for i in range(0, len(solution) - 1):
            result += __xy__(solution[i], solution[i + 1])
        return result

    def __f34__(self, solution=None):
        return sum(solution ** 2 - 10 * cos(2 * pi * solution) + 10)

    def __f56__(self, solution=None):
        def __f8__(x):
            return x ** 2 / 4000 - cos(x / sqrt(x)) + 1

        def __f2__(x, y):
            return 100 * (x ** 2 - y) ** 2 + (x - 1) ** 2

        result = __f8__(__f2__(solution[-1], solution[0]))
        for i in range(0, len(solution) - 1):
            result += __f8__(__f2__(solution[i], solution[i + 1]))
        return result

    def __f78__(self, solution=None, a=0.5, b=3, k_max=20):
        result = 0.0
        for i in range(len(solution)):
            result += sum([a ** k * cos(2 * pi * b ** k * (solution + 0.5)) for k in range(0, k_max)])
        return result - len(solution) * sum([a ** k * cos(2 * pi * b ** k * 0.5) for k in range(0, k_max)])

    def __f910__(self, solution=None):
        result = sum(solution ** 2) / 4000
        temp = 1.0
        for i in range(len(solution)):
            temp *= cos(solution[i] / sqrt(i + 1))
        return result - temp + 1

    def __fi__(self, solution=None, idx=None):
        if idx == 0 or idx == 1:
            return self.__f12__(solution)
        elif idx == 2 or idx == 3:
            return self.__f34__(solution)
        elif idx == 4 or idx == 5:
            return self.__f56__(solution)
        elif idx == 6 or idx == 7:
            return self.__f78__(solution)
        else:
            return self.__f910__(solution)

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        if problem_size == 10 or problem_size == 30 or problem_size == 50:
            self.f_matrix = "hybrid_func3_HM_D" + str(problem_size)
        else:
            print("CEC 2005 F22 function only support problem size 10, 30, 50")
            return 1
        num_funcs = 10
        C = 2000
        xichma = array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        lamda = array([5.0 * 5.0 / 100.0, 5.0 / 100.0, 5.0 * 1.0, 1.0, 5.0 * 1.0, 1.0, 5.0 * 10.0, 10.0, 5.0 * 5.0 / 200.0, 5.0 / 200.0])
        bias = array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
        y = 5 * ones(problem_size)
        shift_data = self.load_matrix_data(self.f_shift_data_file)
        shift_data = shift_data[:, :problem_size]
        matrix = self.load_matrix_data(self.f_matrix)

        weights = ones(num_funcs)
        fits = ones(num_funcs)
        for i in range(0, num_funcs):
            w_i = exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
            z = dot((solution - shift_data[i]) / lamda[i], matrix[i * problem_size:(i + 1) * problem_size, :])
            fit_i = self.__fi__(z, i)
            f_maxi = self.__fi__(dot((y / lamda[i]), matrix[i * problem_size:(i + 1) * problem_size, :]), i)
            fit_i = C * fit_i / f_maxi

            weights[i] = w_i
            fits[i] = fit_i

        sw = sum(weights)
        maxw = max(weights)

        for i in range(0, num_funcs):
            if weights[i] != maxw:
                weights[i] = weights[i] * (1 - maxw ** 10)
            weights[i] = weights[i] / sw

        fx = sum(dot(weights, (fits + bias)))
        return fx + self.f_bias
