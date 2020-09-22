#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:54, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2005.root import Root
from numpy import sum, dot, cos, pi


class Model(Root):
    def __init__(self, f_name="Shifted Rotated Weierstrass Function", f_shift_data_file="data_weierstrass",
                 f_ext='.txt', f_bias=90, f_matrix=None, a=0.5, b=3, k_max=20):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix
        self.a = a
        self.b = b
        self.k_max = k_max

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        if problem_size == 10 or problem_size == 30 or problem_size == 50:
            self.f_matrix = "weierstrass_M_D" + str(problem_size)
        else:
            print("CEC 2005 F11 function only support problem size 10, 30, 50")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        matrix = self.load_matrix_data(self.f_matrix)
        z = dot((solution - shift_data), matrix)

        result = 0.0
        for i in range(0, problem_size):
            result += sum([self.a ** k * cos(2 * pi * self.b ** k * (z[i] + 0.5)) for k in range(0, self.k_max)])
        result -= problem_size * sum([self.a ** k * cos(2 * pi * self.b ** k * 0.5) for k in range(0, self.k_max)])
        return result + self.f_bias

