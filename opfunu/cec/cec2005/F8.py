#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:39, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2005.root import Root
from numpy import sum, dot, cos, exp, pi, e, sqrt


class Model(Root):
    def __init__(self, f_name="Shifted Rotated Ackley's Function with Global Optimum on Bounds", f_shift_data_file="data_ackley",
                 f_ext='.txt', f_bias=-140, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        if problem_size == 10 or problem_size == 30 or problem_size == 50:
            self.f_matrix = "ackley_M_D" + str(problem_size)
        else:
            print("CEC 2005 F8 function only support problem size 10, 30, 50")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        t1 = int(problem_size/2)
        for j in range(0, t1-1):
            shift_data[2*(j+1)-1] = -32 * shift_data[2*(j+1)]
        matrix = self.load_matrix_data(self.f_matrix)

        z = dot((solution - shift_data), matrix)
        result = -20 * exp(-0.2 * sum(z ** 2) / problem_size) - exp(sum(cos(2 * pi * z))) + 20 + e
        return result + self.f_bias





