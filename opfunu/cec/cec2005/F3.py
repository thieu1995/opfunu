#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:17, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2005.root import Root
from numpy import dot


class Model(Root):
    def __init__(self, f_name="Shifted Rotated High Conditioned Elliptic Function", f_shift_data_file="data_high_cond_elliptic_rot",
                 f_ext='.txt', f_bias=-450, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        if problem_size == 10 or problem_size == 30 or problem_size == 50:
            self.f_matrix = "elliptic_M_D" + str(problem_size)
        else:
            print("CEC 2005 F8 function only support problem size 10, 30, 50")
            return 1

        shift_data = self.load_shift_data()[:problem_size]
        matrix = self.load_matrix_data(self.f_matrix)

        z = (dot((solution - shift_data), matrix))**2
        result = 0
        for i in range(0, problem_size):
            result += (10**6) ** (i / (problem_size - 1)) * z[i]**2
        return result + self.f_bias

