#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:07, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2005.root import Root
from numpy import dot, sin, cos
from pandas import read_csv


class Model(Root):
    def __init__(self, f_name="Schwefel's Problem 2.13", f_shift_data_file="data_schwefel_213",
                 f_ext='.txt', f_bias=-460):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def load_shift_data(self):
        data = read_csv(self.support_path_data + self.f_shift_data_file + self.f_ext, delimiter='\s+', index_col=False, header=None)
        data = data.values
        a_matrix = data[:100, :]
        b_matrix = data[100:200, :]
        shift_data = data[200:, :]
        return shift_data, a_matrix, b_matrix

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        shift_data, a_matrix, b_matrix = self.load_shift_data()
        shift_data = shift_data.reshape(-1)[:problem_size]
        a_matrix = a_matrix[:problem_size, :problem_size]
        b_matrix = b_matrix[:problem_size, :problem_size]

        result = 0.0
        for i in range(0, problem_size):
            t1 = dot(a_matrix[i], sin(shift_data)) + dot(b_matrix[i], cos(shift_data))
            t2 = dot(a_matrix[i], sin(solution)) + dot(b_matrix[i], cos(solution))
            result += (t1-t2)**2
        return result + self.f_bias

