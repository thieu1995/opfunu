#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:50, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2005.root import Root
from numpy import dot, max, abs, array
from pandas import read_csv


class Model(Root):
    def __init__(self, f_name="Schwefel's Problem 2.6 with Global Optimum on Bounds", f_shift_data_file="data_schwefel_206",
                 f_ext='.txt', f_bias=-310):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def load_shift_data(self):
        data = read_csv(self.support_path_data + self.f_shift_data_file + self.f_ext, delimiter='\s+', index_col=False, header=None)
        data = data.values
        shift_data = data[:1, :]
        matrix_data = data[1:, :]
        return shift_data, matrix_data

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        shift_data, matrix_data = self.load_shift_data()
        shift_data = shift_data.reshape(-1)[:problem_size]
        t1 = int(0.25 * problem_size) + 1
        t2 = int(0.75 * problem_size)
        shift_data[:t1] = -100
        shift_data[t2:] = 100
        matrix_data = matrix_data[:problem_size, :problem_size]

        result = []
        for i in range(0, problem_size):
            temp = abs(dot(matrix_data[i], solution) - dot(matrix_data[i], shift_data))
            result.append(temp)
        return max(array(result)) + self.f_bias


