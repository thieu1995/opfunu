#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:50, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2008.root import Root
from numpy import cos, sqrt


class Model(Root):
    def __init__(self, f_name="Shifted Griewank’s Function", f_shift_data_file="griewank_shift_func_data", f_ext='.txt', f_bias=-180):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        z = solution - shift_data
        result = sum(z**2/4000)
        temp = 1.0
        for i in range(0, problem_size):
            temp *= cos(z[i] / sqrt(i+1))
        return result - temp + 1 + self.f_bias

