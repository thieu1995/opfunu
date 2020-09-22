#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:43, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2005.root import Root
from numpy import sum
from numpy.random import normal


class Model(Root):
    def __init__(self, f_name="Shifted Schwefel's Problem 1.2 with Noise in Fitness", f_shift_data_file="data_schwefel_102",
                 f_ext='.txt', f_bias=-450, f_noise=True):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_noise = f_noise

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        shift_data = self.load_shift_data()[:problem_size]

        result = 0
        for i in range(0, problem_size):
            result += (sum(solution[:i] - shift_data[:i]))**2
        if self.f_noise:
            return result*(1 + 0.4* abs(normal(0, 1))) + self.f_bias
        else:
            return result + self.f_bias

