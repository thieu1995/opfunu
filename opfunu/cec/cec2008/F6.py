#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:21, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2008.root import Root
from numpy import sqrt, exp, cos, pi, e, sum


class Model(Root):
    def __init__(self, f_name="Shifted Ackley’s Function", f_shift_data_file="ackley_shift_func_data", f_ext='.txt', f_bias=-140):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        z = solution - shift_data

        return -20*exp(-0.2*sqrt(sum(z**2)/problem_size)) - exp(sum(cos(2*pi*solution))/problem_size) + 20 + e + self.f_bias

