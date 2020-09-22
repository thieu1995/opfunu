#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:17, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2008.root import Root


class Model(Root):
    def __init__(self, f_name="Shifted Rosenbrock’s Function", f_shift_data_file="rosenbrock_shift_func_data", f_ext='.txt', f_bias=390, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        z = solution - shift_data + 1
        result = 0
        for i in range(0, problem_size-1):
            result += 100*(z[i]**2 - z[i+1])**2 + (z[i] - 1)**2
        return result + self.f_bias

