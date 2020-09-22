#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:18, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2005.root import Root
from numpy import cos


class Model(Root):
    def __init__(self, f_name="Shifted Expanded Griewank's plus Rosenbrock's Function (F8F2)", f_shift_data_file="data_EF8F2",
                 f_ext='.txt', f_bias=-130):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def __f8__(self, x=None):
        return x ** 2 / 4000 - cos(x) + 1

    def __f2__(self, x=None):
        return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        z = solution - shift_data + 1

        result = 0
        for i in range(0, problem_size):
            if i == problem_size - 1:
                result += self.__f8__(self.__f2__([z[i], z[0]]))
            else:
                result += self.__f8__(self.__f2__(z[i:i + 2]))
        return result + self.f_bias
