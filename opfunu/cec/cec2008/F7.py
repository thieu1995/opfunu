#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:29, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2008.root import Root
from numpy import sum
from numpy.random import seed, choice, uniform


class Model(Root):
    def __init__(self, f_name="FastFractal “DoubleDip” Function", f_shift_data_file=None, f_ext='.txt', f_bias=None, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix

    def _main__(self, solution=None):
        seed(0)
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1

        def __doubledip__(x, c, s):
            if -0.5 < x < 0.5:
                return (-6144*(x - c)**6 + 3088*(x - c)**4 - 392*(x - c)**2 + 1)*s
            else:
                return 0

        def __fractal1d__(x):
            result1 = 0.0
            for k in range(1, 4):
                result2 = 0.0
                upper = 2**(k-1)
                for t in range(1, upper):
                    selected = choice([0, 1, 2])
                    result2 += sum([ __doubledip__(x, uniform(), 1.0 / (2**(k-1)*(2-uniform()))) for _ in range(0, selected)])
                result1 += result2
            return result1

        def __twist__(y):
            return 4*(y**4 - 2*y**3 + y**2)

        result = solution[-1] + __twist__(solution[0])
        for i in range(0, problem_size-1):
            x = solution[i] + __twist__(solution[i%problem_size + 1])
            result += __fractal1d__(x)
        return result
