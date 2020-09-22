#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:45, 03/08/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from opfunu.cec_basic.cec2014 import *
from mealpy.swarm_based.WOA import BaseWOA
from mealpy.swarm_based.SpaSA import BaseSpaSA

# list_funcs = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17,
#               F18, F19, F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30]

list_funcs = [F23,]

problem_size = 10
solution = -1 * np.ones(problem_size)

for F in list_funcs:
    result1 = F(solution)
    print(result1)

test1 = F14(solution, shift_num=1, rotate_num=1)
print(test1)

temp = BaseSpaSA(F30, problem_size=10, domain_range=(-100, 100), log=True, epoch=1000, pop_size=100)
temp._train__()

temp = BaseWOA(F23, problem_size=10, domain_range=(-100, 100), log=True, epoch=1000, pop_size=100)
temp._train__()



