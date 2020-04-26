#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 22:21, 25/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from opfunu.cec.cec2013.unconstraint import Model

problem_size = 10
solution = np.random.uniform(0, 1, problem_size)
func = Model(problem_size)

print(func.F1(solution))
