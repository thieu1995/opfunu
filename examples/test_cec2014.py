#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 02:18, 22/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from opfunu.cec.cec2014.function import *
from opfunu.cec.cec2014.unconstraint2 import Model as MD2
from opfunu.cec.cec2014.unconstraint import Model as MD

problem_size = 10
solution = np.random.uniform(0, 1, problem_size)


result1 = F1(solution)
print(result1)

result2 = MD2(solution)
print(result2.F1())

result3 = MD(problem_size)
print(result3.F1(solution))






