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
from opfunu.cec.cec2014.unconstraint import Model

solution = np.random.uniform(0, 1, 10)


result1 = F1(solution)
print(result1)

result2 = Model(solution)
print(result2.F1())







