#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:45, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

# f8

import numpy as np
from opfunu.cec.cec2005.F1 import Model

t1 = Model()

temp = np.array([0.5, 1, 1.5, 2, 3, 0.9, 1.2, 2, 1, 5])

result = t1._main__(temp)


print(result)