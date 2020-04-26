#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:26, 26/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

# 1, 2, 3, 17, 18, 20, 21,29, 30

import numpy as np
from opfunu.cec_basic.cec2014 import *
from opfunu.cec.cec_2014 import Functions

t1 = Functions()

problem_size = 20
sol = np.random.uniform(0, 1, 20)

print(F30(sol))

print(t1.C30(sol))