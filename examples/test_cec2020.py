#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:19, 22/09/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2020.constant import benchmark_function as BF
from opfunu.cec.cec2020 import engineering
from numpy.random import uniform

for i in range(1, 26):
    out = BF(i)  # Get object contain information about problems
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]

    solution = uniform(xmin, xmax)  ## Create solution based on information above
    problem = "p" + str(i)  ## Choice the problem
    fx, gx, hx = getattr(engineering, problem)(solution)  ## Fitness function, constraint
    print("\n==============" + problem + "=================")
    print("fx:", fx)
    print("gx:", gx)
    print("hx:", hx)

