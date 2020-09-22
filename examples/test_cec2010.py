#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:28, 21/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from opfunu.cec.cec2010.function import *

temp = np.random.uniform(0, 1, 1000)

func = F18
epoch = 100

problem_size = 100

from mealpy.swarm_based.WOA import BaseWOA
from mealpy.swarm_based.SpaSA import BaseSpaSA
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.swarm_based.GWO import BaseGWO
from mealpy.human_based.TLO import BaseTLO
from mealpy.human_based.QSA import ImprovedQSA
from mealpy.physics_based.EFO import BaseEFO

temp1 = BaseTLO(func, problem_size=problem_size, domain_range=(-100, 100), log=True, epoch=epoch, pop_size=50)
temp1._train__()

temp1 = BaseSpaSA(func, problem_size=problem_size, domain_range=(-100, 100), log=True, epoch=epoch, pop_size=50)
temp1._train__()

temp2 = BaseEFO(func, problem_size=problem_size, domain_range=(-100, 100), log=True, epoch=epoch, pop_size=50)
temp2._train__()

temp2 = ImprovedQSA(func, problem_size=problem_size, domain_range=(-100, 100), log=True, epoch=epoch, pop_size=50)
temp2._train__()

