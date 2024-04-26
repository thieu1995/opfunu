#!/usr/bin/env python
# Created by "Thieu" at 00:36, 30/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## Examples with Mealpy <= 2.5.4

from mealpy.bio_based import SMA
from opfunu.name_based import Ackley02


ackey = Ackley02()

problem_dict1 = {
    "fit_func": ackey.evaluate,
    "lb": ackey.lb,
    "ub": ackey.ub,
    "minmax": "min",
    "log_to": None,
    "save_population": False,
}

## Run the algorithm
model = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03)
best_position, best_fitness = model.solve(problem_dict1)
print(f"Best solution: {best_position}, Best fitness: {best_fitness}")

print(ackey.n_fe)
print(ackey.f_global)
print(ackey.x_global)


## Examples with Mealpy >= 3.0.0

from opfunu.cec_based import cec2017
f3 = cec2017.F32017(ndim=30)

from mealpy import GA, FloatVar

problem = {
    "obj_func": f3.evaluate,
    "bounds": FloatVar(lb=f3.lb, ub=f3.ub),
    "minmax": "min",
}
model = GA.BaseGA(epoch=100, pop_size=50)
gbest = model.solve(problem_dict1)
print(f"Solution: {gbest.solution}, Fit: {gbest.target.fitness}")
