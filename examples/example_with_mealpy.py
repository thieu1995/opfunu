#!/usr/bin/env python
# Created by "Thieu" at 00:36, 30/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.bio_based import SMA
from opfunu.name_based import Ackley02


ackey = Ackley02()

problem_dict1 = {
    "fit_func": ackey.evaluate,
    "lb": ackey.lb.tolist(),
    "ub": ackey.ub.tolist(),
    "minmax": "min",
    "log_to": None,
    "save_population": False,
}

## Run the algorithm
model = SMA.BaseSMA(problem_dict1, epoch=100, pop_size=50, pr=0.03)
best_position, best_fitness = model.solve()
print(f"Best solution: {best_position}, Best fitness: {best_fitness}")

print(ackey.n_fe)
print(ackey.f_global)
print(ackey.x_global)
