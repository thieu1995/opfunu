#!/usr/bin/env python
# Created by "Thieu" at 20:18, 19/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np
from mealpy.swarm_based import WOA


print("====================Test CamelThreeHump")
ndim = 2
problem = opfunu.name_based.CamelThreeHump(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test CamelSixHump")
ndim = 2
problem = opfunu.name_based.CamelSixHump(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test ChenBird")
ndim = 2
problem = opfunu.name_based.ChenBird(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test ChenV")
ndim = 2
problem = opfunu.name_based.ChenV(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Chichinadze")
ndim = 2
problem = opfunu.name_based.Chichinadze(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test ChungReynolds")
ndim = 7
problem = opfunu.name_based.ChungReynolds(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Cigar")
ndim = 7
problem = opfunu.name_based.Cigar(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Cola")
ndim = 17
problem = opfunu.name_based.Cola(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Colville")
ndim = 4
problem = opfunu.name_based.Colville(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Corana")
ndim = 4
problem = opfunu.name_based.Corana(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test CosineMixture")
ndim = 7
problem = opfunu.name_based.CosineMixture(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test CrossInTray")
ndim = 2
problem = opfunu.name_based.CrossInTray(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test CrossLegTable")
ndim = 2
problem = opfunu.name_based.CrossLegTable(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test CrownedCross")
ndim = 2
problem = opfunu.name_based.CrownedCross(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Csendes")
ndim = 7
problem = opfunu.name_based.Csendes(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))



print("====================Test Cube")
ndim = 2
problem = opfunu.name_based.Cube(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


problem_dict = {
    "fit_func": problem.evaluate,
    "lb": problem.lb.tolist(),
    "ub": problem.ub.tolist(),
    "minmax": "min",
    "log_to": "None",
}

model = WOA.OriginalWOA(epoch=1000, pop_size=50)
best_position, best_fitness_value = model.solve(problem_dict)
print(best_position, best_fitness_value)
