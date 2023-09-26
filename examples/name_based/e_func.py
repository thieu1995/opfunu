#!/usr/bin/env python
# Created by "Thieu" at 11:04, 21/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np
from mealpy.swarm_based import WOA


print("====================Test Easom")
ndim = 2
problem = opfunu.name_based.Easom(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test ElAttarVidyasagarDutta")
ndim = 2
problem = opfunu.name_based.ElAttarVidyasagarDutta(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test EggCrate")
ndim = 2
problem = opfunu.name_based.EggCrate(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test EggHolder")
ndim = 11
problem = opfunu.name_based.EggHolder(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Exponential")
ndim = 11
problem = opfunu.name_based.Exponential(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Exp2")
ndim = 2
problem = opfunu.name_based.Exp2(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Eckerle4")
ndim = 3
problem = opfunu.name_based.Eckerle4(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


problem_dict = {
    "fit_func": problem.evaluate,
    "lb": problem.lb,
    "ub": problem.ub,
    "minmax": "min",
    "log_to": "None",
}

model = WOA.OriginalWOA(epoch=1000, pop_size=50)
best_position, best_fitness_value = model.solve(problem_dict)
print(best_position, best_fitness_value)
