#!/usr/bin/env python
# Created by "Thieu" at 09:32, 21/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np
from mealpy.swarm_based import WOA


print("====================Test Damavandi")
ndim = 2
problem = opfunu.name_based.Damavandi(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Deb01")
ndim = 2
problem = opfunu.name_based.Deb01(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Deb03")
ndim = 18
problem = opfunu.name_based.Deb03(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Decanomial")
ndim = 2
problem = opfunu.name_based.Decanomial(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Deceptive")
ndim = 5
problem = opfunu.name_based.Deceptive(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test DeckkersAarts")
ndim = 2
problem = opfunu.name_based.DeckkersAarts(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test DeflectedCorrugatedSpring")
ndim = 5
problem = opfunu.name_based.DeflectedCorrugatedSpring(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test DeVilliersGlasser01")
ndim = 4
problem = opfunu.name_based.DeVilliersGlasser01(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test DeVilliersGlasser02")
ndim = 5
problem = opfunu.name_based.DeVilliersGlasser02(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test DixonPrice")
ndim = 5
problem = opfunu.name_based.DixonPrice(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Dolan")
ndim = 5
problem = opfunu.name_based.Dolan(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test DropWave")
ndim = 2
problem = opfunu.name_based.DropWave(ndim=ndim)
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
