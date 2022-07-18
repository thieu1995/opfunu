#!/usr/bin/env python
# Created by "Thieu" at 20:32, 18/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np


print("====================Test Ackley01")
problem = opfunu.name_based.Ackley01(ndim=25)
x = np.ones(25)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Ackley02")
problem = opfunu.name_based.Ackley02(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Ackley03")
problem = opfunu.name_based.Ackley03(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Adjiman")
problem = opfunu.name_based.Adjiman(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Alpine01")
problem = opfunu.name_based.Alpine01(ndim=14)
x = np.ones(14)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Alpine02")
problem = opfunu.name_based.Alpine02(ndim=3)
x = np.ones(3)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test AMGM")
problem = opfunu.name_based.AMGM(ndim=3)
x = np.ones(3)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))






