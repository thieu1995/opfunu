#!/usr/bin/env python
# Created by "Thieu" at 14:52, 07/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np


## Test CEC2015 F1
print("====================F1")
problem = opfunu.cec_based.F12015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F2
print("====================F2")
problem = opfunu.cec_based.F22015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F3
print("====================F3")
problem = opfunu.cec_based.F32015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F4
print("====================F4")
problem = opfunu.cec_based.F42015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F5
print("====================F5")
problem = opfunu.cec_based.F52015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))






