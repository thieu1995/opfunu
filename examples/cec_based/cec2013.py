#!/usr/bin/env python
# Created by "Thieu" at 18:30, 02/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np

## Test CEC2013 F1
print("====================F1")
problem = opfunu.cec_based.F12013(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2013 F2
print("====================F2")
problem = opfunu.cec_based.F22013(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2013 F3
print("====================F3")
problem = opfunu.cec_based.F32013(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2013 F4
print("====================F4")
problem = opfunu.cec_based.F42013(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2013 F5
print("====================F5")
problem = opfunu.cec_based.F52013(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))
















