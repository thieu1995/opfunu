#!/usr/bin/env python
# Created by "Thieu" at 16:37, 12/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np


## Test CEC2019 F1
print("====================F1")
problem = opfunu.cec_based.F12019(ndim=9)
x = np.ones(9)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2019 F2
print("====================F2")
problem = opfunu.cec_based.F22019(ndim=16)
x = np.ones(16)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
# print(problem.is_succeed(problem.x_global))


## Test CEC2019 F3
print("====================F3")
problem = opfunu.cec_based.F32019(ndim=18)
x = np.ones(18)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2019 F4
print("====================F4")
problem = opfunu.cec_based.F42019(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2019 F5
print("====================F5")
problem = opfunu.cec_based.F52019(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2019 F6
print("====================F6")
problem = opfunu.cec_based.F62019(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2019 F7
print("====================F7")
problem = opfunu.cec_based.F72019(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2019 F8
print("====================F8")
problem = opfunu.cec_based.F82019(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2019 F9
print("====================F9")
problem = opfunu.cec_based.F92019(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2019 F10
print("====================F10")
problem = opfunu.cec_based.F102019(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))
