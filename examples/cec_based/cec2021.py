#!/usr/bin/env python
# Created by "Thieu" at 09:23, 13/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np


## Test CEC2021 F1
print("====================F1")
problem = opfunu.cec_based.F12021(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2021 F2
print("====================F2")
problem = opfunu.cec_based.F22021(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2021 F3
print("====================F3")
problem = opfunu.cec_based.F32021(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2021 F4
print("====================F4")
problem = opfunu.cec_based.F42021(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2021 F5
print("====================F5")
problem = opfunu.cec_based.F52021(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2021 F6
print("====================F6")
problem = opfunu.cec_based.F62021(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2021 F7
print("====================F7")
problem = opfunu.cec_based.F72021(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2021 F8
print("====================F8")
problem = opfunu.cec_based.F82021(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2021 F9
print("====================F9")
problem = opfunu.cec_based.F92021(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2021 F10
print("====================F10")
problem = opfunu.cec_based.F102021(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))
