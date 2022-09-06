#!/usr/bin/env python
# Created by "Thieu" at 09:59, 13/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np


## Test CEC2022 F1
print("====================F1")
problem = opfunu.cec_based.F12022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F2
print("====================F2")
problem = opfunu.cec_based.F22022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F3
print("====================F3")
problem = opfunu.cec_based.F32022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F4
print("====================F4")
problem = opfunu.cec_based.F42022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F5
print("====================F5")
problem = opfunu.cec_based.F52022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F6
print("====================F6")
problem = opfunu.cec_based.F62022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F7
print("====================F7")
problem = opfunu.cec_based.F72022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F8
print("====================F8")
problem = opfunu.cec_based.F82022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F9
print("====================F9")
problem = opfunu.cec_based.F92022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F10
print("====================F10")
problem = opfunu.cec_based.F102022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F11
print("====================F11")
problem = opfunu.cec_based.F112022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2022 F12
print("====================F12")
problem = opfunu.cec_based.F122022(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))
