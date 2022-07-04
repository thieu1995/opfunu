#!/usr/bin/env python
# Created by "Thieu" at 15:44, 04/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np


## Test CEC2014 F1
print("====================F1")
problem = opfunu.cec_based.F12014(ndim=100)
x = np.ones(100)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2014 F2
print("====================F2")
problem = opfunu.cec_based.F22014(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2014 F3
print("====================F3")
problem = opfunu.cec_based.F32014(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2014 F4
print("====================F4")
problem = opfunu.cec_based.F42014(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2014 F5
print("====================F5")
problem = opfunu.cec_based.F52014(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2014 F6
print("====================F6")
problem = opfunu.cec_based.F62014(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2014 F7
print("====================F7")
problem = opfunu.cec_based.F72014(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2014 F8
print("====================F8")
problem = opfunu.cec_based.F82014(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2014 F9
print("====================F9")
problem = opfunu.cec_based.F92014(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2014 F10
print("====================F10")
problem = opfunu.cec_based.F102014(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2014 F11
print("====================F11")
problem = opfunu.cec_based.F112014(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


