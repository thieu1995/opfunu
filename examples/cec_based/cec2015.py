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


## Test CEC2015 F6
print("====================F6")
problem = opfunu.cec_based.F62015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F6
print("====================F6")
problem = opfunu.cec_based.F62015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F7
print("====================F7")
problem = opfunu.cec_based.F72015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F8
print("====================F7")
problem = opfunu.cec_based.F82015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F9
print("====================F9")
problem = opfunu.cec_based.F92015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F10
print("====================F10")
problem = opfunu.cec_based.F102015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F11
print("====================F11")
problem = opfunu.cec_based.F112015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F12
print("====================F12")
problem = opfunu.cec_based.F122015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F13
print("====================F13")
problem = opfunu.cec_based.F132015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F14
print("====================F14")
problem = opfunu.cec_based.F142015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2015 F15
print("====================F15")
problem = opfunu.cec_based.F152015(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))
