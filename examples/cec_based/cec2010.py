#!/usr/bin/env python
# Created by "Thieu" at 11:16, 02/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np

## Test CEC2010 F1
print("====================F1")
problem = opfunu.cec_based.F12010(ndim=100)
x = np.ones(100)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F2
print("====================F2")
problem = opfunu.cec_based.F22010(ndim=100)
x = np.ones(100)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F3
print("====================F3")
problem = opfunu.cec_based.F32010(ndim=100)
x = np.ones(100)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F4
print("====================F4")
problem = opfunu.cec_based.F42010(ndim=105)
x = np.ones(105)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F5
print("====================F5")
problem = opfunu.cec_based.F52010(ndim=105)
x = np.ones(105)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F6
print("====================F6")
problem = opfunu.cec_based.F62010(ndim=105)
x = np.ones(105)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F7
print("====================F7")
problem = opfunu.cec_based.F72010(ndim=105)
x = np.ones(105)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F8
print("====================F8")
problem = opfunu.cec_based.F82010(ndim=105)
x = np.ones(105)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F9
print("====================F9")
problem = opfunu.cec_based.F92010(ndim=105)
x = np.ones(105)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F10
print("====================F10")
problem = opfunu.cec_based.F102010(ndim=105)
x = np.ones(105)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F11
print("====================F11")
problem = opfunu.cec_based.F112010(ndim=120)
x = np.ones(120)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F12
print("====================F12")
problem = opfunu.cec_based.F122010(ndim=125)
x = np.ones(125)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F13
print("====================F13")
problem = opfunu.cec_based.F132010(ndim=125)
x = np.ones(125)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F14
print("====================F14")
problem = opfunu.cec_based.F142010(ndim=150)
x = np.ones(150)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F15
print("====================F15")
problem = opfunu.cec_based.F152010(ndim=175)
x = np.ones(175)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F16
print("====================F16")
problem = opfunu.cec_based.F162010(ndim=210)
x = np.ones(210)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F17
print("====================F17")
problem = opfunu.cec_based.F172010(ndim=236)
x = np.ones(236)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F18
print("====================F18")
problem = opfunu.cec_based.F182010(ndim=244)
x = np.ones(244)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F19
print("====================F19")
problem = opfunu.cec_based.F192010(ndim=250)
x = np.ones(250)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2010 F20
print("====================F20")
problem = opfunu.cec_based.F202010(ndim=255)
x = np.ones(255)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))
