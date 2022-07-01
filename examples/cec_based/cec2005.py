#!/usr/bin/env python
# Created by "Thieu" at 20:56, 30/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np

## Test CEC2005 F1
print("====================F1")
problem = opfunu.cec_based.F12005(ndim=25)
x = np.ones(25)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F2
print("====================F2")
problem = opfunu.cec_based.F22005(ndim=23)
x = np.ones(23)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F3
print("====================F3")
problem = opfunu.cec_based.F32005(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F4
print("====================F4")
problem = opfunu.cec_based.F42005(ndim=100)
x = np.ones(100)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F5
print("====================F5")
problem = opfunu.cec_based.F52005(ndim=20)
x = np.ones(20)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F6
print("====================F6")
problem = opfunu.cec_based.F62005(ndim=14)
x = np.ones(14)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F7
print("====================F7")
problem = opfunu.cec_based.F72005(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F8
print("====================F8")
problem = opfunu.cec_based.F82005(ndim=10)
x = np.ones(10) * 2
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F9
print("====================F9")
problem = opfunu.cec_based.F92005(ndim=13)
x = np.ones(13) * 2
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F10
print("====================F10")
problem = opfunu.cec_based.F102005(ndim=30)
x = np.ones(30)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F11
print("====================F11")
problem = opfunu.cec_based.F112005(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F12
print("====================F12")
problem = opfunu.cec_based.F122005(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F13
print("====================F13")
problem = opfunu.cec_based.F132005(ndim=25)
x = np.ones(25)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F14
print("====================F14")
problem = opfunu.cec_based.F142005(ndim=10)
x = np.ones(10)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F15
print("====================F15")
problem = opfunu.cec_based.F152005(ndim=12)
x = np.ones(12)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F16
print("====================F16")
problem = opfunu.cec_based.F162005(ndim=10)
x = np.ones(10)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F17
print("====================F17")
problem = opfunu.cec_based.F172005(ndim=10)
x = np.ones(10)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F18
print("====================F18")
problem = opfunu.cec_based.F182005(ndim=30)
x = np.ones(30)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F19
print("====================F19")
problem = opfunu.cec_based.F192005(ndim=10)
x = np.ones(10)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F20
print("====================F20")
problem = opfunu.cec_based.F202005(ndim=10)
x = np.ones(10)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F21
print("====================F21")
problem = opfunu.cec_based.F212005(ndim=10)
x = np.ones(10)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F22
print("====================F22")
problem = opfunu.cec_based.F222005(ndim=10)
x = np.ones(10)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F23
print("====================F23")
problem = opfunu.cec_based.F232005(ndim=30)
x = np.ones(30)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F24
print("====================F24")
problem = opfunu.cec_based.F242005(ndim=30)
x = np.ones(30)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2005 F25
print("====================F25")
problem = opfunu.cec_based.F252005(ndim=10)
x = np.ones(10)
print(problem.f_global)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.evaluate(problem.x_global))

print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))






