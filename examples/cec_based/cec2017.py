#!/usr/bin/env python
# Created by "Thieu" at 17:54, 08/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np


## Test CEC2017 F1
print("====================F1")
problem = opfunu.cec_based.F12017(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2017 F2
print("====================F2")
problem = opfunu.cec_based.F22017(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


## Test CEC2017 F3
print("====================F3")
problem = opfunu.cec_based.F32017(ndim=50)
x = np.ones(50)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


















