#!/usr/bin/env python
# Created by "Thieu" at 20:40, 18/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np


print("====================Test BartelsConn")
problem = opfunu.name_based.BartelsConn(ndim=25)
x = np.ones(25)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Beale")
problem = opfunu.name_based.Beale(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


