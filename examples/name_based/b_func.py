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


print("====================Test BiggsExp02")
problem = opfunu.name_based.BiggsExp02(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test BiggsExp03")
problem = opfunu.name_based.BiggsExp03(ndim=3)
x = np.ones(3)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test BiggsExp04")
problem = opfunu.name_based.BiggsExp04(ndim=4)
x = np.ones(4)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test BiggsExp05")
problem = opfunu.name_based.BiggsExp05(ndim=5)
x = np.ones(5)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Bird")
problem = opfunu.name_based.Bird(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Bohachevsky1")
problem = opfunu.name_based.Bohachevsky1(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Bohachevsky2")
problem = opfunu.name_based.Bohachevsky2(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Bohachevsky3")
problem = opfunu.name_based.Bohachevsky3(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Booth")
problem = opfunu.name_based.Booth(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test BoxBetts")
problem = opfunu.name_based.BoxBetts(ndim=3)
x = np.ones(3)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Branin01")
problem = opfunu.name_based.Branin01(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Branin02")
problem = opfunu.name_based.Branin02(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Brent")
problem = opfunu.name_based.Brent(ndim=2)
x = np.ones(2)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Brown")
problem = opfunu.name_based.Brown(ndim=13)
x = np.ones(13)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Bukin02")
ndim = 2
problem = opfunu.name_based.Bukin02(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Bukin04")
ndim = 2
problem = opfunu.name_based.Bukin04(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))


print("====================Test Bukin06")
ndim = 2
problem = opfunu.name_based.Bukin06(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))









