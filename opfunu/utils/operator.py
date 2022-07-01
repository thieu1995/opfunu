#!/usr/bin/env python
# Created by "Thieu" at 10:49, 01/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def griewank_func(x):
    x = np.array(x).ravel()
    t1 = np.sum(x**2) / 4000
    t2 = np.prod([np.cos(x[idx] / np.sqrt(idx+1)) for idx in range(0, len(x))])
    return t1 - t2 + 1


def rosenbrock_func(x):
    x = np.array(x).ravel()
    return np.sum([100*(x[idx]**2 - x[idx+1])**2 + (x[idx] - 1)**2 for idx in range(0, len(x)-1)])

