#!/usr/bin/env python
# Created by "Thieu" at 17:55, 22/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Hansen(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Hansen Function"
    latex_formula = r'f(x) = \left[ \sum_{i=0}^4(i+1)\cos(ix_1+i+1)\right ]\left[\sum_{j=0}^4(j+1)\cos[(j+2)x_2+j+1])\right ]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(-7.58989583, -7.70831466) = -176.54179'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = False
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = -176.54179
        self.x_global = np.array([-7.58989583, -7.70831466])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        i = np.arange(5.)
        a = (i + 1) * np.cos(i * x[0] + i + 1)
        b = (i + 1) * np.cos((i + 2) * x[1] + i + 1)
        return np.sum(a) * np.sum(b)




