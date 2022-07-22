#!/usr/bin/env python
# Created by "Thieu" at 17:22, 22/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class FreudensteinRoth(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Freudenstein Roth Function"
    latex_formula = r'f(x) = \left\{x_1 - 13 + \left[(5 - x_2) x_2' +\
        r'- 2 \right] x_2 \right\}^2 + \left \{x_1 - 29 + \left[(x_2 + 1) x_2 - 14 \right] x_2 \right\}^2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(5, 4) = 0'
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
        self.f_global = 0.
        self.x_global = np.array([5., 4.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        f1 = (-13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1]) ** 2
        f2 = (-29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1]) ** 2
        return f1 + f2
