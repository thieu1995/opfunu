#!/usr/bin/env python
# Created by "Thieu" at 17:32, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class VenterSobiezcczanskiSobieski(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

         f(x) = x_1^2 - 100 \cos^2(x_1) - 100 \cos(x_1^2/30)+ x_2^2 - 100 \cos^2(x_2)- 100 \cos(x_2^2/30)

    with :math:`x_i \in [-50, 50]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -400` for :math:`x = [0, 0]`
    """
    name = "VenterSobiezcczanskiSobieski Function"
    latex_formula = r'f(x) = x_1^2 - 100 \cos^2(x_1) - 100 \cos(x_1^2/30)+ x_2^2 - 100 \cos^2(x_2)- 100 \cos(x_2^2/30)'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = True
    scalable = False
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-50., 50.] for _ in range(self.dim_default)]))
        self.f_global = -400
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        u = x[0] ** 2.0 - 100.0 * np.cos(x[0]) ** 2.0
        v = -100.0 * np.cos(x[0] ** 2.0 / 30.0) + x[1] ** 2.0
        w = - 100.0 * np.cos(x[1]) ** 2.0 - 100.0 * np.cos(x[1] ** 2.0 / 30.0)
        return u + v + w
