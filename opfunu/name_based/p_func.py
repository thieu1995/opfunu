#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Parsopoulos(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Parsopoulos}}(x) = \cos(x_1)^2 + \sin(x_2)^2

    with :math:`x_i \in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: This function has infinite number of global minima in R2, at points
    :math:`\left(k\frac{\pi}{2}, \lambda \pi \right)`, where :math:`k = \pm1, \pm3, ...` and :math:`\lambda = 0, \pm1, \pm2, ...`

    In the given domain problem, function has 12 global minima all equal to zero.
    """
    name = "Parsopoulos Function"
    latex_formula = r'f_{\text{Parsopoulos}}(x) = \cos(x_1)^2 + \sin(x_2)^2'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.cos(x[0]) ** 2.0 + np.sin(x[1]) ** 2.0
