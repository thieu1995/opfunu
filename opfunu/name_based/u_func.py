#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Ursem01(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

         f_{\text{Ursem01}}(x) = - \sin(2x_1 - 0.5 \pi) - 3 \cos(x_2) - 0.5 x_1

    with :math:`x_1 \in [-2.5, 3]` and :math:`x_2 \in [-2, 2]`.

    *Global optimum*: :math:`f(x) = -4.81681406371` for :math:`x = [1.69714, 0.0]`
    """
    name = "Qing Function"
    latex_formula = r'f_{\text{Ursem01}}(x) = - \sin(2x_1 - 0.5 \pi) - 3 \cos(x_2) - 0.5 x_1'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([(-2.5, 3.0), (-2.0, 2.0)]))
        self.f_global = -4.81681406371
        self.x_global = np.array([1.69714, 0.0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return -np.sin(2 * x[0] - 0.5 * np.pi) - 3.0 * np.cos(x[1]) - 0.5 * x[0]
