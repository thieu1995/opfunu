#!/usr/bin/env python
# Created by "Thieu" at 17:32, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Watson(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

         f(x) = \sum_{i=0}^{29} \left\{\sum_{j=0}^4 ((j + 1)a_i^j x_{j+1}) - \left[ \sum_{j=0}^5 a_i^j x_{j+1} \right ]^2 - 1 \right\}^2 + x_1^2

    Where, in this exercise, :math:`a_i = i/29`. with :math:`x_i \in [-5, 5]` for :math:`i = 1, ..., 6`.

    *Global optimum*: :math:`f(x) = 0.002288` for :math:`x = [-0.0158, 1.012, -0.2329, 1.260, -1.513, 0.9928]`
    """
    name = "Watson Function"
    latex_formula = r'f(x) = \sum_{i=0}^{29} \left\{\sum_{j=0}^4 ((j + 1)a_i^j x_{j+1}) - \left[ \sum_{j=0}^5 a_i^j x_{j+1} \right ]^2 - 1 \right\}^2 + x_1^2'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = False

    differentiable = True
    scalable = False
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 6
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))
        self.f_global = 0.002288
        self.x_global = np.array([-0.0158, 1.012, -0.2329, 1.260, -1.513, 0.9928])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        i = np.atleast_2d(np.arange(30.)).T
        a = i / 29.
        j = np.arange(5.)
        k = np.arange(6.)
        t1 = np.sum((j + 1) * a ** j * x[1:], axis=1)
        t2 = np.sum(a ** k * x, axis=1)
        inner = (t1 - t2 ** 2 - 1) ** 2
        return np.sum(inner) + x[0] ** 2
