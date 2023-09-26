#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class TestTubeHolder(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{TestTubeHolder}}(x) = - 4 \left | {e^{\left|{\cos \left(\frac{1}{200} x_{1}^{2} +
        \frac{1}{200} x_{2}^{2}\right)} \right|}\sin\left(x_{1}\right) \cos\left(x_{2}\right)}\right|

    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -10.872299901558` for :math:`x= [-\pi/2, 0]`
    """
    name = "Qing Function"
    latex_formula = r'f_{\text{TestTubeHolder}}(x)='
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = -10.87229990155800
        self.x_global = np.array([-np.pi / 2, 0.0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        u = np.sin(x[0]) * np.cos(x[1])
        v = (x[0] ** 2 + x[1] ** 2) / 200
        return -4 * np.abs(u * np.exp(np.abs(np.cos(v))))
