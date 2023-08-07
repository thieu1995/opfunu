#!/usr/bin/env python
# Created by "Thieu" at 17:32, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%
import numpy as np
from opfunu.benchmark import Benchmark


class Zakharov(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f_{\text{Leon}}(\mathbf{x}) =

    with :math:`x_i \in [-5, 10]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, ..., 0]`
    """
    name = "Zakharov Function"
    latex_formula = r'f(x) ='
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-5, 10] for i \in [1, n]'
    latex_formula_global_optimum = r'f([0, ..., 0]) = 0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        z = np.array(x).ravel()
        self.check_solution(z)
        self.n_fe += 1
        idx = np.arange(1,len(z) + 1)
        term = 0.5 * np.sum(idx * z)
        return np.sum(z ** 2) + term ** 2 + term ** 4

