#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Rana(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Rana}}(x) = \sum_{i=1}^{n} \left[x_{i} \sin\left(\sqrt{\lvert{x_{1} - x_{i} + 1}\rvert}\right)
        \cos\left(\sqrt{\lvert{x_{1} + x_{i} + 1}\rvert}\right) + \left(x_{1} + 1\right) \sin\left(\sqrt{\lvert{x_{1} + x_{i} +
        1}\rvert}\right) \cos\left(\sqrt{\lvert{x_{1} - x_{i} +1}\rvert}\right)\right]

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-500.0, 500.0]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x_i) = -928.5478` for :math:`x = [-300.3376, 500]`.
    """
    name = "Qing Function"
    latex_formula = r'f_{\text{Rana}}(x) = '
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-500., 500.] for _ in range(self.dim_default)]))
        self.f_global = -500.8021602966615
        self.x_global = np.array([-300.3376, 500.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        t1 = np.sqrt(np.abs(x[1:] + x[: -1] + 1))
        t2 = np.sqrt(np.abs(x[1:] - x[: -1] + 1))
        v = (x[1:] + 1) * np.cos(t2) * np.sin(t1) + x[:-1] * np.cos(t1) * np.sin(t2)
        return np.sum(v)
