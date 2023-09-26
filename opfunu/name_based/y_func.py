#!/usr/bin/env python
# Created by "Thieu" at 17:32, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class YaoLiu04(Benchmark):
    """
    .. [1]  Yao X., Liu Y. (1997) Fast evolution strategies. In: Angeline P.J., Reynolds R.G., McDonnell J.R., Eberhart R. (eds)
    Evolutionary Programming VI. EP 1997. Lecture Notes in Computer Science, vol 1213. Springer, Berlin, Heidelberg

    .. [2]  Mishra, S. Global Optimization by Differential Evolution and Particle Swarm Methods: Evaluation
    on Some Benchmark Functions. Munich Personal RePEc Archive, 2006, 1005

    .. math::

         f(x) = {max}_i \left\{ \left | x_i \right | , 1 \leq i \leq n \right\}

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """
    name = "Yao-Liu 4 Function"
    latex_formula = r'f(x) = {max}_i \left\{ \left | x_i \right | , 1 \leq i \leq n \right\}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.abs(x).max()
