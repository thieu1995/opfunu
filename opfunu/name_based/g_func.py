#!/usr/bin/env python
# Created by "Thieu" at 17:26, 22/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Giunta(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Giunta Function"
    latex_formula = r'f(x) = 0.6 + \sum_{i=1}^{n} \left[\sin^{2}\left(1' +\
        r'- \frac{16}{15} x_i\right) - \frac{1}{50} \sin\left(4 - \frac{64}{15} x_i\right) - \sin\left(1 - \frac{16}{15} x_i\right)\right]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f([0.4673200277395354, 0.4673200169591304]) = 0.06447042053690566'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1., 1.] for _ in range(self.dim_default)]))
        self.f_global = 0.06447042053690566
        self.x_global = np.array([0.4673200277395354, 0.4673200169591304])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        arg = 16 * x / 15.0 - 1
        return 0.6 + np.sum(np.sin(arg) + np.sin(arg) ** 2 + np.sin(4 * arg) / 50.)


class GoldsteinPrice(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Goldstein Price Function"
    latex_formula = r'f(x) = \left[ 1 + (x_1 + x_2 + 1)^2 (19 - 14 x_1 + 3 x_1^2 - 14 x_2 + 6 x_1 x_2 + 3 x_2^2) ' + \
                    r'\right] \left[ 30 + ( 2x_1 - 3 x_2)^2 (18 - 32 x_1 + 12 x_1^2 + 48 x_2 - 36 x_1 x_2 + 27 x_2^2) \right]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-2, 2], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f([0, -1]) = 3'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-2., 2.] for _ in range(self.dim_default)]))
        self.f_global = 3.
        self.x_global = np.array([0., -1.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        a = (1 + (x[0] + x[1] + 1) ** 2
             * (19 - 14 * x[0] + 3 * x[0] ** 2
                - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))
        b = (30 + (2 * x[0] - 3 * x[1]) ** 2
             * (18 - 32 * x[0] + 12 * x[0] ** 2
                + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
        return a * b






