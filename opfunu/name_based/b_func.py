#!/usr/bin/env python
# Created by "Thieu" at 18:47, 29/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class BartelsConn(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Beale"
    latex_formula = r'f_{\text{BartelsConn}}(x) = \lvert {x_1^2 + x_2^2 + x_1x_2} \rvert + \lvert {\sin(x_1)} \rvert + \lvert {\cos(x_2)} \rvert'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-500, 500], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = False
    scalable = False
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, bounds=None):
        super().__init__()
        self.check_bounds(bounds, np.array([[-500, 500] for _ in range(2)]))
        self.dim_changeable = False
        self.f_global = 1.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.abs(x[0]**2 + x[1]**2 + x[0]*x[1]) + np.abs(np.sin(x[0]))+ np.abs(np.cos(x[1]))


class Beale(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Bartels Conn"
    latex_formula = r'f_{\text{Beale}}(x) = \left(x_1 x_2 - x_1 + 1.5\right)^{2} +' + \
                    '\left(x_1 x_2^{2} - x_1 + 2.25\right)^{2} + \left(x_1 x_2^{3} - x_1 + 2.625\right)^{2}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-4.5, 4.5], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(3.0, 0.5) = 0.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = False
    scalable = False
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, bounds=None):
        super().__init__()
        self.check_bounds(bounds, np.array([[-4.5, 4.5] for _ in range(2)]))
        self.dim_changeable = False
        self.f_global = 0.0
        self.x_global = np.array([3.0, 0.5])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2



