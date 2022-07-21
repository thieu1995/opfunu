#!/usr/bin/env python
# Created by "Thieu" at 11:04, 21/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%


import numpy as np
from opfunu.benchmark import Benchmark


class Easom(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Easom Function"
    latex_formula = r'a - \frac{a}{e^{b \sqrt{\frac{\sum_{i=1}^{n}' +\
        r'x_i^{2}}{n}}}} + e - e^{\frac{\sum_{i=1}^{n} \cos\left(c x_i\right)} {n}}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(pi, pi) = -1'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = True
    scalable = False
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = -1.
        self.x_global = np.pi * np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        a = (x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-a)


class ElAttarVidyasagarDutta(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "El-Attar-Vidyasagar-Dutta Function"
    latex_formula = r'a - \frac{a}{e^{b \sqrt{\frac{\sum_{i=1}^{n}' +\
        r'x_i^{2}}{n}}}} + e - e^{\frac{\sum_{i=1}^{n} \cos\left(c x_i\right)} {n}}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-500, 500], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(3.40918683, -2.17143304) = 1.712780354'
    continuous = True
    linear = False
    convex = True
    unimodal = True
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-500., 500.] for _ in range(self.dim_default)]))
        self.f_global = 1.712780354
        self.x_global = np.array([3.40918683, -2.17143304])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return ((x[0] ** 2 + x[1] - 10) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2 + (x[0] ** 2 + x[1] ** 3 - 1) ** 2)







