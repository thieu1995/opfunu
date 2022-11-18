#!/usr/bin/env python
# Created by "Thieu" at 17:30, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Katsuura(Benchmark):
    """
    .. [1] Adorio, E. MVF - "Multivariate Test Functions Library in C for
    Unconstrained Global Optimization", 2005
    [2] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::
        f_{\text{Katsuura}}(x) = \prod_{i=0}^{n-1} \left [ 1 +
        (i+1) \sum_{k=1}^{d} \lfloor (2^k x_i) \rfloor 2^{-k} \right ]

    Where, in this exercise, :math:`d = 32`.
    Here, :math:`n` represents the number of dimensions and

    :math:`x_i \in [0, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 1` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`.
    """
    name = "Katsuura Function"
    latex_formula = r'f_{\text{Katsuura}}(x) = \prod_{i=0}^{n-1} \left [ 1 + (i+1) \sum_{k=1}^{d} \lfloor (2^k x_i) \rfloor 2^{-k} \right ]'
    latex_formula_dimension = r'd = 32'
    latex_formula_bounds = r'x_i \in [0, 100]'
    latex_formula_global_optimum = r'f(0., 0., ..., 0.) = 1.'
    continuous = True
    linear = False
    convex = True
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 100.] for _ in range(self.dim_default)]))
        self.f_global = 1.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        d = 32
        k = np.atleast_2d(np.arange(1, d + 1)).T
        idx = np.arange(0., self.ndim * 1.)
        inner = np.round(2 ** k * x) * (2. ** (-k))
        return np.prod(np.sum(inner, axis=0) * (idx + 1) + 1)


class Keane(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f_{\text{Keane}}(x) = \frac{\sin^2(x_1 - x_2)\sin^2(x_1 + x_2)}{\sqrt{x_1^2 + x_2^2}}
    with :math:`x_i \in [0, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0.0` for :math:`x = [7.85396153, 7.85396135]`.
    """
    name = "Katsuura Function"
    latex_formula = r'f_{\text{Keane}}(x) = \frac{\sin^2(x_1 - x_2)\sin^2(x_1 + x_2)} {\sqrt{x_1^2 + x_2^2}}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [0, 10]'
    latex_formula_global_optimum = r'f(7.85396153, 7.85396135) = 0.'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = False
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.array([7.85396153, 7.85396135])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        val = np.sin(x[0] - x[1]) ** 2 * np.sin(x[0] + x[1]) ** 2
        return val / np.sqrt(x[0] ** 2 + x[1] ** 2)


class Kowalik(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f_{\text{Kowalik}}(x) = \sum_{i=0}^{10} \left [ a_i
        - \frac{x_1 (b_i^2 + b_i x_2)} {b_i^2 + b_i x_3 + x_4} \right ]^2

    .. math::
        \begin{matrix}
        a = [4, 2, 1, 1/2, 1/4 1/8, 1/10, 1/12, 1/14, 1/16] \\
        b = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
             0.0456, 0.0342, 0.0323, 0.0235, 0.0246]\\
        \end{matrix}
    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-5, 5]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 0.00030748610` for :math:`x = [0.192833, 0.190836, 0.123117, 0.135766]`.
    """
    name = "Kowalik Function"
    latex_formula = r'f_{\text{Kowalik}}(x) = \sum_{i=0}^{10} \left [ a_i - \frac{x_1 (b_i^2 + b_i x_2)} {b_i^2 + b_i x_3 + x_4} \right ]^2'
    latex_formula_dimension = r'd = 4'
    latex_formula_bounds = r'x_i \in [-5, 5]'
    latex_formula_global_optimum = r'f(0.192833, 0.190836, 0.123117, 0.135766) = 0.00030748610'
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
        self.dim_default = 4
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))
        self.f_global = 0.00030748610
        self.x_global = np.array([0.192833, 0.190836, 0.123117, 0.135766])
        self.a = np.asarray([4.0, 2.0, 1.0, 1 / 2.0, 1 / 4.0, 1 / 6.0, 1 / 8.0,
                          1 / 10.0, 1 / 12.0, 1 / 14.0, 1 / 16.0])
        self.b = np.asarray([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
                          0.0456, 0.0342, 0.0323, 0.0235, 0.0246])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        vec = self.b - (x[0] * (self.a ** 2 + self.a * x[1]) / (self.a ** 2 + self.a * x[2] + x[3]))
        return np.sum(vec ** 2)
