#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Qing(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Qing}}(x) = \sum_{i=1}^{n} (x_i^2 - i)^2

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-500, 500]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = \pm \sqrt(i)` for :math:`i = 1, ..., n`
    """
    name = "Qing Function"
    latex_formula = r'f_{\text{Qing}}(x) = \sum_{i=1}^{n} (x_i^2 - i)^2'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-500., 500.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.array([np.sqrt(_) for _ in range(1, self.ndim + 1)])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        i = np.arange(1, self.ndim + 1)
        return np.sum((x ** 2.0 - i) ** 2.0)


class Quadratic(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Quadratic}}(x) = -3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2+ 203.64x_2^2 + 182.25x_1x_2

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -3873.72418` for :math:`x = [0.19388, 0.48513]`
    """
    name = "Quadratic Function"
    latex_formula = r'f_{\text{Quadratic}}(x) = -3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2+ 203.64x_2^2 + 182.25x_1x_2'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = -3873.72418
        self.x_global = np.array([0.19388, 0.48513])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (-3803.84 - 138.08 * x[0] - 232.92 * x[1] + 128.08 * x[0] ** 2.0
                + 203.64 * x[1] ** 2.0 + 182.25 * x[0] * x[1])


class Quartic(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Quartic}}(x) =

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -3873.72418` for :math:`x = [0.19388, 0.48513]`
    """
    name = "Quartic Function"
    latex_formula = r'f_{\text{Quartic}}(x) = '
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = False
    randomized_term = True
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1.28, 1.28] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        i = np.arange(1, self.ndim + 1)
        return np.sum(i*x**4.) + np.random.rand()


class Quintic(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Quintic}}(x) = \sum_{i=1}^{n} \left|{x_{i}^{5} - 3 x_{i}^{4}+ 4 x_{i}^{3} + 2 x_{i}^{2} - 10 x_{i} -4}\right|

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x_i) = 0` for :math:`x_i = -1` for :math:`i = 1, ..., n`
    """
    name = "Quartic Function"
    latex_formula = r'f_{\text{Quintic}}(x) = \sum_{i=1}^{n} \left|{x_{i}^{5} - 3 x_{i}^{4}+ 4 x_{i}^{3} + 2 x_{i}^{2} - 10 x_{i} -4}\right|'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = -1*np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.sum(np.abs(x ** 5 - 3 * x ** 4 + 4 * x ** 3 + 2 * x ** 2 - 10 * x - 4))
