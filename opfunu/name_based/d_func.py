#!/usr/bin/env python
# Created by "Thieu" at 09:33, 21/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Damavandi(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Damavandi Function"
    latex_formula = r'f(x) = \left[ 1 - \lvert{\frac{\sin[\pi (x_1 - 2)]\sin[\pi (x2 - 2)]}' \
                    r'{\pi^2 (x_1 - 2)(x_2 - 2)}}\rvert^5 \right] \left[2 + (x_1 - 7)^2 + 2(x_2 - 7)^2 \right]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [0, 14], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(2, 2) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 14.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = 2 * np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        num = np.sin(np.pi * (x[0] - 2.0)) * np.sin(np.pi * (x[1] - 2.0))
        den = (np.pi ** 2) * (x[0] - 2.0) * (x[1] - 2.0) + self.epsilon
        factor1 = 1.0 - (np.abs(num / den)) ** 5.0
        factor2 = 2 + (x[0] - 7.0) ** 2.0 + 2 * (x[1] - 7.0) ** 2.0
        return factor1 * factor2


class Deb01(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Deb 1 Function"
    latex_formula = r'f(x) = - \frac{1}{N} \sum_{i=1}^n \sin^6(5 \pi x_i)'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0.3, -0.3) = -1'
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
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1, -1.] for _ in range(self.dim_default)]))
        self.f_global = -1.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return -(1.0 / self.ndim) * np.sum(np.sin(5 * np.pi * x) ** 6.0)


class Deb03(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Deb 3 Function"
    latex_formula = r'f(x) = - \frac{1}{N} \sum_{i=1}^n \sin^6 \left[ 5 \pi\left ( x_i^{3/4} - 0.05 \right) \right ]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0.3, -0.3) = -1'
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
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1., 1.] for _ in range(self.dim_default)]))
        self.f_global = -1.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return -(1.0 / self.ndim) * np.sum(np.sin(5 * np.pi * (x ** 0.75 - 0.05)) ** 6.0)


class Decanomial(Benchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """
    name = "Decanomial Function"
    latex_formula = r'f(x) = 0.001 \left(\lvert{x_{2}^{4} + 12 x_{2}^{3}' + \
       r'+ 54 x_{2}^{2} + 108 x_{2} + 81.0}\rvert + \lvert{x_{1}^{10} - 20 x_{1}^{9} + 180 x_{1}^{8} - 960 x_{1}^{7} + 3360 x_{1}^{6}' + \
       r'- 8064 x_{1}^{5} + 13340 x_{1}^{4} - 15360 x_{1}^{3} + 11520 x_{1}^{2} - 5120 x_{1} + 2624.0}\rvert\right)^{2}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(2, -3) = 0'
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
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.array([2, -3])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        val = x[1] ** 4 + 12 * x[1] ** 3 + 54 * x[1] ** 2 + 108 * x[1] + 81.0
        val2 = x[0] ** 10. - 20 * x[0] ** 9 + 180 * x[0] ** 8 - 960 * x[0] ** 7
        val2 += 3360 * x[0] ** 6 - 8064 * x[0] ** 5 + 13340 * x[0] ** 4
        val2 += - 15360 * x[0] ** 3 + 11520 * x[0] ** 2 - 5120 * x[0] + 2624
        return 0.001 * (abs(val) + abs(val2)) ** 2.





