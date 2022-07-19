#!/usr/bin/env python
# Created by "Thieu" at 20:18, 19/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class CamelThreeHump(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Camel Function – Three Hump"
    latex_formula = r'f(x) = 2x_1^2 -1.05x_1^4 + x_1^6/6 + x_1x_2 + x_2^2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-5, 5], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0) = 0'
    continuous = False
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
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))
        self.dim_changeable = False
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2


class CamelSixHump(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Camel Function – Six Hump"
    latex_formula = r'f(x) = (4 - 2.1x_1^2 + x_1^4/3)x_1^2 + x_1x_2 + (4x_2^2 -4)x_2^2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-5, 5], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(-0.0898, 0.7126) = f(0.0898, -0.7126) = -1.0316284229280819'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))
        self.dim_changeable = False
        self.f_global = -1.0316284229280819
        self.x_global = np.array([-0.0898, 0.7126])
        self.x_globals = np.array([[-0.0898, 0.7126], [0.0898, -0.7126]])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 - 4)*x[1]**2


class ChenBird(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Chen Bird Function"
    latex_formula = r'f(x) = '
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-500, 500], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(-113.11622344, 227.73244688) = -1000'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-500., 500.] for _ in range(self.dim_default)]))
        self.dim_changeable = False
        self.f_global = -1000.
        self.x_global = np.array([-113.11622344, 227.73244688])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return -0.001/(0.001**2 + (x[0] - 0.4*x[1] - 0.1)**2) - 0.001/(0.001**2 + (2*x[0] + x[1] - 1.5)**2)


class ChenV(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Chen V Function"
    latex_formula = r'f(x) = '
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-500, 500], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(±0.70710678, ±0.70710678) = -2000.0039999840005'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-500., 500.] for _ in range(self.dim_default)]))
        self.dim_changeable = False
        self.f_global = -2000.0039999840005
        self.x_global = np.array([-0.70710678, -0.70710678])
        self.x_globals = np.array([[-0.70710678, -0.70710678],
                                   [0.70710678, -0.70710678],
                                   [-0.70710678, 0.70710678],
                                   [0.70710678, 0.70710678]])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return -0.001/(0.001**2 + (x[0]**2 + x[1]**2 - 1)**2) - 0.001/(0.001**2 + (x[0]**2 + x[1]**2 - 0.5)**2) - 0.001/(0.001**2 + (x[0]**2 - x[1]**2)**2)


class Chichinadze(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Chichinadze Function"
    latex_formula = r'f(x) = x_{1}^{2} - 12 x_{1} + 8 \sin\left(\frac{5}{2} \pi x_{1}\right)' +\
        r'+ 10 \cos\left(\frac{1}{2} \pi x_{1}\right) + 11 - 0.2 \frac{\sqrt{5}}{e^{\frac{1}{2} \left(x_{2} -0.5 \right)^{2}}}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-30, 30], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(6.189866586965680, 0.5) = -42.94438701899098'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-30., 30.] for _ in range(self.dim_default)]))
        self.dim_changeable = False
        self.f_global = -42.94438701899098
        self.x_global = np.array([6.189866586965680, 0.5])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (x[0] ** 2 - 12 * x[0] + 11 + 10 * np.cos(np.pi * x[0] / 2)
                + 8 * np.sin(5 * np.pi * x[0] / 2) - 1.0 / np.sqrt(5) * np.exp(-((x[1] - 0.5) ** 2) / 2))












