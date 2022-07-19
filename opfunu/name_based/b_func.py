#!/usr/bin/env python
# Created by "Thieu" at 18:47, 29/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class BartelsConn(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Bartels Conn Function"
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

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-500., 500.] for _ in range(self.dim_default)]))
        self.dim_changeable = False
        self.f_global = 1.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.abs(x[0]**2 + x[1]**2 + x[0]*x[1]) + np.abs(np.sin(x[0]))+ np.abs(np.cos(x[1]))


class Beale(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Beale Function"
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

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-4.5, 4.5] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.array([3.0, 0.5])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2


class BiggsExp02(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Biggs EXP2 Function"
    latex_formula = r'\begin{matrix}f_{\text{BiggsExp02}}(x) = \sum_{i=1}^{10} (e^{-t_i x_1} - 5 e^{-t_i x_2} - y_i)^2 \\' + \
        r't_i = 0.1 i\\y_i = e^{-t_i} - 5 e^{-10t_i}\\ \end{matrix}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [0., 20.], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(1.0, 10.) = 0.0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 20.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.array([1.0, 10.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        t = np.arange(1, 11.) * 0.1
        y = np.exp(-t) - 5 * np.exp(-10 * t)
        return np.sum((np.exp(-t * x[0]) - 5 * np.exp(-t * x[1]) - y) ** 2)


class BiggsExp03(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Biggs EXP3 Function"
    latex_formula = r'\begin{matrix}\ f_{\text{BiggsExp03}}(x) = \sum_{i=1}^{10} (e^{-t_i x_1} - x_3e^{-t_i x_2} - y_i)^2\\' + \
        r't_i = 0.1i\\ y_i = e^{-t_i} - 5e^{-10 t_i}\\ \end{matrix}'
    latex_formula_dimension = r'd = 3'
    latex_formula_bounds = r'x_i \in [0., 20.], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(1.0, 10., 5.0) = 0.0'
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
        self.dim_default = 3
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 20.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.array([1.0, 10., 5.0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        t = np.arange(1., 11.) * 0.1
        y = np.exp(-t) - 5 * np.exp(-10 * t)
        return np.sum((np.exp(-t * x[0]) - x[2] * np.exp(-t * x[1]) - y) ** 2)


class BiggsExp04(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Biggs EXP4 Function"
    latex_formula = r'\begin{matrix}\ f_{\text{BiggsExp04}}(x) = \sum_{i=1}^{10} (x_3 e^{-t_i x_1} - x_4 e^{-t_i x_2} - y_i)^2\\' + \
        r't_i = 0.1i\\ y_i = e^{-t_i} - 5 e^{-10 t_i}\\ \end{matrix}'
    latex_formula_dimension = r'd = 4'
    latex_formula_bounds = r'x_i \in [0., 20.], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(1.0, 10., 1.0, 5.0) = 0.0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 20.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.array([1.0, 10., 1.0, 5.0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        t = np.arange(1, 11.) * 0.1
        y = np.exp(-t) - 5 * np.exp(-10 * t)
        return np.sum((x[2] * np.exp(-t * x[0]) - x[3] * np.exp(-t * x[1]) - y) ** 2)


class BiggsExp05(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Biggs EXP5 Function"
    latex_formula = r'\begin{matrix}\ f_{\text{BiggsExp05}}(x) = \sum_{i=1}^{11} (x_3 e^{-t_i x_1} - x_4 e^{-t_i x_2} + 3 e^{-t_i x_5} - y_i)^2\\' + \
        r't_i = 0.1i\\ y_i = e^{-t_i} - 5e^{-10 t_i} + 3e^{-4 t_i}\\ \end{matrix}'
    latex_formula_dimension = r'd = 5'
    latex_formula_bounds = r'x_i \in [0., 20.], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(1.0, 10., 1.0, 5.0, 4.0) = 0.0'
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
        self.dim_default = 5
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 20.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.array([1.0, 10., 1.0, 5.0, 4.0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        t = np.arange(1, 12.) * 0.1
        y = np.exp(-t) - 5 * np.exp(-10 * t) + 3 * np.exp(-4 * t)
        return np.sum((x[2] * np.exp(-t * x[0]) - x[3] * np.exp(-t * x[1]) + 3 * np.exp(-t * x[4]) - y) ** 2)


class Bird(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = " Bird Function"
    latex_formula = r'f_{\text{Bird}}(x) = \left(x_1 - x_2\right)^{2} + e^{\left[1 -' + \
         r'\sin\left(x_1\right) \right]^{2}} \cos\left(x_2\right) + e^{\left[1 - \cos\left(x_2\right)\right]^{2}} \sin\left(x_1\right)'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(4.701055751981055, 3.152946019601391) = f(-1.582142172055011, -3.130246799635430) = -106.7645367198034'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-2*np.pi, 2*np.pi] for _ in range(self.dim_default)]))
        self.f_global = -106.7645367198034
        self.x_global = np.array([4.701055751981055, 3.152946019601391])        # [-1.582142172055011, -3.130246799635430]

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (np.sin(x[0]) * np.exp((1 - np.cos(x[1])) ** 2) + np.cos(x[1]) * np.exp((1 - np.sin(x[0])) ** 2) + (x[0] - x[1]) ** 2)













