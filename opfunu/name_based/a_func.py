#!/usr/bin/env python
# Created by "Thieu" at 18:47, 29/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Ackley01(Benchmark):
    r"""
    .. [1] Adorio, E. MVF - "Multivariate Test Functions Library in C for Unconstrained Global Optimization", 2005
    TODO: the -0.2 factor in the exponent of the first term is given as -0.02 in Jamil et al.
    """
    name = "Ackley 01"
    latex_formula = r'f_{\text{Ackley01}}(x) = -20 e^{-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}} - e^{\frac{1}{n} \sum_{i=1}^n \cos(2 \pi x_i)} + 20 + e'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-35, 35], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, ..., 0) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-35., 35.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        u = np.sum(x ** 2)
        v = np.sum(np.cos(2 * np.pi * x))
        return (-20. * np.exp(-0.2 * np.sqrt(u / self.ndim)) - np.exp(v / self.ndim) + 20. + np.exp(1.))


class Ackley02(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark FunctionsFor Global Optimization Problems Int.
    Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Ackley 02"
    latex_formula = r'f_{\text{Ackley02}(x) = -200 e^{-0.02 \sqrt{x_1^2 + x_2^2}}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-32.0, 32.0], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, ..., 0) = -200'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-32., 32.] for _ in range(self.dim_default)]))
        self.f_global = -200.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return -200 * np.exp(-0.02 * np.sqrt(x[0] ** 2 + x[1] ** 2))


class Ackley03(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark FunctionsFor Global Optimization Problems Int.
    Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Ackley 03"
    latex_formula = r'f_{\text{Ackley03}}(x) = -200 e^{-0.02 \sqrt{x_1^2 + x_2^2}} + 5e^{\cos(3x_1) + \sin(3x_2)}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-32.0, 32.0], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(x1, x2)\approx-195.629028238419, at$$ $$x1=-0.682584587365898, and$$ $$ x2=-0.36075325513719'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-32., 32.] for _ in range(self.dim_default)]))
        self.f_global = -195.62902825923879
        self.x_global = np.array([-0.68255758, -0.36070859])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return -200 * np.exp(-0.02 * np.sqrt(x[0] ** 2 + x[1] ** 2)) + 5 * np.exp(np.cos(3 * x[0]) + np.sin(3 * x[1]))


class Adjiman(Benchmark):
    """
    [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark FunctionsFor Global Optimization Problems Int.
    Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Adjiman Function"
    latex_formula = r'f_{\text{Adjiman}}(x) = \cos(x_1)\sin(x_2) - \frac{x_1}{(x_2^2 + 1)}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_1 \in [-1.0, 2.0], x_2 \in [-1., 1.]'
    latex_formula_global_optimum = r'f(x1, x2)\approx-2.02181, at$$ $$x1=2.0, and$$ $$ x2=0.10578'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1.0, 2.0], [-1.0, 1.0]]))
        self.f_global = -2.02180678
        self.x_global = np.array([2.0, 0.10578])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.cos(x[0]) * np.sin(x[1]) - x[0] / (x[1] ** 2 + 1)


class Alpine01(Benchmark):
    """
    [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark FunctionsFor Global Optimization Problems Int.
    Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Alpine01 Function"
    latex_formula = r'f_{\text{Alpine01}}(x) = \sum_{i=1}^{n} \lvert {x_i \sin \left( x_i\right) + 0.1 x_i} \rvert'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-10.0, 10.0], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(x*)\approx 0, at$$ $$x*=0.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = False
    scalable = False
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


class Alpine02(Benchmark):
    """
    [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark FunctionsFor Global Optimization Problems Int.
    Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Alpine02 Function"
    latex_formula = r'f_{\text{Alpine02}(x) = \prod_{i=1}^{n} \sqrt{x_i} \sin(x_i)'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [0., 10.0], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(x*)\approx -6.12950, at$$ $$x_1=7.91705268, x_2=4.81584232'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 10.] for _ in range(self.dim_default)]))
        self.f_global = -6.12950
        self.x_global = np.array([7.91705268, 4.81584232] + list(np.random.uniform(0, 10, self.ndim-2)))

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.prod(np.sqrt(x) * np.sin(x))


class AMGM(Benchmark):
    """
    [1] The AMGM (Arithmetic Mean - Geometric Mean Equality).
    [2] Gavana, A. Global Optimization Benchmarks and AMPGO, retrieved 2015
    """
    name = "AMGM Function"
    latex_formula = r'f_{\text{AMGM}}(x) = \left ( \frac{1}{n} \sum_{i=1}^{n} x_i - \sqrt[n]{ \prod_{i=1}^{n} x_i} \right )^2'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [0., 10.0], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(x*)\approx 0, at$$ $$x_1=x_2=...=x_n'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        f1 = np.sum(x) / self.ndim
        f2 = np.prod(x) ** (1.0 / self.ndim)
        return (f1 - f2) ** 2




