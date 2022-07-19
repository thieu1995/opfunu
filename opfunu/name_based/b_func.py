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
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-500., 500.] for _ in range(self.dim_default)]))
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
    name = "Bird Function"
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
        self.x_global = np.array([4.701055751981055, 3.152946019601391])
        self.x_globals = np.array([[4.701055751981055, 3.152946019601391], [-1.582142172055011, -3.130246799635430]])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (np.sin(x[0]) * np.exp((1 - np.cos(x[1])) ** 2) + np.cos(x[1]) * np.exp((1 - np.sin(x[0])) ** 2) + (x[0] - x[1]) ** 2)


class Bohachevsky1(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Bohachevsky 1 Function"
    latex_formula = r'f_{\text{Bohachevsky}}(x) = \sum_{i=1}^{n-1}\left[x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i + 1}) + 0.7 \right]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0) = 0'
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
        self.f_global = 0.0
        self.x_global = np.array([0.0, 0.0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7)


class Bohachevsky2(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Bohachevsky 2 Function"
    latex_formula = r'f_{\text{Bohachevsky}}(x) = \sum_{i=1}^{n-1}\left[x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i + 1}) + 0.7 \right]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.array([0.0, 0.0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0]) * np.cos(4 * np.pi * x[1]) + 0.3)


class Bohachevsky3(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Bohachevsky 3 Function"
    latex_formula = r'f_{\text{Bohachevsky}}(x) = \sum_{i=1}^{n-1}\left[x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i + 1}) + 0.7 \right]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.array([0.0, 0.0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0] + 4 * np.pi * x[1]) + 0.3)


class Booth(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Booth Function"
    latex_formula = r'f_{\text{Booth}}(x) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(1, 3) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.array([1.0, 3.0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


class BoxBetts(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Box-Betts Quadratic Sum Function"
    latex_formula = r'f_{\text{BoxBetts}}(x) = \sum_{i=1}^k g(x_i)^2; g(x) = e^{-0.1i x_1} - e^{-0.1i x_2} - x_3\left[e^{-0.1i} - e^{-i}\right]; k=10'
    latex_formula_dimension = r'd = 3'
    latex_formula_bounds = r'x_1 \in [0.9, 1.2], x_2 \in [9, 11.2], x_3 \in [0.9, 1.2]'
    latex_formula_global_optimum = r'f(1, 10, 1) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0.9, 1.2], [9.0, 11.2], [0.9, 1.2]]))
        self.f_global = 0.0
        self.x_global = np.array([1.0, 10.0, 1.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        i = np.arange(1, 11)
        g = (np.exp(-0.1 * i * x[0]) - np.exp(-0.1 * i * x[1]) - (np.exp(-0.1 * i) - np.exp(-i)) * x[2])
        return np.sum(g ** 2)


class Branin01(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Branin RCOS 1 Function"
    latex_formula = r'f_{\text{Branin01}}(x) = \left(- 1.275 \frac{x_1^{2}}{\pi^{2}} + 5' + \
        r'\frac{x_1}{\pi} + x_2 -6\right)^{2} + \left(10 -\frac{5}{4 \pi} \right) \cos\left(x_1\right) + 10'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_1 \in [-5, 10], x_2 \in [0, 15]'
    latex_formula_global_optimum = r'f(x_i) = 0.39788735772973816, x_i = [-\pi, 12.275]; or [\pi, 2.275] or x = [3\pi, 2.475]'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 10.], [0., 15.]]))
        self.f_global = 0.39788735772973816
        self.x_global = np.array([-np.pi, 12.275])
        self.x_globals = np.array([[-np.pi, 12.275], [np.pi, 2.275], [3 * np.pi, 2.475]])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return ((x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)


class Branin02(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Branin RCOS 2 Function"
    latex_formula = r'f_{\text{Branin02}}(x) = \left(- 1.275 \frac{x_1^{2}}{\pi^{2}} + 5 \frac{x_1}{\pi} + x_2 - 6 \right)^{2} + ' \
                    r'\left(10 - \frac{5}{4 \pi} \right) \cos\left(x_1\right) \cos\left(x_2\right) + \log(x_1^2+x_2^2 + 1) + 10'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_1 \in [-5, 15], x_2 \in [-5, 15]'
    latex_formula_global_optimum = r'f(x_i) = 5.559037, x_i = [-3.2, 12.53]'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 15.] for _ in range(self.dim_default)]))
        self.f_global = 5.559037
        self.x_global = np.array([-3.2, 12.53])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return ((x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
                + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) * np.cos(x[1]) + np.log(x[0] ** 2.0 + x[1] ** 2.0 + 1.0) + 10)


class Brent(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Brent Function"
    latex_formula = r'f_{\text{Brent}}(x) = (x_1 + 10)^2 + (x_2 + 10)^2 + e^{(-x_1^2 -x_2^2)}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(x_i) = 0, x_i = [-10, -10]'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.array([-10., -10.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (x[0] + 10.0) ** 2.0 + (x[1] + 10.0) ** 2.0 + np.exp(-x[0] ** 2.0 - x[1] ** 2.0)


class Brown(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Brown Function"
    latex_formula = r'f_{\text{Brown}}(x) = \sum_{i=1}^{n-1}\left[\left(x_i^2\right)^{x_{i + 1}^2 + 1} + \left(x_{i + 1}^2\right)^{x_i^2 + 1}\right]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-1, 4], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(x_i) = 0, x_i = 0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1., 4.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x0 = x[:-1]
        x1 = x[1:]
        return np.sum((x0 ** 2.0) ** (x1 ** 2.0 + 1.0) + (x1 ** 2.0) ** (x0 ** 2.0 + 1.0))


class Bukin02(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Bukin 2 Function"
    latex_formula = r'f_{\text{Bukin02}}(x) = 100 (x_2^2 - 0.01x_1^2 + 1) + 0.01(x_1 + 10)^2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_1 \in [-15, -5], x_2 \in [-3, 3]'
    latex_formula_global_optimum = r'f(x_i) = -124.75, x_i = [-15, 0]'
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
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-15., -5.], [-3., 3.]]))
        self.f_global = -124.75
        self.x_global = np.array([-15., 0.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return 100 * (x[1] ** 2 - 0.01 * x[0] ** 2 + 1.0) + 0.01 * (x[0] + 10.0) ** 2.0


class Bukin04(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Bukin 4 Function"
    latex_formula = r'f_{\text{Bukin04}}(x) = 100 x_2^{2} + 0.01 \lvert{x_1 + 10}\rvert'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_1 \in [-15, -5], x_2 \in [-3, 3]'
    latex_formula_global_optimum = r'f(x_i) = 0, x_i = [-10, 0]'
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
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-15., -5.], [-3., 3.]]))
        self.f_global = 0.0
        self.x_global = np.array([-10., 0.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return 100 * x[1] ** 2 + 0.01 * abs(x[0] + 10)


class Bukin06(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Bukin 6 Function"
    latex_formula = r'f_{\text{Bukin06}}(x) = 100 \sqrt{ \lvert{x_2 - 0.01 x_1^{2}}\rvert} + 0.01 \lvert{x_1 + 10} \rvert'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_1 \in [-15, -5], x_2 \in [-3, 3]'
    latex_formula_global_optimum = r'f(x_i) = 0, x_i = [-10, 1]'
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
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-15., -5.], [-3., 3.]]))
        self.f_global = 0.0
        self.x_global = np.array([-10., 1.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)











