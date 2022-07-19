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
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))
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
        self.f_global = -42.94438701899098
        self.x_global = np.array([6.189866586965680, 0.5])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (x[0] ** 2 - 12 * x[0] + 11 + 10 * np.cos(np.pi * x[0] / 2)
                + 8 * np.sin(5 * np.pi * x[0] / 2) - 1.0 / np.sqrt(5) * np.exp(-((x[1] - 0.5) ** 2) / 2))


class ChungReynolds(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Chung Reynolds Function"
    latex_formula = r'f(x) = (\sum_{i=1}^D x_i^2)^2'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0,...,0) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.sum(x**2)**2


class Cigar(Benchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2022
    """
    name = "Cigar Function"
    latex_formula = r'f(x) = x_1^2 + 10^6\sum_{i=2}^{n} x_i^2'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0,...,0) = 0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)


class Cola(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Cola Function"
    latex_formula = r'f(x) = x_1^2 + 10^6\sum_{i=2}^{n} x_i^2'
    latex_formula_dimension = r'd = 17'
    latex_formula_bounds = r'x_0 \in [0, 4], x_i \in [-4, 4], \forall i \in \llbracket 1, d-1\rrbracket'
    latex_formula_global_optimum = r'f(0,...,0) = 11.7464'
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
        self.dim_default = 17
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0.0, 4.0]] + [[-4., 4.] for _ in range(self.dim_default-1)]))
        self.f_global = 11.7464
        self.x_global = np.array([0.651906, 1.30194, 0.099242, -0.883791, -0.8796, 0.204651, -3.28414, 0.851188,
                                -3.46245, 2.53245, -0.895246, 1.40992, -3.07367, 1.96257, -2.97872, -0.807849, -1.68978])
        self.d = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1.27, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1.69, 1.43, 0, 0, 0, 0, 0, 0, 0, 0],
                 [2.04, 2.35, 2.43, 0, 0, 0, 0, 0, 0, 0],
                 [3.09, 3.18, 3.26, 2.85, 0, 0, 0, 0, 0, 0],
                 [3.20, 3.22, 3.27, 2.88, 1.55, 0, 0, 0, 0, 0],
                 [2.86, 2.56, 2.58, 2.59, 3.12, 3.06, 0, 0, 0, 0],
                 [3.17, 3.18, 3.18, 3.12, 1.31, 1.64, 3.00, 0, 0, 0],
                 [3.21, 3.18, 3.18, 3.17, 1.70, 1.36, 2.95, 1.32, 0, 0],
                 [2.38, 2.31, 2.42, 1.94, 2.85, 2.81, 2.56, 2.91, 2.97, 0.]])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        xi = np.atleast_2d(np.asarray([0.0, x[0]] + list(x[1::2])))
        xj = np.repeat(xi, np.size(xi, 1), axis=0)
        xi = xi.T

        yi = np.atleast_2d(np.asarray([0.0, 0.0] + list(x[2::2])))
        yj = np.repeat(yi, np.size(yi, 1), axis=0)
        yi = yi.T

        inner = (np.sqrt(((xi - xj) ** 2 + (yi - yj) ** 2)) - self.d) ** 2
        inner = np.tril(inner, -1)
        return np.sum(np.sum(inner, axis=1))


class Colville(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Colville Function"
    latex_formula = r'f(x) = \left(x_{1} -1\right)^{2} + 100 \left(x_{1}^{2} - x_{2}\right)^{2} + 10.1 \left(x_{2} -1\right)^{2} + ' \
                    r'\left(x_{3} -1\right)^{2} + 90 \left(x_{3}^{2} - x_{4}\right)^{2} + 10.1 \left(x_{4} -1\right)^{2} + 19.8 \frac{x_{4} -1}{x_{2}}'
    latex_formula_dimension = r'd = 4'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(1,...,1) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10.0, 10.0] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (100 * (x[0] - x[1] ** 2) ** 2 + (1 - x[0]) ** 2 + (1 - x[2]) ** 2 + 90 * (x[3] - x[2] ** 2) ** 2 +
                10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1))


class Corana(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Corana Function"
    latex_formula = r'f(x) = '
    latex_formula_dimension = r'd = 4'
    latex_formula_bounds = r'x_i \in [-5, 5], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(1,...,1) = 0'
    continuous = False
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = False
    scalable = False
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 4
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5.0, 5.0] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        d = [1., 1000., 10., 100.]
        r = 0
        for j in range(4):
            zj = np.floor(np.abs(x[j] / 0.2) + 0.49999) * np.sign(x[j]) * 0.2
            if np.abs(x[j] - zj) < 0.05:
                r += 0.15 * ((zj - 0.05 * np.sign(zj)) ** 2) * d[j]
            else:
                r += d[j] * x[j] * x[j]
        return r


class CosineMixture(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Cosine Mixture Function"
    latex_formula = r'f(x) = -0.1 \sum_{i=1}^n \cos(5 \pi x_i) - \sum_{i=1}^n x_i^2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(-1,...,-1) = -0.9*D'
    continuous = False
    linear = False
    convex = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1.0, 1.0] for _ in range(self.dim_default)]))
        self.f_global = -0.9*self.ndim
        self.x_global = -1 * np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return -0.1 * np.sum(np.cos(5.0 * np.pi * x)) - np.sum(x ** 2.0)


class CrossInTray(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Cross-in-Tray Function"
    latex_formula = r'f(x) = - 0.0001 \left(\left|{e^{\left|{100' +\
        r'- \frac{\sqrt{x_{1}^{2} + x_{2}^{2}}}{\pi}}\right|} \sin\left(x_{1}\right) \sin\left(x_{2}\right)}\right| + 1\right)^{0.1}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(±1.349406608602084, ±1.349406608602084) = -2.062611870822739'
    continuous = True
    linear = False
    convex = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10.0, 10.0] for _ in range(self.dim_default)]))
        self.f_global = -2.062611870822739
        self.x_global = np.array([1.349406608602084, 1.349406608602084])
        self.x_globals = np.array([(1.349406685353340, 1.349406608602084),
                               (-1.349406685353340, 1.349406608602084),
                               (1.349406685353340, -1.349406608602084),
                               (-1.349406685353340, -1.349406608602084)])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (-0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))+ 1) ** (0.1))


class CrossLegTable(Benchmark):
    """
    .. [1] Mishra, S. Global Optimization by Differential Evolution and Particle Swarm Methods:
    Evaluation on Some Benchmark Functions Munich University, 2006
    """
    name = "Cross-Leg-Table Function"
    latex_formula = r'f(x) = -\frac{1}{\left(\left|{e^{\left|{100 - \frac{\sqrt{x_{1}^{2} + x_{2}^{2}}}{\pi}}\right|}' + \
        r'\sin\left(x_{1}\right) \sin\left(x_{2}\right)}\right| + 1\right)^{0.1}}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0) = -1'
    continuous = True
    linear = False
    convex = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10.0, 10.0] for _ in range(self.dim_default)]))
        self.f_global = -1.
        self.x_global = np.array([0., 0.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        u = 100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi
        v = np.sin(x[0]) * np.sin(x[1])
        return -(np.abs(v * np.exp(np.abs(u))) + 1) ** (-0.1)


class CrownedCross(Benchmark):
    """
    .. [1] Mishra, S. Global Optimization by Differential Evolution and Particle Swarm Methods:
    Evaluation on Some Benchmark Functions Munich University, 2006
    """
    name = "Cross-Leg-Table Function"
    latex_formula = r'f(x) = 0.0001 \left(\left|{e^{\left|{100 - \frac{\sqrt{x_{1}^{2} + x_{2}^{2}}}{\pi}}\right|}' + \
        r'\sin\left(x_{1}\right) \sin\left(x_{2}\right)}\right| + 1\right)^{0.1}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0) = 0.0001'
    continuous = True
    linear = False
    convex = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10.0, 10.0] for _ in range(self.dim_default)]))
        self.f_global = 0.0001
        self.x_global = np.array([0., 0.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        u = 100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi
        v = np.sin(x[0]) * np.sin(x[1])
        return 0.0001 * (np.abs(v * np.exp(np.abs(u))) + 1) ** (0.1)


class Csendes(Benchmark):
    """
    .. [1] Mishra, S. Global Optimization by Differential Evolution and Particle Swarm Methods:
    Evaluation on Some Benchmark Functions Munich University, 2006
    """
    name = "Csendes Function"
    latex_formula = r'f(x) = \sum_{i=1}^n x_i^6 \left[ 2 + \sin \left( \frac{1}{x_i} \right ) \right]'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0,..., 0) = 0'
    continuous = True
    linear = False
    convex = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1.0, 1.0] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.sum((x ** 6.0) * (2.0 + np.sin(1.0/(x+self.epsilon))))


class Cube(Benchmark):
    """
    .. [1] Mishra, S. Global Optimization by Differential Evolution and Particle Swarm Methods:
    Evaluation on Some Benchmark Functions Munich University, 2006
    """
    name = "Cube Function"
    latex_formula = r'f(x) = 100(x_2 - x_1^3)^2 + (1 - x1)^2'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(1, 1) = 0'
    continuous = True
    linear = False
    convex = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10.0, 10.0] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.array([1., 1.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return 100.0 * (x[1] - x[0] ** 3.0) ** 2.0 + (1.0 - x[0]) ** 2.0
