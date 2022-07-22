#!/usr/bin/env python
# Created by "Thieu" at 17:55, 22/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Hansen(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Hansen Function"
    latex_formula = r'f(x) = \left[ \sum_{i=0}^4(i+1)\cos(ix_1+i+1)\right ]\left[\sum_{j=0}^4(j+1)\cos[(j+2)x_2+j+1])\right ]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(-7.58989583, -7.70831466) = -176.54179'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = -176.54179
        self.x_global = np.array([-7.58989583, -7.70831466])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        i = np.arange(5.)
        a = (i + 1) * np.cos(i * x[0] + i + 1)
        b = (i + 1) * np.cos((i + 2) * x[1] + i + 1)
        return np.sum(a) * np.sum(b)


class Hartmann3(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Hartman 3 Function"
    latex_formula = r'f(x) = -\sum\limits_{i=1}^{4} c_i e^{-\sum\limits_{j=1}^{n}a_{ij}(x_j - p_{ij})^2}'
    latex_formula_dimension = r'd = 3'
    latex_formula_bounds = r'x_i \in [0, 1], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f([0.11461292,  0.55564907,  0.85254697]) = -3.8627821478'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 1.] for _ in range(self.dim_default)]))
        self.f_global = -3.8627821478
        self.x_global = np.array([0.11461292,  0.55564907,  0.85254697])
        self.a = np.asarray([[3.0, 10., 30.],
                          [0.1, 10., 35.],
                          [3.0, 10., 30.],
                          [0.1, 10., 35.]])
        self.p = np.asarray([[0.3689, 0.1170, 0.2673],
                          [0.4699, 0.4387, 0.7470],
                          [0.1091, 0.8732, 0.5547],
                          [0.03815, 0.5743, 0.8828]])
        self.c = np.asarray([1., 1.2, 3., 3.2])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        XX = np.atleast_2d(x)
        d = np.sum(self.a * (XX - self.p) ** 2, axis=1)
        return -np.sum(self.c * np.exp(-d))


class Hartmann6(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Hartman 6 Function"
    latex_formula = r'f(x) = -\sum\limits_{i=1}^{4} c_i e^{-\sum\limits_{j=1}^{n}a_{ij}(x_j - p_{ij})^2}'
    latex_formula_dimension = r'd = 3'
    latex_formula_bounds = r'x_i \in [0, 1], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f([0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]) = -3.32236801141551'
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
        self.dim_default = 6
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 1.] for _ in range(self.dim_default)]))
        self.f_global = -3.32236801141551
        self.x_global = np.array([0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054])
        self.a = np.asarray([[10., 3., 17., 3.5, 1.7, 8.],
                          [0.05, 10., 17., 0.1, 8., 14.],
                          [3., 3.5, 1.7, 10., 17., 8.],
                          [17., 8., 0.05, 10., 0.1, 14.]])
        self.p = np.asarray([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                          [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                          [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
                          [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
        self.c = np.asarray([1.0, 1.2, 3.0, 3.2])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        XX = np.atleast_2d(x)
        d = np.sum(self.a * (XX - self.p) ** 2, axis=1)
        return -np.sum(self.c * np.exp(-d))


class HelicalValley(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Helical Valley"
    latex_formula = r'f(x) = 100{[z-10\Psi(x_1,x_2)]^2 +(\sqrt{x_1^2+x_2^2}-1)^2}+x_3^2'
    latex_formula_dimension = r'd \in N^+'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f([1.0, 0.0, 0.0]) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.array([1.0, 0.0, 0.0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        theta = 1 / (2. * np.pi) * np.arctan2(x[1], x[0])
        return x[2] ** 2 + 100 * ((x[2] - 10 * theta) ** 2 + (r - 1) ** 2)


class Himmelblau(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Himmelblau Function"
    latex_formula = r'f(x) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2'
    latex_formula_dimension = r'd \in N^+'
    latex_formula_bounds = r'x_i \in [-5, 5], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f([3, 2]) = 0'
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
        self.f_global = 0.0
        self.x_global = np.array([3., 2.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


class Hosaki(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """
    name = "Hosaki Function"
    latex_formula = r'f(x) = \left ( 1 - 8 x_1 + 7 x_1^2 - \frac{7}{3} x_1^3 + \frac{1}{4} x_1^4 \right ) x_2^2 e^{-x_1}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r' 0 <= x_1 <= 5, 0 <= x2 <= 6'
    latex_formula_global_optimum = r'f(4, 2) = −2.3458'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 5.], [0., 6.]]))
        self.f_global = -2.345811576101292
        self.x_global = np.array([4., 2.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        val = (1 - 8 * x[0] + 7 * x[0] ** 2 - 7 / 3. * x[0] ** 3 + 0.25 * x[0] ** 4)
        return val * x[1] ** 2 * np.exp(-x[1])


class HolderTable(Benchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """
    name = "Hosaki Function"
    latex_formula = r'f(x) = - \left|{e^{\left|{1' +\
        r'- \frac{\sqrt{x_{1}^{2} + x_{2}^{2}}}{\pi} }\right|} \sin\left(x_{1}\right) \cos\left(x_{2}\right)}\right|'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r' 0 <= x_1 <= 5, 0 <= x2 <= 6'
    latex_formula_global_optimum = r'f(\pm 9.664590028909654) = -19.20850256788675'
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
        self.f_global = -19.20850256788675
        self.x_global = np.array([8.055023472141116, 9.664590028909654])
        self.x_globals = np.array([[8.055023472141116, 9.664590028909654],
                                   [-8.055023472141116, 9.664590028909654],
                                   [8.055023472141116, -9.664590028909654],
                                   [-8.055023472141116, -9.664590028909654]])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))
