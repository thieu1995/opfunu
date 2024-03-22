#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Matyas(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f_{\text{Matyas}}(x) = 0.26(x_1^2 + x_2^2) - 0.48 x_1 x_2

    Here :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = 0.0`for :math:`x = [0, 0]`
    """
    name = "Matyas Function"
    latex_formula = r'f_{\text{Matyas}}(x) = 0.26(x_1^2 + x_2^2) - 0.48 x_1 x_2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10]'
    latex_formula_global_optimum = r'f(0, 0) = 0'
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
        self.x_global = np.array([0., 0.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


class McCormick(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = - x_{1} + 2 x_{2} + \left(x_{1} - x_{2}\right)^{2} + \sin\left(x_{1} + x_{2}\right) + 1

    Here :math:`x_1 \in [-1.5, 4], x_2 \in [-3, 4]` .
    *Global optimum*: :math:`f(x) = -1.913222954981037`for :math:`x = [-0.5471975602214493, -1.547197559268372]`
    """
    name = "McCormick Function"
    latex_formula = r'f(x) = - x_{1} + 2 x_{2} + \left(x_{1} - x_{2}\right)^{2} + \sin\left(x_{1} + x_{2}\right) + 1'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_1 \in [-1.5, 4], x_2 \in [-3, 4]'
    latex_formula_global_optimum = r'f(-0.5471975602214493, -1.547197559268372) = -1.913222954981037'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1.5, 4.], [-3., 4.]]))
        self.f_global = -1.913222954981037
        self.x_global = np.array([-0.5471975602214493, -1.547197559268372])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1)


class Meyer(Benchmark):
    """
    .. [1] https://www.itl.nist.gov/div898/strd/nls/data/mgh10.shtml
    """
    name = "Meyer Function"
    latex_formula = r'f(x)'
    latex_formula_dimension = r'd = 3'
    latex_formula_bounds = r'x_1 \in [0, 1], x_2 \in [100, 1000], x_3 \in [100, 500]'
    latex_formula_global_optimum = r'f(5.6096364710e-3, 6.1813463463e2, 3.4522363462e2) = 8.7945855171e1'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 1.], [100., 1000.], [100., 500.]]))
        self.f_global = 8.7945855171e1
        self.x_global = np.array([5.6096364710e-3, 6.1813463463e2, 3.4522363462e2])
        self.a = np.asarray([3.478E+04, 2.861E+04, 2.365E+04, 1.963E+04, 1.637E+04,
                             1.372E+04, 1.154E+04, 9.744E+03, 8.261E+03, 7.030E+03,
                             6.005E+03, 5.147E+03, 4.427E+03, 3.820E+03, 3.307E+03, 2.872E+03])
        self.b = np.asarray([5.000E+01, 5.500E+01, 6.000E+01, 6.500E+01, 7.000E+01,
                             7.500E+01, 8.000E+01, 8.500E+01, 9.000E+01, 9.500E+01,
                             1.000E+02, 1.050E+02, 1.100E+02, 1.150E+02, 1.200E+02, 1.250E+02])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        vec = x[0] * np.exp(x[1] / (self.b + x[2]))
        return np.sum((self.a - vec) ** 2)


class Michalewicz(Benchmark):
    """
    .. [1] Adorio, E. MVF - "Multivariate Test Functions Library in C for
    Unconstrained Global Optimization", 2005

    .. math::
        f(x) = - \sum_{i=1}^{2} \sin\left(x_i\right) \sin^{2 m}\left(\frac{i x_i^{2}}{\pi}\right)

    Here :math:`x_i \in [0, \pi]`.
    *Global optimum*: :math:`f(x) = -1.8013`for :math:`x = [0, 0]`
    """
    name = "McCormick Function"
    latex_formula = r'f(x) = - x_{1} + 2 x_{2} + \left(x_{1} - x_{2}\right)^{2} + \sin\left(x_{1} + x_{2}\right) + 1'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [0, \pi]`'
    latex_formula_global_optimum = r'f(0, 0) = -1.8013'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., np.pi] for _ in range(self.dim_default)]))
        self.f_global = -1.8013
        self.x_global = np.array([0, 0])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        m = 10.0
        idx = np.arange(1, self.ndim + 1)
        return -np.sum(np.sin(x) * np.sin(idx * x ** 2 / np.pi) ** (2 * m))


class MieleCantrell(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = (e^{-x_1} - x_2)^4 + 100(x_2 - x_3)^6 + \tan^4(x_3 - x_4) + x_1^8

    Here :math:`x_i \in [-1, 1] for i \in [1, 4]`.
    *Global optimum*: :math:`f(x) = 0`for :math:`x = [0, 1, 1, 1]`
    """
    name = "Miele Cantrell Function"
    latex_formula = r'f(x) = (e^{-x_1} - x_2)^4 + 100(x_2 - x_3)^6 + \tan^4(x_3 - x_4) + x_1^8'
    latex_formula_dimension = r'd = 4'
    latex_formula_bounds = r'x_i \in [-1, 1] for i \in [1, 4]'
    latex_formula_global_optimum = r'f(0, 1, 1, 1) = 0'
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
        self.dim_default = 4
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1., 1.] for _ in range(self.dim_default)]))
        self.f_global = 0
        self.x_global = np.array([0, 1, 1, 1])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (np.exp(-x[0]) - x[1]) ** 4 + 100 * (x[1] - x[2]) ** 6 + np.tan(x[2] - x[3]) ** 4 + x[0] ** 8


class Mishra01(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = (1 + x_n)^{x_n}
        x_n = n - \sum_{i=1}^{n-1} x_i

    Here :math:`x_i \in [0, 1] for i \in [1, n]`.
    *Global optimum*: :math:`f(x) = 2`for :math:`x_i = 1 for all i \in [1, n]`
    """
    name = "Mishra 1 Function"
    latex_formula = r'f(x) = (1 + x_n)^{x_n}; x_n = n - \sum_{i=1}^{n-1} x_i'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [0, 1] for i \in [1, n]'
    latex_formula_global_optimum = r'f(1) = 2'
    continuous = True
    linear = False
    convex = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 1.] for _ in range(self.dim_default)]))
        self.f_global = 2.0
        self.x_global = np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        xn = self.ndim - np.sum(x[0:-1])
        return (1 + xn) ** xn


class Mishra02(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = (1 + x_n)^{x_n}
        x_n = n - \sum_{i=1}^{n-1} \frac{(x_i + x_{i+1})}{2}

    Here :math:`x_i \in [0, 1] for i \in [1, n]`.
    *Global optimum*: :math:`f(x) = 2`for :math:`x_i = 1 for all i \in [1, n]`
    """
    name = "Mishra 2 Function"
    latex_formula = r'f(x) = (1 + x_n)^{x_n}; x_n = n - \sum_{i=1}^{n-1} \frac{(x_i + x_{i+1})}{2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [0, 1] for i \in [1, n]'
    latex_formula_global_optimum = r'f(1) = 2'
    continuous = True
    linear = False
    convex = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[0., 1. + 1e-9] for _ in range(self.dim_default)]))
        self.f_global = 2.0
        self.x_global = np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        xn = self.ndim - np.sum((x[:-1] + x[1:]) / 2.0)
        return (1 + xn) ** xn


class Mishra03(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \sqrt{\lvert \cos{\sqrt{\lvert x_1^2 + x_2^2 \rvert}} \rvert} + 0.01(x_1 + x_2)

    Here :math:`x_i \in [0, 1] for i \in [1, n]`.
    *Global optimum*: :math:`f(-9.99378322, -9.99918927) = -0.19990562`
    """
    name = "Mishra 3 Function"
    latex_formula = r'f(x) = \sqrt{\lvert \cos{\sqrt{\lvert x_1^2 + x_2^2 \rvert}} \rvert} + 0.01(x_1 + x_2)'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10] for i \in [1, 2]'
    latex_formula_global_optimum = r'f(-9.99378322, -9.99918927) = -0.19990562'
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
        self.f_global = -0.19990562
        self.x_global = np.array([-9.99378322, -9.99918927])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return 0.01 * (x[0] + x[1]) + np.sqrt(np.abs(np.cos(np.sqrt(np.abs(x[0] ** 2 + x[1] ** 2)))))


class Mishra04(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \sqrt{\lvert \sin{\sqrt{\lvert x_1^2 + x_2^2 \rvert}} \rvert} + 0.01(x_1 + x_2)

    Here :math:`x_i \in [-10, 10] for i \in [1, n]`.
    *Global optimum*: :math:`f(-8.71499636, -9.0533148) = -0.17767`
    """
    name = "Mishra 4 Function"
    latex_formula = r'f(x) = \sqrt{\lvert \sin{\sqrt{\lvert x_1^2 + x_2^2 \rvert}} \rvert} + 0.01(x_1 + x_2)'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10] for i \in [1, 2]'
    latex_formula_global_optimum = r'f(-8.71499636, -9.0533148) = -0.17767'
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
        self.f_global = -0.17767
        self.x_global = np.array([-8.71499636, -9.0533148])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return 0.01 * (x[0] + x[1]) + np.sqrt(np.abs(np.sin(np.sqrt(abs(x[0] ** 2 + x[1] ** 2)))))


class Mishra05(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) + \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2 + 0.01(x_1 + x_2)

    Here :math:`x_i \in [-10, 10] for i \in [1, 2]`.
    *Global optimum*: :math:`f(-1.98682, -10) = -1.019829519930646`
    """
    name = "Mishra 5 Function"
    latex_formula = r'f(x) = \left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) + \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2 + 0.01(x_1 + x_2)'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10] for i \in [1, 2]'
    latex_formula_global_optimum = r'f(-1.98682, -10) = -1.019829519930646'
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
        self.f_global = -1.019829519930646
        self.x_global = np.array([-1.98682, -10])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (0.01 * x[0] + 0.1 * x[1]
                + (np.sin((np.cos(x[0]) + np.cos(x[1])) ** 2) ** 2 + np.cos((np.sin(x[0]) + np.sin(x[1])) ** 2) ** 2 + x[0]) ** 2)


class Mishra06(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = -\log{\left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) - \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2} + 0.01 \left[(x_1 -1)^2 + (x_2 - 1)^2 \right]

    Here :math:`x_i \in [-10, 10] for i \in [1, 2]`.
    *Global optimum*: :math:`f(2.88631, 1.82326) = -2.28395`
    """
    name = "Mishra 6 Function"
    latex_formula = r'f(x) = -\log{\left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) - \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2} + 0.01 \left[(x_1 -1)^2 + (x_2 - 1)^2 \right]'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10] for i \in [1, 2]'
    latex_formula_global_optimum = r'f(2.88631, 1.82326) = -2.28395'
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
        self.f_global = -2.28395
        self.x_global = np.array([2.88631, 1.82326])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        a = 0.1 * ((x[0] - 1) ** 2 + (x[1] - 1) ** 2)
        u = (np.cos(x[0]) + np.cos(x[1])) ** 2
        v = (np.sin(x[0]) + np.sin(x[1])) ** 2
        return a - np.log((np.sin(u) ** 2 - np.cos(v) ** 2 + x[0]) ** 2)


class Mishra07(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \left [\prod_{i=1}^{n} x_i - n! \right]^2

    Here :math:`x_i \in [-10, 10] for i \in [1, n]`.
    *Global optimum*: :math:`f(\sqrt{n}) = 0, `
    """
    name = "Mishra 7 Function"
    latex_formula = r'f(x) = \left [\prod_{i=1}^{n} x_i - n! \right]^2'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10] \forall i \in [1, n]'
    latex_formula_global_optimum = r'f(\sqrt{n}) = 0'
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
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0
        self.x_global = np.sqrt(self.ndim) * np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (np.prod(x) - np.math.factorial(self.ndim)) ** 2.0


class Mishra08(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = 0.001 \left[\lvert x_1^{10} - 20x_1^9 + 180x_1^8 - 960 x_1^7 + 3360x_1^6 - 8064x_1^5 + 13340x_1^4 - 15360x_1^3
       + 11520x_1^2 - 5120x_1 + 2624 \rvert \lvert x_2^4 + 12x_2^3 + 54x_2^2 + 108x_2 + 81 \rvert \right]^2

    Here :math:`x_i \in [-10, 10] for i \in [1, 2]`.
    *Global optimum*: :math:`f(2, -3) = 0, `
    """
    name = "Mishra 8 Function"
    latex_formula = r'f(x) = 0.001 \left[\lvert x_1^{10} - 20x_1^9 + 180x_1^8 - 960 x_1^7 + 3360x_1^6 - 8064x_1^5 + 13340x_1^4 - 15360x_1^3 + 11520x_1^2 - 5120x_1 + 2624 \rvert \lvert x_2^4 + 12x_2^3 + 54x_2^2 + 108x_2 + 81 \rvert \right]^2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10] \forall i \in [1, 2]'
    latex_formula_global_optimum = r'f(2, -3) = 0'
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
        self.f_global = 0.0
        self.x_global = np.array([2., -3.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        val = np.abs(x[0] ** 10 - 20 * x[0] ** 9 + 180 * x[0] ** 8 - 960 * x[0] ** 7 + 3360 * x[0] ** 6 - 8064 * x[0] ** 5
                     + 13340 * x[0] ** 4 - 15360 * x[0] ** 3 + 11520 * x[0] ** 2 - 5120 * x[0] + 2624)
        val += np.abs(x[1] ** 4 + 12 * x[1] ** 3 + 54 * x[1] ** 2 + 108 * x[1] + 81)
        return 0.001 * val ** 2


class Mishra09(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \left[ ab^2c + abc^2 + b^2 + (x_1 + x_2 - x_3)^2 \right]^2

    Where, in this exercise:

    .. math::
        \begin{cases} a = 2x_1^3 + 5x_1x_2 + 4x_3 - 2x_1^2x_3 - 18 \\
        b = x_1 + x_2^3 + x_1x_2^2 + x_1x_3^2 - 22 \\
        c = 8x_1^2 + 2x_2x_3 + 2x_2^2 + 3x_2^3 - 52 \end{cases}


    Here :math:`x_i \in [-10, 10] for i \in [1, 2, 3]`.
    *Global optimum*: :math:`f(1, 2, 3) = 0, `
    """
    name = "Mishra 9 Function"
    latex_formula = r'\left[ ab^2c + abc^2 + b^2 + (x_1 + x_2 - x_3)^2 \right]^2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10] \forall i \in [1, 2, 3]'
    latex_formula_global_optimum = r'f(1, 2, 3) = 0'
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
        self.x_global = np.array([1., 2., 3.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        a = (2 * x[0] ** 3 + 5 * x[0] * x[1] + 4 * x[2] - 2 * x[0] ** 2 * x[2] - 18)
        b = x[0] + x[1] ** 3 + x[0] * x[1] ** 2 + x[0] * x[2] ** 2 - 22.0
        c = (8 * x[0] ** 2 + 2 * x[1] * x[2] + 2 * x[1] ** 2 + 3 * x[1] ** 3 - 52)
        return (a * c * b ** 2 + a * b * c ** 2 + b ** 2 + (x[0] + x[1] - x[2]) ** 2) ** 2


class Mishra10(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \left[ \lfloor x_1 \perp x_2 \rfloor - \lfloor x_1 \rfloor - \lfloor x_2 \rfloor \right]^2

    Here :math:`x_i \in [-10, 10] for i \in [1, 2]`.
    *Global optimum*: :math:`f(2, 2) = 0, `
    """
    name = "Mishra 10 Function"
    latex_formula = r'\left[ \lfloor x_1 \perp x_2 \rfloor - \lfloor x_1 \rfloor - \lfloor x_2 \rfloor \right]^2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10] \forall i \in [1, 2]'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.array([2., 2.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x_int = x.astype(int)
        return ((x_int[0] + x_int[1]) - (x_int[0] * x_int[1])) ** 2.0


class Mishra11(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \left [ \frac{1}{n} \sum_{i=1}^{n} \lvert x_i \rvert - \left(\prod_{i=1}^{n} \lvert x_i \rvert \right )^{\frac{1}{n}} \right]^2

    Here :math:`x_i \in [-10, 10] for i \in [1, 2]`.
    *Global optimum*: :math:`f(0) = 0, `
    """
    name = "Mishra 11 Function"
    latex_formula = r'\left [ \frac{1}{n} \sum_{i=1}^{n} \lvert x_i \rvert - \left(\prod_{i=1}^{n} \lvert x_i \rvert \right )^{\frac{1}{n}} \right]^2'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10] \forall i \in [1, n]'
    latex_formula_global_optimum = r'f(0) = 0'
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
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim, dtype=float)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return ((1.0 / self.ndim) * np.sum(np.abs(x)) - (np.prod(np.abs(x))) ** 1.0 / self.ndim) ** 2.0


class MultiModal(Benchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::
        f(x) = \left( \sum_{i=1}^n \lvert x_i \rvert \right) \left( \prod_{i=1}^n \lvert x_i \rvert \right)

    Here :math:`x_i \in [-10, 10] for i \in [1, n]`.
    *Global optimum*: :math:`f(0) = 0, `
    """
    name = "Mishra 11 Function"
    latex_formula = r'\left( \sum_{i=1}^n \lvert x_i \rvert \right) \left( \prod_{i=1}^n \lvert x_i \rvert \right)'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10] \forall i \in [1, n]'
    latex_formula_global_optimum = r'f(0) = 0'
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
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim, dtype=float)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.sum(np.abs(x)) * np.prod(np.abs(x))
