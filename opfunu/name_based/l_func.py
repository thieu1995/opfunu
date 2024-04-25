#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Langermann(Benchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::
        f_{\text{Langermann}}(x) = - \sum_{i=1}^{5}
        \frac{c_i \cos\left\{\pi \left[\left(x_{1}- a_i\right)^{2}
        + \left(x_{2} - b_i \right)^{2}\right]\right\}}{e^{\frac{\left( x_{1}
        - a_i\right)^{2} + \left( x_{2} - b_i\right)^{2}}{\pi}}}
    Where:
    .. math::
        \begin{matrix}
        a = [3, 5, 2, 1, 7]\\
        b = [5, 2, 1, 4, 9]\\
        c = [1, 2, 5, 2, 3] \\
        \end{matrix}
    Here :math:`x_i \in [0, 10]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = -5.1621259`for :math:`x = [2.00299219, 1.006096]`
    """
    name = "Langermann Function"
    latex_formula = r'f_{\text{Langermann}}(x) = - \sum_{i=1}^{5} \frac{c_i \cos\left\{\pi \left[\left(x_{1}- a_i\right)^{2}  + \left(x_{2} - b_i \right)^{2}\right]\right\}}{e^{\frac{\left( x_{1} - a_i\right)^{2} + \left( x_{2} - b_i\right)^{2}}{\pi}}}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [0, 10]'
    latex_formula_global_optimum = r'f(2.00299219, 1.006096) = -5.1621259'
    continuous = True
    linear = False
    convex = False
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
        self.f_global = -5.1621259
        self.x_global = np.array([2.00299219, 1.006096])

        self.a = np.array([3, 5, 2, 1, 7])
        self.b = np.array([5, 2, 1, 4, 9])
        self.c = np.array([1, 2, 5, 2, 3])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return (-np.sum(self.c * np.exp(-(1 / np.pi) * ((x[0] - self.a) ** 2 +
                (x[1] - self.b) ** 2)) * np.cos(np.pi * ((x[0] - self.a) ** 2 + (x[1] - self.b) ** 2))))


class LennardJones(Benchmark):
    """
    .. [1] http://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html

    .. math::
        f_{\text{LennardJones}}(\mathbf{x}) = \sum_{i=0}^{n-2}\sum_{j>1}^{n-1}
        \frac{1}{r_{ij}^{12}} - \frac{1}{r_{ij}^{6}}
    Where, in this exercise:
    .. math::
        r_{ij} = \sqrt{(x_{3i}-x_{3j})^2 + (x_{3i+1}-x_{3j+1})^2) + (x_{3i+2}-x_{3j+2})^2}

    Valid for any dimension, :math:`n = 3*k, k=2 , 3, 4, ..., 20`. :math:`k` is the number of atoms in 3-D space
    constraints: unconstrained type: multi-modal with one global minimum; non-separable Value-to-reach: :math:`minima[k-2] + 0.0001`.
    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-4, 4]` for :math:`i = 1 ,..., n`.
    *Global optimum*:
    .. math::
        \text{minima} = [-1.,-3.,-6.,-9.103852,-12.712062,-16.505384,\\
                         -19.821489, -24.113360, -28.422532,-32.765970,\\
                         -37.967600,-44.326801, -47.845157,-52.322627,\\
                         -56.815742,-61.317995, -66.530949, -72.659782,\\
                         -77.1777043]\\
    """
    name = "LennardJones Function"
    latex_formula = r'f_{\text{LennardJones}}(\mathbf{x}) = \sum_{i=0}^{n-2}\sum_{j>1}^{n-1}\frac{1}{r_{ij}^{12}} - \frac{1}{r_{ij}^{6}}'
    latex_formula_dimension = r'd \in [6:60]'
    latex_formula_bounds = r'x_i \in [-4, 4]'
    latex_formula_global_optimum = r'f = [-1.,-3.,-6.,-9.103852,-12.712062,-16.505384, -19.821489, -24.113360, -28.422532,-32.765970, -37.967600,' \
                                   r'-44.326801, -47.845157,-52.322627, -56.815742,-61.317995, -66.530949, -72.659782, 77.1777043]'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = False
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_supported = list(range(6, 61))
        self.dim_changeable = True
        self.dim_default = 6
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-4., 4.] for _ in range(self.dim_default)]))
        self.minima = [-1.0, -3.0, -6.0, -9.103852, -12.712062,
                       -16.505384, -19.821489, -24.113360, -28.422532,
                       -32.765970, -37.967600, -44.326801, -47.845157,
                       -52.322627, -56.815742, -61.317995, -66.530949,
                       -72.659782, -77.1777043]
        self.f_global = self.minima[int(self.ndim/3) - 2]
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        if self.ndim not in self.dim_supported:
            raise ValueError(f"{self.__class__.__name__} problem is only supported ndim in {self.dim_supported}!")
        self.check_solution(x)
        self.n_fe += 1
        k = int(self.ndim / 3)
        s = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                a = 3 * i
                b = 3 * j
                xd = x[a] - x[b]
                yd = x[a + 1] - x[b + 1]
                zd = x[a + 2] - x[b + 2]
                ed = xd * xd + yd * yd + zd * zd
                ud = ed * ed * ed
                if ed > 0.0:
                    s += (1.0 / ud - 2.0) / ud
        return s


class Leon(Benchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f_{\text{Leon}}(\mathbf{x}) = \left(1 - x_{1}\right)^{2}
        + 100 \left(x_{2} - x_{1}^{2} \right)^{2}

    with :math:`x_i \in [-1.2, 1.2]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 1]`

    """
    name = "Leon Function"
    latex_formula = r'f_{\text{Leon}}(\mathbf{x}) = \left(1 - x_{1}\right)^{2} + 100 \left(x_{2} - x_{1}^{2} \right)^{2}'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-1.2, 1.2]'
    latex_formula_global_optimum = r'f(1, 1) = 0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-1.2, 1.2] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return 100. * (x[1] - x[0] ** 2.0) ** 2.0 + (1 - x[0]) ** 2.0


class Levy03(Benchmark):
    """
    .. [1] Mishra, S. Global Optimization by Differential Evolution and Particle Swarm Methods: Evaluation
    on Some Benchmark Functions. Munich Personal RePEc Archive, 2006, 1005

    .. math::
        f_{\text{Levy03}}(\mathbf{x}) = \sin^2(\pi y_1)+\sum_{i=1}^{n-1}(y_i-1)^2[1+10\sin^2(\pi y_{i+1})]+(y_n-1)^2

    .. math::
        y_i=1+\frac{x_i-1}{4}
    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-10, 10]` for :math:`i=1,...,n`.
    *Global optimum*: :math:`f(x_i) = 0` for :math:`x_i = 1` for :math:`i=1,...,n`
    """
    name = "Levy 3 Function"
    latex_formula = r'f_{\text{Levy03}}(\mathbf{x}) = \sin^2(\pi y_1)+\sum_{i=1}^{n-1}(y_i-1)^2[1+10\sin^2(\pi y_{i+1})]+(y_n-1)^2'
    latex_formula_dimension = r'd \in N^+'
    latex_formula_bounds = r'x_i \in [-10, 10]'
    latex_formula_global_optimum = r'f(1,... 1) = 0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

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
        self.f_global = 0.0
        self.x_global = np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        y = 1 + (x - 1) / 4
        v = np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
        z = (y[-1] - 1) ** 2
        return np.sin(np.pi * y[0]) ** 2 + v + z


class Levy05(Benchmark):
    """
    .. [1] Mishra, S. Global Optimization by Differential Evolution and Particle Swarm Methods: Evaluation
    on Some Benchmark Functions. Munich Personal RePEc Archive, 2006, 1005

    .. math::
        f_{\text{Levy05}}(\mathbf{x}) = \sum_{i=1}^{5} i \cos \left[(i-1)x_1 + i \right] \times \sum_{j=1}^{5} j \cos \left[(j+1)x_2 + j \right] + (x_1 + 1.42513)^2 + (x_2 + 0.80032)^2
    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-10, 10]` for :math:`i=1,...,n`.
    *Global optimum*: :math:`f(x_i) = -176.1375779` for :math:`\mathbf{x} = [-1.30685, -1.42485]`.
    """
    name = "Levy 5 Function"
    latex_formula = r'f(\mathbf{x}) = \sum_{i=1}^{5} i \cos \left[(i-1)x_1 + i \right] \times \sum_{j=1}^{5} j \cos \left[(j+1)x_2 + j \right] + (x_1 + 1.42513)^2 + (x_2 + 0.80032)^2'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10]'
    latex_formula_global_optimum = r'f(-1.30685, -1.42485) = -176.1375779'
    continuous = True
    linear = False
    convex = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = -176.1375779
        self.x_global = np.array([-1.30685, -1.42485])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        idx = np.arange(1, 6)
        a = idx * np.cos((idx - 1) * x[0] + idx)
        b = idx * np.cos((idx + 1) * x[1] + idx)
        return np.sum(a) * np.sum(b) + (x[0] + 1.42513) ** 2 + (x[1] + 0.80032) ** 2


class Levy13(Benchmark):
    """
    .. [1] Mishra, S. Global Optimization by Differential Evolution and Particle Swarm Methods: Evaluation
    on Some Benchmark Functions. Munich Personal RePEc Archive, 2006, 1005

    .. math::
        f_{\text{Levy13}}(x) = \left(x_{1} -1\right)^{2} \left[\sin^{2}\left(3 \pi x_{2}\right) + 1\right] + \left(x_{2}
        - 1\right)^{2} \left[\sin^{2}\left(2 \pi x_{2}\right)+ 1\right] + \sin^{2}\left(3 \pi x_{1}\right)
    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 1]`
    """
    name = "Levy 5 Function"
    latex_formula = r'f_{\text{Levy13}}(x) = \left(x_{1} -1\right)^{2} \left[\sin^{2}\left(3 \pi x_{2}\right) + 1\right] + \left(x_{2} - 1\right)^{2} \left[\sin^{2}\left(2 \pi x_{2}\right)+ 1\right] + \sin^{2}\left(3 \pi x_{1}\right)'
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10]'
    latex_formula_global_optimum = r'f(1., 1.) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.array([1., 1.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        u = np.sin(3 * np.pi * x[0]) ** 2
        v = (x[0] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1])) ** 2)
        w = (x[1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * x[1])) ** 2)
        return u + v + w

