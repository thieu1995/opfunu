#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class NeedleEye(Benchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::
        f_{\text{NeedleEye}}(x) =
            \begin{cases}
            1 & \textrm{if }\hspace{5pt} \lvert x_i \rvert  <  eye \hspace{5pt}
            \forall i \\
            \sum_{i=1}^n (100 + \lvert x_i \rvert) & \textrm{if } \hspace{5pt}
            \lvert x_i \rvert > eye \\
            0 & \textrm{otherwise}\\
            \end{cases}

    Here :math:`x_i \in [-10, 10]` for :math:`i = 1, 2,...,n`.
    *Global optimum*: :math:`f(x) = 1.0`for :math:`x = [0, 0,...,0]`
    """
    name = "NeedleEye Function"
    latex_formula = r'f_{\text{Matyas}}(x) = \begin{cases} 1 & \textrm{if }\hspace{5pt} \lvert x_i \rvert  <  eye \hspace{5pt} \forall i \\ ' \
                    r'\sum_{i=1}^n (100 + \lvert x_i \rvert) & \textrm{if } \hspace{5pt}\lvert x_i \rvert > eye \\ 0 & \textrm{otherwise}\\\end{cases}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 1.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        f = fp = 0.0
        eye = 0.0001
        for val in x:
            if abs(val) >= eye:
                fp = 1.0
                f += 100.0 + abs(val)
            else:
                f += 1.0
        if fp < 1e-6:
            f = f / self.ndim
        return f


class NewFunction01(Benchmark):
    """
    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions. Munich Personal RePEc Archive, 2006, 1005

    .. math::
        f_{\text{NewFunction01}}(x) = \left | {\cos\left(\sqrt{\left|{x_{1}^{2}
       + x_{2}}\right|}\right)} \right |^{0.5} + (x_{1} + x_{2})/100

    Here :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = -0.18459899925`for :math:`x = [-8.46669057, -9.99982177]`
    """
    name = "NewFunction01 Function"
    latex_formula = "f_{\text{NewFunction01}}(x) = \left | {\cos\left(\sqrt{\left|{x_{1}^{2}+ x_{2}}\right|}\right)} \right |^{0.5} + (x_{1} + x_{2})/100"
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10]'
    latex_formula_global_optimum = r'f([-8.46669057, -9.99982177]) = -0.18459899925'
    continuous = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = -0.18459899925
        self.x_global = np.array([-8.46669057, -9.99982177])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return ((np.abs(np.cos(np.sqrt(np.abs(x[0] ** 2 + x[1]))))) ** 0.5 + 0.01 * (x[0] + x[1]))


class NewFunction02(Benchmark):
    """
    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions. Munich Personal RePEc Archive, 2006, 1005

    .. math::
        f_{\text{NewFunction02}}(x) = \left | {\sin\left(\sqrt{\lvert{x_{1}^{2}
       + x_{2}}\rvert}\right)} \right |^{0.5} + (x_{1} + x_{2})/100

    Here :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = -0.19933159253`for :math:`x = [-9.94103375, -9.99771235]`
    """
    name = "NewFunction02 Function"
    latex_formula = "f_{\text{NewFunction02}}(x) = \left | {\sin\left(\sqrt{\lvert{x_{1}^{2} + x_{2}}\rvert}\right)} \right |^{0.5} + (x_{1} + x_{2})/100"
    latex_formula_dimension = r'd = 2'
    latex_formula_bounds = r'x_i \in [-10, 10]'
    latex_formula_global_optimum = r'f([-9.94103375, -9.99771235]) = -0.19933159253'
    continuous = False
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = -0.19933159253
        self.x_global = np.array([-9.94103375, -9.99771235])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return ((np.abs(np.sin(np.sqrt(np.abs(x[0] ** 2 + x[1]))))) ** 0.5 + 0.01 * (x[0] + x[1]))
