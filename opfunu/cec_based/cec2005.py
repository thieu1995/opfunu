#!/usr/bin/env python
# Created by "Thieu" at 06:36, 30/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec import CecBenchmark


class F12005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """
    name = "F1: Shifted Sphere Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="data_sphere.txt", f_bias=-450.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.check_shift_data(f_shift)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.check_solution(x, self.dim_max, None)
        ndim = len(x)
        return np.sum((x - self.f_shift[:ndim]) ** 2) + self.f_bias


class F22005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """
    name = "F2: Shifted Schwefel’s Problem 1.2"
    latex_formula = r'F_2(x) = \sum_{i=1}^D (\sum_{j=1}^i z_j)^2 + bias, z=x-o, \\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_2(x^*) = bias = -450.0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="data_schwefel_102.txt", f_bias=-450.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.check_shift_data(f_shift)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.check_solution(x, self.dim_max, None)
        ndim = len(x)
        results = [np.sum(x[:idx] - self.f_shift[:idx])**2 for idx in range(0, ndim)]
        return np.sum(results) + self.f_bias

