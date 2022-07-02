#!/usr/bin/env python
# Created by "Thieu" at 09:55, 02/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec import CecBenchmark
from opfunu.utils import operator


class F12010(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """
    name = "F1: Shifted Elliptic Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'
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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="f01_o"):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 100
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2010")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_global = 0
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift,}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.elliptic_func(x - self.f_shift)


class F22010(F12010):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """
    name = "F1: Shifted Elliptic Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f02_o"):
        super().__init__(ndim, bounds, f_shift)
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.rastrigin_func(x - self.f_shift)



























