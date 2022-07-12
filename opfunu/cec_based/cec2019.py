#!/usr/bin/env python
# Created by "Thieu" at 16:32, 12/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec import CecBenchmark
from opfunu.utils import operator


class F12019(CecBenchmark):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """
    name = "F1: Storn’s Chebyshev Polynomial Fitting Problem"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0'

    continuous = False
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = False
    scalable = False
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Multimodal with one global minimum", "Very highly conditioned", "fully parameter-dependent"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_1", f_bias=1.):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 9
        self.dim_max = 9
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-8192., 8192.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2019")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.storn_chebyshev_polynomial_fitting_func(x) + self.f_bias


class F22019(CecBenchmark):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """
    name = "F2: Inverse Hilbert Matrix Problem"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0'

    continuous = False
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = False
    scalable = False
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Multimodal with one global minimum", "Very highly conditioned", "fully parameter-dependent"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_2", f_bias=1.):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 16
        self.dim_max = 16
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-16384., 16384.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2019")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.inverse_hilbert_matrix_func(x - self.f_shift) + self.f_bias




