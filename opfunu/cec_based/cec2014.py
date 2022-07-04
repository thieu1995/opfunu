#!/usr/bin/env python
# Created by "Thieu" at 15:45, 04/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec import CecBenchmark
from opfunu.utils import operator


class F12014(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F1: Rotated High Conditioned Elliptic Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 100.0'

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
    rotated = True

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Quadratic ill-conditioned", "Smooth local irregularities"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_1", f_matrix="M_1_D", f_bias=100.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [10, 20, 30, 50, 100]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2014")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.elliptic_func(z) + self.f_bias


class F22014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F2: Rotated Bent Cigar Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 200.0'

    characteristics = ["Smooth but narrow ridge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_2", f_matrix="M_2_D", f_bias=200.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [10, 20, 30, 50, 100]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2014")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.bent_cigar_func(z) + self.f_bias


class F32014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F3: Rotated Discus Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 300.0'

    characteristics = ["With one sensitive direction"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_3", f_matrix="M_3_D", f_bias=300.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [10, 20, 30, 50, 100]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2014")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.discus_func(z) + self.f_bias











