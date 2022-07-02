#!/usr/bin/env python
# Created by "Thieu" at 18:15, 02/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec import CecBenchmark
from opfunu.utils import operator


class F12013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """
    name = "F1: Sphere Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -1400.0'

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

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data", f_bias=-1400.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_data(f_shift, kind="matrix")[0, :self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return np.sum((x - self.f_shift) ** 2) + self.f_bias


class F22013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """
    name = "F2: Rotated High Conditioned Elliptic Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -1300.0'

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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data", f_matrix="M_D", f_bias=-1300.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_data(f_shift, kind="matrix")[0, :self.ndim]
        self.f_matrix = self.check_matrix_data(f"{f_matrix}{self.ndim}")[:self.ndim, :self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = operator.tosz_func(np.dot(self.f_matrix, x - self.f_shift))
        return operator.elliptic_func(z) + self.f_bias


class F32013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """
    name = "F3: Rotated Bent Cigar Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -1200.0'

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

    characteristics = ["Smooth but narrow ridge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data", f_matrix="M_D", f_bias=-1200.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_data(f_shift, kind="matrix")[0, :self.ndim]
        self.f_matrix = self.check_matrix_data(f"{f_matrix}{self.ndim}")[:2*self.ndim, :self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        M1 = self.f_matrix[:self.ndim, :]
        M2 = self.f_matrix[self.ndim:, :]
        z = operator.tasy_func(np.dot(M1, x - self.f_shift), beta=0.5)
        return operator.bent_cigar_func(np.dot(M2, z)) + self.f_bias


class F42013(F22013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """
    name = "F4: Rotated Discus Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -1100.0'

    characteristics = ["Asymmetrical", "Smooth local irregularities", "With one sensitive direction"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data", f_matrix="M_D", f_bias=-1100.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = operator.tosz_func(np.dot(self.f_matrix, x - self.f_shift))
        return operator.discus_func(z) + self.f_bias


class F52013(F12013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """
    name = "F5: Different Powers Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -1000.0'

    continuous = False
    differentiable = False

    characteristics = ["Sensitivities of the zi-variables are different"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data", f_bias=-1000.):
        super().__init__(ndim, bounds, f_shift, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.different_powers_func(x - self.f_shift) + self.f_bias





