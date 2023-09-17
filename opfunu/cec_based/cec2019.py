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
        self.x_global = np.zeros(self.ndim)
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.chebyshev_func(x) + self.f_bias


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
        # the f_global and x_global was obtained by executing the cec2019 c code
        self.f_global = int(np.sqrt(self.ndim)) + f_bias
        self.x_global = np.zeros(self.ndim)
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.inverse_hilbert_func(x) + self.f_bias


class F32019(CecBenchmark):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization

    **Note: The CEC 2019 implementation and this implementation results match when x* = [0,...,0] and
    """
    name = "F3: Lennard-Jones Minimum Energy Cluster Problem"
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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_3", f_bias=1.):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 18
        self.dim_max = 18
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-4., 4.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2019")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        # f_global calculated by verifying the cec2019 C value for f(x*) where x*==f_shift
        self.f_global = 12.712062001703194 + self.f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.lennard_jones_func(x) + self.f_bias


class F42019(CecBenchmark):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """
    name = "F4: Shifted and Rotated Rastrigin’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0'

    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Local optima’s number is huge", "The penultimate optimum is far from the global optimum"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_4", f_matrix="M_1_D", f_bias=1.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 10
        self.dim_supported = [2, 10]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2019")
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
        return operator.rastrigin_func(z) + self.f_bias


class F52019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """
    name = "F5: Shifted and Rotated Griewank’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0'

    convex = True
    modality = False  # Number of ambiguous peaks, unknown # peaks

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_5", f_matrix="M_5_D", f_bias=1.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.griewank_func(z) + self.f_bias


class F62019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """
    name = "F6: Shifted and Rotated Weierstrass Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0'

    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks

    characteristics = ["Local optima’s number is huge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_6", f_matrix="M_6_D", f_bias=1.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.weierstrass_norm_func(z) + self.f_bias


class F72019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """
    name = "F7: Shifted and Rotated Schwefel’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0'

    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks

    characteristics = ["Local optima’s number is huge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_7", f_matrix="M_7_D", f_bias=1.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.modified_schwefel_func(z) + self.f_bias


class F82019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """
    name = "F8: Shifted and Rotated Expanded Schaffer’s F6 Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0'

    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks

    characteristics = ["Local optima’s number is huge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_8", f_matrix="M_8_D", f_bias=1.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 0.005*(x - self.f_shift))
        return operator.expanded_scaffer_f6_func(z) + self.f_bias


class F92019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """
    name = "F9: Shifted and Rotated Happy Cat Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0'

    convex = False
    modality = False  # Number of ambiguous peaks, unknown # peaks

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_9", f_matrix="M_9_D", f_bias=1.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.happy_cat_func(z, shift=-1.0) + self.f_bias


class F102019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """
    name = "F10: Shifted and Rotated Ackley Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0'

    convex = False
    modality = False  # Number of ambiguous peaks, unknown # peaks

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_10", f_matrix="M_10_D", f_bias=1.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.ackley_func(z) + self.f_bias
