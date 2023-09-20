#!/usr/bin/env python
# Created by "Thieu" at 22:19, 01/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec import CecBenchmark
from opfunu.utils import operator


class F12008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
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
    rotated = False

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="sphere_shift_func_data", f_bias=-450.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 500
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2008")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.sphere_func(x - self.f_shift) + self.f_bias


class F22008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """
    name = "F2: Schwefel’s Problem 2.21"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0'
    continuous = False
    linear = True
    convex = True
    unimodal = True
    separable = False

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = False

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="schwefel_shift_func_data", f_bias=-450.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 500
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2008")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return np.max(np.abs(x - self.f_shift)) + self.f_bias


class F32008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """
    name = "F3: Shifted Rosenbrock’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = False

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="rosenbrock_shift_func_data", f_bias=-390.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 500
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2008")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.rosenbrock_func(x - self.f_shift, shift=1.0) + self.f_bias


class F42008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """
    name = "F4: Shifted Rastrigin’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
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

    def __init__(self, ndim=None, bounds=None, f_shift="rastrigin_shift_func_data", f_bias=-330.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 500
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2008")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        return operator.rastrigin_func(z) + self.f_bias


class F52008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """
    name = "F5: Shifted Griewank’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = False

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="griewank_shift_func_data", f_bias=-180.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 500
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-600., 600.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2008")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        return operator.griewank_func(z) + self.f_bias


class F62008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """
    name = "F6: Shifted Ackley’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
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

    def __init__(self, ndim=None, bounds=None, f_shift="ackley_shift_func_data", f_bias=-140.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 500
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-32., 32.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2008")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        return operator.ackley_func(z) + self.f_bias

class F72008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """
    name = "F7: FastFractal “DoubleDip” Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = unknown, F_1(x^*) = unknown'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = False
    scalable = True
    randomized_term = True
    parametric = True
    shifted = True
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="rastrigin_shift_func_data", f_bias=0.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 500
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-1., 1.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2008")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_shift = self.f_shift / np.max(self.f_shift)
        self.f_bias = f_bias
        self.f_global = -1e32
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        ndim = len(x)
        results = [operator.fractal_1d_func(x[idx] + operator.twist_func(x[idx+1])) for idx in range(0, ndim-1)]
        return np.sum(results) + operator.fractal_1d_func(x[-1] + operator.twist_func(x[0])) + self.f_bias
