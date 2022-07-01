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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="data_sphere", f_bias=-450.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return np.sum((x - self.f_shift) ** 2) + self.f_bias


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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="data_schwefel_102", f_bias=-450.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        ndim = len(x)
        results = [np.sum(x[:idx] - self.f_shift[:idx])**2 for idx in range(0, ndim)]
        return np.sum(results) + self.f_bias


class F32005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """
    name = "F3: Shifted Rotated High Conditioned Elliptic Function"
    latex_formula = r'F_3(x) = \sum_{i=1}^D (10^6)^{\frac{i-1}{D-1}} z_i^2 + bias; \\ z=(x-o).M; x=[x_1, ..., x_D], \\o=[o_1, ..., o_D]: \text{the shifted global optimum}\\ M: \text{orthogonal matrix}'
    latex_formula_dimension = r'D \in [10, 30, 50]'
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_3(x^*) = bias = -450.0'
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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="data_high_cond_elliptic_rot", f_matrix="elliptic_M_D", f_bias=-450.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [10, 30, 50]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f"{f_matrix}{self.ndim}")
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_matrix": self.f_matrix, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        ndim = len(x)
        z = (np.dot((x - self.f_shift), self.f_matrix)) ** 2
        results = [(10**6)**(idx / (ndim - 1)) * z[idx]**2 for idx in range(0, ndim)]
        return np.sum(results) + self.f_bias


class F42005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """
    name = "F4: Shifted Schwefel’s Problem 1.2 with Noise in Fitness"
    latex_formula = r'F_4(x) = \Big(\sum_{i=1}^D (\sum_{j=1}^i)^2\Big)*\Big(1 + 0.4|N(0, 1)|\Big)+ bias;\\ z=(x-o).M; x=[x_1, ..., x_D], \\o=[o_1, ..., o_D]: \text{the shifted global optimum}\\ N(0,1): \text{gaussian noise}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_4(x^*) = bias = -450.0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = False

    differentiable = True
    scalable = True
    randomized_term = True
    parametric = True
    shifted = True
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="data_schwefel_102", f_bias=-450.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        ndim = len(x)
        results = [np.sum(x[:idx] - self.f_shift[:idx])**2 for idx in range(0, ndim)]
        return np.sum(results) * (1 + 0.4 * np.abs(np.random.normal(0, 1))) + self.f_bias


class F52005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """
    name = "F5: Schwefel’s Problem 2.6 with Global Optimum on Bounds"
    latex_formula = r'F_5(x) = max{\Big| A_ix - B_i \Big|} + bias; i=1,...,D; x=[x_1, ..., x_D];' + \
        r'\\A: \text{is D*D matrix}, a_{ij}: \text{are integer random numbers in range [-500, 500]};' + \
        r'\\det(A) \neq 0; A_i: \text{is the } i^{th} \text{ row of A.}' + \
        r'\\B_i = A_i * o, o=[o_1, ..., o_D]: \text{the shifted global optimum}' + \
        r'\\ \text{After load the data file, set } o_i=-100, \text{ for } i=1,2,...[D/4], \text{and }o_i=100 \text{ for } i=[3D/4,...,D]'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_5(x^*) = bias = -310.0'
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

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="data_schwefel_206", f_bias=-310.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        shift_data, matrix_data = self.load_shift_and_matrix_data(f_shift)
        self.f_shift = shift_data[:self.ndim]
        self.f_matrix = matrix_data[:self.ndim, :self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        ndim = len(x)
        self.f_shift[:int(0.25 * ndim) + 1] = -100
        self.f_shift[int(0.75 * ndim):] = 100
        results = [np.abs(np.dot(self.f_matrix[idx], x) - np.dot(self.f_matrix[idx], self.f_shift)) for idx in range(0, ndim)]
        return np.max(results) + self.f_bias


class F62005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """
    name = "F6: Shifted Rosenbrock’s Function"
    latex_formula = r'F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;' + \
        '\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0'
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

    def __init__(self, ndim=None, bounds=None, f_shift="data_rosenbrock", f_bias=390.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        ndim = len(x)
        z = x - self.f_shift + 1
        results = [(100 * (z[idx]**2 - z[idx+1])**2 + (z[idx] - 1)**2) for idx in range(0, ndim-1)]
        return np.sum(results) + self.f_bias


class F72005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """
    name = "F7: Shifted Rotated Griewank’s Function without Bounds"
    latex_formula = r'F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;' + \
        '\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = True
    parametric = True
    shifted = True
    rotated = True

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self, ndim=None, bounds=None, f_shift="data_griewank", f_matrix="griewank_M_D", f_bias=-180.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [10, 30, 50]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[0., 600.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f"{f_matrix}{self.ndim}")
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_matrix": self.f_matrix, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        ndim = len(x)
        z = np.dot((x - self.f_shift), self.f_matrix)
        vt1 = np.sum(z**2) / 4000 + 1
        results = [np.cos(z[idx] / np.sqrt(idx+1)) for idx in range(0, ndim)]

        return vt1 - np.sum(results) + self.f_bias





