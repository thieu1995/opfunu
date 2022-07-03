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
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
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
        self.dim_default = 1000
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
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F2: Shifted Rastrigin’s Function"
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


class F32010(F12010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F3: Shifted Ackley’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f03_o"):
        super().__init__(ndim, bounds, f_shift)
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-32., 32.] for _ in range(self.dim_default)]))

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.ackley_func(x - self.f_shift)


class F42010(CecBenchmark):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F4: Single-group Shifted and m-rotated Elliptic Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    continuous = True
    linear = False
    convex = False
    unimodal = True
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

    def __init__(self, ndim=None, bounds=None, f_shift="f04_op", f_matrix="f04_m", m_group=50):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 1000
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2010")
        f_shift = self.load_matrix_data(f_shift)
        self.f_matrix = self.check_matrix_data(f_matrix, False)
        self.f_shift = f_shift[:1, :].ravel()[:self.ndim]
        if self.ndim == 1000:
            self.P = (f_shift[1:, :].ravel() - np.ones(self.ndim)).astype(int)
        else:
            np.random.seed(0)
            self.P = np.random.permutation(self.ndim)
        self.m_group = self.check_m_group(m_group)
        self.f_global = 0
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "P": self.P, "f_matrix": self.f_matrix, "m_group": self.m_group}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        idx1 = self.P[:self.m_group]
        idx2 = self.P[self.m_group:]
        z_rot_elliptic = np.dot(z[idx1], self.f_matrix[:self.m_group, :self.m_group])
        z_elliptic = z[idx2]
        return operator.elliptic_func(z_rot_elliptic)*10**6 + operator.elliptic_func(z_elliptic)


class F52010(F42010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F5: Single-group Shifted and m-rotated Rastrigin’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f05_op", f_matrix="f05_m", m_group=50):
        super().__init__(ndim, bounds, f_shift, f_matrix, m_group)
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        idx1 = self.P[:self.m_group]
        idx2 = self.P[self.m_group:]
        z_rot_ras = np.dot(z[idx1], self.f_matrix[:self.m_group, :self.m_group])
        z_ras = z[idx2]
        return operator.rastrigin_func(z_rot_ras)*10**6 + operator.rastrigin_func(z_ras)


class F62010(F42010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F6: Single-group Shifted and m-rotated Ackley’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f06_op", f_matrix="f06_m", m_group=50):
        super().__init__(ndim, bounds, f_shift, f_matrix, m_group)
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-32., 32.] for _ in range(self.dim_default)]))

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        idx1 = self.P[:self.m_group]
        idx2 = self.P[self.m_group:]
        z_rot_ras = np.dot(z[idx1], self.f_matrix[:self.m_group, :self.m_group])
        z_ras = z[idx2]
        return operator.ackley_func(z_rot_ras)*10**6 + operator.ackley_func(z_ras)


class F72010(CecBenchmark):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F7: Single-group Shifted m-dimensional Schwefel’s Problem 1.2"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    continuous = True
    linear = False
    convex = False
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

    def __init__(self, ndim=None, bounds=None, f_shift="f07_op", m_group=50):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 1000
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2010")
        f_shift = self.load_matrix_data(f_shift)
        self.f_shift = f_shift[:1, :].ravel()[:self.ndim]
        if self.ndim == 1000:
            self.P = (f_shift[1:, :].ravel() - np.ones(self.ndim)).astype(int)
        else:
            np.random.seed(0)
            self.P = np.random.permutation(self.ndim)
        self.m_group = self.check_m_group(m_group)
        self.f_global = 0
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "P": self.P, "m_group": self.m_group}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z_schwefel = z[self.P[:self.m_group]]
        z_sphere = z[self.P[self.m_group:]]
        return operator.schwefel_12_func(z_schwefel)*10**6 + operator.sphere_func(z_sphere)


class F82010(F72010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F8: Single-group Shifted m-dimensional Rosenbrock’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f08_op", m_group=50):
        super().__init__(ndim, bounds, f_shift, m_group)
        self.x_global = self.f_shift.copy()
        self.x_global[self.P[:self.m_group]] = self.f_shift[self.P[:self.m_group]] + 1
        self.x_global[self.P[self.m_group:]] = self.f_shift[self.P[self.m_group:]]

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z_rosen = z[self.P[:self.m_group]]
        z_sphere = z[self.P[self.m_group:]]
        return operator.rosenbrock_func(z_rosen)*10**6 + operator.sphere_func(z_sphere)


class F92010(CecBenchmark):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F9: D/2m-group Shifted and m-rotated Elliptic Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    continuous = True
    linear = False
    convex = False
    unimodal = True
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

    def __init__(self, ndim=None, bounds=None, f_shift="f09_op", f_matrix="f09_m", m_group=50):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 1000
        self.dim_max = 1000
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2010")
        self.f_matrix = self.check_matrix_data(f_matrix, False)
        f_shift = self.load_matrix_data(f_shift)
        self.f_shift = f_shift[:1, :].ravel()[:self.ndim]
        if self.ndim == 1000:
            self.P = (f_shift[1:, :].ravel() - np.ones(self.ndim)).astype(int)
        else:
            np.random.seed(0)
            self.P = np.random.permutation(self.ndim)
        self.m_group = self.check_m_group(m_group)
        self.f_global = 0
        self.x_global = self.f_shift
        self.count_up = int(self.ndim / (2*self.m_group))
        self.paras = {"f_shift": self.f_shift, "P": self.P, "f_matrix": self.f_matrix, "m_group": self.m_group}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        result = 0.0
        for k in range(0, self.count_up):
            idx1 = self.P[k*self.m_group: (k + 1)*self.m_group]
            z1 = np.dot(z[idx1], self.f_matrix[:len(idx1), :len(idx1)])
            result += operator.elliptic_func(z1)
        z2 = z[self.P[int(self.ndim/2):]]
        return result + operator.elliptic_func(z2)


class F102010(F92010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F10: D/2m-group Shifted and m-rotated Rastrigin’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f10_op", f_matrix="f10_m", m_group=50):
        super().__init__(ndim, bounds, f_shift, f_matrix, m_group)
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        result = 0.0
        for k in range(0, self.count_up):
            idx1 = self.P[k*self.m_group: (k + 1)*self.m_group]
            z1 = np.dot(z[idx1], self.f_matrix[:len(idx1), :len(idx1)])
            result += operator.rastrigin_func(z1)
        z2 = z[self.P[int(self.ndim/2):]]
        return result + operator.rastrigin_func(z2)


class F112010(F92010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F11: D/2m-group Shifted and m-rotated Ackley’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f11_op", f_matrix="f11_m", m_group=50):
        super().__init__(ndim, bounds, f_shift, f_matrix, m_group)
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-32., 32.] for _ in range(self.dim_default)]))

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        result = 0.0
        for k in range(0, self.count_up):
            idx1 = self.P[k*self.m_group: (k + 1)*self.m_group]
            z1 = np.dot(z[idx1], self.f_matrix[:len(idx1), :len(idx1)])
            result += operator.ackley_func(z1)
        z2 = z[self.P[int(self.ndim/2):]]
        return result + operator.ackley_func(z2)


class F122010(F72010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F12: D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    def __init__(self, ndim=None, bounds=None, f_shift="f11_op", m_group=50):
        super().__init__(ndim, bounds, f_shift, m_group)
        self.count_up = int(self.ndim / (2 * self.m_group))

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        result = 0.0
        for k in range(0, self.count_up):
            idx1 = self.P[k*self.m_group: (k + 1)*self.m_group]
            result += operator.schwefel_12_func(z[idx1])
        z2 = z[self.P[int(self.ndim/2):]]
        return result + operator.sphere_func(z2)


class F132010(F72010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F13: D/2m-group Shifted m-dimensional Rosenbrock’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'
    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f13_op", m_group=50):
        super().__init__(ndim, bounds, f_shift, m_group)
        self.count_up = int(self.ndim / (2 * self.m_group))
        self.x_global = self.f_shift.copy()
        self.x_global[self.P[:int(self.ndim/2)]] = self.f_shift[self.P[:int(self.ndim/2)]] + 1
        self.x_global[self.P[int(self.ndim/2):]] = self.f_shift[self.P[int(self.ndim/2):]]

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        result = 0.0
        for k in range(0, self.count_up):
            idx1 = self.P[k*self.m_group: (k + 1)*self.m_group]
            result += operator.rosenbrock_func(z[idx1])
        z2 = z[self.P[int(self.ndim/2):]]
        return result + operator.sphere_func(z2)


class F142010(F92010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F14: D/m-group Shifted and m-rotated Elliptic Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    def __init__(self, ndim=None, bounds=None, f_shift="f14_op", f_matrix="f14_m", m_group=50):
        super().__init__(ndim, bounds, f_shift, f_matrix, m_group)
        self.count_up = int(self.ndim / self.m_group)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        result = 0.0
        for k in range(0, self.count_up):
            idx1 = self.P[k*self.m_group: (k + 1)*self.m_group]
            z1 = np.dot(z[idx1], self.f_matrix[:len(idx1), :len(idx1)])
            result += operator.elliptic_func(z1)
        return result


class F152010(F92010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F15: D/m-group Shifted and m-rotated Rastrigin’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f15_op", f_matrix="f15_m", m_group=50):
        super().__init__(ndim, bounds, f_shift, f_matrix, m_group)
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))
        self.count_up = int(self.ndim / self.m_group)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        result = 0.0
        for k in range(0, self.count_up):
            idx1 = self.P[k*self.m_group: (k + 1)*self.m_group]
            z1 = np.dot(z[idx1], self.f_matrix[:len(idx1), :len(idx1)])
            result += operator.rastrigin_func(z1)
        return result


class F162010(F92010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F16: D/m-group Shifted and m-rotated Ackley’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f16_op", f_matrix="f16_m", m_group=50):
        super().__init__(ndim, bounds, f_shift, f_matrix, m_group)
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-32., 32.] for _ in range(self.dim_default)]))
        self.count_up = int(self.ndim / self.m_group)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        result = 0.0
        for k in range(0, self.count_up):
            idx1 = self.P[k*self.m_group: (k + 1)*self.m_group]
            z1 = np.dot(z[idx1], self.f_matrix[:len(idx1), :len(idx1)])
            result += operator.ackley_func(z1)
        return result


class F172010(F72010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F17: D/m-group Shifted m-dimensional Schwefel’s Problem 1.2"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = True

    def __init__(self, ndim=None, bounds=None, f_shift="f17_op", m_group=50):
        super().__init__(ndim, bounds, f_shift, m_group)
        self.count_up = int(self.ndim / self.m_group)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        result = 0.0
        for k in range(0, self.count_up):
            idx1 = self.P[k*self.m_group: (k + 1)*self.m_group]
            result += operator.ackley_func(z[idx1])
        return result


class F182010(F72010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F18: D/m-group Shifted m-dimensional Rosenbrock’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f18_op", m_group=50):
        super().__init__(ndim, bounds, f_shift, m_group)
        self.count_up = int(self.ndim / self.m_group)
        self.x_global = self.f_shift + 1

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        result = 0.0
        for k in range(0, self.count_up):
            idx1 = self.P[k*self.m_group: (k + 1)*self.m_group]
            result += operator.rosenbrock_func(z[idx1])
        return result


class F192010(F12010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F19: Shifted Schwefel’s Problem 1.2"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    separable = False

    def __init__(self, ndim=None, bounds=None, f_shift="f19_o"):
        super().__init__(ndim, bounds, f_shift)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.schwefel_12_func(x - self.f_shift)


class F202010(F12010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """
    name = "F20: Shifted Rosenbrock’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = 0'

    separable = False
    unimodal = False

    def __init__(self, ndim=None, bounds=None, f_shift="f20_o"):
        super().__init__(ndim, bounds, f_shift)
        self.x_global = self.f_shift + 1

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.rosenbrock_func(x - self.f_shift)
