#!/usr/bin/env python
# Created by "Thieu" at 09:25, 13/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec import CecBenchmark
from opfunu.utils import operator


class F12021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F1: Shifted and Rotated Bent Cigar Function (F1 CEC-2017)"
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

    characteristics = ["Smooth but narrow ridge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_1", f_matrix="M_1_D", f_bias=100.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [2, 10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2021")
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


class F22021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F2: Shifted and Rotated Schwefel’s Function (F11 CEC-2014)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1100.0'

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

    characteristics = ["Local optima’s number is huge", "The penultimate local optimum is far from the global optimum."]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_2", f_matrix="M_2_D", f_bias=1100.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [2, 10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 1000.*(x - self.f_shift)/100)
        return operator.bent_cigar_func(z) + self.f_bias


class F32021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F3: Shifted and Rotated Lunacek bi-Rastrigin Function (F7 CEC-2017)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 700.0'

    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Asymmetrical", "Continuous everywhere yet differentiable nowhere"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_3", f_matrix="M_3_D", f_bias=700.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [2, 10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 600.*(x - self.f_shift)/100)
        return operator.bent_cigar_func(z) + self.f_bias


class F42021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F4: Expanded Rosenbrock’s plus Griewangk’s Function (F15 CEC-2014)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1900.0'

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

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Optimal point locates in flat area"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_4", f_matrix="M_4_D", f_bias=1900.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [2, 10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5. * (x - self.f_shift) / 100) + 1
        return operator.expanded_griewank_rosenbrock_func(z) + self.f_bias


class F52021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F17: Hybrid Function 1 (F17 CEC-2014)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1700.0'

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

    characteristics = ["Different properties for different variables subcomponents"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_5", f_matrix="M_5_D", f_shuffle="shuffle_data_5_D", f_bias=1700.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.n_funcs = 3
        self.p = np.array([0.3, 0.3, 0.4])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.idx1, self.idx2, self.idx3 = self.f_shuffle[:self.n1], self.f_shuffle[self.n1:self.n2], self.f_shuffle[self.n2:self.ndim]
        self.g1 = operator.modified_schwefel_func
        self.g2 = operator.rastrigin_func
        self.g3 = operator.elliptic_func
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix, "f_shuffle": self.f_shuffle}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3]))
        mz = np.dot(self.f_matrix, z1)
        return self.g1(mz[:self.n1]) + self.g2(mz[self.n1:self.n2]) + self.g3(mz[self.n2:]) + self.f_bias


class F62021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F16: Hybrid Function 2 (F15 CEC-2017)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1600.0'

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

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_6", f_matrix="M_6_D", f_shuffle="shuffle_data_6_D", f_bias=1600.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.n_funcs = 4
        self.p = np.array([0.2, 0.2, 0.3, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.idx1, self.idx2 = self.f_shuffle[:self.n1], self.f_shuffle[self.n1:self.n2]
        self.idx3, self.idx4 = self.f_shuffle[self.n2:self.n3], self.f_shuffle[self.n3:self.ndim]
        self.g1 = operator.expanded_scaffer_f6_func
        self.g2 = operator.hgbat_func
        self.g3 = operator.rosenbrock_func
        self.g4 = operator.modified_schwefel_func
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix, "f_shuffle": self.f_shuffle}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        mz = np.dot(self.f_matrix, x - self.f_shift)
        return self.g1(mz[self.idx1]) + self.g2(mz[self.idx2]) + self.g3(mz[self.idx3]) + self.g4(mz[self.idx4]) + self.f_bias


class F72021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F7: Hybrid Function 3 (F21 CEC-2014)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2100.0'

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

    characteristics = ["Different properties for different variables subcomponents"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_7", f_matrix="M_7_D", f_shuffle="shuffle_data_7_D", f_bias=2100.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.n_funcs = 5
        self.p = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.idx1, self.idx2, self.idx3 = self.f_shuffle[:self.n1], self.f_shuffle[self.n1:self.n2], self.f_shuffle[self.n2:self.n3]
        self.idx4, self.idx5 = self.f_shuffle[self.n3:self.n4], self.f_shuffle[self.n4:self.ndim]
        self.g1 = operator.expanded_scaffer_f6_func
        self.g2 = operator.hgbat_func
        self.g3 = operator.rosenbrock_func
        self.g4 = operator.modified_schwefel_func
        self.g5 = operator.elliptic_func
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix, "f_shuffle": self.f_shuffle}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3], z[self.idx4], z[self.idx5]))
        mz = np.dot(self.f_matrix, z1)
        return self.g1(mz[:self.n1]) + self.g2(mz[self.n1:self.n2]) + self.g3(mz[self.n2:self.n3]) +\
               self.g4(mz[self.n3:self.n4]) + self.g5(mz[self.n4:]) + self.f_bias


class F82021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F8: Composition Function 1 (F21 CEC-2017)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2200.0'

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
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_8", f_matrix="M_8_D", f_bias=2200.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [2, 10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_matrix(f_shift)[:, :self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, :self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 3
        self.xichmas = [10, 20, 30]
        self.lamdas = [1., 10., 1.]
        self.bias = [0, 100, 200]
        self.g0 = operator.rastrigin_func
        self.g1 = operator.griewank_func
        self.g2 = operator.modified_schwefel_func
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rastrigin’s Function F5’
        z0 = np.dot(self.f_matrix[:self.ndim, :], x - self.f_shift[0])
        g0 = self.lamdas[0] * self.g0(z0) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Griewank’s Function F15’
        z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], x - self.f_shift[1])
        g1 = self.lamdas[1] * self.g1(z1) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Modifed Schwefel's Function F10’
        # z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], x - self.f_shift[2])
        z2 = 1000*(x - self.f_shift[2])/100
        g2 = self.lamdas[2] * self.g2(z2) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        ws = np.array([w0, w1, w2])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2])
        return np.dot(ws, gs) + self.f_bias


