#!/usr/bin/env python
# Created by "Thieu" at 09:53, 13/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec import CecBenchmark
from opfunu.utils import operator


class F12022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F1: Shifted and full Rotated Zakharov Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 300.0'

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
    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_1", f_matrix="M_1_D", f_bias=300.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [2, 10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2022")
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
        return operator.zakharov_func(z) + self.f_bias


class F22022(F12022):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F2: Shifted and Rotated Rosenbrock’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 400.0'

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Local optima’s number is huge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_2", f_matrix="M_2_D", f_bias=400.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 2.048*(x - self.f_shift)/100) + 1
        return operator.rosenbrock_func(z) + self.f_bias


class F32022(F12022):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F3: Shifted and full Rotated Expanded Schaffer’s F7"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 600.0'

    unimodal = False
    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Asymmetrical", "Local optima’s number is huge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_3", f_matrix="M_3_D", f_bias=600.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 0.5*(x - self.f_shift)/100)
        return operator.rotated_expanded_schaffer_func(z) + self.f_bias


class F42022(F12022):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F4: Shifted and Rotated Non-Continuous Rastrigin’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 800.0'

    unimodal = False
    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Asymmetrical", "Local optima’s number is huge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_4", f_matrix="M_4_D", f_bias=800.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5.12*(x - self.f_shift)/100)
        return operator.non_continuous_rastrigin_func(z) + self.f_bias


class F52022(F12022):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F5: Shifted and Rotated Levy Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 900.0'

    unimodal = False
    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Local optima’s number is huge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_5", f_matrix="M_5_D", f_bias=900.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5.12*(x - self.f_shift)/100)
        return operator.levy_func(z, shift=1.0) + self.f_bias


class F62022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F6: Hybrid Function 1"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1800.0'

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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_6", f_matrix="M_6_D", f_shuffle="shuffle_data_6_D", f_bias=1800.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.n_funcs = 3
        self.p = np.array([0.4, 0.4, 0.2])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.idx1, self.idx2, self.idx3 = self.f_shuffle[:self.n1], self.f_shuffle[self.n1:self.n2], self.f_shuffle[self.n2:self.ndim]
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix, "f_shuffle": self.f_shuffle}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3]))
        mz = np.dot(self.f_matrix, z1)
        return (operator.bent_cigar_func(mz[:self.n1]) +
                operator.hgbat_func(mz[self.n1:self.n2], shift=-1.0) +
                operator.rastrigin_func(mz[self.n2:]) + self.f_bias)


class F72022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F7: Hybrid Function 2"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2000.0'

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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_7", f_matrix="M_7_D", f_shuffle="shuffle_data_7_D", f_bias=2000.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.n_funcs = 6
        self.p = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.2])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.n5 = int(np.ceil(self.p[4] * self.ndim)) + self.n4
        self.idx1, self.idx2, self.idx3 = self.f_shuffle[:self.n1], self.f_shuffle[self.n1:self.n2], self.f_shuffle[self.n2:self.n3]
        self.idx4, self.idx5, self.idx6 = self.f_shuffle[self.n3:self.n4], self.f_shuffle[self.n4:self.n5], self.f_shuffle[self.n5:self.ndim]
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix, "f_shuffle": self.f_shuffle}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3], z[self.idx4], z[self.idx5], z[self.idx6]))
        mz = np.dot(self.f_matrix, z1)
        return (operator.hgbat_func(mz[:self.n1], shift=-1.0) +
                operator.katsuura_func(mz[self.n1:self.n2]) +
                operator.ackley_func(mz[self.n2:self.n3]) +
                operator.rastrigin_func(mz[self.n3:self.n4]) +
                operator.modified_schwefel_func(mz[self.n4:self.n5]) +
                operator.schaffer_f7_func(mz[self.n5:self.ndim]) + self.f_bias)


class F82022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F8: Hybrid Function 3"
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
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Different properties for different variables subcomponents"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_8", f_matrix="M_8_D", f_shuffle="shuffle_data_8_D", f_bias=2200.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.n_funcs = 6
        self.p = np.array([0.3, 0.2, 0.2, 0.1, 0.2])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.idx1, self.idx2, self.idx3 = self.f_shuffle[:self.n1], self.f_shuffle[self.n1:self.n2], self.f_shuffle[self.n2:self.n3]
        self.idx4, self.idx5 = self.f_shuffle[self.n3:self.n4], self.f_shuffle[self.n4:self.ndim]
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix, "f_shuffle": self.f_shuffle}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3], z[self.idx4], z[self.idx5]))
        mz = np.dot(self.f_matrix, z1)
        return (operator.katsuura_func(mz[:self.n1]) +
                operator.happy_cat_func(mz[self.n1:self.n2], shift=-1.0) +
                operator.grie_rosen_cec_func(mz[self.n2:self.n3]) +
                operator.modified_schwefel_func(mz[self.n3:self.n4]) +
                operator.ackley_func(mz[self.n4:self.ndim]) + self.f_bias)


class F92022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F9: Composition Function 1"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2300.0'

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
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_9", f_matrix="M_9_D", f_bias=2300.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [2, 10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_matrix(f_shift)[:, :self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, :self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [10, 20, 30, 40, 50] # aka delta in original CEC2022 logic
        self.lamdas = [1, 1e-6, 1e-6, 1e-6, 1e-6]
        self.bias = [0, 200, 300, 100, 400]
        self.g0 = operator.rosenbrock_func
        self.g1 = operator.elliptic_func
        self.g2 = operator.bent_cigar_func
        self.g3 = operator.discus_func
        self.g4 = operator.elliptic_func
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rotated Rosenbrock’s Function f2
        z0 = np.dot(self.f_matrix[:self.ndim, :], 2.048*(x - self.f_shift[0])/100) + 1
        g0 = self.lamdas[0] * self.g0(z0) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. High Conditioned Elliptic Function f8
        z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], x - self.f_shift[0])
        g1 = self.lamdas[1] * self.g1(z1) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Rotated Bent Cigar Function f6
        z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], x - self.f_shift[0])
        g2 = self.lamdas[2] * self.g2(z2) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        # 4. Rotated Discus Function f14
        z3 = np.dot(self.f_matrix[3 * self.ndim:4 * self.ndim, :], x - self.f_shift[0])
        g3 = self.lamdas[3] * self.g3(z3) + self.bias[3]
        w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

        # 5. High Conditioned Elliptic Function f8
        z4 = np.dot(self.f_matrix[4 * self.ndim:5 * self.ndim, :], x - self.f_shift[0])
        g4 = self.lamdas[4] * self.g4(z4) + self.bias[4]
        w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

        ws = np.array([w0, w1, w2, w3, w4])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3, g4])
        return np.dot(ws, gs) + self.f_bias


class F102022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F10: Composition Function 2"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2400.0'

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
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_10", f_matrix="M_10_D", f_bias=2400.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [2, 10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_matrix(f_shift)[:, :self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, :self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 3
        self.xichmas = [20, 10, 10]
        self.lamdas = [1, 1, 1]
        self.bias = [0, 200, 100]
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rotated Schwefel's Function f12
        z0 = np.dot(self.f_matrix[:self.ndim, :], (1000./100)*(x - self.f_shift[0]))
        g0 = self.lamdas[0] * operator.modified_schwefel_func(z0) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Rotated Rastrigin’s Function f4
        z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], (5.12/100)*(x - self.f_shift[0]))
        g1 = self.lamdas[1] * operator.rastrigin_func(z1) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. HGBat Function f7
        z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], (5/100)*(x - self.f_shift[0]))
        g2 = self.lamdas[2] * operator.hgbat_func(z2, shift=-1.0) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        ws = np.array([w0, w1, w2])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2])
        return np.dot(ws, gs) + self.f_bias


class F112022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F11: Composition Function 3"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2600.0'

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
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_11", f_matrix="M_11_D", f_bias=2600.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [2, 10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_matrix(f_shift)[:, :self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, :self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [20, 20, 30, 30, 20]
        self.lamdas = [1e-26, 10, 1e-6, 10, 5e-4]
        self.bias = [0, 200, 300, 400, 200]
        self.g0 = operator.rotated_expanded_schaffer_func
        self.g1 = operator.modified_schwefel_func
        self.g2 = operator.griewank_func
        self.g3 = operator.rosenbrock_func
        self.g4 = operator.rastrigin_func
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Expanded Schaffer’s F6 Function f3
        z0 = np.dot(self.f_matrix[:self.ndim, :], 0.5*(x - self.f_shift[0])/100)
        g0 = self.lamdas[0] * self.g0(z0) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Modified Schwefel's Function f12
        z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], 1000.*(x - self.f_shift[0])/100)
        g1 = self.lamdas[1] * self.g1(z1) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Griewank’s Function f15
        z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], 600.*(x - self.f_shift[0])/100)
        g2 = self.lamdas[2] * self.g2(z2) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        # 4. Rosenbrock’s Function f2
        z3 = np.dot(self.f_matrix[3 * self.ndim:4 * self.ndim, :], 2.048*(x - self.f_shift[0])/100)
        g3 = self.lamdas[3] * self.g3(z3) + self.bias[3]
        w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

        # 5. Rastrigin’s Function f4
        z4 = np.dot(self.f_matrix[4 * self.ndim:5 * self.ndim, :], x - self.f_shift[0])
        g4 = self.lamdas[4] * self.g4(z4) + self.bias[4]
        w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

        ws = np.array([w0, w1, w2, w3, w4])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3, g4])
        return np.dot(ws, gs) + self.f_bias


class F122022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """
    name = "F12: Composition Function 4"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2700.0'

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
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_12", f_matrix="M_12_D", f_bias=2700.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 20
        self.dim_supported = [2, 10, 20]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_matrix(f_shift)[:, :self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, :self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 6
        self.xichmas = [10, 20, 30, 40, 50, 60]
        self.lamdas = [10, 10, 2.5, 1e-26, 1e-6, 5e-4]
        self.bias = [0, 300, 500, 100, 400, 200]
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. HGBat Function f7
        z0 = np.dot(self.f_matrix[:self.ndim, :], 5.*(x - self.f_shift[0])/100)
        g0 = self.lamdas[0] * operator.hgbat_func(z0, shift=-1.0) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Rastrigin’s Function f4
        z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], 5.12*(x - self.f_shift[0])/100)
        g1 = self.lamdas[1] * operator.rastrigin_func(z1) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Modified Schwefel's Function f12
        z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], 1000.*(x - self.f_shift[0])/100)
        g2 = self.lamdas[2] * operator.modified_schwefel_func(z2) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        # 4. Bent Cigar Function f6
        z3 = np.dot(self.f_matrix[3 * self.ndim:4 * self.ndim, :], x - self.f_shift[0])
        g3 = self.lamdas[3] * operator.bent_cigar_func(z3) + self.bias[3]
        w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

        # 5.  High Conditioned Elliptic Function f8
        z4 = np.dot(self.f_matrix[4 * self.ndim:5 * self.ndim, :], x - self.f_shift[0])
        g4 = self.lamdas[4] * operator.elliptic_func(z4) + self.bias[4]
        w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

        # 6.  Expanded Schaffer’s F6 Function f3
        z5 = np.dot(self.f_matrix[5 * self.ndim:6 * self.ndim, :], x - self.f_shift[0])
        g5 = self.lamdas[5] * operator.rotated_expanded_schaffer_func(z5) + self.bias[5]
        w5 = operator.calculate_weight(x - self.f_shift[5], self.xichmas[5])

        ws = np.array([w0, w1, w2, w3, w4, w5])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3, g4, g5])
        return np.dot(ws, gs) + self.f_bias
