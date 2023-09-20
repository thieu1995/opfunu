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
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

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
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.discus_func(z) + self.f_bias


class F42014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F4: Shifted and Rotated Rosenbrock’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 400.0'

    unimodal = False

    characteristics = ["Having a very narrow valley from local optimum to global optimum"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_4", f_matrix="M_4_D", f_bias=400.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 2.048*(x - self.f_shift)/100)
        return operator.rosenbrock_func(z, shift=1.0) + self.f_bias


class F52014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F5: Shifted and Rotated Ackley’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 500.0'

    unimodal = False

    characteristics = ["Having a very narrow valley from local optimum to global optimum"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_5", f_matrix="M_5_D", f_bias=500.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.ackley_func(z) + self.f_bias


class F62014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F6: Shifted and Rotated Weierstrass Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 600.0'

    unimodal = False
    convex = False
    modality = True

    characteristics = ["Continuous but differentiable only on a set of points"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_6", f_matrix="M_6_D", f_bias=600.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 0.5*(x - self.f_shift)/100)
        return operator.weierstrass_norm_func(z) + self.f_bias


class F72014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F7: Shifted and Rotated Griewank’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 700.0'

    unimodal = False

    characteristics = ["Continuous but differentiable only on a set of points"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_7", f_matrix="M_7_D", f_bias=700.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 600.*(x - self.f_shift)/100)
        return operator.griewank_func(z) + self.f_bias


class F82014(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F8: Shifted Rastrigin’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 800.0'

    continuous = True
    linear = False
    convex = False
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

    characteristics = ["Local optima’s number is huge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_8", f_bias=800.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [10, 20, 30, 50, 100]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2014")
        self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.rastrigin_func(5.12 * (x - self.f_shift) / 100) + self.f_bias


class F92014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F9: Shifted and Rotated Rastrigin’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 900.0'

    unimodal = False
    separable = False
    parametric = True
    shifted = True
    rotated = True
    modality = True

    characteristics = ["Local optima’s number is huge"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_9", f_matrix="M_9_D", f_bias=900.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5.12*(x - self.f_shift)/100)
        return operator.rastrigin_func(z) + self.f_bias


class F102014(F82014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F10: Shifted Schwefel’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1000.0'

    characteristics = ["Local optima’s number is huge", "The second better local optimum is far from the global optimum"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_10", f_bias=1000.):
        super().__init__(ndim, bounds, f_shift, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        return operator.modified_schwefel_func(1000*(x - self.f_shift)/100) + self.f_bias


class F112014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F11: Shifted and Rotated Schwefel’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1100.0'

    convex = False
    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Local optima’s number is huge", "The second better local optimum is far from the global optimum"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_11", f_matrix="M_11_D", f_bias=1100.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 1000*(x - self.f_shift)/100)
        return operator.modified_schwefel_func(z) + self.f_bias


class F122014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F12: Shifted and Rotated Katsuura Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1200.0'

    convex = False
    unimodal = False
    separable = False
    differentiable = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Continuous everywhere yet differentiable nowhere"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_12", f_matrix="M_12_D", f_bias=1200.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5*(x - self.f_shift)/100)
        return operator.katsuura_func(z) + self.f_bias


class F132014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F13: Shifted and Rotated HappyCat Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1300.0'

    continuous = False
    convex = True
    unimodal = False
    separable = False
    differentiable = False

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_13", f_matrix="M_13_D", f_bias=1300.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5*(x - self.f_shift)/100)
        return operator.happy_cat_func(z, shift=-1.0) + self.f_bias


class F142014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F14: Shifted and Rotated HGBat Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1400.0'

    continuous = True
    convex = True
    unimodal = False
    separable = False
    differentiable = True
    modality = False

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_14", f_matrix="M_14_D", f_bias=1400.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5*(x - self.f_shift)/100)
        return operator.hgbat_func(z, shift=-1.0) + self.f_bias


class F152014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F15: Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1500.0'

    continuous = True
    linear = False
    convex = False
    unimodal = False

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_15", f_matrix="M_15_D", f_bias=1500.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5*(x - self.f_shift)/100)
        return operator.expanded_griewank_rosenbrock_func(z) + self.f_bias


class F162014(F12014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F16: Shifted and Rotated Expanded Scaffer’s F6 Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1600.0'

    continuous = True
    separable = False
    convex = False
    unimodal = False
    modality = True

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_16", f_matrix="M_16_D", f_bias=1600.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.expanded_scaffer_f6_func(z) + self.f_bias


class F172014(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F17: Hybrid Function 1"
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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_17", f_matrix="M_17_D", f_shuffle="shuffle_data_17_D", f_bias=1700.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [10, 20, 30, 50, 100]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2014")
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


class F182014(F172014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F18: Hybrid Function 2"
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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_18", f_matrix="M_18_D", f_shuffle="shuffle_data_18_D", f_bias=1800.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_shuffle, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3]))
        mz = np.dot(self.f_matrix, z1)
        return (operator.bent_cigar_func(mz[:self.n1]) +
                operator.hgbat_func(mz[self.n1:self.n2], shift=-1.0) +
                operator.rastrigin_func(mz[self.n2:]) + self.f_bias)


class F192014(F172014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F19: Hybrid Function 3"
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

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Different properties for different variables subcomponents"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_19", f_matrix="M_19_D", f_shuffle="shuffle_data_19_D", f_bias=1900.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_shuffle, f_bias)
        self.n_funcs = 4
        self.p = np.array([0.2, 0.2, 0.3, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.idx1, self.idx2 = self.f_shuffle[:self.n1], self.f_shuffle[self.n1:self.n2]
        self.idx3, self.idx4 = self.f_shuffle[self.n2:self.n3], self.f_shuffle[self.n3:self.ndim]

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3], z[self.idx4]))
        mz = np.dot(self.f_matrix, z1)
        return (operator.griewank_func(mz[:self.n1]) +
                operator.weierstrass_func(mz[self.n1:self.n2]) +
                operator.rosenbrock_func(mz[self.n2:self.n3], shift=1.0) +
                operator.expanded_scaffer_f6_func(mz[self.n3:]) + self.f_bias)


class F202014(F192014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F20: Hybrid Function 4"
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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_20", f_matrix="M_20_D", f_shuffle="shuffle_data_20_D", f_bias=2000.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_shuffle, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3], z[self.idx4]))
        mz = np.dot(self.f_matrix, z1)
        return (operator.hgbat_func(mz[:self.n1], shift=-1.0) +
                operator.discus_func(mz[self.n1:self.n2]) +
                operator.expanded_griewank_rosenbrock_func(mz[self.n2:self.n3]) +
                operator.rastrigin_func(mz[self.n3:]) + self.f_bias)

class F212014(F172014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F21: Hybrid Function 5"
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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_21", f_matrix="M_21_D", f_shuffle="shuffle_data_21_D", f_bias=2100.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_shuffle, f_bias)
        self.n_funcs = 5
        self.p = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.idx1, self.idx2, self.idx3 = self.f_shuffle[:self.n1], self.f_shuffle[self.n1:self.n2], self.f_shuffle[self.n2:self.n3]
        self.idx4, self.idx5 = self.f_shuffle[self.n3:self.n4], self.f_shuffle[self.n4:self.ndim]


    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3], z[self.idx4], z[self.idx5]))
        mz = np.dot(self.f_matrix, z1)
        return (operator.expanded_scaffer_f6_func(mz[:self.n1]) +
                operator.hgbat_func(mz[self.n1:self.n2], shift=-1.0) +
                operator.rosenbrock_func(mz[self.n2:self.n3], shift=1.0) +
                operator.modified_schwefel_func(mz[self.n3:self.n4]) +
                operator.elliptic_func(mz[self.n4:]) + self.f_bias)


class F222014(F212014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F22: Hybrid Function 6"
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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_22", f_matrix="M_22_D", f_shuffle="shuffle_data_22_D", f_bias=2200.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_shuffle, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = x - self.f_shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3], z[self.idx4], z[self.idx5]))
        mz = np.dot(self.f_matrix, z1)
        return (operator.katsuura_func(mz[:self.n1]) +
                operator.happy_cat_func(mz[self.n1:self.n2], shift=-1.0) +
                operator.expanded_griewank_rosenbrock_func(mz[self.n2:self.n3]) +
                operator.modified_schwefel_func(mz[self.n3:self.n4]) +
                operator.ackley_func(mz[self.n4:]) + self.f_bias)

class F232014(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F23: Composition Function 1"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2300.0'

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
    rotated = True

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_23", f_matrix="M_23_D", f_bias=2300.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [10, 20, 30, 50, 100]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2014")
        self.f_shift = self.check_shift_matrix(f_shift)[:, :self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, :self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [10, 20, 30, 40, 50]
        self.lamdas = [1., 1e-6, 1e-26, 1e-6, 1e-6]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = F42014(self.ndim, None, self.f_shift[0], self.f_matrix[:self.ndim,:], f_bias=0)
        self.g1 = F12014(self.ndim, None, self.f_shift[1], self.f_matrix[self.ndim:2*self.ndim,:], f_bias=0)
        self.g2 = F22014(self.ndim, None, self.f_shift[2], self.f_matrix[2*self.ndim:3*self.ndim,:], f_bias=0)
        self.g3 = F32014(self.ndim, None, self.f_shift[3], self.f_matrix[3*self.ndim:4*self.ndim,:], f_bias=0)
        self.g4 = F12014(self.ndim, None, self.f_shift[4], self.f_matrix[4*self.ndim:5*self.ndim, :],f_bias=0)
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rotated Rosenbrock’s Function F4’
        g0 = self.lamdas[0] * self.g0.evaluate(x) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. High Conditioned Elliptic Function F1’
        g1 = self.lamdas[1] * operator.elliptic_func(x) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Rotated Bent Cigar Function F2’
        g2 = self.lamdas[2] * self.g2.evaluate(x) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        # 4. Rotated Discus Function F3’
        g3 = self.lamdas[3] * self.g3.evaluate(x) + self.bias[3]
        w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

        # 5. High Conditioned Elliptic Function F1’
        g4 = self.lamdas[4] * operator.elliptic_func(x) + self.bias[4]
        w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

        ws = np.array([w0, w1, w2, w3, w4])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3, g4])
        return np.dot(ws, gs) + self.f_bias


class F242014(F232014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F24: Composition Function 2"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2400.0'

    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks

    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_24", f_matrix="M_24_D", f_bias=2400.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)
        self.n_funcs = 3
        self.xichmas = [20, 20, 20]
        self.lamdas = [1., 1., 1.]
        self.bias = [0, 100, 200]
        self.g0 = F102014(self.ndim, None, self.f_shift[0], f_bias=0)
        self.g1 = F92014(self.ndim, None, self.f_shift[1], f_bias=0)
        self.g2 = F142014(self.ndim, None, self.f_shift[2], f_bias=0)
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Schwefel's Function F10’
        g0 = self.lamdas[0] * self.g0.evaluate(x) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Rotated Rastrigin’s Function F9'
        g1 = self.lamdas[1] * self.g1.evaluate(x) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Rotated HGBat Function F14'
        g2 = self.lamdas[2] * self.g2.evaluate(x) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        ws = np.array([w0, w1, w2])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2])
        return np.dot(ws, gs) + self.f_bias


class F252014(F232014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F25: Composition Function 3"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2500.0'

    convex = False
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_25", f_matrix="M_25_D", f_bias=2500.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)
        self.n_funcs = 3
        self.xichmas = [10, 30, 50]
        self.lamdas = [0.25, 1., 1e-7]
        self.bias = [0, 100, 200]
        self.g0 = F112014(self.ndim, None, self.f_shift[0], f_bias=0)
        self.g1 = F92014(self.ndim, None, self.f_shift[1], f_bias=0)
        self.g2 = F12014(self.ndim, None, self.f_shift[2], f_bias=0)
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rotated Schwefel's Function F11'
        g0 = self.lamdas[0] * self.g0.evaluate(x) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Rotated Rastrigin’s Function F9’
        g1 = self.lamdas[1] * self.g1.evaluate(x) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Rotated High Conditioned Elliptic Function F1’
        g2 = self.lamdas[2] * self.g2.evaluate(x) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        ws = np.array([w0, w1, w2])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2])
        return np.dot(ws, gs) + self.f_bias


class F262014(F232014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F26: Composition Function 4"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2600.0'

    convex = False
    modality = True
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_26", f_matrix="M_26_D", f_bias=2600.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)
        self.n_funcs = 5
        self.xichmas = [10, 10, 10, 10, 10]
        self.lamdas = [0.25, 1, 1e-7, 2.5, 10]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = F112014(self.ndim, None, self.f_shift[0], f_bias=0)
        self.g1 = F132014(self.ndim, None, self.f_shift[1], f_bias=0)
        self.g2 = F12014(self.ndim, None, self.f_shift[2], f_bias=0)
        self.g3 = F62014(self.ndim, None, self.f_shift[3], f_bias=0)
        self.g4 = F72014(self.ndim, None, self.f_shift[4], f_bias=0)
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rotated Schwefel's Function F11’
        g0 = self.lamdas[0] * self.g0.evaluate(x) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Rotated HappyCat Function F13’
        g1 = self.lamdas[1] * self.g1.evaluate(x) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Rotated High Conditioned Elliptic Function F1’
        g2 = self.lamdas[2] * self.g2.evaluate(x) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        # 4. Rotated Weierstrass Function F6’
        g3 = self.lamdas[3] * self.g3.evaluate(x) + self.bias[3]
        w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

        # 5. Rotated Griewank’s Function F7’
        g4 = self.lamdas[4] * self.g4.evaluate(x) + self.bias[4]
        w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

        ws = np.array([w0, w1, w2, w3, w4])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3, g4])
        return np.dot(ws, gs) + self.f_bias


class F272014(F232014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F27: Composition Function 5"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2700.0'

    convex = False
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_27", f_matrix="M_27_D", f_bias=2700.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)
        self.n_funcs = 5
        self.xichmas = [10, 10, 10, 20, 20]
        self.lamdas = [10, 10, 2.5, 25, 1e-6]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = F142014(self.ndim, None, self.f_shift[0], f_bias=0)
        self.g1 = F92014(self.ndim, None, self.f_shift[1], f_bias=0)
        self.g2 = F112014(self.ndim, None, self.f_shift[2], f_bias=0)
        self.g3 = F62014(self.ndim, None, self.f_shift[3], f_bias=0)
        self.g4 = F12014(self.ndim, None, self.f_shift[4], f_bias=0)
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rotated HGBat Function F14’
        g0 = self.lamdas[0] * self.g0.evaluate(x) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Rotated Rastrigin’s Function F9’
        g1 = self.lamdas[1] * self.g1.evaluate(x) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Rotated Schwefel's Function F11’
        g2 = self.lamdas[2] * self.g2.evaluate(x) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        # 4. Rotated Weierstrass Function F6’
        g3 = self.lamdas[3] * self.g3.evaluate(x) + self.bias[3]
        w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

        # 5. Rotated High Conditioned Elliptic Function F1’
        g4 = self.lamdas[4] * self.g4.evaluate(x) + self.bias[4]
        w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

        ws = np.array([w0, w1, w2, w3, w4])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3, g4])
        return np.dot(ws, gs) + self.f_bias


class F282014(F232014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F28: Composition Function 6"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2800.0'

    convex = False
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_28", f_matrix="M_28_D", f_bias=2800.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)
        self.n_funcs = 5
        self.xichmas = [10, 20, 30, 40, 50]
        self.lamdas = [2.5, 10, 2.5, 5e-4, 1e-6]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = F152014(self.ndim, None, self.f_shift[0], self.f_matrix[:self.ndim,:], f_bias=0)
        self.g1 = F132014(self.ndim, None, self.f_shift[1], self.f_matrix[self.ndim:2*self.ndim,:], f_bias=0)
        self.g2 = F112014(self.ndim, None, self.f_shift[2], self.f_matrix[2*self.ndim:3*self.ndim,:], f_bias=0)
        self.g3 = F162014(self.ndim, None, self.f_shift[3], self.f_matrix[3*self.ndim:4*self.ndim,:], f_bias=0)
        self.g4 = F12014(self.ndim, None, self.f_shift[4], self.f_matrix[4*self.ndim:5*self.ndim,:], f_bias=0)
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rotated Expanded Griewank’s plus Rosenbrock’s Function F15’
        g0 = self.lamdas[0] * self.g0.evaluate(x) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Rotated HappyCat Function F13’
        g1 = self.lamdas[1] * self.g1.evaluate(x) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Rotated Schwefel's Function F11’
        g2 = self.lamdas[2] * self.g2.evaluate(x) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        # 4. Rotated Expanded Scaffer’s F6 Function F16'
        g3 = self.lamdas[3] * self.g3.evaluate(x) + self.bias[3]
        w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

        # 5. Rotated High Conditioned Elliptic Function F1’
        g4 = self.lamdas[4] * self.g4.evaluate(x) + self.bias[4]
        w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

        ws = np.array([w0, w1, w2, w3, w4])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3, g4])
        return np.dot(ws, gs) + self.f_bias


class F292014(F232014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F29: Composition Function 7"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2900.0'

    convex = False
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_29", f_matrix="M_29_D", f_shuffle="shuffle_data_29_D", f_bias=2900.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)
        self.n_funcs = 3
        self.xichmas = [10, 30, 50]
        self.lamdas = [1, 1, 1]
        self.bias = [0, 100, 200]
        self.x_global = self.f_shift[0]
        self.g0 = F172014(self.ndim, None, self.f_shift[0], f_shuffle=f_shuffle, f_bias=0)
        self.g1 = F182014(self.ndim, None, self.f_shift[1], f_shuffle=f_shuffle,f_bias=0)
        self.g2 = F192014(self.ndim, None, self.f_shift[2], f_shuffle=f_shuffle, f_bias=0)
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Hybrid Function 1 F17’
        g0 = self.lamdas[0] * self.g0.evaluate(x) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Hybrid Function 2 F18’
        g1 = self.lamdas[1] * self.g1.evaluate(x) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Hybrid Function 3 F19’
        g2 = self.lamdas[2] * self.g2.evaluate(x) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        ws = np.array([w0, w1, w2])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2])
        return np.dot(ws, gs) + self.f_bias


class F302014(F232014):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F30: Composition Function 8"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 3000.0'

    convex = False
    characteristics = ["Asymmetrical", "Different properties around different local optima",
                       "Different properties for different variables subcomponents"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_30", f_matrix="M_30_D", f_shuffle="shuffle_data_30_D", f_bias=3000.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)
        self.n_funcs = 3
        self.xichmas = [10, 30, 50]
        self.lamdas = [1, 1, 1]
        self.bias = [0, 100, 200]
        self.x_global = self.f_shift[0]
        self.g0 = F202014(self.ndim, None, self.f_shift[0], self.f_matrix[:self.ndim, :], f_shuffle=f_shuffle, f_bias=0)
        self.g1 = F212014(self.ndim, None, self.f_shift[1], self.f_matrix[self.ndim:2*self.ndim, :], f_shuffle=f_shuffle,f_bias=0)
        self.g2 = F222014(self.ndim, None, self.f_shift[2], self.f_matrix[2*self.ndim:3*self.ndim, :], f_shuffle=f_shuffle, f_bias=0)
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Hybrid Function 1 F17’
        g0 = self.lamdas[0] * self.g0.evaluate(x) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Hybrid Function 2 F18’
        g1 = self.lamdas[1] * self.g1.evaluate(x) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Hybrid Function 3 F19’
        g2 = self.lamdas[2] * self.g2.evaluate(x) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        ws = np.array([w0, w1, w2])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2])
        return np.dot(ws, gs) + self.f_bias
