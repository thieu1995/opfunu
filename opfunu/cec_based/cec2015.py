#!/usr/bin/env python
# Created by "Thieu" at 14:45, 07/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec import CecBenchmark
from opfunu.utils import operator


class F12015(CecBenchmark):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """
    name = "F1: Rotated Bent Cigar Function"
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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_1_D", f_matrix="M_1_D", f_bias=100.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 30
        self.dim_supported = [10, 30]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2015")
        self.f_shift = self.check_matrix_data(f_shift, needed_dim=True).ravel()
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


class F22015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """
    name = "F2: Rotated Discus Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 200.0'

    characteristics = ["With one sensitive direction"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_2_D", f_matrix="M_2_D", f_bias=200.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.discus_func(z) + self.f_bias


class F32015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """
    name = "F3: Shifted and Rotated Weierstrass Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 300.0'

    convex = False
    unimodal = False

    characteristics = ["Continuous but differentiable only on a set of points"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_3_D", f_matrix="M_3_D", f_bias=300.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 0.5*(x - self.f_shift)/100)
        return operator.weierstrass_norm_func(z) + self.f_bias


class F42015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """
    name = "F4: Shifted and Rotated Schwefel’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 400.0'

    convex = False
    unimodal = False
    modality = True

    characteristics = ["Local optima’s number is huge",  "The second better local optimum is far from the global optimum"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_4_D", f_matrix="M_4_D", f_bias=400.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 1000*(x - self.f_shift)/100)
        return operator.modified_schwefel_func(z) + self.f_bias


class F52015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """
    name = "F5: Shifted and Rotated Katsuura Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 500.0'

    convex = False
    unimodal = False
    differentiable = False
    modality = True

    characteristics = ["Continuous everywhere yet differentiable nowhere"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_5_D", f_matrix="M_5_D", f_bias=500.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5*(x - self.f_shift)/100)
        return operator.katsuura_func(z) + self.f_bias


class F62015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """
    name = "F6: Shifted and Rotated HappyCat Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 600.0'

    convex = False
    unimodal = False
    separable = False
    differentiable = False

    characteristics = ["Continuous everywhere yet differentiable nowhere"]

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_6_D", f_matrix="M_6_D", f_bias=600.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5*(x - self.f_shift)/100)
        return operator.happy_cat_func(z, shift=-1.0) + self.f_bias

class F72015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """
    name = "F7: Shifted and Rotated HGBat Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 700.0'

    unimodal = False
    separable = False
    differentiable = True

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_7_D", f_matrix="M_7_D", f_bias=700.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5*(x - self.f_shift)/100)
        return operator.hgbat_func(z, shift=-1.0) + self.f_bias


class F82015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """
    name = "F8: Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 800.0'

    unimodal = False
    separable = False
    differentiable = True

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_8_D", f_matrix="M_8_D", f_bias=800.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, 5.*(x - self.f_shift)/100)
        return operator.expanded_griewank_rosenbrock_func(z) + self.f_bias


class F92015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """
    name = "F9: Shifted and Rotated Expanded Scaffer’s F6 Function"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 900.0'

    unimodal = False
    convex = False
    separable = False
    differentiable = True
    modality = True
    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_9_D", f_matrix="M_9_D", f_bias=900.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        z = np.dot(self.f_matrix, x - self.f_shift)
        return operator.expanded_scaffer_f6_func(z) + self.f_bias


class F102015(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F10: Hybrid Function 1 (N=3)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1000.0'

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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_10_D", f_matrix="M_10_D", f_shuffle="shuffle_data_10_D", f_bias=1000.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 30
        self.dim_supported = [10, 30]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2015")
        self.f_shift = self.check_matrix_data(f_shift, needed_dim=True).ravel()
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
        mz = np.dot(self.f_matrix, x - self.f_shift)
        return self.g1(mz[self.idx1]) + self.g2(mz[self.idx2]) + self.g3(mz[self.idx3]) + self.f_bias


class F112015(F102015):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F11: Hybrid Function 2 (N=4)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1100.0'

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_11_D", f_matrix="M_11_D", f_shuffle="shuffle_data_11_D", f_bias=1100.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_shuffle, f_bias)
        self.n_funcs = 4
        self.p = np.array([0.2, 0.2, 0.3, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.idx1, self.idx2 = self.f_shuffle[:self.n1], self.f_shuffle[self.n1:self.n2]
        self.idx3, self.idx4 = self.f_shuffle[self.n2:self.n3], self.f_shuffle[self.n3:self.ndim]
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix, "f_shuffle": self.f_shuffle}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        mz = np.dot(self.f_matrix, x - self.f_shift)
        return (operator.griewank_func(mz[self.idx1]) +
                operator.weierstrass_func(mz[self.idx2]) +
                operator.rosenbrock_func(mz[self.idx3], shift=1.0) +
                operator.expanded_scaffer_f6_func(mz[self.idx4]) + self.f_bias)


class F122015(F102015):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F12: Hybrid Function 3 (N=5)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1200.0'

    characteristics = []

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_11_D", f_matrix="M_11_D", f_shuffle="shuffle_data_11_D", f_bias=1200.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_shuffle, f_bias)
        self.n_funcs = 5
        self.p = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.idx1, self.idx2 = self.f_shuffle[:self.n1], self.f_shuffle[self.n1:self.n2]
        self.idx3, self.idx4, self.idx5 = self.f_shuffle[self.n2:self.n3], self.f_shuffle[self.n3:self.n4], self.f_shuffle[self.n4:self.ndim]
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix, "f_shuffle": self.f_shuffle}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)
        mz = np.dot(self.f_matrix, x - self.f_shift)
        return (operator.katsuura_func(mz[self.idx1]) +
                operator.happy_cat_func(mz[self.idx2], shift=-1.0) +
                operator.expanded_griewank_rosenbrock_func(mz[self.idx3]) +
                operator.modified_schwefel_func(mz[self.idx4]) +
                operator.ackley_func(mz[self.idx5]) + self.f_bias)


class F132015(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F13: Composition Function 1 (N=5)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1300.0'

    continuous = False
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

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_13_D", f_matrix="M_13_D", f_bias=1300.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 30
        self.dim_supported = [10, 30]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2015")
        self.f_shift = self.check_matrix_data(f_shift, needed_dim=True).ravel().reshape((5, -1))
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [10, 20, 30, 40, 50]
        self.lamdas = [1., 1e-6, 1e-26, 1e-6, 1e-6]
        self.bias = [0, 100, 200, 300, 400]
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rotated Rosenbrock’s Function f10
        z0 = np.dot(self.f_matrix[:self.ndim, :], x - self.f_shift[0])
        g0 = self.lamdas[0] * operator.rosenbrock_func(z0, shift=1.0) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. High Conditioned Elliptic Function f13
        g1 = self.lamdas[1] * operator.elliptic_func(x) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Rotated Bent Cigar Function f1
        z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], x - self.f_shift[2])
        g2 = self.lamdas[2] * operator.bent_cigar_func(z2) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        # 4. Rotated Discus Function f2
        z3 = np.dot(self.f_matrix[3 * self.ndim:4 * self.ndim, :], x - self.f_shift[3])
        g3 = self.lamdas[3] * operator.discus_func(z3) + self.bias[3]
        w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

        # 5. High Conditioned Elliptic Function f13
        g4 = self.lamdas[4] * operator.elliptic_func(x) + self.bias[4]
        w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

        ws = np.array([w0, w1, w2, w3, w4])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3, g4])
        return np.dot(ws, gs) + self.f_bias


class F142015(F132015):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F14: Composition Function 2 (N=3)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1400.0'

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_14_D", f_matrix="M_14_D", f_bias=1400.):
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias)
        self.f_shift = self.check_matrix_data(f_shift, needed_dim=True).ravel().reshape((3, -1))
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 3
        self.xichmas = [10, 30, 50]
        self.lamdas = [0.25, 1.0, 1e-7]
        self.bias = [0, 100, 200]
        self.g0 = operator.modified_schwefel_func
        self.g1 = operator.rastrigin_func
        self.g2 = operator.elliptic_func
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rotated Schwefel's Function f4
        z0 = np.dot(self.f_matrix[:self.ndim, :], x - self.f_shift[0])
        g0 = self.lamdas[0] * self.g0(z0) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Rotated Rastrigin’s Function f12
        z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], x - self.f_shift[1])
        g1 = self.lamdas[1] * self.g1(z1) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Rotated High Conditioned Elliptic Function f13
        z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], x - self.f_shift[2])
        g2 = self.lamdas[2] * self.g2(z2) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        ws = np.array([w0, w1, w2])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2])
        return np.dot(ws, gs) + self.f_bias


class F152015(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """
    name = "F15: Composition Function 3 (N=5)"
    latex_formula = r'F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}'
    latex_formula_dimension = r'2 <= D <= 100'
    latex_formula_bounds = r'x_i \in [-100.0, 100.0], \forall i \in  [1, D]'
    latex_formula_global_optimum = r'\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1500.0'

    modality = False

    def __init__(self, ndim=None, bounds=None, f_shift="shift_data_15_D", f_matrix="M_15_D", f_bias=1500.):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 30
        self.dim_supported = [10, 30]
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2015")
        self.f_shift = self.check_matrix_data(f_shift, needed_dim=True).ravel().reshape((5, -1))
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [10, 10, 10, 20, 20]
        self.lamdas = [10, 10, 2.5, 25, 1e-6]
        self.bias = [0, 100, 200, 300, 400]
        self.paras = {"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix}

    def evaluate(self, x, *args):
        self.n_fe += 1
        self.check_solution(x, self.dim_max, self.dim_supported)

        # 1. Rotated HGBat Function f7
        z0 = np.dot(self.f_matrix[:self.ndim, :], x - self.f_shift[0])
        g0 = self.lamdas[0] * operator.hgbat_func(z0, shift=-1.0) + self.bias[0]
        w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

        # 2. Rotated Rastrigin’s Function f12
        z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], x - self.f_shift[1])
        g1 = self.lamdas[1] * operator.rastrigin_func(z1) + self.bias[1]
        w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

        # 3. Rotated Schwefel's Function f4
        z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], x - self.f_shift[2])
        g2 = self.lamdas[2] * operator.modified_schwefel_func(z2) + self.bias[2]
        w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

        # 4. Rotated Weierstrass Function f3
        z3 = np.dot(self.f_matrix[3 * self.ndim:4 * self.ndim, :], x - self.f_shift[3])
        g3 = self.lamdas[3] * operator.weierstrass_func(z3) + self.bias[3]
        w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

        # 5. Rotated High Conditioned Elliptic Function f13
        z4 = np.dot(self.f_matrix[4 * self.ndim:5 * self.ndim, :], x - self.f_shift[4])
        g4 = self.lamdas[4] * operator.elliptic_func(z4) + self.bias[4]
        w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

        ws = np.array([w0, w1, w2, w3, w4])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3, g4])
        return np.dot(ws, gs) + self.f_bias
