#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:13, 26/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy import dot, array, sum, matmul, where, sqrt, sign, min, cos, pi, exp, round, ceil, ones, concatenate
from opfunu.cec.utils import BasicFunction


class Model(BasicFunction):
    def __init__(self, solution=None, cec_type="cec2014", bound=(-100, 100), dimensions=(10, 20, 30, 50, 100)):
        BasicFunction.__init__(self, cec_type)
        self.problem_size = len(solution)
        self.dimensions = dimensions
        self.check_dimensions(self.problem_size)
        self.bound = bound
        self.solution = solution

    def __load_shift_matrix_1__(self, f_shift=None, f_matrix=None):
        shift = self.load_shift_data__(f_shift)[:self.problem_size]
        matrix = self.load_matrix_data__(f_matrix)
        return shift, matrix

    def __load_shift_matrix_2__(self, f_shift=None, f_matrix=None):
        shift = self.load_matrix_data__(f_shift)[:, :self.problem_size]
        matrix = self.load_matrix_data__(f_matrix)
        return shift, matrix

    def __load_shift_matrix_shuffle__(self, f_shift=None, f_matrix=None, f_shuffle=None):
        shift = self.load_shift_data__(f_shift)[:self.problem_size]
        matrix = self.load_matrix_data__(f_matrix)
        shuffle = (self.load_shift_data__(f_shuffle)[:self.problem_size] - ones(self.problem_size)).astype(int)
        return shift, matrix, shuffle


    def F1(self, name="Rotated High Conditioned Elliptic Function", f_shift="shift_data_1.txt", f_matrix="M_1_D", f_bias=100):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        t1 = self.solution - shift
        z = dot(matrix, t1)
        return self.elliptic__(z) + f_bias

    def F2(self, name="Rotated Bent Cigar Function", f_shift="shift_data_2.txt", f_matrix="M_2_D", f_bias=200):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = dot(matrix, self.solution - shift)
        return self.bent_cigar__(z) + f_bias

    def F3(self, name="Rotated Discus Function", f_shift="shift_data_3.txt", f_matrix="M_3_D", f_bias=300):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = dot(matrix, self.solution - shift)
        return self.discus__(z) + f_bias

    def F4(self, name="Shifted and Rotated Rosenbrock’s Function", f_shift="shift_data_4.txt", f_matrix="M_4_D", f_bias=400):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 2.048 * (self.solution - shift) / 100
        y = dot(matrix, z) + 1
        return self.rosenbrock__(y) + f_bias

    def F5(self, name="Shifted and Rotated Ackley’s Function", f_shift="shift_data_5.txt", f_matrix="M_5_D", f_bias=500):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = dot(matrix, self.solution - shift)
        return self.ackley__(z) + f_bias

    def F6(self, name="Shifted and Rotated Weierstrass Function", f_shift="shift_data_6.txt", f_matrix="M_6_D", f_bias=600):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 0.5 * (self.solution - shift) / 100
        y = dot(matrix, z)
        return self.weierstrass__(y) + f_bias

    def F7(self, name="Shifted and Rotated Griewank’s Function", f_shift="shift_data_7.txt", f_matrix="M_7_D", f_bias=700):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 600 * (self.solution - shift) / 100
        y = dot(matrix, z)
        return self.griewank__(y) + f_bias

    def F8(self, name="Shifted Rastrigin’s Function", f_shift="shift_data_8.txt", f_matrix="M_8_D", f_bias=800):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 5.12 * (self.solution - shift) / 100
        return self.rastrigin__(z) + f_bias

    def F9(self, name="Shifted and Rotated Rastrigin’s Function", f_shift="shift_data_9.txt", f_matrix="M_9_D", f_bias=900):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 5.12 * (self.solution - shift) / 100
        y = dot(matrix, z)
        return self.griewank__(y) + f_bias

    def F10(self, name="Shifted Schwefel’s Function", f_shift="shift_data_10.txt", f_matrix="M_10_D", f_bias=1000):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 1000 * (self.solution - shift) / 100
        return self.modified_schwefel__(z) + f_bias

    def F11(self, name="Shifted and Rotated Schwefel’s Function", f_shift="shift_data_11.txt", f_matrix="M_11_D", f_bias=1100):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 1000 * (self.solution - shift) / 100
        y = dot(matrix, z)
        return self.modified_schwefel__(y) + f_bias

    def F12(self, name="Shifted and Rotated Katsuura Function", f_shift="shift_data_12.txt", f_matrix="M_12_D", f_bias=1200):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 5 * (self.solution - shift) / 100
        y = dot(matrix, z)
        return self.katsuura__(y) + f_bias

    def F13(self, name="Shifted and Rotated HappyCat Function", f_shift="shift_data_13.txt", f_matrix="M_13_D", f_bias=1300):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 5 * (self.solution - shift) / 100
        y = dot(matrix, z)
        return self.happy_cat__(y) + f_bias

    def F14(self, name="Shifted and Rotated HGBat Function", f_shift="shift_data_14.txt", f_matrix="M_14_D", f_bias=1400):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 5 * (self.solution - shift) / 100
        y = dot(matrix, z)
        return self.hgbat__(y) + f_bias

    def F15(self, name="Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function",
            f_shift="shift_data_15.txt", f_matrix="M_15_D", f_bias=1500):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = 5 * (self.solution - shift) / 100
        y = dot(matrix, z) + 1
        return self.expanded_griewank__(y) + f_bias

    def F16(self, name="Shifted and Rotated Expanded Scaffer’s F6 Function", f_shift="shift_data_16.txt", f_matrix="M_16_D", f_bias=1600):
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_1__(f_shift, f_matrix)
        z = dot(matrix, self.solution - shift) + 1
        return self.expanded_scaffer__(z) + f_bias

    ### Hybrid functions
    def F17(self, name="Hybrid Function 1", f_shift="shift_data_17.txt", f_matrix="M_17_D", shuffle=17, f_bias=1700):
        p = array([0.3, 0.3, 0.4])
        n1 = int(ceil(p[0] * self.problem_size))
        n2 = int(ceil(p[1] * self.problem_size)) + n1
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(self.problem_size) + ".txt"
        shift, matrix, shuffle = self.__load_shift_matrix_shuffle__(f_shift, f_matrix, f_shuffle)
        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n2]
        idx3 = shuffle[n2:]
        z = self.solution - shift
        z1 = concatenate((z[idx1], z[idx2], z[idx3]))
        mz = dot(matrix, z1)
        return self.modified_schwefel__(mz[:n1]) + self.rastrigin__(mz[n1:n2]) + self.elliptic__(mz[n2:]) + f_bias

    def F18(self, name="Hybrid Function 2", f_shift="shift_data_18.txt", f_matrix="M_18_D", shuffle=18, f_bias=1800):
        p = array([0.3, 0.3, 0.4])
        n1 = int(ceil(p[0] * self.problem_size))
        n2 = int(ceil(p[1] * self.problem_size)) + n1
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(self.problem_size) + ".txt"
        shift, matrix, shuffle = self.__load_shift_matrix_shuffle__(f_shift, f_matrix, f_shuffle)
        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n2]
        idx3 = shuffle[n2:]
        z = self.solution - shift
        z1 = concatenate((z[idx1], z[idx2], z[idx3]))
        mz = dot(matrix, z1)
        return self.bent_cigar__(mz[:n1]) + self.hgbat__(mz[n1:n2]) + self.rastrigin__(mz[n2:]) + f_bias

    def F19(self, name="Hybrid Function 3", f_shift="shift_data_19.txt", f_matrix="M_19_D", shuffle=19, f_bias=1900):
        p = array([0.2, 0.2, 0.3, 0.3])
        n1 = int(ceil(p[0] * self.problem_size))
        n2 = int(ceil(p[1] * self.problem_size)) + n1
        n3 = int(ceil(p[2] * self.problem_size)) + n2
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(self.problem_size) + ".txt"
        shift, matrix, shuffle = self.__load_shift_matrix_shuffle__(f_shift, f_matrix, f_shuffle)
        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n2]
        idx3 = shuffle[n2:n3]
        idx4 = shuffle[n3:]
        z = self.solution - shift
        z1 = concatenate((z[idx1], z[idx2], z[idx3], z[idx4]))
        mz = dot(matrix, z1)
        return self.griewank__(mz[:n1]) + self.weierstrass__(mz[n1:n2]) + self.rosenbrock__(mz[n2:n3]) + self.expanded_scaffer__(mz[n3:]) + f_bias

    def F20(self, name="Hybrid Function 4", f_shift="shift_data_20.txt", f_matrix="M_20_D", shuffle=20, f_bias=2000):
        p = array([0.2, 0.2, 0.3, 0.3])
        n1 = int(ceil(p[0] * self.problem_size))
        n2 = int(ceil(p[1] * self.problem_size)) + n1
        n3 = int(ceil(p[2] * self.problem_size)) + n2
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(self.problem_size) + ".txt"
        shift, matrix, shuffle = self.__load_shift_matrix_shuffle__(f_shift, f_matrix, f_shuffle)
        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n2]
        idx3 = shuffle[n2:n3]
        idx4 = shuffle[n3:]
        z = self.solution - shift
        z1 = concatenate((z[idx1], z[idx2], z[idx3], z[idx4]))
        mz = dot(matrix, z1)
        return self.hgbat__(mz[:n1]) + self.discus__(mz[n1:n2]) + self.expanded_griewank__(mz[n2:n3]) + self.rastrigin__(mz[n3:]) + f_bias

    def F21(self, name="Hybrid Function 5", f_shift="shift_data_21.txt", f_matrix="M_21_D", shuffle=21, f_bias=2100):
        p = array([0.1, 0.2, 0.2, 0.2, 0.3])
        n1 = int(ceil(p[0] * self.problem_size))
        n2 = int(ceil(p[1] * self.problem_size)) + n1
        n3 = int(ceil(p[2] * self.problem_size)) + n2
        n4 = int(ceil(p[3] * self.problem_size)) + n3
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(self.problem_size) + ".txt"
        shift, matrix, shuffle = self.__load_shift_matrix_shuffle__(f_shift, f_matrix, f_shuffle)
        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n2]
        idx3 = shuffle[n2:n3]
        idx4 = shuffle[n3:n4]
        idx5 = shuffle[n4:]
        z = self.solution - shift
        z1 = concatenate((z[idx1], z[idx2], z[idx3], z[idx4], z[idx5]))
        mz = dot(matrix, z1)
        return self.expanded_scaffer__(mz[:n1]) + self.hgbat__(mz[n1:n2]) + self.rosenbrock__(mz[n2:n3]) + \
               self.modified_schwefel__(mz[n3:n4]) + self.elliptic__(mz[n4:]) + f_bias

    def F22(self, name="Hybrid Function 6", f_shift="shift_data_22.txt", f_matrix="M_22_D", shuffle=22, f_bias=2200):
        p = array([0.1, 0.2, 0.2, 0.2, 0.3])
        n1 = int(ceil(p[0] * self.problem_size))
        n2 = int(ceil(p[1] * self.problem_size)) + n1
        n3 = int(ceil(p[2] * self.problem_size)) + n2
        n4 = int(ceil(p[3] * self.problem_size)) + n3
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(self.problem_size) + ".txt"
        shift, matrix, shuffle = self.__load_shift_matrix_shuffle__(f_shift, f_matrix, f_shuffle)
        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n2]
        idx3 = shuffle[n2:n3]
        idx4 = shuffle[n3:n4]
        idx5 = shuffle[n4:]
        z = self.solution - shift
        z1 = concatenate((z[idx1], z[idx2], z[idx3], z[idx4], z[idx5]))
        mz = dot(matrix, z1)
        return self.katsuura__(mz[:n1]) + self.happy_cat__(mz[n1:n2]) + self.expanded_griewank__(mz[n2:n3]) + \
               self.modified_schwefel__(mz[n3:n4]) + self.ackley__(mz[n4:]) + f_bias

    ## Composition functions

    def __calculate_weights__(self, z, xichma):
        weight = 1
        temp = sum(z ** 2)
        if temp != 0:
            weight = (1.0 / sqrt(temp)) * exp(-temp / (2 * self.problem_size * xichma ** 2))
        return weight

    def F23(self, name="Composition Function 1", f_shift="shift_data_23.txt", f_matrix="M_23_D", f_bias=2300):
        xichma = array([10, 20, 30, 40, 50])
        lamda = array([1, 1e-6, 1e-26, 1e-6, 1e-6])
        bias = array([0, 100, 200, 300, 400])
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_2__(f_shift, f_matrix)

        # 1. Rotated Rosenbrock’s Function F4’
        z = self.solution - shift[0]
        g1 = lamda[0] * self.rosenbrock__(dot(matrix[:self.problem_size, :], z)) + bias[0]
        w1 = self.__calculate_weights__(z, xichma[0])

        # 2. High Conditioned Elliptic Function F1’
        t2 = self.solution - shift[1]
        g2 = lamda[1] * self.elliptic__(solution=self.solution) + bias[1]
        w2 = self.__calculate_weights__(t2, xichma[1])

        # 3. Rotated Bent Cigar Function F2’
        t3 = self.solution - shift[2]
        g3 = lamda[2] * self.bent_cigar__(dot(matrix[2 * self.problem_size: 3 * self.problem_size, :], t3)) + bias[2]
        w3 = self.__calculate_weights__(t3, xichma[2])

        # 4. Rotated Discus Function F3’
        t4 = self.solution - shift[3]
        g4 = lamda[3] * self.discus__(dot(matrix[3 * self.problem_size: 4 * self.problem_size, :], t4)) + bias[3]
        w4 = self.__calculate_weights__(t4, xichma[3])

        # 4. High Conditioned Elliptic Function F1’
        t5 = self.solution - shift[4]
        g5 = lamda[4] * self.elliptic__(self.solution) + bias[4]
        w5 = self.__calculate_weights__(t5, xichma[4])

        sw = sum([w1, w2, w3, w4, w5])
        result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
        return result + f_bias

    def F24(self, name="Composition Function 2", f_shift="shift_data_24.txt", f_matrix="M_24_D", f_bias=2400):
        xichma = array([20, 20, 20])
        lamda = array([1, 1, 1])
        bias = array([0, 100, 200])
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_2__(f_shift, f_matrix)

        # 1. Schwefel's Function F10'
        t1 = self.solution - shift[0]
        g1 = lamda[0] * self.modified_schwefel__(self.solution) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        # 2. Rotated Rastrigin’s Function F9’
        t2 = self.solution - shift[1]
        g2 = lamda[1] * self.rastrigin__(dot(matrix[self.problem_size: 2 * self.problem_size, :], t2)) + bias[1]
        w2 = self.__calculate_weights__(t2, xichma[1])

        # 3. Rotated HGBat Function F14’
        t3 = self.solution - shift[2]
        g3 = lamda[2] * self.hgbat__(dot(matrix[2 * self.problem_size: 3 * self.problem_size, :], t3)) + bias[2]
        w3 = self.__calculate_weights__(t3, xichma[2])

        sw = sum([w1, w2, w3])
        result = (w1 * g1 + w2 * g2 + w3 * g3) / sw
        return result + f_bias

    def F25(self, name="Composition Function 3", f_shift="shift_data_25.txt", f_matrix="M_25_D", f_bias=2500):
        xichma = array([10, 30, 50])
        lamda = array([0.25, 1, 1e-7])
        bias = array([0, 100, 200])
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_2__(f_shift, f_matrix)

        # 1. Rotated Schwefel's Function F11’
        t1 = self.solution - shift[0]
        g1 = lamda[0] * self.modified_schwefel__(dot(matrix[:self.problem_size, :], t1)) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        # 2. Rotated Rastrigin’s Function F9’
        t2 = self.solution - shift[1]
        g2 = lamda[1] * self.rastrigin__(dot(matrix[self.problem_size: 2 * self.problem_size, :], t2)) + bias[1]
        w2 = self.__calculate_weights__(t2, xichma[1])

        # 3. Rotated High Conditioned Elliptic Function F1'
        t3 = self.solution - shift[2]
        g3 = lamda[2] * self.elliptic__(dot(matrix[2 * self.problem_size: 3 * self.problem_size, :], t3)) + bias[2]
        w3 = self.__calculate_weights__(t3, xichma[2])

        sw = sum([w1, w2, w3])
        result = (w1 * g1 + w2 * g2 + w3 * g3) / sw
        return result + f_bias

    def F26(self, name="Composition Function 4", f_shift="shift_data_26.txt", f_matrix="M_26_D", f_bias=2600):
        xichma = array([10, 10, 10, 10, 10])
        lamda = array([0.25, 1, 1e-7, 2.5, 10])
        bias = array([0, 100, 200, 300, 400])
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_2__(f_shift, f_matrix)

        # 1. Rotated Schwefel's Function F11’
        t1 = self.solution - shift[0]
        g1 = lamda[0] * self.modified_schwefel__(dot(matrix[:self.problem_size, :], t1)) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        # 2. Rotated HappyCat Function F13’
        t2 = self.solution - shift[1]
        g2 = lamda[1] * self.happy_cat__(dot(matrix[self.problem_size:2 * self.problem_size, :], t2)) + bias[1]
        w2 = self.__calculate_weights__(t2, xichma[1])

        # 3. Rotated High Conditioned Elliptic Function F1’
        t3 = self.solution - shift[2]
        g3 = lamda[2] * self.elliptic__(dot(matrix[2 * self.problem_size: 3 * self.problem_size, :], t3)) + bias[2]
        w3 = self.__calculate_weights__(t3, xichma[2])

        # 4. Rotated Weierstrass Function F6’
        t4 = self.solution - shift[3]
        g4 = lamda[3] * self.weierstrass__(dot(matrix[3 * self.problem_size: 4 * self.problem_size, :], t4)) + bias[3]
        w4 = self.__calculate_weights__(t4, xichma[3])

        # 5. Rotated Griewank’s Function F7’
        t5 = self.solution - shift[4]
        g5 = lamda[4] * self.griewank__(dot(matrix[4 * self.problem_size:, :], t5)) + bias[4]
        w5 = self.__calculate_weights__(t5, xichma[4])

        sw = sum([w1, w2, w3, w4, w5])
        result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
        return result + f_bias


    def F27(self, name="Composition Function 5", f_shift="shift_data_27.txt", f_matrix="M_27_D", f_bias=2700):
        xichma = array([10, 10, 10, 20, 20])
        lamda = array([10, 10, 2.5, 25, 1e-6])
        bias = array([0, 100, 200, 300, 400])
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_2__(f_shift, f_matrix)

        # 1. Rotated HGBat Function F14'
        t1 = self.solution - shift[0]
        g1 = lamda[0] * self.hgbat__(dot(matrix[:self.problem_size, :], t1)) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        # 2. Rotated Rastrigin’s Function F9’
        t2 = self.solution - shift[1]
        g2 = lamda[1] * self.rastrigin__(dot(matrix[self.problem_size:2 * self.problem_size, :], t2)) + bias[1]
        w2 = self.__calculate_weights__(t2, xichma[1])

        # 3. Rotated Schwefel's Function F11’
        t3 = self.solution - shift[2]
        g3 = lamda[2] * self.modified_schwefel__(dot(matrix[2 * self.problem_size: 3 * self.problem_size, :], t3)) + bias[2]
        w3 = self.__calculate_weights__(t3, xichma[2])

        # 4. Rotated Weierstrass Function F6’
        t4 = self.solution - shift[3]
        g4 = lamda[3] * self.weierstrass__(dot(matrix[3 * self.problem_size: 4 * self.problem_size, :], t4)) + bias[3]
        w4 = self.__calculate_weights__(t4, xichma[3])

        # 5. Rotated High Conditioned Elliptic Function F1’
        t5 = self.solution - shift[4]
        g5 = lamda[4] * self.elliptic__(dot(matrix[4 * self.problem_size:, :], t5)) + bias[4]
        w5 = self.__calculate_weights__(t5, xichma[4])

        sw = sum([w1, w2, w3, w4, w5])
        result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
        return result + f_bias

    def F28(self, name="Composition Function 6", f_shift="shift_data_28.txt", f_matrix="M_28_D", f_bias=2800):
        xichma = array([10, 20, 30, 40, 50])
        lamda = array([2.5, 10, 2.5, 5e-4, 1e-6])
        bias = array([0, 100, 200, 300, 400])
        f_matrix = f_matrix + str(self.problem_size) + ".txt"
        shift, matrix = self.__load_shift_matrix_2__(f_shift, f_matrix)

        # 1. Rotated Expanded Griewank’s plus Rosenbrock’s Function F15’
        t1 = self.solution - shift[0]
        g1 = lamda[0] * self.F15(f_bias=0) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        # 2. Rotated HappyCat Function F13’
        t2 = self.solution - shift[1]
        g2 = lamda[1] * self.happy_cat__(dot(matrix[self.problem_size:2 * self.problem_size, :], t2)) + bias[1]
        w2 = self.__calculate_weights__(t2, xichma[1])

        # 3. Rotated Schwefel's Function F11’
        t3 = self.solution - shift[2]
        g3 = lamda[2] * self.modified_schwefel__(dot(matrix[2 * self.problem_size: 3 * self.problem_size, :], t3)) + bias[2]
        w3 = self.__calculate_weights__(t3, xichma[2])

        # 4. Rotated Expanded Scaffer’s F6 Function F16’
        t4 = self.solution - shift[3]
        g4 = lamda[3] * self.expanded_scaffer__(dot(matrix[3 * self.problem_size: 4 * self.problem_size, :], t4)) + bias[3]
        w4 = self.__calculate_weights__(t4, xichma[3])

        # 5. Rotated High Conditioned Elliptic Function F1’
        t5 = self.solution - shift[4]
        g5 = lamda[4] * self.elliptic__(dot(matrix[4 * self.problem_size:, :], t5)) + bias[4]
        w5 = self.__calculate_weights__(t5, xichma[4])

        sw = sum([w1, w2, w3, w4, w5])
        result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
        return result + f_bias

    def F29(self, name="Composition Function 7", f_shift="shift_data_29.txt", f_matrix="M_29_D", f_bias=2900):
        num_funcs = 3
        func_names = ["F17", "F18", "F19"]
        xichma = array([10, 30, 50])
        lamda = array([1, 1, 1])
        bias = array([0, 100, 200])
        shift = self.load_matrix_data__(f_shift)[:, :self.problem_size]

        weights = ones(num_funcs)
        fits = ones(num_funcs)
        for i in range(0, num_funcs):
            func = self.__getattribute__(func_names[i])
            t1 = lamda[i] * func(f_shift=f_shift, f_matrix=f_matrix, shuffle=29, f_bias=0) + bias[i]
            fits[i] = t1
            z = self.solution - shift[i]
            weights[i] = self.__calculate_weights__(z, xichma[i])
        sw = sum(weights)
        result = 0.0
        for i in range(0, num_funcs):
            result += (weights[i] / sw) * fits[i]
        return result + f_bias

    def F30(self, name="Composition Function 8", f_shift="shift_data_30.txt", f_matrix="M_30_D", f_bias=3000):
        num_funcs = 3
        func_names = ["F20", "F21", "F22"]
        xichma = array([10, 30, 50])
        lamda = array([1, 1, 1])
        bias = array([0, 100, 200])
        shift = self.load_matrix_data__(f_shift)[:, :self.problem_size]

        weights = ones(num_funcs)
        fits = ones(num_funcs)
        for i in range(0, num_funcs):
            func = self.__getattribute__(func_names[i])
            t1 = lamda[i] * func(f_shift=f_shift, f_matrix=f_matrix, shuffle=29, f_bias=0) + bias[i]
            fits[i] = t1
            z = self.solution - shift[i]
            weights[i] = self.__calculate_weights__(z, xichma[i])
        sw = sum(weights)
        result = 0.0
        for i in range(0, num_funcs):
            result += (weights[i] / sw) * fits[i]
        return result + f_bias

