#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:16, 26/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import dot, array, sum, matmul, where, sqrt, sign, min, cos, pi, exp, round
from opfunu.cec.utils import BasicFunction


class Model(BasicFunction):
    def __init__(self, problem_size=None, cec_type="cec2013", f_shift="shift_data", f_matrix="M_D", bound=(-100, 100),
                 dimensions=(2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)):
        BasicFunction.__init__(self, cec_type)
        self.problem_size = problem_size
        self.dimensions = dimensions
        self.check_dimensions(self.problem_size)
        self.bound = bound
        self.f_shift = f_shift + ".txt"
        self.f_matrix = f_matrix + str(self.problem_size) + ".txt"
        self.shift = self.load_matrix_data__(self.f_shift)[:, :problem_size]
        self.matrix = self.load_matrix_data__(self.f_matrix)

    def F1(self, solution=None, name="Sphere Function", shift=None, f_bias=-1400):
        if shift is None:
            shift = self.shift[0]
        return sum((solution - shift)**2) + f_bias

    def F2(self, solution=None, name="Rotated High Conditioned Elliptic Function", shift=None, matrix=None, f_bias=-1300):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:self.problem_size, :]
        t1 = dot(matrix, solution - shift)
        t2 = self.osz_func__(t1)
        return self.elliptic__(t2) + f_bias

    def F3(self, solution=None, name="Rotated Bent Cigar Function", shift=None, matrix=None, f_bias=-1200):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:2*self.problem_size, :]
        t1 = dot(matrix[:self.problem_size, :], solution - shift)
        t2 = self.asy_func__(t1, beta=0.5)
        t3 = dot(matrix[self.problem_size:2 * self.problem_size, :], t2)
        return self.bent_cigar__(t3) + f_bias

    def F4(self, solution=None, name="Rotated Discus Function", shift=None, matrix=None, f_bias=-1100):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:self.problem_size, :]
        t1 = dot(matrix, solution - shift)
        t2 = self.osz_func__(t1)
        return self.discus__(t2) + f_bias

    def F5(self, solution=None, name="Different Powers Function", shift=None, f_bias=-1000):
        if shift is None:
            shift = self.shift[0]
        return self.different_powers__(solution - shift) + f_bias

    def F6(self, solution=None, name="Rotated Rosenbrock’s Function", shift=None, matrix=None, f_bias=-900):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:self.problem_size, :]
        t1 = 2.048 * (solution - shift) / 100
        t2 = dot(matrix, t1) + 1
        return self.rosenbrock__(t2) + f_bias

    def F7(self, solution=None, name="Rotated Schaffers F7 Function", shift=None, matrix=None, f_bias=-800):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:2*self.problem_size, :]
        t2 = dot(matrix[:self.problem_size, :], solution - shift)
        t3 = self.asy_func__(t2, 0.5)
        t4 = self.create_diagonal_matrix__(self.problem_size, alpha=10)
        t5 = matmul(t4, matrix[self.problem_size: 2 * self.problem_size, :])
        t6 = dot(t5, t3)
        return self.schaffers_f7__(t6) + f_bias

    def F8(self, solution=None, name="Rotated Ackley’s Function", shift=None, matrix=None, f_bias=-700):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:2 * self.problem_size, :]
        t2 = dot(matrix[:self.problem_size, :], solution - shift)
        t3 = self.asy_func__(t2, 0.5)
        t4 = self.create_diagonal_matrix__(self.problem_size, alpha=10)
        t5 = matmul(t4, matrix[self.problem_size: 2 * self.problem_size, :])
        t6 = dot(t5, t3)
        return self.ackley__(t6) + f_bias

    def F9(self, solution=None, name="Rotated Weierstrass Function", shift=None, matrix=None, f_bias=-600):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:2 * self.problem_size, :]
        t1 = 0.5 * (solution - shift) / 100
        t2 = dot(matrix[:self.problem_size, :], t1)
        t3 = self.asy_func__(t2, 0.5)
        t4 = self.create_diagonal_matrix__(self.problem_size, alpha=10)
        t5 = matmul(t4, matrix[self.problem_size: 2 * self.problem_size, :])
        t6 = dot(t5, t3)
        return self.weierstrass__(t6) + f_bias

    def F10(self, solution=None, name="Rotated Griewank’s Function", shift=None, matrix=None, f_bias=-500):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:2 * self.problem_size, :]
        t1 = 600 * (solution - shift) / 100
        t2 = self.create_diagonal_matrix__(self.problem_size, alpha=100)
        t3 = matmul(t2, matrix[:self.problem_size, :])
        t4 = dot(t3, t1)
        return self.griewank__(t4) + f_bias

    def F11(self, solution=None, name="Rastrigin’s Function", shift=None, f_bias=-400):
        if shift is None:
            shift = self.shift[0]
        t2 = self.osz_func__(5.12 * (solution - shift) / 100)
        t3 = self.asy_func__(t2, beta=0.2)
        t4 = self.create_diagonal_matrix__(self.problem_size, alpha=10)
        t5 = dot(t4, t3)
        return self.rastrigin__(t5) + f_bias

    def F12(self, solution=None, name="Rotated Rastrigin’s Function", shift=None, matrix=None, f_bias=-300):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:2 * self.problem_size, :]
        t1 = 5.12 * (solution - shift) / 100
        t2 = dot(matrix[:self.problem_size, :], t1)
        t3 = self.osz_func__(t2)
        t4 = self.asy_func__(t3, beta=0.2)
        t5 = self.create_diagonal_matrix__(self.problem_size, alpha=10)
        t6 = matmul(matrix[:self.problem_size, :], t5)
        t7 = matmul(t6, matrix[self.problem_size: 2 * self.problem_size, :])
        t8 = dot(t7, t4)
        return self.rastrigin__(t8) + f_bias

    def F13(self, solution=None, name="Non-continuous Rotated Rastrigin’s Function", shift=None, matrix=None, f_bias=-200):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:2 * self.problem_size, :]
        t1 = 5.12 * (solution - shift) / 100
        t2 = dot(matrix[:self.problem_size, :], t1)
        t3 = where(abs(t2) > 0.5, round(2 * t2) / 2, t2)
        t4 = self.osz_func__(t3)
        t5 = self.asy_func__(t4, beta=0.2)
        t6 = self.create_diagonal_matrix__(self.problem_size, alpha=10)
        t7 = matmul(matrix[:self.problem_size, :], t6)
        t8 = matmul(t7, matrix[self.problem_size: 2 * self.problem_size, :])
        t9 = dot(t8, t5)
        return self.rastrigin__(t9) + f_bias

    def F14(self, solution=None, name="Schwefel’s Function", shift=None, f_bias=-100):
        if shift is None:
            shift = self.shift[0]
        t1 = 1000 * (solution - shift) / 100
        t2 = self.create_diagonal_matrix__(self.problem_size, alpha=10)
        t3 = dot(t2, t1) + 4.209687462275036e+002
        return self.modified_schwefel__(t3) + f_bias

    def F15(self, solution=None, name="Rotated Schwefel’s Function", shift=None, matrix=None, f_bias=100):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:self.problem_size, :]
        t1 = 1000 * (solution - shift) / 100
        t2 = self.create_diagonal_matrix__(self.problem_size, alpha=10)
        t3 = matmul(t2, matrix[:self.problem_size, :])
        t4 = dot(t3, t1) + 4.209687462275036e+002
        return self.modified_schwefel__(t4) + f_bias

    def F16(self, solution=None, name="Rotated Katsuura Function", shift=None, matrix=None, f_bias=200):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:2*self.problem_size, :]
        t1 = 5 * (solution - shift) / 100
        t2 = dot(matrix[:self.problem_size, :], t1)
        t3 = self.create_diagonal_matrix__(self.problem_size, alpha=100)
        t4 = matmul(matrix[self.problem_size:2 * self.problem_size, :], t3)
        t5 = dot(t4, t2)
        return self.katsuura__(t5) + f_bias

    def F17(self, solution=None, name="Lunacek bi-Rastrigin Function", shift=None, f_bias=300):
        if shift is None:
            shift = self.shift[0]
        d = 1
        s = 1 - 1.0 / (2 * sqrt(self.problem_size + 20) - 8.2)
        miu0 = 2.5
        miu1 = -sqrt((miu0 ** 2 - d) / s)
        t1 = 10 * (solution - shift) / 100
        t2 = 2 * sign(solution) * t1 + miu0
        t3 = self.create_diagonal_matrix__(self.problem_size, alpha=100)
        t4 = dot(t3, (t2 - miu0))
        vp1 = sum((t2 - miu0) ** 2)
        vp2 = d * self.problem_size + s * sum((t2 - miu1) ** 2) + 10 * (self.problem_size - sum(cos(2 * pi * t4)))
        return min([vp1, vp2]) + f_bias

    def F18(self, solution=None, name="Rotated Lunacek bi-Rastrigin Function", shift=None, matrix=None, f_bias=400):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:2*self.problem_size, :]
        d = 1
        s = 1 - 1.0 / (2 * sqrt(self.problem_size + 20) - 8.2)
        miu0 = 2.5
        miu1 = -sqrt((miu0 ** 2 - d) / s)
        t1 = 10 * (solution - shift) / 100
        t2 = 2 * sign(t1) * t1 + miu0
        t3 = self.create_diagonal_matrix__(self.problem_size, alpha=100)
        t4 = matmul(matrix[self.problem_size:2 * self.problem_size, :], t3)
        t5 = dot(matrix[:self.problem_size, :], (t2 - miu0))
        t6 = dot(t4, t5)
        vp1 = sum((t2 - miu0) ** 2)
        vp2 = d * self.problem_size + s * sum((t2 - miu1) ** 2) + 10 * (self.problem_size - sum(cos(2 * pi * t6)))
        return min([vp1, vp2]) + f_bias

    def F19(self, solution=None, name="Rotated Expanded Griewank’s plus Rosenbrock’s Function", shift=None, matrix=None, f_bias=500):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:self.problem_size, :]
        t1 = 5 * (solution - shift) / 100
        t2 = dot(matrix[:self.problem_size, :], t1) + 1
        return self.expanded_griewank__(t2) + f_bias

    def F20(self, solution=None, name="Rotated Expanded Scaffer’s F6 Function", shift=None, matrix=None, f_bias=600):
        if shift is None and matrix is None:
            shift = self.shift[0]
            matrix = self.matrix[:2*self.problem_size, :]
        t1 = dot(matrix[:self.problem_size, :], (solution - shift))
        t2 = self.asy_func__(t1, beta=0.5)
        t3 = dot(matrix[self.problem_size: 2 * self.problem_size, :], t2)
        return self.expanded_scaffer__(t3) + f_bias

    ## Composition functions
    def __calculate_weights__(self, z, xichma):
        weight = 1
        temp = sum(z ** 2)
        if temp != 0:
            weight = (1.0 / sqrt(temp)) * exp(-temp / (2 * self.problem_size * xichma ** 2))
        return weight

    def F21(self, solution=None, name="Composition Function 1", f_bias=700):
        xichma = array([10, 20, 30, 40, 50])
        lamda = array([1, 1e-6, 1e-26, 1e-6, 0.1])
        bias = array([0, 100, 200, 300, 400])

        # g1: Rotated Rosenbrock’s Function f6’
        g1 = lamda[0] * self.F6(solution, shift=self.shift[0], matrix=self.matrix[:self.problem_size, :], f_bias=0) + bias[0]
        t1 = solution - self.shift[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        # g2: Rotated Different Powers Function f5’
        t1 = solution - self.shift[1]
        t2 = dot(self.matrix[self.problem_size: 2*self.problem_size, :], t1)
        g2 = lamda[1] * self.different_powers__(t2) + bias[1]
        w2 = self.__calculate_weights__(t1, xichma[1])

        # g3 Rotated Bent Cigar Function f3’
        t1 = solution - self.shift[2]
        g3 = lamda[2] * self.F3(solution, shift=self.shift[2], matrix=self.matrix[2*self.problem_size:4*self.problem_size, :], f_bias=0) + bias[2]
        w3 = self.__calculate_weights__(t1, xichma[2])

        # g4: Rotated Discus Function f4’
        t1 = solution - self.shift[3]
        g4 = lamda[3] * self.F4(solution, shift=self.shift[3], matrix=self.matrix[3*self.problem_size:4*self.problem_size, :], f_bias=0) + bias[3]
        w4 = self.__calculate_weights__(t1, xichma[3])

        # g5: Sphere Function f1
        t1 = solution - self.shift[3]
        g5 = lamda[4] * self.F1(solution, shift=self.shift[4], f_bias=0) + bias[4]
        w5 = self.__calculate_weights__(t1, xichma[4])

        sw = sum([w1, w2, w3, w4, w5])
        result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
        return result + f_bias

    def F22(self, solution=None, name="Composition Function 2", f_bias=800):
        xichma = array([20, 20, 20])
        lamda = array([1, 1, 1])
        bias = array([0, 100, 200])

        # g1-3: Schwefel's Function f14’
        t1 = solution - self.shift[0]
        g1 = lamda[0] * self.F14(solution, shift=self.shift[0], f_bias=0) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        t1 = solution - self.shift[1]
        g2 = lamda[1] * self.F14(solution, shift=self.shift[1], f_bias=0) + bias[1]
        w2 = self.__calculate_weights__(t1, xichma[1])

        t1 = solution - self.shift[2]
        g3 = lamda[2] * self.F14(solution, shift=self.shift[2], f_bias=0) + bias[2]
        w3 = self.__calculate_weights__(t1, xichma[2])

        sw = sum([w1, w2, w3])
        result = (w1 * g1 + w2 * g2 + w3 * g3) / sw
        return result + f_bias

    def F23(self, solution=None, name="Composition Function 3", f_bias=900):
        xichma = array([20, 20, 20])
        lamda = array([1, 1, 1])
        bias = array([0, 100, 200])

        # g1-3: Schwefel's Function f15’
        t1 = solution - self.shift[0]
        g1 = lamda[0] * self.F15(solution, shift=self.shift[0], matrix=self.matrix[:self.problem_size, :], f_bias=0) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        t1 = solution - self.shift[1]
        g2 = lamda[1] * self.F15(solution, shift=self.shift[1], matrix=self.matrix[:self.problem_size, :], f_bias=0) + bias[1]
        w2 = self.__calculate_weights__(t1, xichma[1])

        t1 = solution - self.shift[2]
        g3 = lamda[2] * self.F15(solution, shift=self.shift[2], matrix=self.matrix[:self.problem_size, :], f_bias=0) + bias[2]
        w3 = self.__calculate_weights__(t1, xichma[2])

        sw = sum([w1, w2, w3])
        result = (w1 * g1 + w2 * g2 + w3 * g3) / sw
        return result + f_bias

    def F24(self, solution=None, name="Composition Function 4", f_bias=1000):
        xichma = array([20, 20, 20])
        lamda = array([0.25, 1, 2.5])
        bias = array([0, 100, 200])

        # g1-3: Schwefel's Function f15’, f12', f9'
        t1 = solution - self.shift[0]
        g1 = lamda[0] * self.F15(solution, shift=self.shift[0], matrix=self.matrix[:self.problem_size, :], f_bias=0) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        t1 = solution - self.shift[1]
        g2 = lamda[1] * self.F12(solution, shift=self.shift[1], matrix=self.matrix[self.problem_size:3*self.problem_size, :], f_bias=0) + bias[1]
        w2 = self.__calculate_weights__(t1, xichma[1])

        t1 = solution - self.shift[2]
        g3 = lamda[2] * self.F9(solution, shift=self.shift[2], matrix=self.matrix[2*self.problem_size:4 * self.problem_size, :],f_bias=0) + bias[2]
        w3 = self.__calculate_weights__(t1, xichma[2])

        sw = sum([w1, w2, w3])
        result = (w1 * g1 + w2 * g2 + w3 * g3) / sw
        return result + f_bias

    def F25(self, solution=None, name="Composition Function 5", f_bias=1100):
        xichma = array([10, 30, 50])
        lamda = array([0.25, 1, 2.5])
        bias = array([0, 100, 200])

        # g1-3: Schwefel's Function f15’, f12', f9'
        t1 = solution - self.shift[0]
        g1 = lamda[0] * self.F15(solution, shift=self.shift[0], matrix=self.matrix[:self.problem_size, :], f_bias=0) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        t1 = solution - self.shift[1]
        g2 = lamda[1] * self.F12(solution, shift=self.shift[1], matrix=self.matrix[self.problem_size:3 * self.problem_size, :], f_bias=0) + bias[1]
        w2 = self.__calculate_weights__(t1, xichma[1])

        t1 = solution - self.shift[2]
        g3 = lamda[2] * self.F9(solution, shift=self.shift[2], matrix=self.matrix[2 * self.problem_size:4 * self.problem_size, :], f_bias=0) + bias[2]
        w3 = self.__calculate_weights__(t1, xichma[2])

        sw = sum([w1, w2, w3])
        result = (w1 * g1 + w2 * g2 + w3 * g3) / sw
        return result + f_bias

    def F26(self, solution=None, name="Composition Function 6", f_bias=1200):
        xichma = array([10, 10, 10, 10, 10])
        lamda = array([0.25, 1, 1e-7, 2.5, 10])
        bias = array([0, 100, 200, 300, 400])

        # g1: Rotated Schwefel's Function f15’
        t1 = solution - self.shift[0]
        g1 = lamda[0] * self.F15(solution, shift=self.shift[0], matrix=self.matrix[:self.problem_size, :], f_bias=0) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        # g2: Rotated Rastrigin’s Function f12’
        t1 = solution - self.shift[1]
        g2 = lamda[1] * self.F12(solution, shift=self.shift[1], matrix=self.matrix[self.problem_size:3 * self.problem_size, :], f_bias=0) + bias[1]
        w2 = self.__calculate_weights__(t1, xichma[1])

        # g3: Rotated High Conditioned Elliptic Function f2’
        t1 = solution - self.shift[2]
        g3 = lamda[2] * self.F2(solution, shift=self.shift[2], matrix=self.matrix[2*self.problem_size:3 * self.problem_size, :], f_bias=0) + bias[2]
        w3 = self.__calculate_weights__(t1, xichma[2])

        # g4: Rotated Weierstrass Function f9’
        t1 = solution - self.shift[3]
        g4 = lamda[3] * self.F9(solution, shift=self.shift[3], matrix=self.matrix[3 * self.problem_size:5 * self.problem_size, :], f_bias=0) + bias[3]
        w4 = self.__calculate_weights__(t1, xichma[3])

        # g5: Rotated Griewank’s Function f10
        t1 = solution - self.shift[4]
        g5 = lamda[4] * self.F10(solution, shift=self.shift[4], matrix=self.matrix[4 * self.problem_size:6 * self.problem_size, :], f_bias=0) + bias[4]
        w5 = self.__calculate_weights__(t1, xichma[4])

        sw = sum([w1, w2, w3, w4, w5])
        result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
        return result + f_bias

    def F27(self, solution=None, name="Composition Function 7", f_bias=1300):
        xichma = array([10, 10, 10, 20, 20])
        lamda = array([100, 10, 2.5, 2.5, 0.1])
        bias = array([0, 100, 200, 300, 400])

        # g1: Rotated Schwefel's Function f15’
        t1 = solution - self.shift[0]
        g1 = lamda[0] * self.F10(solution, shift=self.shift[0], matrix=self.matrix[:2 * self.problem_size, :], f_bias=0) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        # g2: Rotated Rastrigin’s Function f12’
        t1 = solution - self.shift[1]
        g2 = lamda[1] * self.F12(solution, shift=self.shift[1], matrix=self.matrix[self.problem_size:3 * self.problem_size, :], f_bias=0) + bias[1]
        w2 = self.__calculate_weights__(t1, xichma[1])

        # g3: Rotated High Conditioned Elliptic Function f2’
        t1 = solution - self.shift[2]
        g3 = lamda[2] * self.F15(solution, shift=self.shift[2], matrix=self.matrix[2*self.problem_size:3 * self.problem_size, :], f_bias=0) + bias[2]
        w3 = self.__calculate_weights__(t1, xichma[2])

        # g4: Rotated Weierstrass Function f9’
        t1 = solution - self.shift[3]
        g4 = lamda[3] * self.F9(solution, shift=self.shift[3], matrix=self.matrix[3 * self.problem_size:5 * self.problem_size, :], f_bias=0) + bias[3]
        w4 = self.__calculate_weights__(t1, xichma[3])

        # g5: Rotated Griewank’s Function f10
        t1 = solution - self.shift[4]
        g5 = lamda[4] * self.F1(solution, shift=self.shift[4], f_bias=0) + bias[4]
        w5 = self.__calculate_weights__(t1, xichma[4])

        sw = sum([w1, w2, w3, w4, w5])
        result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
        return result + f_bias

    def F28(self, solution=None, name="Composition Function 8", f_bias=1400):
        xichma = array([10, 20, 30, 40, 50])
        lamda = array([2.5, 2.5e-3, 2.5, 2.5e-4, 0.1])
        bias = array([0, 100, 200, 300, 400])
        t1 = solution - self.shift[:1, :self.problem_size]

        # g1: Rotated Schwefel's Function f15’
        t1 = solution - self.shift[0]
        g1 = lamda[0] * self.F19(solution, shift=self.shift[0], matrix=self.matrix[:self.problem_size, :], f_bias=0) + bias[0]
        w1 = self.__calculate_weights__(t1, xichma[0])

        # g2: Rotated Rastrigin’s Function f12’
        t1 = solution - self.shift[1]
        g2 = lamda[1] * self.F7(solution, shift=self.shift[1], matrix=self.matrix[self.problem_size:3 * self.problem_size, :], f_bias=0) + bias[1]
        w2 = self.__calculate_weights__(t1, xichma[1])

        # g3: Rotated High Conditioned Elliptic Function f2’
        t1 = solution - self.shift[2]
        g3 = lamda[2] * self.F15(solution, shift=self.shift[2], matrix=self.matrix[2*self.problem_size:3 * self.problem_size, :], f_bias=0) + bias[2]
        w3 = self.__calculate_weights__(t1, xichma[2])

        # g4: Rotated Weierstrass Function f9’
        t1 = solution - self.shift[3]
        g4 = lamda[3] * self.F20(solution, shift=self.shift[3], matrix=self.matrix[3*self.problem_size:5 * self.problem_size, :], f_bias=0) + bias[3]
        w4 = self.__calculate_weights__(t1, xichma[3])

        # g5: Rotated Griewank’s Function f10
        t1 = solution - self.shift[4]
        g5 = lamda[4] * self.F1(solution, shift=self.shift[4], f_bias=0) + bias[4]
        w5 = self.__calculate_weights__(t1, xichma[4])

        sw = sum([w1, w2, w3, w4, w5])
        result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
        return result + f_bias
