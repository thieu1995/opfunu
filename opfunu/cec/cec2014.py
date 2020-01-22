#!/usr/bin/env python                                                                                   #
# ------------------------------------------------------------------------------------------------------#
# Created by "Thieu Nguyen" at 03:16, 22/01/2020                                                        #
#                                                                                                       #
#       Email:      nguyenthieu2102@gmail.com                                                           #
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  #
#       Github:     https://github.com/thieunguyen5991                                                  #
#-------------------------------------------------------------------------------------------------------#


import numpy as np

class Functions:
    """
        This class of functions taken from CEC 2014 competition
    """

    def __CEC_1__(self, solution=None, shift=0):
        """
        Rotated High Conditioned Elliptic Function
        x1 = x2 = ... = xn = o
        f(x*) = 100
        """
        res = 0
        constant = np.power(10, 6)
        dim = len(solution)
        for i in range(dim):
            res += np.power(constant, i / dim) * np.square((solution[i] - shift))
        return res

    def __CEC_2__(self, solution=None, shift=0):
        """
        Bent cigar function
        f(x*) =  200
        """
        res = 0
        constant = np.power(10, 6)
        dim = len(solution)
        res = np.square((solution[0] - shift))
        for i in range(1, dim):
            res += constant * np.square((solution[i] - shift))
        return res

    def __CEC_3__(self, solution=None, shift=0):
        """
        Discus Function
        f(x*) = 300
        """
        x = solution - shift
        constant = np.power(10, 6)
        dim = len(solution)
        res = constant * np.square(x[0])
        for i in range(1, dim):
            res += np.square(x[i])
        return res

    def __CEC_4__(self, solution=None, shift=0):
        """
        rosenbrock Function
        f(x*) = 400
        """
        x = solution - shift
        constant = np.power(10, 6)
        dim = len(solution)
        res = 0
        for i in range(dim - 1):
            res += 100 * np.square(x[i] ** 2 - x[i + 1]) + np.square(x[i] - 1)
        return res

    def __CEC_5__(self, solution=None, shift=0):
        """
        Ackley’s Function
        """
        x = solution - shift
        dim = len(solution)
        res = 0
        A = 0
        B = 0
        A += -0.2 * np.sqrt(np.sum(np.square(x)) / dim)
        B += np.sum(np.cos(2 * np.pi * x)) / dim
        res = -20 * np.exp(A) - np.exp(B) + 20 + np.e
        # print("res", res)
        return res

    def __CEC_6__(self, solution=None, shift=0):
        """
        Weierstrass Function
        """
        x = solution - shift
        dim = len(solution)
        res = 0
        kmax = 1
        a = 0.5
        b = 3
        A = 0
        B = 0
        for i in range(dim):
            for k in range(kmax + 1):
                A += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (x[i] + 0.5))
        for k in range(kmax + 1):
            B += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
        res = A - dim * B
        return res

    def __CEC_7__(self, solution=None, shift=0):
        x = solution - shift
        A = np.sum(np.square(x)) / 4000
        B = 1
        if isinstance(x, np.ndarray):
            dim = len(x)
            for i in range(dim):
                B *= np.cos(x[i] / np.sqrt(i + 1))
        else:
            B = np.cos(x)
        res = A - B + 1
        return res

    def __CEC_8__(self, solution=None, shift=0):
        x = solution - shift
        dim = len(x)
        res = np.sum(np.square(x)) - 10 * np.sum(np.cos(2 * np.pi * x)) + 10 * dim
        return res

    def __g9__(self, z, dim):
        if np.abs(z) <= 500:
            return z * np.sin(np.power(np.abs(z), 1 / 2))
        elif z > 500:
            return (500 - z % 500) * np.sin(np.sqrt(np.abs(500 - z % 500))) - np.square(z - 500) / (10000 * dim)
        else:
            return (z % 500 - 500) * np.sin(np.sqrt(np.abs(z % 500 - 500))) - np.square(z + 500) / (10000 * dim)

    def __CEC_9__(self, solution=None, shift=0):
        x = solution - shift
        dim = len(x)
        B = 0
        A = 418.9829 * dim
        z = x + 4.209687462275036e+002
        for i in range(dim):
            B += self.__g9__(z[i], dim)
        res = A - B
        return res

    def __CEC_10__(self, solution=None, shift=0):
        x = solution - shift
        dim = len(x)
        A = 1
        for i in range(dim):
            temp = 1
            for j in range(32):
                temp += i * (np.abs(np.power(2, j + 1) * x[i] - round(np.power(2, j + 1) * x[i]))) / np.power(2, j)
            A *= np.power(temp, 10 / np.power(dim, 1.2))
        B = 10 / np.square(dim)
        res = B * A - B
        return res

    def __CEC_11__(self, solution=None, shift=0):
        x = solution - shift
        dim = len(x)

        A = np.power(np.abs(np.sum(np.square(x)) - dim), 1 / 4)
        B = (0.5 * np.sum(np.square(x)) + np.sum(x)) / dim
        res = A + B + 0.5
        return res

    def __CEC_12__(self, solution=None, shift=0):
        x = solution - shift
        dim = len(x)
        A = np.power(np.abs(np.square(np.sum(np.square(x))) - np.square(np.sum(x))), 1 / 2)
        B = (0.5 * np.sum(np.square(x)) + np.sum(x)) / dim
        res = A + B + 0.5
        return res

    def __CEC_13__(self, solution=None, shift=0):
        x = solution - shift
        res = 0
        dim = len(x)
        for i in range(dim):
            res += self.__CEC_7__(self.__CEC_4__(x[i: (i + 2) % dim], shift=0), shift=0)
        return res

    def __CEC_14__(self, solution=None, shift=0):
        x = solution - shift
        res = 0
        dim = len(x)

        def g(x, y):
            return 0.5 + (np.square(np.sin(np.sqrt(x * x + y * y))) - 0.5) / np.square(1 + 0.001 * np.square((x * x + y * y)))

        for i in range(dim):
            res += g(x[i], x[(i + 1) % dim])
        return res

    def __shift__(self, solution, shift_number):
        return np.array(solution) - shift_number

    def __rotate__(self, solution, original_x, rotate_rate=1):
        return solution


    def C1(self, solution, shift_num=1, rate=1):
        x = self.__shift__(solution, shift_num)
        return self.__CEC_1__(x) + 100 * rate

    def C2(self, solution, shift_num=1, rate=1):
        x = self.__shift__(solution, shift_num)
        return self.__CEC_2__(x) + 200 * rate

    def C3(self, solution, shift_num=1, rate=1):
        x = self.__shift__(solution, shift_num)
        return self.__CEC_3__(x) + 300 * rate

    def C4(self, solution, shift_num=2, rate=1):
        x = 2.48 / 100 * self.__shift__(solution, shift_num)
        x = self.__rotate__(x, solution) + 1
        return self.__CEC_4__(x) + 400 * rate

    def C5(self, solution, shift_num=1, rate=1):
        x = self.__shift__(solution, shift_num)
        x = self.__rotate__(x, solution)
        return self.__CEC_5__(x) + 500 * rate

    def C6(self, solution, shift_num=1, rate=1):
        x = 0.5 / 100 * self.__shift__(solution, shift_num)
        return self.__CEC_6__(x) + 600 * rate

    def C7(self, solution, shift_num=1, rate=1):
        x = 600 / 100 * self.__shift__(solution, shift_num)
        return self.__CEC_7__(x) + 700 * rate

    def C8(self, solution, shift_num=1, rate=1):
        x = 5.12 / 100 * self.__shift__(solution, shift_num)
        return self.__CEC_8__(x) + 800 * rate

    def C9(self, solution, shift_num=1, rate=1):
        x = 5.12 / 100 * self.__shift__(solution, shift_num)
        x = self.__rotate__(x, solution)
        return self.__CEC_8__(x) + 900 * rate

    def C10(self, solution, shift_num=1, rate=1):
        x = 1000 / 100 * self.__shift__(solution, shift_num)
        return self.__CEC_9__(x) + 1000 * rate

    def C11(self, solution, shift_num=1, rate=1):
        x = 1000 / 100 * self.__shift__(solution, shift_num)
        x = self.__rotate__(x, solution)
        return self.__CEC_9__(x) + 1100 * rate

    def C12(self, solution, shift_num=1, rate=1):
        x = 5 / 100 * self.__shift__(solution, shift_num)
        x = self.__rotate__(x, solution)
        return self.__CEC_10__(x) + 1200 * rate

    def C13(self, solution, shift_num=1, rate=1):
        x = 5 / 100 * self.__shift__(solution, shift_num)
        x = self.__rotate__(x, solution)
        return self.__CEC_11__(x) + 1300 * rate

    def C14(self, solution, shift_num=1, rate=1):
        x = 5 / 100 * self.__shift__(solution, shift_num)
        x = self.__rotate__(x, solution)
        return self.__CEC_12__(x) + 1400 * rate

    def C15(self, solution, shift_num=2, rate=1):
        x = 5 / 100 * self.__shift__(solution, shift_num)
        x = self.__rotate__(x, solution) + 1
        return self.__CEC_13__(x) + 1500 * rate

    def C16(self, solution, shift_num=1, rate=1):
        x = 5 / 100 * self.__shift__(solution, shift_num)
        x = self.__rotate__(x, solution) + 1
        return self.__CEC_14__(x) + 1600 * rate

    def C17(self, solution, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.3 * dim)
        n2 = int(0.3 * dim) + n1
        D = np.arange(dim)

        # np.random.shuffle(D)
        x = self.__shift__(solution, shift_num)
        return self.__CEC_9__(x[D[: n1]]) + self.__CEC_8__(x[D[n1: n2]]) + self.__CEC_1__(x[D[n2:]]) + 1700 * rate

    def C18(self, solution, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.3 * dim)
        n2 = int(0.3 * dim) + n1
        D = np.arange(dim)
        # np.random.shuffle(D)
        x = self.__shift__(solution, shift_num)
        return self.__CEC_2__(x[D[: n1]]) + self.__CEC_12__(x[D[n1: n2]]) + self.__CEC_8__(x[D[n2:]]) + 1800 * rate

    def C19(self, solution, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.2 * dim)
        n2 = int(0.2 * dim) + n1
        n3 = int(0.3 * dim) + n2
        D = np.arange(dim)
        # np.random.shuffle(D)
        x = self.__shift__(solution, shift_num)
        return self.__CEC_7__(x[D[: n1]]) + self.__CEC_6__(x[D[n1: n2]]) + self.__CEC_4__(x[D[n2: n3]]) + self.__CEC_14__(x[D[n3:]]) + 1900 * rate

    def C20(self, solution, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.2 * dim)
        n2 = int(0.2 * dim) + n1
        n3 = int(0.3 * dim) + n2
        D = np.arange(dim)
        # np.random.shuffle(D)
        x = self.__shift__(solution, shift_num)
        return self.__CEC_12__(x[D[: n1]]) + self.__CEC_3__(x[D[n1: n2]]) + self.__CEC_13__(x[D[n2: n3]]) + self.__CEC_8__(x[D[n3:]]) + 2000 * rate

    def C21(self, solution, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.1 * dim)
        n2 = int(0.2 * dim) + n1
        n3 = int(0.2 * dim) + n2
        n4 = int(0.2 * dim) + n3
        D = np.arange(dim)
        # np.random.shuffle(D)
        x = self.__shift__(solution, shift_num)
        return self.__CEC_14__(x[D[: n1]]) + self.__CEC_12__(x[D[n1: n2]]) + self.__CEC_4__(x[D[n2: n3]]) + \
               self.__CEC_9__(x[D[n3: n4]]) + self.__CEC_1__(x[D[n4:]]) + 2100 * rate

    def C22(self, solution, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.1 * dim)
        n2 = int(0.2 * dim) + n1
        n3 = int(0.2 * dim) + n2
        n4 = int(0.2 * dim) + n3
        D = np.arange(dim)
        # np.random.shuffle(D)
        x = self.__shift__(solution, shift_num)
        return self.__CEC_10__(x[D[: n1]]) + self.__CEC_11__(x[D[n1: n2]]) + self.__CEC_13__(x[D[n2: n3]]) + \
               self.__CEC_9__(x[D[n3: n4]]) + self.__CEC_5__(x[D[n4:]]) + 2200 * rate

    def C23(self, solution, rate=1):
        shift_arr = [1, 2, 3, 4, 5]
        sigma = [10, 20, 30, 40, 50]
        lamda = [1, 1.0e-6, 1.0e-26, 1.0e-6, 1.0e-6]
        bias = [0, 100, 200, 300, 400]
        fun = [self.C4, self.C1, self.C2, self.C3, self.C1]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = self.__shift__(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2300 * rate

    def C24(self, solution, rate=1):
        shift_arr = [1, 2, 3]
        sigma = [20, 20, 20]
        lamda = [1, 1, 1]
        bias = [0, 100, 200]
        fun = [self.C10, self.C9, self.C14]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = self.__shift__(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2400 * rate

    def C25(self, solution, rate=1):
        shift_arr = [1, 2, 3]
        sigma = [10, 30, 50]
        lamda = [0.25, 1, 1.0e-7]
        bias = [0, 100, 200]
        fun = [self.C11, self.C9, self.C1]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = self.__shift__(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2500 * rate

    def C26(self, solution, rate=1):
        shift_arr = [1, 2, 3, 4, 5]
        sigma = [10, 10, 10, 10, 10]
        lamda = [0.25, 1.0, 1.0e-7, 2.5, 10.0]
        bias = [0, 100, 200, 300, 400]
        fun = [self.C11, self.C13, self.C1, self.C6, self.C7]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = self.__shift__(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2600 * rate

    def C27(self, solution, rate=1):
        shift_arr = [1, 2, 3, 4, 5]
        sigma = [10, 10, 10, 20, 20]
        lamda = [10, 10, 2.5, 25, 1.0e-6]
        bias = [0, 100, 200, 300, 400]
        fun = [self.C14, self.C9, self.C11, self.C6, self.C1]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = self.__shift__(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2700 * rate

    def C28(self, solution, rate=1):
        shift_arr = [1, 2, 3, 4, 5]
        sigma = [10, 20, 30, 40, 50]
        lamda = [2.5, 10, 2.5, 5.0e-4, 1.0e-6]
        bias = [0, 100, 200, 300, 400]
        fun = [self.C15, self.C13, self.C11, self.C16, self.C1]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = self.__shift__(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2800 * rate

    def C29(self, solution, rate=1):
        shift_arr = [4, 5, 6]
        sigma = [10, 30, 50]
        lamda = [1, 1, 1]
        bias = [0, 100, 200]
        fun = [self.C17, self.C18, self.C19]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = self.__shift__(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2900 * rate

    def C30(self, solution, rate=1):
        shift_arr = [1, 2, 3]
        sigma = [10, 30, 50]
        lamda = [1, 1, 1]
        bias = [0, 100, 200]
        fun = [self.C20, self.C21, self.C22]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = self.__shift__(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 3000 * rate
