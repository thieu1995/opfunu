#!/usr/bin/env python
# Created by "Thieu" at 15:00, 29/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# Paper: Prairie Dog Optimization Algorithm (PDO-2022)

import numpy as np
from opfunu.engineer import Engineer
from opfunu.utils.encoder import LabelEncoder


class WeldedBeamProblem(Engineer):
    """
    x = [x1, x2, x3, x4]

    WBD is subjected to 4 design constraints: shear, beam blending stress, bar buckling load beam, and deflection
    variables: h=x1, l=x2, t=x3, b=x4
    l: length, h: height, t: thickness, b: weld thickness of the bar

    https://sci-hub.se/10.1016/s0166-3615(99)00046-9
    """

    name = "Welded Beam Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 4
        self._n_objs = 1
        self._n_cons = 11
        self._bounds = [(0.1, 2.), (0.1, 10.), (0.1, 10.), (0.1, 2.)]
        self.xichma_max = 30000
        self.P = 6000
        self.L = 14
        self.delta_max = 0.25
        self.E = 30 * 10 ** 6
        self.theta_max = 13600
        self.G = 12 * 10 ** 6
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = x[0] ** 2 * x[1] * 1.10471 + 0.04811 * x[2] * x[3] * (14.0 + x[1])
        return np.array([f1])

    def get_cons(self, x):
        Pc_X = 4.013 * self.E * np.sqrt(x[2] ** 2 * x[3] ** 6 / 36) / self.L ** 2 * (1. - x[2] * np.sqrt(self.E / (4 * self.G)) / (2 * self.L))
        J = 2 * (np.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 4 + (x[0] + x[2] / 2) ** 2))
        M = self.P * (self.L + x[1] / 2)
        R = np.sqrt(x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4)
        t2 = M * R / J
        t1 = self.P / (np.sqrt(2) * x[0] * x[1])
        t_X = np.sqrt(t1 ** 2 + 2 * t1 * t2 * x[1] / (2 * R) + t2 ** 2)
        xichma_X = 6 * self.P * self.L / (x[3] * x[2] ** 2)
        delta_X = 4 * self.P * self.L ** 3 / (self.E * x[2] ** 3 * x[3])
        g1 = t_X - self.theta_max
        g2 = xichma_X - self.xichma_max
        g3 = x[0] - x[3]
        g4 = 0.10471 * x[0] ** 2 + 0.04811 * x[2] * x[3] * (14.0 + x[1]) - 5.0
        g5 = 0.125 - x[0]
        g6 = delta_X - self.delta_max
        g7 = self.P - Pc_X
        return np.array([g1, g2, g3, g4, g5, g6, g7])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class PressureVesselProblem(Engineer):
    """
    x = [x1, x2, x3, x4]

    Variables: the inner radius (R=x3), the thickness of the head (Th=x2),
        the length of the cylindrical section of the vessel (L=x4), and the thickness of the shell (Ts=x1)

    https://sci-hub.se/10.1115/1.2912596
    """

    name = "Pressure Vessel Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 4
        self._n_objs = 1
        self._n_cons = 4
        self._bounds = [(0., 99.), (0., 99.), (10., 200.), (10., 200.)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 0.6224 * x[2] * x[0] * x[3] + 1.7781 * x[2] ** 2 * x[1] + 3.1611 * x[0] ** 2 * x[3] + 19.8621 * x[2] * x[0] ** 2
        return np.array([f1])

    def get_cons(self, x):
        g1 = -x[0] + 0.0193 * x[2]
        g2 = -x[2] + 0.00954 * x[2]
        g3 = -np.pi * x[1] ** 2 * x[3] - 4. / 3 * np.pi * x[2] ** 3 + 750 * 1728
        g4 = -240 + x[3]
        return np.array([g1, g2, g3, g4])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class CompressionSpringProblem(Engineer):
    """
    x = [x1, x2, x3, x4]

    CSD aims to minimize the weight of a tension/compression spring given the values of 3 parameters:
        the wire diameter (d=x1), number of active coils (P=x3), and mean coil diameter (D=x2).

    https://sci-hub.se/10.1016/s0166-3615(99)00046-9
    """

    name = "Compression Spring Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 3
        self._n_objs = 1
        self._n_cons = 4
        self._bounds = [(0.05, 2.), (0.25, 1.3), (2., 15.)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = (x[2] + 2)*x[1]*x[0]**2
        return np.array([f1])

    def get_cons(self, x):
        g1 = 1 - x[1] ** 3 * x[2] / (71785 * x[0] ** 4)
        g2 = (4 * x[1] ** 2 - x[0] * x[1]) / (12566 * (x[2] * x[0] ** 3 - x[0] ** 4)) + 1. / (5108 * x[0] ** 2) - 1
        g3 = 1 - 140.45 * x[0] / (x[1] ** 2 * x[2])
        g4 = (x[0] + x[1]) / 1.5 - 1
        return np.array([g1, g2, g3, g4])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class SpeedReducerProblem(Engineer):
    """
    Depicts a gearbox that sits between the propeller and engine of an aeroplane
    [x1, x2, x3, x4, x5, x6, x7] = [b, m, z, l1, l2, d1, d2]
    """

    name = "Speed Reducer Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 7
        self._n_objs = 1
        self._n_cons = 11
        self._bounds = [(2.6, 3.6), (0.7, 0.8), (17, 28.99), (7.3, 8.3), (7.3, 8,3), (2.9, 3.9), (5.0, 5.5)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 0.7854*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) - 1.508*x[0]*(x[5]**2 + x[6]**2) +\
            7.4777*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)
        return np.array([f1])

    def amend_position(self, x, lb=None, ub=None):
        x[2] = int(x[2])
        return x

    def get_cons(self, x):
        g1 = 27. / (x[0] * x[1] ** 2 * x[2]) - 1
        g2 = 397.5 / (x[0] * x[1] ** 2 * x[2] ** 2) - 1
        g3 = 1.93 * x[3] ** 2 / (x[1] * x[5] ** 4 * x[2]) - 1
        g4 = 1.93 * x[4] ** 2 / (x[1] * x[6] ** 4 * x[2]) - 1
        g5 = np.sqrt((745 * x[3] / (x[1] * x[2])) ** 2 + 16 * 10 ** 6) / (110 * x[5] ** 3) - 1
        g6 = np.sqrt((745 * x[4] / (x[1] * x[2])) ** 2 + 157.5 * 10 ** 6) / (85 * x[6] ** 3) - 1
        g7 = x[1] * x[2] / 40 - 1
        g8 = 5 * x[1] / x[0] - 1
        g9 = x[0] / (12. * x[1]) - 1
        g10 = (1.5 * x[5] + 1.9) / x[3] - 1
        g11 = (1.1 * x[6] + 1.9) / x[4] - 1
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class ThreeBarTrussProblem(Engineer):
    """
    Minimize three-bar structure weight subject to supporting a total load P acting vertically downwards

    [x1, x2]
    """

    name = "Three Bar Truss Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 2
        self._n_objs = 1
        self._n_cons = 3
        self._bounds = [(0., 1.0), (0., 1.)]
        self.L = 100
        self.P = 2
        self.xichma = 2
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = (2*np.sqrt(2)*x[0] + x[1]) * self.L
        return np.array([f1])

    def get_cons(self, x):
        g1 = (np.sqrt(2) * x[0] + x[1]) / (np.sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]) * self.P - self.xichma
        g2 = x[1] * self.P / (np.sqrt(x[0] ** 2 + 2 * x[0] * x[1])) - self.xichma
        g3 = self.P / (np.sqrt(2) * x[1] + x[0]) - self.xichma
        return np.array([g1, g2, g3])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class GearTrainProblem(Engineer):
    """
    Unconstrained discrete design optimization problem
        [x1, x2, x3, x4] = [n_A, n_B, n_C, n_D]
    """

    name = "Gear Train Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 4
        self._n_objs = 1
        self._n_cons = 0
        self._bounds = [(12, 60.99), (12, 60.99), (12, 60.99), (12, 60.99)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = (1. / 6.931 - x[2]*x[1] / (x[0] * x[3]))**2
        return np.array([f1])

    def amend_position(self, x, lb=None, ub=None):
        return np.asarray(x, int)

    def get_cons(self, x):
        return np.array([])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class CantileverBeamProblem(Engineer):
    """
    Minimize a cantilever beam's weight.
        [x1, x2, x3, x4, x5]
    """

    name = "Cantilever Beam Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 5
        self._n_objs = 1
        self._n_cons = 1
        self._bounds = [(0.01, 100.),] * 5
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 0.0624 * np.sum(x)
        return np.array([f1])

    def get_cons(self, x):
        g1 = 61./x[0]**3 + 37./x[1]**3 + 19./x[2]**3 + 7./x[3]**3 + 1./x[4]**3 - 1
        return np.array([g1, ])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class IBeamProblem(Engineer):
    """
    Minimizes the vertical deflection of a beam
        [x1, x2, x3, x4] = [b, h, t_w, t_f]
    """

    name = "I Beam Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 4
        self._n_objs = 1
        self._n_cons = 2
        self._bounds = [(10, 50.), (10, 80.), (0.9, 5.), (0.9, 5.)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 500. / ( (x[2]*(x[1]-2*x[3])**3)/12 + (x[0]*x[3]**3/6) + 2*x[0]*x[3]*(x[1] - x[3])**2 )
        return np.array([f1])

    def get_cons(self, x):
        g1 = 2 * x[0] * x[2] + x[2] * (x[1] - 2 * x[3]) - 300
        g2 = (18 * x[1] * 10 ** 4) / (x[2] * (x[1] - 2 * x[3]) ** 3 + 2 * x[0] * x[2] * (4 * x[3] ** 2 + 3 * x[1] * (x[1] - 2 * x[3]))) + \
             15 * x[0] * 10 ** 3 / ((x[1] - 2 * x[3]) * x[2] ** 2 + 2 * x[2] * x[0] ** 3) - 56
        return np.array([g1, g2])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class TubularColumnProblem(Engineer):
    """
    [x1, x2] = [d, t]

    https://apmonitor.com/me575/index.php/Main/TubularColumn
    """

    name = "Tubular Column Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 2
        self._n_objs = 1
        self._n_cons = 6
        self._bounds = [(2., 14.), (0.2, 0.8)]
        self.xichma_y = 450
        self.E = 0.65 * 10 ** 6
        self.P = 2300
        self.pro = 0.002
        self.L = 300
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 9.8*x[0]*x[1] + 2*x[0]
        return np.array([f1])

    def get_cons(self, x):
        g1 = self.P / (np.pi * x[0] * x[1] * self.xichma_y) - 1
        g2 = (8 * self.P * self.L ** 2) / (np.pi ** 3 * self.E * x[0] * x[1] * (x[0] ** 2 + x[1] ** 2)) - 1
        g3 = 2. / x[0] - 1
        g4 = x[0] / 14 - 1
        g5 = 0.2 / x[1] - 1
        g6 = x[1] / 8 - 1
        return np.array([g1, g2, g3, g4, g5, g6])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class PistonLeverProblem(Engineer):
    """
    [x1, x2, x3, x4] = [H, B, D, X]
    """

    name = "Piston Lever Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 4
        self._n_objs = 1
        self._n_cons = 4
        self._bounds = [(0.05, 500), (0.05, 500.), (0.05, 120.), (0.05, 500.)]
        self.L = 240
        self.M_max = 1.8 * 10 ** 6
        self.P = 1500
        self.Q = 10000
        self.theta = np.pi / 4
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        L1 = np.sqrt((x[3] - x[1]) ** 2 + x[0] ** 2)
        L2 = np.sqrt((x[3] * np.sin(self.theta) + x[0]) ** 2 + (x[1] - x[3] * np.cos(self.theta)) ** 2)
        f1 = 0.25*np.pi*x[2]**2 * (L2 - L1)
        return np.array([f1])

    def get_cons(self, x):
        L1 = np.sqrt((x[3] - x[1]) ** 2 + x[0] ** 2)
        L2 = np.sqrt((x[3] * np.sin(self.theta) + x[0]) ** 2 + (x[1] - x[3] * np.cos(self.theta)) ** 2)
        R = np.abs(-x[3] * (x[3] * np.sin(self.theta) + x[0]) + x[0] * (x[1] - x[3] * np.cos(self.theta))) / np.sqrt((x[3] - x[1]) ** 2 + x[0] ** 2)
        F = np.pi * self.P * x[2] ** 2 / 4
        g1 = self.Q * self.L * np.cos(self.theta) - R * F
        g2 = self.Q * (self.L - x[3]) - self.M_max
        g3 = 1.2 * (L2 - L1) - L1
        g4 = x[2] / 2 - x[1]
        return np.array([g1, g2, g3, g4])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class CorrugatedBulkheadProblem(Engineer):
    """
    [x1, x2, x3, x4] = [width, depth, length, thickness]
    """

    name = "Corrugated Bulkhead Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 4
        self._n_objs = 1
        self._n_cons = 6
        self._bounds = [(0., 100), (0., 100.), (0., 100.), (0., 5.)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 5.885*x[3]*(x[0] + x[2]) / (x[0] + np.sqrt(np.abs(x[2]**2 - x[1]**2)))
        return np.array([f1])

    def get_cons(self, x):
        g1 = -x[3] * x[2] * (0.4 * x[0] + x[2] / 6) + 8.94 * (x[0] + np.sqrt(np.abs(x[2] ** 2 - x[1] ** 2)))
        g2 = -x[3] * x[1] ** 2 * (0.2 * x[0] + x[2] / 12) + 2.2 * (8.94 * (x[0] + np.sqrt(np.abs(x[2] ** 2 - x[1] ** 2)))) ** (4. / 3)
        g3 = -x[3] + 0.0156 * x[0] + 0.15
        g4 = -x[3] + 0.0156 * x[2] + 0.15
        g5 = -x[3] + 1.05
        g6 = -x[2] + x[1]
        return np.array([g1, g2, g3, g4, g5, g6])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class ReinforcedConcreateBeamProblem(Engineer):
    """
    [x1, x2, x3]
    """

    name = "Reinforced Concreate Beam Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 3
        self._n_objs = 1
        self._n_cons = 2
        self.x0 = [6.0, 6.16, 6.32, 6.6, 7.0, 7.11, 7.2, 7.8, 7.9, 8.0, 8.4]
        self.le = LabelEncoder()
        self.le.fit(self.x0)
        self._bounds = [(0., 10.99), (28., 40.), (5., 10.)]
        self.check_penalty_func(f_penalty)

    def amend_position(self, x, lb=None, ub=None):
        x[0] = self.le.inverse_transform([int(x[0])])
        return x

    def get_objs(self, x):
        f1 = 2.9*x[0] + 0.6*x[1]*x[2]
        return np.array([f1])

    def get_cons(self, x):
        g1 = x[1] / x[2] - 4
        g2 = 180 + 7.375 * x[0] ** 2 / x[2] - x[0] * x[1]
        return np.array([g1, g2])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[0]) != int:
            x = self.amend_position(x, self.lb, self.ub)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


WBP = WeldedBeamProblem
PVP = PressureVesselProblem
CSP = CompressionSpringProblem
SRD = SpeedReducerProblem
TBTD = ThreeBarTrussProblem
GTD = GearTrainProblem
CBD = CantileverBeamProblem
IBD = IBeamProblem
TCD = TubularColumnProblem
PLD = PistonLeverProblem
CBHD = CorrugatedBulkheadProblem
RCB = ReinforcedConcreateBeamProblem
