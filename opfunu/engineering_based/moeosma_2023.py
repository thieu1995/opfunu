#!/usr/bin/env python
# Created by "Thieu" at 16:38, 05/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# Paper: Multi‑objective equilibrium optimizer slime mould algorithm and its application in solving engineering problems

import numpy as np
from opfunu.engineer import Engineer
from opfunu.utils.encoder import LabelEncoder


class SpeedReducerProblem(Engineer):
    """
    x = [x1, x2, x3, x4, x5, x6, x7] = [b, m, z, l1, l2, d1, d2]
    """

    name = "Speed Reducer Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 7
        self._n_objs = 2
        self._n_cons = 11
        self._bounds = [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.3, 8.3), (2.9, 3.9), (5.0, 5.5)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 0.7854*x[0]*x[1]**2*(14.9334*x[2] + 3.3333*x[2]**2 - 43.0934) - 1.508*x[0]*(x[5]**2 + x[6]**2) \
            + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2) + 7.4777*(x[5]**3 + x[6]**3)
        f2 = np.sqrt((745*x[3]/(x[1]*x[2]))**2 + 16.9*10**6)/(0.1*x[5]**3)
        return np.array([f1, f2])

    def get_cons(self, x):
        g1 = 27 / (x[0]*x[2]*x[1]**2) - 1
        g2 = 397.5 / (x[0]*x[1]**2*x[2]**2) - 1
        g3 = 1.93*x[3]**3 / (x[1]*x[2]*x[5]**4) - 1
        g4 = 1.93*x[4]**3 / (x[1]*x[2]*x[6]**4) - 1
        g5 = x[1]*x[2] / 40 - 1
        g6 = x[0] / (12*x[1]) - 1
        g7 = 5*x[1] / x[0] - 1
        g8 = (1.5*x[5] + 1.9) / x[3] - 1
        g9 = (1.1*x[6] + 1.9) / x[4] - 1
        g10 = np.sqrt((745*x[3]/(x[1]*x[2]))**2 + 16.9*10**6)/(0.1*x[5]**3) - 1100 - 1
        g11 = np.sqrt((745*x[4]/(x[1]*x[2]))**2 + 157.5*10**6)/(0.1*x[6]**3) - 850 - 1
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])

    def amend_position(self, x, lb=None, ub=None):
        x[2] = int(x[2])
        return x

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class SpringProblem(Engineer):
    """
    x = [x1, x2, x3] = [d, D, N]
    """

    name = "Spring Design Problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 3
        self._n_objs = 2
        self._n_cons = 8
        self.x0 = [0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014,
              0.015, 0.0162, 0.0173, 0.018, 0.020, 0.023, 0.025,
              0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063,
              0.072, 0.080, 0.092, 0.0105, 0.120, 0.135, 0.148,
              0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263,
              0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.500]
        self.le = LabelEncoder()
        self.le.fit(self.x0)
        self._bounds = [(0, 41.99), (1.0, 30.0), (1, 32)]
        self.check_penalty_func(f_penalty)

    def amend_position(self, x, lb=None, ub=None):
        x[0] = self.le.inverse_transform([int(x[0])])
        x[2] = int(x[2])
        return x

    def get_objs(self, x):
        d, D, N = x
        G = 11.5 * 10 ** 6
        K = G * (d ** 4) / (8 * N * (D ** 3))
        lf = 1.05 * d * (N + 2) + 1000 / K
        C = D / d
        cf = (4 * C - 1) / (4 * C - 4) + 0.615 / C
        f1 =  0.25 * np.pi ** 2 * (d ** 2) * D * (N + 2)
        f2 = 8000 * cf * D / (np.pi * d ** 3)
        return np.array([f1, f2])

    def get_cons(self, x):
        G = 11.5 * 10 ** 6
        K = G * (x[0] ** 4) / (8 * x[2] * (x[1] ** 3))
        lf = 1.05 * x[0] * (x[2] + 2) + 1000 / K
        C = x[0] / x[0]
        cf = (4 * C - 1) / (4 * C - 4) + 0.615 / C
        f1 = 0.25 * np.pi ** 2 * (x[0] ** 2) * x[1] * (x[2] + 2)
        f2 = 8000 * cf * x[1] / (np.pi * x[0] ** 3)
        g1 = 0.25 * np.pi**2 * x[0]**2 * x[1]*(x[2] + 2) - 30
        g2 = f2 - 189000
        g3 = lf - 14
        g4 = 0.2 - x[0]
        g5 = x[0] + x[1] - 3
        g6 = 3 - C
        g7 = 300 / K - 6
        g8 = 1.25 - 700/K
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        if type(x[0]) != int:
            x = self.amend_position(x, self.lb, self.ub)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class HydrostaticThrustBearingProblem(Engineer):
    """
    x = [x1, x2, x3, x4] = [R, R0, mu, Q]
    """

    name = "Hydrostatic thrust bearing design problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 4
        self._n_objs = 2
        self._n_cons = 7
        self.gamma = 0.0307
        self.g = 386.4
        self.N = 750
        self.Ws = 101000
        self.Pmax = 1000
        self.DeltaTmax = 50
        self.hmin = 0.001
        self.C = 0.5
        self.C1 = 10.04
        self.n = -3.55
        self._bounds = [(1., 16.), (1., 16.), (1e-6, 16e-6), (1., 16.)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        R, R0, mu, Q = x
        P0 = 6 * mu * Q
        W = np.pi * P0 / 2
        P = (np.log10(np.log10(8.122 * 10**6 * mu + 0.8)) - self.C1) / self.n
        DeltaT = 2 * (10**P - 560)
        Ef = 9336 * Q * self.gamma * self.C * DeltaT
        h = (2*np.pi*self.N/60)**2 * 2*np.pi*mu / Ef * (R**4 - R0**4)/4
        f1 = 1. / 12 * (Q * P0 / 0.7 + Ef)
        f2 = self.gamma / (self.g * P0) * (Q / (2 * np.pi * R * h))
        return np.array([f1, f2])

    def get_cons(self, x):
        R, R0, mu, Q = x
        P0 = 6 * mu * Q * np.log(R / R0)
        W = np.pi * P0 / 2
        P = (np.log10(np.log10(8.122 * 10 ** 6 * mu + 0.8)) - self.C1) / self.n
        DeltaT = 2 * (10 ** P - 560)
        Ef = 9336 * Q * self.gamma * self.C * DeltaT
        h = (2 * np.pi * self.N / 60) ** 2 * 2 * np.pi * mu / Ef * (R ** 4 - R0 ** 4) / 4
        g1 = self.Ws - W
        g2 =  P0 - self.Pmax
        g3 = DeltaT - self.DeltaTmax
        g4 = self.hmin - h
        g5 = R0 - R
        g6 = self.gamma / (self.g * P0) * (Q / (2 * np.pi * R * h)) - 0.001
        g7 = W / (np.pi * (R ** 2 - R0 ** 2)) - 5000
        return np.array([g1, g2, g3, g4, g5, g6, g7])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class VibratingPlatformProblem(Engineer):
    """
    x = [d1, d2, d3, b, L] = [x0, x1, x2, x3, x4]

    Original Ref: On improving multiobjective genetic algorithms for design optimization
    """

    name = "Vibrating platform design problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 5
        self._n_objs = 2
        self._n_cons = 5
        # Define constants
        self.E1 = 70 * 10 ** 9
        self.E2 = 1.6 * 10 ** 9
        self.E3 = 200 * 10 ** 9
        self.rho1 = 2770
        self.rho2 = 100
        self.rho3 = 7780
        self.c1 = 1500
        self.c2 = 500
        self.c3 = 800
        self._bounds = [(0.05, 0.5), (0.2, 0.5), (0.2, 0.6), (0.35, 0.5), (3, 6)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        d1, d2, d3, b, L = x
        EI = (2*b/3)*(self.E1*d1**3 - self.E2*(d1**3 - d2**3) - self.E3*(d2**3 - d3**3))
        mu = 2*b*(self.rho1*d1 - self.rho2*(d1 - d2) - self.rho3*(d2 - d3))
        f1 = -np.pi/(2*L**2) * np.sqrt(EI / mu)
        f2 = 2 * b * L * (self.c1 * d1 - self.c2 * (d1 - d2) - self.c3 * (d2 - d3))
        return np.array([f1, f2])

    def get_cons(self, x):
        d1, d2, d3, b, L = x
        mu = 2 * b * (self.rho1 * d1 - self.rho2 * (d1 - d2) - self.rho3 * (d2 - d3))
        g1 = mu * L - 2800
        g2 = d1 - d2
        g3 = d2 - d1 - 0.15
        g4 = d2 - d3
        g5 = d3 - d2 - 0.01
        return np.array([g1, g2, g3, g4, g5])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class CarSideImpactProblem(Engineer):
    """
    x = [x1, x2, x3, x4, x5, x6, x7]

    Original Ref:
    """

    name = "Car side impact design problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 7
        self._n_objs = 3
        self._n_cons = 10
        self._bounds = [(0.5, 1.5), (0.45, 1.35), (0.5, 1.5), (0.5, 1.5), (0.875, 2.625), (0.4, 1.2), (0.4, 1.2)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        c1 = np.array([4.90, 6.67, 6.98, 4.01, 1.78, 1e-5, 2.73])
        V_mbp = 10.58 - 0.67275 * x[1] - 0.674 * x[0] * x[1]
        V_fd = 16.45 - 0.489 * x[2] * x[6] - 0.843 * x[4]*x[5]
        f1 = np.dot(c1, x) + 1.98
        f2 = 4.72 - 0.19 * x[1] * x[2] - 0.5*x[3]
        f3 = 0.5 * (V_mbp + V_fd)
        return np.array([f1, f2, f3])

    def get_cons(self, x):
        V_mbp = 10.58 - 0.67275 * x[1] - 0.674 * x[0] * x[1]
        V_fd = 16.45 - 0.489 * x[2] * x[6] - 0.843 * x[4] * x[5]
        g1 = 1.16 - 0.0092928 * x[2] - 0.3717 * x[1] * x[3] - 1
        g2 = 0.261 - 0.06486 * x[0] + 0.0154464 * x[5] - 0.0159 * x[0] * x[1] - 0.019 * x[1] * x[6] + 0.0144 * x[2] * x[4] - 0.32
        g3 = 0.214 - 0.0587118 * x[0] + 0.018 * x[1]**2 + 0.030408 * x[2] + 0.00817 * x[4] + 0.03099 * x[1] * x[5] - 0.018 * x[1] * x[6] - 0.00364 * x[4] * x[5] - 0.32
        g4 = 0.74 - 0.61 * x[1] + 0.227 * x[1]**2 - 0.031296 * x[2] - 0.031872 * x[6] - 0.32
        g5 = 28.98 + 3.818 * x[2] + 1.27296 * x[5] - 2.68065 * x[6] - 4.2 * x[0] * x[1] - 32
        g6 = 33.86 - 3.795 * x[1] + 2.95 * x[2] - 3.4431 * x[6] - 5.057 * x[0] * x[1] + 1.45728 - 32
        g7 = 46.36 - 4.4505 * x[0] - 9.9 * x[1] - 32
        g8 = 4.72 - 0.19 * x[1] * x[2] - 0.5*x[3] - 4
        g9 = V_mbp - 9.9
        g10 = V_fd - 15.7
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class WaterResourceManagementProblem(Engineer):
    """
    x =  [x1, x2, x3]

    Original Ref:
    """

    name = "Water resource management problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 3
        self._n_objs = 5
        self._n_cons = 7
        self._bounds = [(0.01, 0.45), (0.01, 0.1), (0.01, 0.1)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 106780.37*(x[1] + x[2]) + 61704.67
        f2 = 3000*x[0]
        f3 = 30570 * 2289*x[1]/ (0.06 * 2289)**0.65
        f4 = 250 * 2289 * np.exp(2.74 - 39.75*x[1] + 9.9*x[2])
        f5 = 25*(1.39/ (x[0]*x[1]) + 4940*x[2] - 80)
        return np.array([f1, f2, f3, f4, f5])

    def get_cons(self, x):
        t = x[0] * x[1]
        g1 = 4.94*x[2] + 0.00139/ t - 1.08
        g2 = 1.082*x[2] + 0.000306/ t - 1.0986
        g3 = 49408.24*x[2] + 12.307/t - 54051.02
        g4 = 8046.33*x[2] + 2.098 / t - 16696.71
        g5 = 7883.39*x[2] + 2.138/t - 10705.04
        g6 = 1721.26*x[2] + 0.417*t - 2136.54
        g7 = 631.13*x[2] + 0.164/t - 604.48
        return np.array([g1, g2, g3, g4, g5, g6, g7])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class BulkCarriersProblem(Engineer):
    """
    x = [L, B, D, T, Vk, CB] = [x1, x2, x3, x4, x5, x6]

    Original Ref:
    """

    name = "Bulk carriers design problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 6
        self._n_objs = 3
        self._n_cons = 9
        self._bounds = [(150, 274.32), (20, 32.31), (13, 25), (10, 11.71), (14., 18.), (0.63, 0.75)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        L, B, D, T, Vk, CB = x
        # Calculate intermediate variables
        Sd = 5000 * Vk / 24
        Ws = 0.034 * L ** 1.7 * B ** 0.7 * D ** 0.4 * CB ** 0.5
        Wo = L ** 0.8 * B ** 0.6 * D ** 0.3 * CB ** 0.1
        a = 4456.51 - 8105.61 * CB + 4977.06 * CB ** 2
        b = -6960.32 + 12817 * CB - 10847.2 * CB ** 2
        Fn = 0.5144 * Vk / (9.8065 * L) ** 0.5
        P = (1.025 * L * B * T * CB) ** (2. / 3) * Vk ** 3 / (a + b * Fn)
        Wm = 0.17 * P ** 0.9
        Wls = Ws + Wo + Wm
        Dwt = 1.025 * L * B * T * CB - Wls
        Dc = 0.19 * 24 * P / 1000 + 0.2
        Dcwt = Dwt - Dc * (Sd + 5) - 2 * Dwt ** 0.5
        Rtpa = 350 / (Sd + 2 * Dcwt / 8000 + 2 * 0.5)
        Ca = Dcwt * Rtpa
        Cv = Rtpa * (105 * Dc * Sd + 6.3 * Dwt ** 0.8)
        Cr = 40000 * Dwt ** 0.3
        Cc = 2.6 * (2000 * Ws ** 0.85 + 3500 * Wo + 2400 * P ** 0.8)
        # Calculate objective functions
        f1 = (Cc + Cr + Cv) / Ca
        f2 = Wls
        f3 = -Ca
        return np.array([f1, f2, f3])

    def get_cons(self, x):
        L, B, D, T, Vk, CB = x
        Ws = 0.034 * L ** 1.7 * B ** 0.7 * D ** 0.4 * CB ** 0.5
        Wo = L ** 0.8 * B ** 0.6 * D ** 0.3 * CB ** 0.1
        a = 4456.51 - 8105.61 * CB + 4977.06 * CB ** 2
        b = -6960.32 + 12817 * CB - 10847.2 * CB ** 2
        Fn = 0.5144 * Vk / (9.8065 * L) ** 0.5
        P = (1.025 * L * B * T * CB) ** (2. / 3) * Vk ** 3 / (a + b * Fn)
        Wm = 0.17 * P ** 0.9
        Wls = Ws + Wo + Wm
        Dwt = 1.025 * L * B * T * CB - Wls
        g1 = -L / B + 6
        g2 = L / D - 15
        g3 = -L / T - 19
        g4 = T - 0.45 * Dwt ** 0.31
        g5 = T - 0.7 * D - 0.7
        g6 = Fn - 0.32
        g7 = -0.53 * T - ((0.085 * CB - 0.002) * B ** 2) / (T * CB) + (1 + 0.52 * D) + 0.07 * B
        g8 = -Dwt + 3000
        g9 = Dwt - 500000
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class MultiProductBatchPlantProblem(Engineer):
    """
    x = [N1, N2, N3, V1, V2, V3, TL1, TL2, B1, B2] = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

    Original Ref:
    """

    name = "Multi-product batch plant problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 10
        self._n_objs = 3
        self._n_cons = 3
        self.N = 2
        self.M = 3
        self.alpha = np.array([250, 250, 250])
        self.beta = np.array([0.6, 0.6, 0.6])
        self.H = 6000
        self.Q = np.array([40000, 20000])
        self.S = np.array([[2, 3, 4], [4, 6, 3]])
        self.t = np.array([[8, 20, 8], [16, 4, 4]])
        self._bounds = [(1, 3.99)] * self.N + [(250, 2500)] * self.M + [(6, 20), (4, 16), (40, 700), (10, 450)]
        self.check_penalty_func(f_penalty)

    def amend_position(self, x, lb=None, ub=None):
        x[0] = int(x[0])
        x[1] = int(x[1])
        x[2] = int(x[2])
        return x

    def get_objs(self, x):
        f1 = np.sum(self.alpha * x[:self.M] * (x[self.M:2*self.M] ** self.beta))
        f2 = 65 * (self.Q[0]/x[8] + self.Q[1]/x[9]) + 0.08*self.Q[0] + 0.1*self.Q[1]
        f3 = self.Q[0] * x[6] / x[8] + self.Q[2] * x[7] / x[9]
        return np.array([f1, f2, f3])

    def get_cons(self, x):
        g1 = self.Q[0] * x[6] / x[8] + self.Q[2] * x[7] / x[9] - self.H
        b = np.array(x[-2:])
        v = np.array(x[self.M:2*self.M])
        g2 = np.sum(b*self.S - v)
        g3 = np.sum([self.t[i,j] - x[j] * x[2*self.M+i+self.N] for i in range(self.N) for j in range(self.N, 2*self.M)])
        return np.array([g1, g2, g3])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


SRP = SpeedReducerProblem
SP = SpringProblem
HTBP = HydrostaticThrustBearingProblem
VPP = VibratingPlatformProblem
CSP = CarSideImpactProblem
WRMP = WaterResourceManagementProblem
BCP = BulkCarriersProblem
MPBPP = MultiProductBatchPlantProblem
