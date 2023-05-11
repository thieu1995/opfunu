#!/usr/bin/env python
# Created by "Thieu" at 16:25, 09/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# Paper: IHAOAVOA: An improved hybrid aquila optimizer and African vultures optimization algorithm for global optimization problems

import numpy as np
from opfunu.engineer import Engineer


class TensionCompressionSpringProblem(Engineer):
    """
    x = [x1, x2, x3] = [d, D, N]
    """

    name = "Tension/compression spring design problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 3
        self._n_objs = 1
        self._n_cons = 4
        self._bounds = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = (x[2] + 2) * x[1]*x[0]**2
        return np.array([f1, ])

    def get_cons(self, x):
        g1 = 1 - (x[1]**3 * x[2]) / (71785 * x[0]**4)
        g2 = (4*x[1]**2 - x[0]*x[1])/(12566 * (x[1]*x[0]**3 - x[0]**4)) + 1. / (5108 * x[0]**2)
        g3 = 1 - 140.45*x[0] / (x[1]**2 * x[2])
        g4 = (x[0] + x[1]) / 1.5 - 1
        return np.array([g1, g2, g3, g4])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class WeldedBeamProblem(Engineer):
    """
    x = [x1, x2, x3, x4] = [h, l, t, b]
    """

    name = "Welded beam design problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 4
        self._n_objs = 1
        self._n_cons = 7
        self._bounds = [(0.1, 2.0), (0.1, 10.), (0.1, 10.), (0.1, 2.0)]
        self.L = 14
        self.E = 30*10**6
        self.G = 12*10**6
        self.theta_max = 0.25
        self.tau_max = 13600
        self.xichma_max = 30000
        self.P = 6000
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 1.10471 * x[0]**2 * x[1] + 0.04811 * x[2] * x[3] * (14 + x[1])
        return np.array([f1, ])

    def get_cons(self, x):
        Pc = 4.013*self.E*np.sqrt(x[2]**2 * x[3]**6 / 36) / self.L**2 * (1 - x[2]*np.sqrt(self.E/(4*self.G)) / (2*self.L))
        theta_z = 6 * self.P * self.L**3 / (self.E * x[2]**2 * x[3])
        xichma_z = 6*self.P*self.L / (self.E * x[2]**2 * x[3])
        jj = 2*np.sqrt(2)*x[0]*x[1]*(x[1]**2 / 4 + ((x[0] + x[2])/2)**2)
        R = np.sqrt(x[1]**2/4 + (x[0]+x[2])**2 / 4)
        M = self.P * (self.L + x[1]/2)
        tau2 = M * R / jj
        tau1 = self.P / (np.sqrt(2) * x[0]*x[1])
        tau = np.sqrt(tau1**2 + 2*tau1*tau2*x[1]/(2*R) + tau2**2)

        g1 = tau - self.tau_max
        g2 = xichma_z - self.xichma_max
        g3 = theta_z - self.theta_max
        g4 = x[0] - x[3]
        g5 = self.P - Pc
        g6 = 0.125 - x[0]
        g7 = 1.10471 * x[0]**2 + 0.04811 * x[2]*x[3]*(14 + x[1]) - 5
        return np.array([g1, g2, g3, g4, g5, g6, g7])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class CantilevelBeamProblem(Engineer):
    """
    x = [x1, x2, x3, x4, x5]
    """

    name = "Cantilever beam design problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 5
        self._n_objs = 1
        self._n_cons = 1
        self._bounds = [(0.01, 100.0), ] * 5
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 0.6224 * np.sum(x)
        return np.array([f1, ])

    def get_cons(self, x):
        g1 = 61 / x[0]**3 + 27/x[1]**3 + 19/x[2]**3 + 7/x[3]**3 + 1/x[4]**3
        return np.array([g1, ])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class SpeedReducerProblem(Engineer):
    """
    x = [x1, x2, x3, x4, x5, x6, x7]

    Ref: https://www.hindawi.com/journals/mpe/2013/419043/
    """

    name = "Speed reducer design problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 7
        self._n_objs = 1
        self._n_cons = 11
        self._bounds = [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.8, 8.3), (2.9, 3.9), (5.0, 5.5)]
        self.check_penalty_func(f_penalty)

    def get_objs(self, x):
        f1 = 0.7584*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2]-43.0934) - 1.508*x[0]*(x[5]**2+x[6]**2) + 7.4777*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)
        return np.array([f1, ])

    def get_cons(self, x):
        g1 = 27/(x[0]*x[1]**2*x[2]) - 1
        g2 = 397.5 / (x[0]*x[1]**2*x[2]**2) - 1
        g3 = 1.93*x[3]**3/ (x[1] * x[2]*x[5]**4) - 1
        g4 = 1.93 * x[4]**3 / (x[1] * x[2] * x[6]**4) - 1
        g5 = np.sqrt((745*x[3]/(x[1]*x[2]))**2 + 16.9*10**6) / (110*x[5]**3) - 1
        g6 = np.sqrt((745*x[4]/(x[1]*x[2]))**2 - 157.5*10**6) / (85 * x[6]**3) - 1
        g7 = x[1]*x[2]/40 - 1
        g8 = 5*x[1]/x[0] - 1
        g9 = x[0]/(12 * x[1]) - 1
        g10 = (1.5*x[5] + 1.9) / x[3] - 1
        g11 = (1.1*x[6] + 1.9) / x[4] -1
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)


class RollingElementBearingProblem(Engineer):
    """
    x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10] = [Dm, Db, Z, fi, f0, Kdmin, Kdmax, theta, e, C]
    """

    name = "Rolling element bearing design problem"

    def __init__(self, f_penalty=None):
        super().__init__()
        self._n_dims = 10
        self._n_objs = 1
        self._n_cons = 9
        self.D = 160
        self.d = 90
        self.Bw = 30
        self.ri = self.ro = 11.033
        self._bounds = [(125., 150.), (10.5, 31.5), (4, 50), (0.515, 0.6), (0.515, 0.6), (0.4, 0.5), (0.6, 0.7), (0.3, 0.4), (0.02, 0.1), (0.6, 0.85)]
        self.check_penalty_func(f_penalty)

    def amend_position(self, x, lb=None, ub=None):
        x[2] = int(x[2])
        return x

    def get_objs(self, x):
        gama = x[1] / x[0]
        t1 = 37.91*(1+(1.04*((1-gama)/(1+gama))**1.72*(x[3]/x[4] * (2*x[4] - 1)/(2*x[3] - 1))**0.41)**(10./3))**(-0.3)
        fc = t1 * (gama**0.3 *(1 - gama)**1.39 / ((1+gama)**(1./3))) * (2*x[3]/(2*x[3] - 1))**0.41
        if x[1] <= 25.4:
            f1 = fc * x[2]**(2./3) * x[1]**1.8
        else:
            f1 = 3.647 * fc * x[2]**(2./3) * x[1]**1.4
        return np.array([f1, ])

    def get_cons(self, x):
        T = self.D - self.d - 2*x[1]
        xx = ((self.D - self.d) / 2 - 3 * (T/4))**2 + (self.D/2 - T/4 - x[1])**2 - (self.d/2 + T/4)**2
        yy = 2*((self.D - self.d)/2 - 3*T/4)*(self.D/2 - T/4 - x[1])
        theta0 = 2*np.pi - 1. / np.cos(xx / yy)

        g1 = theta0 / (2. / np.sin(x[1] / x[0])) - x[2] + 1
        g2 = x[5]*(self.D - self.d) - 2*x[1]
        g3 = 2*x[1] - x[6]*(self.D - self.d)
        g4 = x[-1]*self.Bw - x[1]
        g5 = 0.5*(self.D + self.d) - x[0]
        g6 = x[0] - (0.5 + x[-2])*(self.D + self.d)
        g7 = x[-3]*x[1] - 0.5*(self.D - x[0] - x[1])
        g8 = 0.515 - x[3]
        g9 = 0.515 - x[4]
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9])

    def evaluate(self, x):
        self.n_fe += 1
        self.check_solution(x)
        list_objs = self.get_objs(x)
        list_cons = self.get_cons(x)
        return self.f_penalty(list_objs, list_cons)
