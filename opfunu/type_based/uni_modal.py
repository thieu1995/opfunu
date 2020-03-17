#!/usr/bin/env python                                                                                   #
# ------------------------------------------------------------------------------------------------------#
# Created by "Thieu Nguyen" at 16:32, 07/12/2019                                                        #
#                                                                                                       #
#       Email:      nguyenthieu2102@gmail.com                                                           #
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  #
#       Github:     https://github.com/thieunguyen5991                                                  #
#-------------------------------------------------------------------------------------------------------#

import numpy as np

class Functions:
    """
        This class of functions is belongs to uni-modal function, all functions will be scaled to n-dimension space
    """

    def _brown__(self, solution=None):
        """
        Class: uni-modal, convex, differentiable, non-separable
        Global: one global minimum fx = 0 at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/brownfcn.html

        @param solution: A numpy array with x_i in [-1, 4]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d - 1):
            result += (solution[i] ** 2) ** (solution[i + 1] ** 2 + 1) + (solution[i + 1] ** 2) ** (solution[i] ** 2 + 1)
        return result


    def _chung_reynolds__(self, solution=None):
        """
        Class: continuous, differentiable, partially-separable, scalable
        Global: one global minimum fx = 0 at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/brownfcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.sum(solution**2)**2


    def _dixon_price__(self, solution=None):
        """
        Class: continuous, differentiable, non-separable, scalable
        Global: one global minimum fx = 0
        Link: http://benchmarkfcns.xyz/benchmarkfcns/brownfcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        d = len(solution)
        result = (solution[0] - 1)**2
        for i in range(1, d):
            result += (i+1)*(2*solution[i]**2 - solution[i-1])**2
        return result


    def _powell_singular_2__(self, solution=None):
        """
        Class: continuous, differentiable, non-separable, scalable
        Global: one global minimum fx = 0
        Link: http://benchmarkfcns.xyz/benchmarkfcns/brownfcn.html

        @param solution: A numpy array with x_i in [-4, 5]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(1, d-2):
            result += (solution[i-1]+10*solution[i])**2 + 5*(solution[i+1]-solution[i+2])**2 + \
                   (solution[i]-2*solution[i+1])**4 + 10 * (solution[i-1]-solution[i+2])**4
        return result


    def _powell_result__(self, solution=None):
        """
        Class: continuous, differentiable, separable, scalable
        Global: one global minimum fx = 0
        Link: http://benchmarkfcns.xyz/benchmarkfcns/brownfcn.html

        @param solution: A numpy array with x_i in [-1, 1]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            result += np.abs(solution[i])**(i+2)
        return result


    def _rosenbrock__(self, solution=None):
        """
        Class: non-convex, differentiable, non-separable, continuous
        Global: 1 global optima, fx = 0, x = [1, ..., 1]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html

        @param solution: A numpy array with x_i in [-30, 30]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d-1):
            result += 100*(solution[i+1] - solution[i]**2)**2 + (solution[i]-1)**2
        return result


    def _rotate_ellipse__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Scalable
        Global: 1 global optima, fx = 0, x = [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html

        @param solution: A numpy array
        @return: fx
        """
        return np.sum(solution**4)


    def _schwefel__(self, solution=None, alpha=0.5):
        """
        Class: Continuous, Differentiable, Partially-Separable, Scalable
        Global: 1 global optima, fx = 0, x = [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.sum(solution**2)**alpha


    def _schwefel_1_2__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: 1 global optima, fx = 0, x = [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            result+= np.sum(solution[:i]**2)
        return result


    def _schwefel_2_20__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: 1 global optima, fx = 0, x = [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return -np.sum(np.abs(solution))


    def _schwefel_2_21__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: 1 global optima, fx = 0, x = [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.max(np.abs(solution))


    def _schwefel_2_22__(self, solution=None):
        """
        Class: convex, non-differentiable, separable, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel222fcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.sum(np.abs(solution)) + np.prod(np.abs(solution))


    def _schwefel_2_23__(self, solution=None):
        """
        Class: differentiable, non-separable, continuous, scalable
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel222fcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        return np.sum(solution**10)


    def _step__(self, solution=None):
        """
        Class: Discontinuous, Non-Differentiable, Separable, Scalable
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel222fcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.sum(np.floor(np.abs(solution)))


    def _step_2__(self, solution=None):
        """
        Class: Discontinuous, Non-Differentiable, Separable, Scalable
        Global: one global minimum fx = 0, at [-0.5, ..., -0.5]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel222fcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.sum(np.floor(solution+0.5)**2)


    def _step_3__(self, solution=None):
        """
        Class: Discontinuous, Non-Differentiable, Separable, Scalable
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel222fcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.sum(np.floor(solution**2))


    def _stepint__(self, solution=None):
        """
        Class: Discontinuous, Non-Differentiable, Separable, Scalable
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel222fcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.sum(np.floor(solution**2))


    def _streched_v_sin_wave__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel222fcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d-1):
            result+= (solution[i+1]**2 + solution[i]**2)**0.25 * (np.sin(50*(solution[i+1]**2+solution[i]**2)**0.1)**2+0.1)
        return result


    def _sum_squres__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Scalable
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/resultsquaresfcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        d = len(solution)
        result = 0.0
        for i in range(0, d):
            result = (i+1)*solution[i]**2
        return result






















