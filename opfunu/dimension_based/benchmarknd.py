#!/usr/bin/env python                                                                                   #
# ------------------------------------------------------------------------------------------------------#
# Created by "Thieu Nguyen" at 17:44, 06/12/2019                                                        #
#                                                                                                       #
#       Email:      nguyenthieu2102@gmail.com                                                           #
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  #
#       Github:     https://github.com/thieu1995                                                  #
#-------------------------------------------------------------------------------------------------------#

import numpy as np

class Functions:
    """
        This class of functions is belongs to n-dimensional space
    """

    def _ackley__(self, solution=None, a=20, b=0.2, c=2 * np.pi):
        """
        Class: multimodal, continuous, non-convex, differentiable, n-dimensional space.
        Global: one global minimum fx = 0, at [0, 0,...0]

        @param solution: A numpy array like: [1, 2, 10, 4, ...]
        @return: fx
        """
        result1 = np.sum(solution ** 2)
        result2 = np.sum(np.cos(c * solution))
        lin = 1 / len(solution)
        return -a * np.exp(-b * np.sqrt(lin * result1)) - np.exp(lin * result2) + a + np.exp(1)


    def _ackley_n4__(self, solution=None):
        """
        Class: multimodal, non-convex, differentiable, non-separable, n-dimensional space.
        Global: on 2-d space, 1 global min fx = -4.590101633799122, at [−1.51, −0.755]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/ackleyn4fcn.html

        @param solution: A numpy array include 2 items like: [-35, 35, -35, ...]
        """
        d = len(solution)
        score = 0.0
        for i in range(0, d-1):
            score += ( np.exp(-0.2*np.sqrt(solution[i]**2 + solution[i+1]**2)) + 3*(np.cos(2*solution[i]) + np.sin(2*solution[i+1])) )
        return score


    def _alpine_n1__(self, solution=None):
        """
        Class: multimodal, non-convex, differentiable, non-separable, n-dimensional space.
        Global: one global minimum fx = 0, at [0, 0,...0]

        @param solution: A numpy array like: [1, 2, 10, 4, ...]
        @return: fx
        """
        return np.sum(np.abs(solution*np.sin(solution) + 0.1 * solution))


    def _alpine_n2__(self, solution=None):
        """
        Class: multimodal, non-convex, differentiable, non-separable, n-dimensional space.
        Global: one global minimum fx = 2.808^n, at [7.917, ..., 7.917]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/alpinen2fcn.html

        @param solution: A numpy array like: [1, 2, 10, 4, ...]
        @return: fx
        """
        return np.prod(np.sqrt(solution)*np.sin(solution))


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
        for i in range(0, d-1):
            result += (solution[i]**2)**(solution[i+1]**2+1) + (solution[i+1]**2)**(solution[i]**2+1)
        return result


    def _exponential__(self, solution=None):
        """
        Class: uni-modal, convex, differentiable, non-separable, continuous
        Global: one global minimum fx = 0, at [0,...,0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/exponentialfcn.html

        @param solution: A numpy array with x_i in [-1, 1]
        @return: fx
        """
        return -np.exp(0-.5*np.sum(solution**2))


    def _griewank__(self, solution=None):
        """
        Class: uni-modal, non-convex, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/griewankfcn.html

        @param solution: A numpy array with x_i in [-600, 600]
        @return: fx
        """
        d = len(solution)
        result = 1 + np.sum(solution**2) / 4000
        prod = 1.0
        for i in range(0, d):
            prod *= np.cos(solution[i]/np.sqrt(i+1))
        return result - prod


    def _happy_cat__(self, solution=None, alpha=1.0/8):
        """
        Class: multimodal, non-convex, differentiable, non-separable, parametric
        Global: one global minimum fx = 0, at [-1, ..., -1]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/happycatfcn.html

        @param solution: A numpy array with x_i in [-2, 2]
        @return: fx
        """

        return ((np.sum(solution**2) - len(solution))**2)**alpha + (0.5*np.sum(solution**2)+np.sum(solution))/len(solution) + 0.5


    def _periodic__(self, solution=None):
        """
        Class: multimodal, non-convex, differentiable, non-separable, continuous
        Global: one global minimum fx = 0.9, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/periodicfcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        return 1 + np.sum(np.sin(solution)**2) - 0.1*np.exp(np.sum(solution**2))


    def _powell_sum__(self, solution=None):
        """
        Class: uni-modal, convex, non-differentiable, separable, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/powellsumfcn.html

        @param solution: A numpy array with x_i in [-1, 1]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            result += np.abs(solution[i])**(i+2)
        return result


    def _qing__(self, solution=None):
        """
        Class: multimodal, non-convex, differentiable, non-separable, continuous
        Global: one global minimum fx = 0, at (±√i,…,±√i)
        Link: http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html

        @param solution: A numpy array with x_i in [-500, 500]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            result += (solution[i]**2 - i - 1)**2
        return result


    def _quartic__(self, solution=None):
        """
        Class: multimodal, non-convex, differentiable, separable, continuous, random
        Global: one global minimum fx = 0 + random, at (0, ...,0)
        Link: http://benchmarkfcns.xyz/benchmarkfcns/quarticfcn.html

        @param solution: A numpy array with x_i in [-1.28, 1.28]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            result+= (i+1)*solution[i]**4
        return result+np.random.uniform(0, 1)


    def _rastrigin__(self, solution=None):
        """
        Class: multimodal, convex, differentiable, separable, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/rastriginfcn.html

        @param solution: A numpy array with x_in in [-5.12, 5.12]
        @return: fx
        """
        return 10*len(solution) + np.sum(solution**2-10*np.cos(2*solution*np.pi))


    def _ridge__(self, solution=None, d=2, alpha=0.5):
        """
        Class: uni-model, non-convex, differentiable, non-separable
        Global:
        Link: http://benchmarkfcns.xyz/benchmarkfcns/ridgefcn.html

        @param solution: A numpy array with x_i in [-5, 5]
        @return: fx
        """
        t1 = solution[1:]
        return solution[0] + d*np.sum(t1**2)**alpha


    def _rosenbrock__(self, solution=None, a=1, b=100):
        """
        Class: multimodal, non-convex, differentiable, non-separable, continuous
        Global: 1 global optima, fx = 0, x = [1, ..., 1]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html

        @param solution: A numpy array with x_i in [-5, 10]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d-1):
            result += b*(solution[i+1] - solution[i]**2)**2 + (a-solution[i])**2
        return result


    def _salomon__(self, solution=None):
        """
        Class: multimodal, non-convex, differentiable, non-separable, continuous
        Global: 1 global optima, fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/salomonfcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return 1 - np.cos(2*np.pi*np.sqrt(np.sum(solution**2))) + 0.1*np.sqrt(np.sum(solution**2))


    def _schwefel_2_20__(self, solution=None):
        """
        Class: uni-modal, convex, non-differentiable, separable, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel220fcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.sum(np.abs(solution))


    def _schwefel_2_21__(self, solution=None):
        """
        Class: uni-modal, convex, non-differentiable, separable, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel221fcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.max(np.abs(solution))


    def _schwefel_2_22__(self, solution=None):
        """
        Class: uni-modal, convex, non-differentiable, separable, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel222fcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return np.sum(np.abs(solution)) + np.prod(np.abs(solution))


    def _schwefel_2_23__(self, solution=None):
        """
        Class: uni-modal, convex, differentiable, separable, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefel221fcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        return np.sum(solution**10)


    def _schwefel__(self, solution=None):
        """
        Class: multi-modal, non-convex, non-differentiable, non-separable, continuous
        Global: one global minimum fx = 0, at [420.9687, ..., 420.9687]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schwefelfcn.html

        @param solution: A numpy array with x_i in [-500, 500]
        @return: fx
        """
        return 418.9829*len(solution) - np.sum(solution*np.sin(np.sqrt(np.abs(solution))))


    def _shubert_3__(self, solution=None):
        """
        Class: multi-modal, non-convex, differentiable, separable, continuous
        Global: one global minimum fx = -29.6733337
        Link: http://benchmarkfcns.xyz/benchmarkfcns/shubert3fcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            for j in range(1, 6):
                result+= j*np.sin((j+1)*solution[i] + j)
        return result

    def _shubert_4__(self, solution=None):
        """
        Class: multi-modal, non-convex, differentiable, separable, continuous
        Global: one global minimum fx = -25.740858
        Link: http://benchmarkfcns.xyz/benchmarkfcns/shubert4fcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            for j in range(1, 6):
                result += j * np.cos((j + 1) * solution[i] + j)
        return result


    def _shubert__(self, solution=None):
        """
        Class: multi-modal, non-convex, differentiable, non-separable, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/shubertfcn.html

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        d = len(solution)
        prod = 1.0
        for i in range(0, d):
            result = 0
            for j in range(1, 6):
                result += np.cos((j + 1) * solution[i] + j)
            prod *= result
        return prod


    def _sphere__(self, solution=None):
        """
        Class: uni-modal, convex, differentiable, separable, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/spherefcn.html

        @param solution: A numpy array with x_i in [-5.12, 5.12]
        @return: fx
        """
        return np.sum(solution**2)


    def _styblinski__(self, solution=None):
        """
        Class: multi-modal, non-convex, continuous
        Global: one global minimum fx = -39.16599 * d , at [-2.903534, ..., -2.903534]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/styblinskitankfcn.html

        @param solution: A numpy array with x_i in [-5, 5]
        @return: fx
        """
        return 0.5*np.sum(solution**4 - 16*solution**2 + 5*solution)


    def _sum_squres__(self, solution=None):
        """
        Class: uni-modal, convex, differentiable, separable, continuous
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/sumsquaresfcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        d = len(solution)
        result = 0.0
        for i in range(0, d):
            result = (i+1)*solution[i]**2
        return result


    def _xin_she_yang__(self, solution=None):
        """
        Class: multi-modal, non-convex, non-differentiable, separable, random
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/xinsheyangn1fcn.html

        @param solution: A numpy array with x_i in [-5, 5]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            result += np.random.uniform(0, 1) * np.abs(solution[i])**(i+1)
        return result


    def _xin_she_yang_n2__(self, solution=None):
        """
        Class: multi-modal, non-convex, non-differentiable, non-separable
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/xinsheyangn2fcn.html

        @param solution: A numpy array with x_i in [-2pi, 2pi]
        @return: fx
        """
        return np.sum(np.abs(solution))*np.exp(-np.sum(np.sin(solution**2)))


    def _xin_she_yang_n3__(self, solution=None, m=5, beta=15):
        """
        Class: uni-modal, non-convex, differentiable, non-separable, parametric
        Global: one global minimum fx = -1, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/xinsheyangn3fcn.html

        @param solution: A numpy array with x_i in [-2pi, 2pi]
        @return: fx
        """
        t1 = np.exp(-np.sum( np.power(solution/beta, 2*m)))
        t2 = -2*np.exp(-np.sum(solution**2))
        t3 = np.prod(np.cos(solution)**2)
        return t1 + t2*t3


    def _xin_she_yang_n4__(self, solution=None):
        """
        Class: multi-modal, non-convex, non-differentiable, non-separable
        Global: one global minimum fx = -1, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/xinsheyangn4fcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        t1 = np.sum(np.sin(solution)**2)
        t2 = -np.exp(-np.sum(solution**2))
        t3 = -np.exp(np.sum(np.sin(np.sqrt(np.abs(solution)))**2))
        return (t1 + t2) * t3


    def _zakharov__(self, solution=None):
        """
        Class: uni-modal, convex, continuous
        Global: one global minimum fx = -1, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/zakharov.html

        @param solution: A numpy array with x_i in [-5, 10]
        @return: fx
        """
        t1 = np.sum(solution**2)
        t2 = 0
        d = len(solution)
        for i in range(0, d):
            t2 += 0.5*(i+1)*solution[i]
        return t1 + t2**2 + t2**4






