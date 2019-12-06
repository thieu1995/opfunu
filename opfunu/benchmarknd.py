#!/usr/bin/env python                                                                                   #
# ------------------------------------------------------------------------------------------------------#
# Created by "Thieu Nguyen" at 17:44, 06/12/2019                                                        #
#                                                                                                       #
#       Email:      nguyenthieu2102@gmail.com                                                           #
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  #
#       Github:     https://github.com/thieunguyen5991                                                  #
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
        sum1 = np.sum(solution ** 2)
        sum2 = np.sum(np.cos(c * solution))
        lin = 1 / len(solution)
        return -a * np.exp(-b * np.sqrt(lin * sum1)) - np.exp(lin * sum2) + a + np.exp(1)


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


