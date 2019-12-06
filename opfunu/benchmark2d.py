#!/usr/bin/env python                                                                                   #
# ------------------------------------------------------------------------------------------------------#
# Created by "Thieu Nguyen" at 02:54, 06/12/2019                                                        #
#                                                                                                       #
#       Email:      nguyenthieu2102@gmail.com                                                           #
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  #
#       Github:     https://github.com/thieunguyen5991                                                  #
#-------------------------------------------------------------------------------------------------------#

import numpy as np

class Functions:
    """
        This class of functions is belongs to 2-dimensional space
    """

    def _ackley_n2__(self, solution=None):
        """
        Class: unimodal, convex, differentiable, non-separable
        Global: one global minimum fx = -200, at [0, 0]

        @param solution: A numpy array include 2 items like: [10, 22]
        """
        n = len(solution)
        assert (n == 2, 'Ackley N. 2 function is only defined on a 2D space.')
        return -200*np.exp(-0.2*np.sqrt(np.sum(solution**2)))


    def _ackley_n3__(self, solution=None):
        """
        Class: multimodal, non-convex, differentiable, non-separable
        Global: one global minimum fx = −195.629028238419, at [±0.682584587365898,−0.36075325513719]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/ackleyn3fcn.html

        @param solution: A numpy array include 2 items like: [10, 22]
        """
        d = len(solution)
        assert (d == 2, 'Ackley N. 3 function is only defined on a 2D space.')
        return -200*np.exp(-0.2*np.sqrt(np.sum(solution**2))) + 5*np.exp(np.cos(3*solution[0]) + np.sin(3*solution[1]))


    def _adjiman__(self, solution=None):
        """
        Class: multimodal, non-convex, differentiable, non-separable
        Global: if x in [-1, 2], y in [-1, 1] cube => global min fx = -2.02181, at [0, 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/adjimanfcn.html

        @param solution: A numpy array include 2 items like: [10, 22]
        """
        d = len(solution)
        assert (d == 2, 'Adjiman function is only defined on a 2D space.')
        return np.cos(solution[0]) * np.sin(solution[1]) - solution[0] / (solution[1]**2 + 1)

    def _bartels_conn__(self, solution=None):
        """
        Class: multimodal, non-convex, non-differentiable, non-separable
        Global: one global minimum fx = 1, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/bartelsconnfcn.html

        @param solution: A numpy array include 2 items like: [10, 22]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Bartels conn function is only defined on a 2D space.')
        return np.abs(solution[0]**2 + solution[1]**2 + solution[0] * solution[1]) + np.abs(np.sin(solution[0])) + \
               np.abs(np.cos(solution[1]))

    def _beale__(self, solution=None):
        """
        Class: multimodal, non-convex, continuous
        Global: one global minimum fx = 0, at [3, 0.5]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/bealefcn.html

        @param solution: A numpy array include 2 items in range: [-4.5, 4.5], [-4.5, 4.5]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Beale function is only defined on a 2D space.')
        return (1.5-solution[0]+solution[0]*solution[1])**2 + (2.25-solution[0]+solution[0]*solution[1]**2)**2 +\
               (2.625-solution[0]+solution[0]*solution[1]**3)**2

    def _bird__(self, solution=None):
        """
        Class: multimodal, non-convex, non-separable, differentiable
        Global: 2 global minimum fx= -106.764537, at ( 4.70104 , 3.15294 ) and ( − 1.58214 , − 3.13024 ) .
        Link: http://benchmarkfcns.xyz/benchmarkfcns/birdfcn.html

        @param solution: A numpy array include 2 items in range: [-2pi, 2pi], [-2pi, 2pi]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Bird function is only defined on a 2D space.')
        return np.sin(solution[0])*np.exp((1 - np.cos(solution[1]))**2) + \
            np.cos(solution[1]) * np.exp( (1-np.sin(solution[0]))**2 ) + (solution[0] - solution[1])**2


