#!/usr/bin/env python                                                                                   #
# ------------------------------------------------------------------------------------------------------#
# Created by "Thieu Nguyen" at 02:52, 07/12/2019                                                        #
#                                                                                                       #
#       Email:      nguyenthieu2102@gmail.com                                                           #
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  #
#       Github:     https://github.com/thieu1995                                                  #
#-------------------------------------------------------------------------------------------------------#


import numpy as np

class Functions:
    """
        This class of functions is belongs to 3-dimensional space
    """

    def _wolfe__(self, solution=None):
        """
        Class: multi-modal, non-convex, continuous, differentiable, non-separable
        Global: 1 global minimum fx = 0, [0, 0, 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/wolfefcn.html

        @param solution: A numpy array include 3 items like: [0.2, 0.22, 0.5], limited range: [0, 2]
        """
        n = len(solution)
        assert (n == 3, 'Wolfe function is only defined on a 3D space.')
        return 4/3 * (solution[0]**2 + solution[1]**2 - solution[0]*solution[1])**0.75 + solution[2]
