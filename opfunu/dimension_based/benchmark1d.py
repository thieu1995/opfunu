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
        This class of functions is belongs to 1-dimensional space
    """

    def _gramacy_lee__(self, solution=None):
        """
        Class: uni-modal, non-convex, continuous
        Global: 1 global minimum fx = −0.869011134989500, atx 0.548563444114526
        Link: http://benchmarkfcns.xyz/benchmarkfcns/gramacyleefcn.html

        @param solution: A numpy array include 1 items like: [0.5], limited range: [-0.5, 2.5]

        """
        n = len(solution)
        assert (n == 1, 'Gramacy and Lee function is only defined on a 1D space.')
        return np.sin(10*np.pi*solution) / (2*solution) + (solution-1)**4
