#!/usr/bin/env python                                                                                   #
# ------------------------------------------------------------------------------------------------------#
# Created by "Thieu Nguyen" at 02:54, 06/12/2019                                                        #
#                                                                                                       #
#       Email:      nguyenthieu2102@gmail.com                                                           #
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  #
#       Github:     https://github.com/thieu1995                                                  #
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

    def _bohachevskyn_n1__(self, solution=None):
        """
        Class: unimodal, convex, continuous
        Global: global minimum fx= 0, at ( 0, 0 )
        Link: http://benchmarkfcns.xyz/benchmarkfcns/bohachevskyn1fcn.html

        @param solution: A numpy array include 2 items in range: [-100, 100], [-100, 100]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Bohachevskyn N.1 function is only defined on a 2D space.')
        return solution[0]**2 + 2*solution[1]**2 - 0.3*np.cos(3*solution[0]*np.pi) - 0.4*np.cos(4*solution[1]*np.pi) + 0.7


    def _bohachevskyn_n2__(self, solution=None):
        """
        Class: multi-modal, non-convex, non-separable, differentiable
        Global: global minimum fx= 0, at ( 0, 0 )
        Link: http://benchmarkfcns.xyz/benchmarkfcns/bohachevskyn2fcn.html

        @param solution: A numpy array include 2 items in range: [-100, 100], [-100, 100]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Bohachevskyn N.2 function is only defined on a 2D space.')
        return solution[0] ** 2 + 2 * solution[1] ** 2 - 0.3 * np.cos(3 * solution[0] * np.pi) * np.cos(4 * solution[1] * np.pi) + 0.3


    def _booth__(self, solution=None):
        """
        Class: unimodal, convex, non-separable, differentiable, continuous
        Global: one global minimum fx= 0, at ( 1, 3 )
        Link: http://benchmarkfcns.xyz/benchmarkfcns/boothfcn.html

        @param solution: A numpy array include 2 items in range: [-10, 10], [-10, 10]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Brooth function is only defined on a 2D space.')
        return  (solution[0]+2*solution[1]-7)**2 + (2*solution[0]+solution[1]-5)**2


    def _brent__(self, solution=None):
        """
        Class: unimodal, convex, non-separable, differentiable
        Global: one global minimum fx= e^(-200), at ( -10 -10 )
        Link: http://benchmarkfcns.xyz/benchmarkfcns/brentfcn.html

        @param solution: A numpy array include 2 items in range: [-20, 0], [-20, 0]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Brent function is only defined on a 2D space.')
        return (solution[0]+10)**2 + (solution[1]+10)**2 + np.exp(-solution[0]**2-solution[1]**2)


    def _bukin_n6__(self, solution=None):
        """
        Class: multimodal, convex, non-separable, non-differentiable, continuous
        Global: one global minimum fx= 0, at ( -10, 1)
        Link: http://benchmarkfcns.xyz/benchmarkfcns/bukinn6fcn.html

        @param solution: A numpy array include 2 items in range: [-15, -5], [-3, 3]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Bukin N.6 function is only defined on a 2D space.')
        return 100*np.sqrt(np.abs(solution[1] - 0.01*solution[0]**2)) + 0.01*np.abs(solution[0]+10)


    def _cross_in_tray__(self, solution=None):
        """
        Class: multimodal, non-convex, non-separable, non-differentiable, continuous
        Global: 4 global minimum fx= -2.06261218, at (±1.349406685353340,±1.349406608602084)
        Link: http://benchmarkfcns.xyz/benchmarkfcns/crossintrayfcn.html

        @param solution: A numpy array include 2 items in range: [-10, 10], [-10, 10]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Bukin N.6 function is only defined on a 2D space.')
        t1 = np.exp( np.abs(100 - np.sqrt(np.sum(solution**2))/np.pi ) )
        t2 = np.sin(solution[0]) * np.cos(solution[1])
        return -0.0001*(np.abs(t1*t2) + 1)**0.1


    def _deckkers_aarts__(self, solution=None):
        """
        Class: multimodal, non-convex, non-separable, differentiable, continuous
        Global: 1 global minimum fx = −24771.09375, at ( 0 , ± 15 ) .
        Link: http://benchmarkfcns.xyz/benchmarkfcns/deckkersaartsfcn.html

        @param solution: A numpy array include 2 items in range: [-20, 20], [-20, 20]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Deckkers Aarts function is only defined on a 2D space.')
        t1 = solution[0]**2
        t2 = solution[1]**2
        return 10**5*t1 + t2 - (t1 + t2)**2 + 10**(-5) * (t1 + t2)**4


    def _drop_wave__(self, solution=None):
        """
        Class: uni-modal, non-convex, continuous
        Global: 1 global minimum fx = −1 at ( 0 , 0 ) .
        Link: http://benchmarkfcns.xyz/benchmarkfcns/dropwavefcn.html

        @param solution: A numpy array include 2 items in range: [-5.2, 5.2], [-5.2, 5.2]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Drop wave function is only defined on a 2D space.')
        return -(1+np.cos(12*np.sqrt(np.sum(solution*2)))) / (0.5 * np.sum(solution**2) + 2)


    def _easom__(self, solution=None):
        """
        Class: multi-modal, non-convex, continuous, differentiable, separable
        Global: 1 global minimum fx = −1 at ( pi, pi ) .
        Link: http://benchmarkfcns.xyz/benchmarkfcns/easomfcn.html

        @param solution: A numpy array include 2 items in range: [-100, 100], [-100, 100]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Easom function is only defined on a 2D space.')
        return -np.cos(solution[0])*np.cos(solution[1])*np.exp(-(solution[0] - np.pi)**2 - (solution[1] - np.pi)**2)


    def _egg_crate__(self, solution=None):
        """
        Class: multi-modal, non-convex, continuous, differentiable, separable
        Global: global minimum fx = 0 at ( 0, 0 ) .
        Link: http://benchmarkfcns.xyz/benchmarkfcns/eggcratefcn.html

        @param solution: A numpy array include 2 items in range: [-5, 5]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Egg Crate function is only defined on a 2D space.')
        return np.sum(solution**2) + 25 * (np.sin(solution[0])**2 + np.sin(solution[1])**2)


    def _goldstein_price__(self, solution=None):
        """
        Class: multi-modal, non-convex, continuous, differentiable, non-separable
        Global: global minimum fx = 3 at ( 0, -1 ) .
        Link: http://benchmarkfcns.xyz/benchmarkfcns/goldsteinpricefcn.html

        @param solution: A numpy array include 2 items in range: [-2, 2]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Goldstein price function is only defined on a 2D space.')
        t1 = 18 - 32*solution[0] + 12*solution[0]**2 + 4*solution[1] - 36*solution[0]*solution[1] + 27*solution[1]**2
        t2 = 19 - 14*solution[0]+3*solution[0]**2 - 14*solution[1] + 6*solution[0]*solution[1] + 3*solution[1]**2
        t3 = (np.sum(solution) + 1)**2
        return (1+t3*t2) * (30 + (2*solution[0]-3*solution[1])**2 * t1)


    def _himmelblau__(self, solution=None):
        """
        Class: multi-modal, non-convex, continuous
        Global: 4 global optima, fx = 0 at (3, 2), (-2.85118, 3.283186), (−3.779310,−3.283186), (3.584458,−1.848126)

        Link: http://benchmarkfcns.xyz/benchmarkfcns/himmelblaufcn.html

        @param solution: A numpy array include 2 items in range: [-6, 6]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Himmelblau function is only defined on a 2D space.')
        return (solution[0]**2+solution[1]-11)**2 + (solution[0] + solution[1]**2 - 7)**2


    def _holder_table__(self, solution=None):
        """
        Class: multi-modal, non-convex, continuous, non-differentiable, non-separable
        Global: 4 global optima, fx = -19.2085 at (±8.05502,±9.66459)

        Link: http://benchmarkfcns.xyz/benchmarkfcns/holdertablefcn.html

        @param solution: A numpy array include 2 items in range: [-10, 10]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Holder Table function is only defined on a 2D space.')
        return -np.abs(np.sin(solution[0])*np.cos(solution[1])*np.abs(1 - np.sqrt(np.sum(solution**2))/np.pi))


    def _keane__(self, solution=None):
        """
        Class: multi-modal, non-convex, continuous, differentiable, non-separable
        Global: 2 global optima, fx = 0.673667521146855 at (1.393249070031784,0), (0,1.393249070031784)

        Link: http://benchmarkfcns.xyz/benchmarkfcns/kealefcn.html

        @param solution: A numpy array include 2 items in range: [0, 10]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Keane function is only defined on a 2D space.')
        return -np.sin(solution[0]-solution[1])**2 * np.sin(solution[0] + solution[1])**2 / np.sqrt(np.sum(solution**2))


    def _leon__(self, solution=None):
        """
        Class: uni-modal, non-convex, continuous, differentiable, non-separable
        Global: 1 global optima, fx = 0 at [0, 10]

        Link: http://benchmarkfcns.xyz/benchmarkfcns/leonfcn.html

        @param solution: A numpy array include 2 items in range: [0, 10]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Leon function is only defined on a 2D space.')
        return 100*(solution[1]-solution[0]**3)**2 + (1-solution[0])**2


    def _levi_n13__(self, solution=None):
        """
        Class: multi-modal, non-convex, continuous, differentiable, non-separable
        Global: 1 global optima, fx = 0 at [1, 1]

        Link: http://benchmarkfcns.xyz/benchmarkfcns/levin13fcn.html

        @param solution: A numpy array include 2 items in range: [-10, 10]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Levi N.13 function is only defined on a 2D space.')
        return np.sin(3*solution[0]*np.pi)**2 + (solution[0]-1)**2*(1+np.sin(3*solution[1]**np.pi)**2) +\
               (solution[1]-1)**2*(1 + np.sin(2*solution[1]*np.pi)**2)


    def _matyas__(self, solution=None):
        """
        Class: uni-modal, convex, continuous, differentiable, non-separable
        Global: 1 global optima, fx = 0 at [0, 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/matyasfcn.html

        @param solution: A numpy array include 2 items in range: [-10, 10]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Matyas function is only defined on a 2D space.')
        return 0.26*np.sum(solution**2) - 0.48*solution[0]*solution[1]


    def _mc_cormick__(self, solution=None):
        """
        Class: multi-modal, convex, continuous, differentiable, non-scalable
        Global: 1 global optima, fx = -1.9133 at [-0.547, -1.547]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/mccormickfcn.html

        @param solution: A numpy array include 2 items in range: [-1.5, 4], [-3, 3]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Mc Cormick function is only defined on a 2D space.')
        return np.sin(solution[0]+solution[1]) + (solution[0] - solution[1])**2 - 1.5*solution[0] + 2.5*solution[1] + 1


    def _schaffer_n1__(self, solution=None):
        """
        Class: uni-modal, non-convex, continuous, differentiable, non-separable
        Global: 1 global optima, fx = 0 at [0, 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schaffern1fcn.html

        @param solution: A numpy array include 2 items in range: [-100, 100]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Scheffer N.1 function is only defined on a 2D space.')
        return 0.5 + (np.sin(np.sum(solution**2)**2)**2 - 0.5) / (1 + 0.001*np.sum(solution**2))**2


    def _schaffer_n2__(self, solution=None):
        """
        Class: uni-modal, non-convex, continuous, differentiable, non-separable
        Global: 1 global optima, fx = 0 at [0, 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schaffern2fcn.html

        @param solution: A numpy array include 2 items in range: [-100, 100]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Scheffer N.2 function is only defined on a 2D space.')
        return 0.5 + (np.sin(solution[0]**2 - solution[1]**2)**2 - 0.5) / (1 + 0.001*np.sum(solution**2))**2


    def _schaffer_n3__(self, solution=None):
        """
        Class: uni-modal, non-convex, continuous, differentiable, non-separable
        Global: 1 global optima, fx = 0.00156685 at [0, 1.253115]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schaffern3fcn.html

        @param solution: A numpy array include 2 items in range: [-100, 100]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Scheffer N.3 function is only defined on a 2D space.')
        return 0.5 + (np.sin(np.cos(np.abs( solution[0]**2 - solution[1]**2 ))) - 0.5) / (1 + 0.001*np.sum(solution**2))**2


    def _schaffer_n4__(self, solution=None):
        """
        Class: uni-modal, non-convex, continuous, differentiable, non-separable
        Global: 1 global optima, fx = 0.292579 at [0, 1.253115]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/schaffern4fcn.html

        @param solution: A numpy array include 2 items in range: [-100, 100]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Scheffer N.4 function is only defined on a 2D space.')
        return 0.5 + (np.cos(np.sin(np.abs( solution[0]**2 - solution[1]**2 ))) - 0.5) / (1 + 0.001*np.sum(solution**2))**2


    def _three_hump_camel__(self, solution=None):
        """
        Class: multi-modal, non-convex, continuous, differentiable, non-separable
        Global: 1 global optima, fx = 0 at [0, 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/threehumpcamelfcn.html

        @param solution: A numpy array include 2 items in range: [-5, 5]
        @return: fx
        """
        d = len(solution)
        assert (d == 2, 'Scheffer N.3 function is only defined on a 2D space.')
        return 2*solution[0]**2 - 1.05*solution[0]**4 + solution[0]**6/6 + solution[0]*solution[1] + solution[1]**2



