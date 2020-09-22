#!/usr/bin/env python                                                                                   #
# ------------------------------------------------------------------------------------------------------#
# Created by "Thieu Nguyen" at 16:32, 07/12/2019                                                        #
#                                                                                                       #
#       Email:      nguyenthieu2102@gmail.com                                                           #
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  #
#       Github:     https://github.com/thieu1995                                                  #
#-------------------------------------------------------------------------------------------------------#

import numpy as np

class Functions:
    """
        This class of functions is belongs to multi-modal function, all functions will be scaled to n-dimension space
    """

    def _ackley__(self, solution=None):
        """
        Class: Continuous, Non-convex, Differentiable
        Global: one global minimum fx = 0, at [0, 0,...0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/ackleyfcn.html

        @param solution: A numpy array like with x_i in [-32, 32]
        @return: fx
        """
        a, b, c = 20, 0.2, 2*np.pi
        d = len(solution)
        sum_1 = -a*np.exp(-b* np.sqrt(np.sum(solution ** 2) / d))
        sum_2 = np.exp(np.sum(np.cos(c*solution))/d)
        return sum_1 - sum_2 + a + np.exp(1)


    def _ackley_4__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: −4.590101633799122
        Link: http://benchmarkfcns.xyz/benchmarkfcns/ackleyn4fcn.html

        @param solution: A numpy array like with x_i in [-35, 35]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d-1):
            result += np.exp(-0.2*np.sqrt(solution[i]**2 + solution[i+1]**2)) + 3*(np.cos(2*solution[i]) + np.sin(2*solution[i+1]))
        return result


    def _alpine_1__(self, solution=None):
        """
        Class: Continuous, Non-Differentiable, Separable, Non-Scalable
        Global: 1 global minimum, fx = 0, at [0, ..., 0]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/alpinen1fcn.html

        @param solution: A numpy array like with x_i in [0, 10]
        @return: fx
        """
        return np.sum(np.dot(solution, np.sin(solution)) + 0.1 * solution)


    def _alpine_2__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Scalable
        Global: 1 global minimum, fx = 2.808^D, at [7.917, ..., 7.917]
        Link: http://benchmarkfcns.xyz/benchmarkfcns/alpinen2fcn.html

        @param solution: A numpy array like with x_i in [0, 10]
        @return: fx
        """
        return np.prod(np.sqrt(solution)*np.sin(solution))


    def _cosine_mixture__(self, solution=None):
        """
        Class: Discontinuous, Non-Differentiable, Separable, Scalable
        Global:
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array like with x_i in [-1, 1]
        @return: fx
        """
        return -0.1*np.sum(np.cos(5*np.pi*solution)) - np.sum(solution**2)


    def _csendes__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Scalable
        Global: fx = 0, at [0, ...,0]
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array like with x_i in [-1, 1]
        @return: fx
        """
        return np.sum(solution**6*(2+np.sin(1.0/solution)))


    def _deb_1__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Scalable
        Global:
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array like with x_i in [-1, 1]
        @return: fx
        """
        return -np.sum(np.sin(5*np.pi*solution)**6) / len(solution)


    def _deb_3__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Scalable
        Global:
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array like with x_i in [-1, 1]
        @return: fx
        """
        return -np.sum(np.sin(5*np.pi*(solution**0.75 - 0.05))**6) / len(solution)


    def _egg_holder__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: 959.64
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array like with x_i in [-512, 512]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d-1):
            result += -(solution[i+1]+47) * np.sin(np.sqrt(np.abs(solution[i+1] + solution[i] / 2 + 47))) -\
                solution[i]*np.sin(np.sqrt(np.abs(solution[i] - solution[i+1] - 47)))
        return result


    def _exponential__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: one global minimum fx = 1, at [0,...,0]
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-1, 1]
        @return: fx
        """
        return -np.exp(-0.5*np.sum(solution**2))


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


    def _mishra_1__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: one global minimum fx = 2
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [0, 1]
        @return: fx
        """
        d = len(solution)
        result = d - np.sum(solution[:d-1])
        return (1+result)**result


    def _mishra_2__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: one global minimum fx = 2
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [0, 1]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d-1):
            result += 0.5*(solution[i]+solution[i+1])
        result = d - result
        return (1+result)**result


    def _mishra_7__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Non-Scalable
        Global: one global minimum fx = 0
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array
        @return: fx
        """
        return (np.prod(solution) - np.math.factorial(len(solution)))**2


    def _mishra_11__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Non-Scalable
        Global: one global minimum fx = 0
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array
        @return: fx
        """
        d = len(solution)
        return (np.sum(np.abs(solution))/d - (np.prod(np.abs(solution)))**(1/d))**2


    def _pathological__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Non-Scalable
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d-1):
            result += 0.5 + ( np.sin(np.sqrt(100*solution[i]**2 + solution[i+1]**2))**2 -0.5 ) / \
                (1 + 0.001*(solution[i]**2 - 2*solution[i]*solution[i+1] + solution[i+1]**2)**2)
        return result


    def _pinter__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-separable, Scalable
        Global: global minimum fx = 0, at [0, ..., 0]
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        d = len(solution)
        result1 = 0
        result2 = 0
        result3 = 0
        for i in range(0, d):
            result1 += (i+1)*solution[i]**2
            if i==0:
                result2 += 20*(i+1)*np.sin(np.sin(solution[i])*solution[d-1] + np.sin(solution[i+1]))**2
                result3 += (i+1)*np.log10(1+(i+1)* (solution[d-1]**2 - 2*solution[i]+3*solution[i+1]-np.cos(solution[i])+1)**2 )
            if i==d-1:
                result2 += 20 * (i + 1) * np.sin(np.sin(solution[i]) * solution[i - 1] + np.sin(solution[0])) ** 2
                result3 += (i + 1) * np.log10(1 + (i + 1) * (solution[i - 1] ** 2 - 2 * solution[i] + 3 * solution[0] - np.cos(solution[i]) + 1) ** 2)
            result2 += 20*(i+1)*np.sin(solution[i-1]*np.sin(solution[i]))**2
            result3 += (i+1)*np.log10(1 + (i+1) * (solution[i-1]**2-2*solution[i]+3*solution[i+1]-np.cos(solution) + 1)**2 )
        return result1 + result2 + result3


    def _qing__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable Scalable
        Global: one global minimum fx = 0, at (±√i,…,±√i)
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-500, 500]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            result += (solution[i]**2 - i - 1)**2
        return result


    def _quintic__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Non-Scalable
        Global: one global minimum fx = 0
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        return np.sum(np.abs(solution**5-3*solution**4+4*solution**3+2*solution**2-10*solution-4))


    def _rana__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: one global minimum fx = 0
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-500, 500]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d-1):
            t1 = np.sqrt(np.abs(solution[i+1] + solution[i] + 1))
            t2 = np.sqrt(np.abs(solution[i+1] - solution[i] + 1))
            result += (solution[i+1]+1)*np.cos(t2)*np.sin(t1) + solution[i]*np.cos(t1)*np.sin(t2)
        return result


    def _salomon__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: 1 global optima, fx = 0, at [0, ..., 0]
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        return 1 - np.cos(2*np.pi*np.sqrt(np.sum(solution**2))) + 0.1*np.sqrt(np.sum(solution**2))


    def _schwefel_2_4__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Non-Scalable
        Global: one global minimum fx = 0, at [1, ..., 1]
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [0, 10]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            result += (solution[i]-1)**2 + (solution[0] - solution[i]**2)**2
        return result


    def _schwefel_2_25__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Non-Scalable
        Global: one global minimum fx = 0, at [1, ..., 1]
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [0, 10]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(1, d):
            result += (solution[i]-1)**2 + (solution[0] - solution[i]**2)**2
        return result


    def _schwefel_2_26__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Scalable
        Global: one global minimum fx = -418.983
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-500, 500]
        @return: fx
        """
        return -np.sum(solution*np.sin(np.sqrt(np.abs(solution)))) / len(solution)


    def _shubert__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Non-Scalable
        Global: one global minimum fx = -186.7309
        Link: http://benchmarkfcns.xyz/benchmarkfcns/shubertfcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        d = len(solution)
        result = 1
        for i in range(0, d):
            temp = 0
            for j in range(1, 6):
                temp += np.cos(solution[i]*(j+1) + j)
            result *= temp
        return result


    def _shubert_3__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Non-Scalable
        Global: one global minimum fx = -29.6733337
        Link: http://benchmarkfcns.xyz/benchmarkfcns/shubert3fcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        d = len(solution)
        result = 0.0
        for i in range(0, d):
            temp = 0
            for j in range(1, 6):
                temp += j*np.sin(solution[i]*(j+1) + j)
            result += temp
        return result


    def _shubert_4__(self, solution=None):
        """
        Class: Continuous, Differentiable, Separable, Non-Scalable
        Global: one global minimum fx = -25.740858
        Link: http://benchmarkfcns.xyz/benchmarkfcns/shubert4fcn.html

        @param solution: A numpy array with x_i in [-10, 10]
        @return: fx
        """
        d = len(solution)
        result = 0.0
        for i in range(0, d):
            temp = 0
            for j in range(1, 6):
                temp += j*np.cos(solution[i]*(j+1) + j)
            result += temp
        return result


    def _schaffer_f6__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-100, 100]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d-1):
            result += 0.5 + (np.sin(np.sqrt(solution[i]**2+solution[i+1]**2))**2 -0.5) / \
                      (1 + 0.001*(solution[i]**2 + solution[i+1]**2))**2
        return result


    def _styblinski_tang_(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Non-Scalable
        Global: one global minimum fx = -78.332
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-5, 5]
        @return: fx
        """
        return 0.5*np.sum(solution**4 - 16*solution**2 + 5*solution)


    def _trid_6__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Non-Scalable
        Global: one global minimum fx = -50
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-36, 36]
        @return: fx
        """
        result = np.sum((solution - 1) ** 2)
        for i in range(1, len(solution)):
            result -= solution[i] * solution[i-1]
        return result


    def _trigonometric_1__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: one global minimum fx = 0, at [0, ..., 0]
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [0, pi]
        @return: fx
        """
        d = len(solution)
        result = 0
        for i in range(0, d):
            result += ( d - np.sum(np.cos(solution)) + (i+1)*(1 - np.cos(solution[i]) - np.sin(solution[i])) )**2
        return result


    def _trigonometric_2__(self, solution=None):
        """
        Class: Continuous, Differentiable, Non-Separable, Scalable
        Global: one global minimum fx = 1, at [0.9, ..., 0.9]
        Link: A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

        @param solution: A numpy array with x_i in [-500, 500]
        @return: fx
        """
        d = len(solution)
        result = 1
        for i in range(0, d):
            result += 8 * np.sin(7*(solution[i] - 0.9)**2) + 6 * np.sin(14*(solution[0]-0.9)**2) + (solution[i]-0.9)**2
        return result

