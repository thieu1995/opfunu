#!/usr/bin/env python
# Created by "Thieu" at 16:39, 05/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from abc import ABC, abstractmethod


class Engineer(ABC):
    """
    Defines an abstract class for engineering design problems.

    All subclasses should implement the ``evaluate`` method for a particular optimization problem.

    Attributes
    ----------
    bounds : list
        The lower/upper bounds of the problem. This a 2D-matrix of [lower, upper] array that contain the lower and upper bounds.
        By default, each problem has its own bounds. But user can try to put different bounds to test the problem.
    n_dims : int
        The dimensionality of the problem. It is calculated from bounds
    lb : np.ndarray
        The lower bounds for the problem
    ub : np.ndarray
        The upper bounds for the problem
    f_global : float
        The global optimum of the evaluated function.
    x_global : np.ndarray
        A list of vectors that provide the locations of the global minimum.
        Note that some problems have multiple global minima, not all of which may be listed.
    n_fe : int
        The number of function evaluations that the object has been asked to calculate.
    dim_changeable : bool
        Whether we can change the benchmark function `x` variable length (i.e., the dimensionality of the problem)
    """

    name = "Benchmark name"
    latex_formula = r'f(\mathbf{x})'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, ..., 0)=-1, \text{ for}, m=5, \beta=15'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    def __init__(self):
        self._bounds = None
        self._n_dims = None
        self._n_objs = 1
        self._n_cons = 0

        self.f_penalty = None
        self.f_global = None
        self.x_global = None
        self.n_fe = 0
        self.paras = {}
        self.epsilon = 1e-8
        self.w = 1e8

    @abstractmethod
    def get_objs(self, x):
        """
        Compute the values of the objective functions for a given set of input values.
        """
        pass

    @abstractmethod
    def get_cons(self, x):
        """
        Compute the values of the constraint functions for a given set of input values.
        """
        pass

    def get_paras(self):
        """
        Return the parameters of the problem. Depended on function
        """
        default = {"bounds": self._bounds, "n_dims": self._n_dims, }
        return {**default, **self.paras}

    @property
    def bounds(self):
        """
        The lower/upper bounds to be used for optimization problem. This a 2D-matrix of [lower, upper] array that contain the lower and upper
        bounds for the problem. The problem should not be asked for evaluation outside these bounds. ``len(bounds) == n_dims``.
        """
        return self._bounds

    @property
    def n_dims(self):
        """
        The dimensionality of the problem.
        """
        return self._n_dims

    @property
    def n_objs(self):
        """
        The number of objective functions of the problem.
        """
        return self._n_objs

    @property
    def n_cons(self):
        """
        The number of constraint functions of the problem.
        """
        return self._n_cons

    @property
    def lb(self):
        """
        The lower bounds for the problem

        Returns
        -------
        lb : 1D-vector
            The lower bounds for the problem
        """
        return np.array([x[0] for x in self.bounds])

    @property
    def ub(self):
        """
        The upper bounds for the problem

        Returns
        -------
        ub : 1D-vector
            The upper bounds for the problem
        """
        return np.array([x[1] for x in self.bounds])

    def amend_position(self, x, lb=None, ub=None):
        """
        Amend position to fit the format of the problem

        Parameters
        ----------
        x : np.ndarray
            The current position (solution)
        """
        return x

    def create_solution(self):
        """
        Create a random solution for the current problem

        Returns
        -------
        solution: np.ndarray
            The random solution
        """
        return np.random.uniform(self.lb, self.ub)

    def check_solution(self, x):
        """
        Raise the error if the problem size is not equal to the solution length

        Parameters
        ----------
        x : np.ndarray
            The solution
        """
        if len(x) != self._n_dims:
            raise ValueError(f"The length of solution should has {self._n_dims} variables!")

    def default_penalty(self, list_objs=None, list_cons=None):
        list_objs_new = np.zeros_like(list_objs)
        for idx, val in enumerate(list_objs):
            temp = val + self.w * np.sum([max(0, f_con) for f_con in list_cons])
            list_objs_new[idx] = temp
        return list_objs_new

    def check_penalty_func(self, func=None):
        if callable(func):
            self.f_penalty = func
        else:
            self.f_penalty = self.default_penalty

    def evaluate(self, x):
        """
        Evaluation of the benchmark function.

        Parameters
        ----------
        x : np.ndarray, list, tuple
            The candidate vector for evaluating the benchmark problem. Must have ``len(x) == self.n_dims``.

        Returns
        -------
        val : float
              the evaluated benchmark function
        """
        pass
