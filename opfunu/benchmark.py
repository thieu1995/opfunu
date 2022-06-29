#!/usr/bin/env python
# Created by "Thieu" at 16:47, 28/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


class Benchmark:
    """
    Defines an abstract class for optimization benchmark problem.

    All subclasses should implement the ``evaluate`` method for a particular optimization problem.

    Attributes
    ----------
    bounds : list
        The lower/upper bounds of the problem. This a 2D-matrix of [lower, upper] array that contain the lower and upper bounds.
        By default, each problem has its own bounds. But user can try to put different bounds to test the problem.
    ndim : int
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
        self._ndim = None
        self.dim_changeable = False
        self.f_global = None
        self.x_global = None
        self.n_fe = 0

    def check_bounds(self, bounds=None, default_bounds=None):
        """
        Check the bounds when initializing the object.

        Parameters
        ----------
        bounds : list
                 List of lower bound and upper bound, should use default None value
        default_bounds : list
                 List of initial lower bound and upper bound values
        """
        if bounds is None:
            self._bounds = default_bounds
        else:
            self._bounds = np.array(bounds).T
        self._ndim = self._bounds.shape[1]

    def check_solution(self, x):
        """
        Raise the error if the problem size is not equal to the solution length

        Parameters
        ----------
        x : np.ndarray
            The solution
        """
        if not self.dim_changeable and (len(x) != self._ndim):
            raise ValueError(f"The length of solution should has {self._ndim} variables!")

    def get_param(self):
        """
        Return the parameters of the problem. Depended on function
        """
        return {"bounds": self._bounds, "ndim": self._ndim, }

    def evaluate(self, x):
        """
        Evaluation of the benchmark function.

        Parameters
        ----------
        x : np.ndarray
            The candidate vector for evaluating the benchmark problem. Must have ``len(x) == self.ndim``.

        Returns
        -------
        val : float
              the evaluated benchmark function
        """

        raise NotImplementedError

    def is_ndim_compatible(self, ndim):
        """
        Method to support searching the functions with input ndim

        Parameters
        ----------
        ndim : int
                The number of dimensions

        Returns
        -------
        val: bool
             Always true if dim_changeable = True, Else return ndim == self.ndim
        """
        assert (ndim is None) or (isinstance(ndim, int) and (not ndim < 0)), "The dimension ndim must be None or a positive integer"
        if ndim is None:
            return True
        else:
            if self.dim_changeable:
                return ndim > 0
            else:
                return ndim == self.ndim

    def change_dimensions(self, ndim):
        """
        Changes the dimensionality of the benchmark problem if suitable.

        Parameters
        ----------
        ndim : int
               The new dimensionality for the problem.
        """

        if self.dim_changeable:
            self._ndim = ndim
            self._bounds = np.array([self._bounds[0]] * ndim)
        else:
            raise ValueError('dimensionality cannot be changed for this problem')

    @property
    def bounds(self):
        """
        The lower/upper bounds to be used for optimization problem. This a 2D-matrix of [lower, upper] array that contain the lower and upper
        bounds for the problem. The problem should not be asked for evaluation outside these bounds. ``len(bounds) == ndim``.
        """
        return self._bounds

    @property
    def ndim(self):
        """
        The dimensionality of the problem.

        Returns
        -------
        ndim : int
            The dimensionality of the problem
        """
        return self._ndim

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
