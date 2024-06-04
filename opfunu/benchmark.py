#!/usr/bin/env python
# Created by "Thieu" at 16:47, 28/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.utils.visualize import draw_2d, draw_3d, draw_latex


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
        self.dim_default = 2
        self.dim_supported = []
        self.f_global = None
        self.x_global = None
        self.n_fe = 0
        self.paras = {}
        self.epsilon = 1e-8

    def check_ndim_and_bounds(self, ndim=None, bounds=None, default_bounds=None):
        """
        Check the bounds when initializing the object.

        Parameters
        ----------
        ndim : int
            The number of dimensions (variables)
        bounds : list, tuple, np.ndarray
            List of lower bound and upper bound, should use default None value
        default_bounds : np.ndarray
            List of initial lower bound and upper bound values
        """
        if ndim is None:
            self._bounds = default_bounds if bounds is None else np.array(bounds).T
            self._ndim = self._bounds.shape[0]
        else:
            if bounds is None:
                if self.dim_changeable:
                    if type(ndim) is int and ndim > 1:
                        self._ndim = int(ndim)
                        self._bounds = np.array([default_bounds[0] for _ in range(self._ndim)])
                    else:
                        raise ValueError('ndim must be an integer and > 1!')
                else:
                    self._ndim = self.dim_default
                    self._bounds = default_bounds
                    print(f"{self.__class__.__name__} is fixed problem with {self.dim_default} variables!")
            else:
                if self.dim_changeable:
                    self._bounds = np.array(bounds).T
                    self._ndim = self._bounds.shape[0]
                    print(f"{self.__class__.__name__} problem is set with {self._ndim} variables!")
                else:
                    self._bounds = np.array(bounds).T
                    if self._bounds.shape[0] != self.dim_default:
                        raise ValueError(f"{self.__class__.__name__} is fixed problem with {self._ndim} variables. Please setup the correct bounds!")
                    else:
                        self._ndim = self.dim_default

    def check_solution(self, x):
        """
        Raise the error if the problem size is not equal to the solution length

        Parameters
        ----------
        x : np.ndarray
            The solution
        """
        if not self.dim_changeable and (len(x) != self._ndim):
            raise ValueError(f"The length of solution should have {self._ndim} variables!")

    def get_paras(self):
        """
        Return the parameters of the problem. Depended on function
        """
        default = {"bounds": self._bounds, "ndim": self._ndim, }
        return {**default, **self.paras}

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

    def is_succeed(self, x, tol=1.e-5):
        """
        Check if a candidate solution at the global minimum.

        Parameters
        ----------
        x : np.ndarray
            The candidate vector for testing if the global minimum has been reached. Must have ``len(x) == self.ndim``
        tol : float
            The evaluated function and known global minimum must differ by less than this amount to be at a global minimum.

        Returns
        -------
        is_succeed : bool
            Answer the question: is the candidate vector at the global minimum?
        """

        # the solution should still be in bounds, otherwise immediate fail.
        if np.any(x > self.ub) or np.any(x < self.lb):
            return False

        val = self.evaluate(np.squeeze(x))
        if np.abs(val - self.f_global) < tol:
            return True

        # you found a lower global minimum.  This shouldn't happen.
        if val < self.f_global:
            raise ValueError("Found a lower global minimum", x, val, self.f_global)
        return False

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

    def create_solution(self) -> np.ndarray:
        """
        Create a random solution for the current problem

        Returns
        -------
        solution: 1D-vector
            The random solution
        """
        return np.random.uniform(self.lb, self.ub)

    def plot_latex(self, latex, title="Latex equation", figsize=(8, 3), dpi=500, filename=None, exts=(".png", ".pdf"), verbose=True):
        """
        Draw latex equation.

        Parameters
        ----------
        latex : equation
            Your latex equation, you can test on the website: https://latex.codecogs.com/
        title : str
            Title for the figure
        figsize : tuple, default=(10, 8)
            The figure size with format of tuple (width, height)
        dpi : int, default=500
            The dot per inches (DPI) - indicate the number of dots per inch.
        filename : str, default = None
            Set the file name, If None, the file will not be saved
        exts : list, tuple, np.ndarray
            The list of extensions to save file, for example: (".png", ".pdf", ".jpg")
        verbose : bool
            Show the figure or not. It will not show on linux system.
        """
        draw_latex(latex, title=title, figsize=figsize, dpi=dpi, filename=filename, exts=exts, verbose=verbose)

    def plot_2d(self, selected_dims=None, n_points=500, ct_cmap="viridis", ct_levels=30, ct_alpha=0.7, fixed_strategy="mean", fixed_values=None,
                title="Contour map of the function", x_label=None, y_label=None, figsize=(10, 8), filename=None, exts=(".png", ".pdf"), verbose=True):
        """
        Draw 2D contour of the function.

        Parameters
        ----------
        selected_dims : list, tuple, np.ndarray
            The selected dimensions you want to draw.
            If your function has only 2 dimensions, it will select (1, 2) automatically.
        n_points : int
            The number of points that will be used to draw the contour
        ct_cmap : str
            The cmap of matplotlib.pyplot.contourf function (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html)
        ct_levels : int
            The levels parameter of contourf function
        ct_alpha : float
            The alpha parameter of contourf function
        fixed_strategy : str
            The selected strategy to set values for other dimensions.
            When your function has > 2 dimensions, you need to set a fixed value for other dimensions to be able to calculate value.
            List of available strategy: ["min", "max", "mean', "values", "zero"]
                + min: Set the other dimensions by its lower bound
                + max: Set the other dimensions by its upper bound
                + mean: Set the other dimensions by it average value (lower bound + upper bound) / 2
                + zero: Set the other dimensions by 0
                + values: Set the other dimensions by your passed values through the parameter: `fixed_values`.

        fixed_values : list, tuple, np.ndarray
            Fixed values for all dimensions (length should be the same as lower bound), the selected dimensions will be replaced in the drawing process.
        title : str
            Title for the figure
        x_label : str
            Set the x label
        y_label : str
            Set the y label
        figsize : tuple, default=(10, 8)
            The figure size with format of tuple (width, height)
        filename : str, default = None
            Set the file name, If None, the file will not be saved
        exts : list, tuple, np.ndarray
            The list of extensions to save file, for example: (".png", ".pdf", ".jpg")
        verbose : bool
            Show the figure or not. It will not show on linux system.
        """
        draw_2d(self.evaluate, self.lb, self.ub, selected_dims=selected_dims, n_points=n_points,
                ct_cmap=ct_cmap, ct_levels=ct_levels, ct_alpha=ct_alpha, fixed_strategy=fixed_strategy, fixed_values=fixed_values,
                title=title, x_label=x_label, y_label=y_label, figsize=figsize, filename=filename, exts=exts, verbose=verbose)

    def plot_3d(self, selected_dims=None, n_points=500, ct_cmap="viridis", ct_levels=30, ct_alpha=0.7, fixed_strategy="mean", fixed_values=None,
                title="3D visualization of the function", x_label=None, y_label=None, figsize=(10, 8), filename=None, exts=(".png", ".pdf"), verbose=True):
        """
        Draw 3D of the function.

        Parameters
        ----------
        selected_dims : list, tuple, np.ndarray
            The selected dimensions you want to draw.
            If your function has only 2 dimensions, it will select (1, 2) automatically.
        n_points : int
            The number of points that will be used to draw the contour
        ct_cmap : str
            The cmap of matplotlib.pyplot.contourf function (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html)
        ct_levels : int
            The levels parameter of contourf function
        ct_alpha : float
            The alpha parameter of contourf function
        fixed_strategy : str
            The selected strategy to set values for other dimensions.
            When your function has > 2 dimensions, you need to set a fixed value for other dimensions to be able to calculate value.
            List of available strategy: ["min", "max", "mean', "values", "zero"]
                + min: Set the other dimensions by its lower bound
                + max: Set the other dimensions by its upper bound
                + mean: Set the other dimensions by it average value (lower bound + upper bound) / 2
                + zero: Set the other dimensions by 0
                + values: Set the other dimensions by your passed values through the parameter: `fixed_values`.

        fixed_values : list, tuple, np.ndarray
            Fixed values for all dimensions (length should be the same as lower bound), the selected dimensions will be replaced in the drawing process.
        title : str
            Title for the figure
        x_label : str
            Set the x label
        y_label : str
            Set the y label
        figsize : tuple, default=(10, 8)
            The figure size with format of tuple (width, height)
        filename : str, default = None
            Set the file name, If None, the file will not be saved
        exts : list, tuple, np.ndarray
            The list of extensions to save file, for example: (".png", ".pdf", ".jpg")
        verbose : bool
            Show the figure or not. It will not show on linux system.
        """
        draw_3d(self.evaluate, self.lb, self.ub, selected_dims=selected_dims, n_points=n_points,
                ct_cmap=ct_cmap, ct_levels=ct_levels, ct_alpha=ct_alpha, fixed_strategy=fixed_strategy, fixed_values=fixed_values,
                title=title, x_label=x_label, y_label=y_label, figsize=figsize, filename=filename, exts=exts, verbose=verbose)
