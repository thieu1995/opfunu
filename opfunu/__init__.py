#!/usr/bin/env python
# Created by "Thieu" at 11:23, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
#
# Examples:
# >>> from opfunu.cec_based.cec2014 import F12014
# >>>
# >>> f1 = F12014(ndim=30, f_bias=100)
# >>>
# >>> lower_bound = f1.lb                       # Numpy array
# >>> lower_bound_as_list = f1.lb.to_list()     # Python list
# >>> upper_bound = f1.ub
# >>> fitness = f1.evaluate
# >>>
# >>> solution = np.random.uniform(0, 1, 30)
# >>> print(f1.evaluate(solution))
# >>> print(fitness.evaluate(solution))
# >>>
# >>> print(f1.get_paras())         # Print the parameters of function if has
# >>>
# >>> Plot 2d or plot 3d contours
# >>> Warning !! change n_points to reduce the computing time
# >>>
# >>> import opfunu
# >>> f2 = opfunu.cec_based.F22005(ndim=2)
# >>> f2.plot_2d(selected_dims=(2, 3), n_points=300)
# >>> f2.plot_3d(selected_dims=(1, 4), n_points=300)

__version__ = "1.0.4"

import inspect
import re
from .utils.visualize import draw_2d, draw_3d, draw_latex
from . import name_based
from . import cec_based

FUNC_DATABASE = inspect.getmembers(name_based, inspect.isclass)
CEC_DATABASE = inspect.getmembers(cec_based, inspect.isclass)
ALL_DATABASE = FUNC_DATABASE + CEC_DATABASE
EXCLUDES = ["Benchmark", "CecBenchmark", "ABC"]


def get_functions_by_classname(name=None):
    """
    Parameters
    ----------
    name : The exact classname of the function (no difference among lowercase, uppercase, mix)

    Returns
    -------
        List of the functions, but all the classname are different, so the result is list of 1 function or list of empty
    """
    functions = [cls for classname, cls in ALL_DATABASE if (classname not in EXCLUDES and (classname.lower() == name.lower()))]
    return functions


def get_functions_based_classname(name=None):
    """
    Parameters
    ----------
    name : The substring of classname of the function

    Returns
    -------
        List of the functions
    """
    functions = [cls for classname, cls in ALL_DATABASE if (classname not in EXCLUDES and re.search(name.lower(), classname.lower()))]
    return functions


def get_functions_by_ndim(ndim=None):
    """
    Parameters
    ----------
    ndim : The exact number of dimensions that function has and not able to change

    Returns
    -------
        List of the functions
    """
    functions = [cls for classname, cls in ALL_DATABASE if classname not in EXCLUDES]
    if type(ndim) is int and ndim > 1:
        return list(filter(lambda f: (f().dim_default == ndim and f().dim_changeable == False), functions))
    return functions


def get_functions_based_ndim(ndim=None):
    """
    Parameters
    ----------
    ndim : Number of dimensions that function supported

    Returns
    -------
        List of the functions
    """
    functions = [cls for classname, cls in ALL_DATABASE if classname not in EXCLUDES]
    if type(ndim) is int and ndim > 1:
        return list(filter(lambda f: (f().dim_default == ndim or f().dim_changeable == True), functions))
    return functions


def get_all_name_based_functions():
    return [cls for classname, cls in FUNC_DATABASE if classname not in EXCLUDES]


def get_all_cec_based_functions():
    return [cls for classname, cls in CEC_DATABASE if classname not in EXCLUDES]


def get_name_based_functions(ndim, continuous=None, linear=None, convex=None, unimodal=None, separable=None,
                  differentiable=None, scalable=None, randomized_term=None, parametric=None, modality=None):
    functions = [cls for classname, cls in FUNC_DATABASE if classname not in EXCLUDES]
    functions = list(filter(lambda f: f().is_ndim_compatible(ndim), functions))

    functions = list(filter(lambda f: (continuous is None) or (f.continuous == continuous), functions))
    functions = list(filter(lambda f: (linear is None) or (f.linear == linear), functions))
    functions = list(filter(lambda f: (convex is None) or (f.convex == convex), functions))
    functions = list(filter(lambda f: (unimodal is None) or (f.unimodal == unimodal), functions))
    functions = list(filter(lambda f: (separable is None) or (f.separable == separable), functions))
    functions = list(filter(lambda f: (differentiable is None) or (f.differentiable == differentiable), functions))
    functions = list(filter(lambda f: (scalable is None) or (f.scalable == scalable), functions))
    functions = list(filter(lambda f: (randomized_term is None) or (f.randomized_term == randomized_term), functions))
    functions = list(filter(lambda f: (parametric is None) or (f.parametric == parametric), functions))
    functions = list(filter(lambda f: (modality is None) or (f.modality == modality), functions))
    return functions


def get_cec_based_functions(ndim=None, continuous=None, linear=None, convex=None, unimodal=None, separable=None, differentiable=None,
             scalable=None, randomized_term=None, parametric=True, shifted=True, rotated=None , modality=None):
    functions = [cls for classname, cls in CEC_DATABASE if classname not in EXCLUDES]
    functions = list(filter(lambda f: f().is_ndim_compatible(ndim), functions))

    functions = list(filter(lambda f: (continuous is None) or (f.continuous == continuous), functions))
    functions = list(filter(lambda f: (linear is None) or (f.linear == linear), functions))
    functions = list(filter(lambda f: (convex is None) or (f.convex == convex), functions))
    functions = list(filter(lambda f: (unimodal is None) or (f.unimodal == unimodal), functions))
    functions = list(filter(lambda f: (separable is None) or (f.separable == separable), functions))
    functions = list(filter(lambda f: (differentiable is None) or (f.differentiable == differentiable), functions))
    functions = list(filter(lambda f: (scalable is None) or (f.scalable == scalable), functions))
    functions = list(filter(lambda f: (randomized_term is None) or (f.randomized_term == randomized_term), functions))
    functions = list(filter(lambda f: (parametric is None) or (f.parametric == parametric), functions))
    functions = list(filter(lambda f: (shifted is None) or (f.shifted == shifted), functions))
    functions = list(filter(lambda f: (rotated is None) or (f.rotated == rotated), functions))
    functions = list(filter(lambda f: (modality is None) or (f.modality == modality), functions))
    return functions
