#!/usr/bin/env python
# Created by "Thieu" at 11:23, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import inspect
from .utils import *
from . import name_based
from . import cec_based

FUNC_DATABASE = inspect.getmembers(name_based, inspect.isclass)
CEC_DATABASE = inspect.getmembers(cec_based, inspect.isclass)
EXCLUDES = ["Benchmark", "CecBenchmark", "ABC"]


def get_functions(ndim, continuous=None, linear=None, convex=None, unimodal=None, separable=None,
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


def get_cecs(ndim=None, continuous=None, linear=None, convex=None, unimodal=None, separable=None, differentiable=None,
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

