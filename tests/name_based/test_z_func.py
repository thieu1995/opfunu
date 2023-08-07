#!/usr/bin/env python
# Created by "Thieu" at 17:39, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%
import numpy as np

import opfunu.name_based


def test_Zakharov_results():
    ndim = 7
    problem = opfunu.name_based.Zakharov(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert isinstance(problem.ub, np.ndarray)
    assert len(problem.ub) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Zakharov_GlobalMin_results():
    ndim = 7
    problem = opfunu.name_based.Zakharov(ndim=ndim)
    x = problem.x_global
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert abs(problem.f_global - result) <= problem.epsilon
