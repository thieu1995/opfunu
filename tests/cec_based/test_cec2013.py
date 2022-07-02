#!/usr/bin/env python
# Created by "Thieu" at 19:57, 02/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import opfunu
import pytest


def test_F12013_results():
    ndim = 50
    problem = opfunu.cec_based.F12013(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F22013_results():
    ndim = 50
    problem = opfunu.cec_based.F22013(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim



























