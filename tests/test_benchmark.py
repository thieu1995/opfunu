#!/usr/bin/env python
# Created by "Thieu" at 20:21, 30/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from opfunu.benchmark import Benchmark


def test_Benchmark_class():
    ndim = 10
    default_bounds = np.array([[-15, ] * ndim, [15, ] * ndim]).T
    x = np.random.uniform(-15, 15, ndim)
    problem = Benchmark()
    problem.check_ndim_and_bounds(ndim, None, default_bounds)

    assert len(problem.lb) == ndim
    assert isinstance(problem.lb, np.ndarray)
    assert type(problem.bounds) == np.ndarray
    assert problem.bounds.shape[0] == ndim
    with pytest.raises(NotImplementedError):
        problem.evaluate(x)
