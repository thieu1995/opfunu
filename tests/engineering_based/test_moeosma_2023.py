#!/usr/bin/env python
# Created by "Thieu" at 13:57, 07/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.engineering_based import moeosma_2023
import numpy as np
import pytest


def test_SpeedReducerProblem():
    p1 = moeosma_2023.SpeedReducerProblem()
    x0 = [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0]

    assert len(p1.bounds) == len(p1.lb)
    assert p1.n_dims == len(p1.bounds)
    assert p1.n_objs == 2
    assert p1.n_cons == 11

    assert np.all(p1.get_objs(x0) - np.array([2352.44784872, 1695.96387746]))
    assert len(p1.get_cons(x0)) == p1.n_cons
    assert np.all(p1.evaluate(x0) - np.array([7.49307543e+10, 7.49307536e+10]))

