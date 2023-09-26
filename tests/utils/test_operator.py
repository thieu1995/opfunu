#!/usr/bin/env python
# Created by "Travis" at 10:00, 13/09/2023 ---------%
#       Github: https://github.com/firestrand       %
# --------------------------------------------------%

import numpy as np
from opfunu.utils import operator


def test_lennard_jones_func_zero_result():
    """
    The CEC2019 version when zero results in penalization of 1.0e20
    """
    x = np.zeros(18)
    assert abs(operator.lennard_jones_func(x) - 1.5e21) <= 1e-8
