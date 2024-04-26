#!/usr/bin/env python
# Created by "Travis" at 10:00, 13/09/2023 ---------%
#       Github: https://github.com/firestrand       %
# --------------------------------------------------%

import numpy as np
from opfunu import get_all_cec_based_functions


def test_whenNdimNone_thenDefaultNdimUsed():
    allFunctions = get_all_cec_based_functions()
    for f in allFunctions:
        f_default = f()
        assert f_default.ndim == f_default.dim_default, f'{f.__name__} failed to have ndim == dim_default'


def test_whenEvaulateDefaultNdim_thenHasResult():
    known_failing = []
    all_functions = [x for x in get_all_cec_based_functions() if x.__name__ not in known_failing]
    failing = []
    for f in all_functions:
        f_default = f()
        x = np.random.rand(f_default.dim_default)
        try:
            f_default.evaluate(x)
        except x:
            print(f'{f_default.__name__}:{f_default.dim_default}:{x}')
            failing.append(f.__name__)

    assert len(failing) == 0, f'{failing} failed to have x_global result in f_global'


def test_whenEvaulateWith_x_global_then_f_global():
    # The following are broken or have incorrect or unknown correct , values.
    known_failing = ['F72008']
    all_functions = [x for x in get_all_cec_based_functions() if x.__name__ not in known_failing]
    failing = []
    for f in all_functions:
        f_default = f()
        x_global = f_default.x_global
        if abs(f_default.evaluate(x_global) - f_default.f_global) >= f_default.epsilon:
            failing.append(f.__name__)
    assert len(failing) == 0, f'{failing} failed to have x_global result in f_global'
