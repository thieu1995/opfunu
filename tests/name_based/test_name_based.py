import numpy as np

from opfunu import get_all_named_functions


def test_whenNdimNone_thenDefaultNdimUsed():
    allFunctions = get_all_named_functions()
    for f in allFunctions:
        f_default = f()
        assert f_default.ndim == f_default.dim_default, f'{f.__name__} failed to have ndim == dim_default'


def test_whenEvaulateWith_x_global_then_f_global():
    # The following are broken or have incorrect or unknown correct x_global values.
    known_failing = ['Damavandi', 'EasomExpanded', 'LennardJones', 'Meyer', 'Michalewicz']
    allFunctions = [x for x in get_all_named_functions() if x.__name__ not in known_failing]
    for f in allFunctions:
        f_default = f()
        x_global = f_default.x_global
        assert abs(f_default.evaluate(x_global) - f_default.f_global) <= f_default.epsilon, \
            f'{f.__name__} failed to have x_global result in f_global'


def test_ndim_min_as_ndim_works():
    known_failing = []
    allFunctions = [x for x in get_all_named_functions() if x.__name__ not in known_failing]
    for f in allFunctions:
        ndim = f.ndim_min
        x = np.random.uniform(0, 1, ndim)
        f_default = f(f.ndim_min)
        assert f_default.evaluate(x) is not None, \
            f'{f.__name__} ndim_min for class invalid'

def test_ndim_max_as_ndim_works():
    known_failing = []
    allFunctions = [x for x in get_all_named_functions() if x.__name__ not in known_failing]
    for f in allFunctions:
        f_default = f(f.ndim_max)
        x = np.random.uniform(0, 1, f_default.ndim)
        assert f_default.evaluate(x) is not None, \
            f'{f.__name__} ndim_max for class invalid'
