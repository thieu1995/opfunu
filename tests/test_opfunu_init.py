import numpy as np

from opfunu import get_functions_by_ndim


def test_get_functions_by_ndim_when_15_all_valid():
    ndim = 15
    random_value = np.random.uniform(-100, 100, 15)
    results = get_functions_by_ndim(ndim)
    assert len(results) >= 1
    for f in results:
        f_actual = f(ndim=ndim)
        assert f_actual.evaluate(random_value) is not None
def test_get_functions_by_ndim_when_None_all_valid():
    results = get_functions_by_ndim(None)
    assert len(results) > 1