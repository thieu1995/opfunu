import numpy as np

from opfunu.utils.operator import rosenbrock_func, elliptic_func, bent_cigar_func, discus_func, calculate_weight, \
    griewank_func


def test_griewank_func_optimium_result():
    ndim = 10
    x = np.zeros(ndim)
    assert abs(griewank_func(x) - 0) <= 1e-8

def test_rosenbrock_func_optimium_result():
    ndim = 2
    x = np.ones(ndim)
    assert abs(rosenbrock_func(x) - 0) <= 1e-8

def test_elliptic_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(elliptic_func(x) - 0) <= 1e-8

def test_bent_cigar_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(bent_cigar_func(x) - 0) <= 1e-8

def test_discus_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(discus_func(x) - 0) <= 1e-8
