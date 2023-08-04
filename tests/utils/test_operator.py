import numpy as np
import opfunu

from opfunu.utils.operator import rosenbrock_func, elliptic_func, bent_cigar_func, discus_func, calculate_weight, \
    griewank_func, rastrigin_func, schwefel_12_func, hgbat_func, calculate_weight_cec, lunacek_bi_rastrigin_func, \
    lunacek_bi_rastrigin_gen_func


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

def test_rastrigin_func_optimum_result():
    ndim = 10
    x = np.zeros(ndim)
    assert abs(rastrigin_func(x) - 0) <= 1e-8

def test_schwefel_12_func_optimum_result():
    ndim = 10
    x = np.zeros(ndim)
    assert abs(schwefel_12_func(x) - 0) <= 1e-8

def test_hgbat_func_optimum_result():
    ndim = 10
    x = np.ones(ndim) * -1.
    assert abs(hgbat_func(x) - 0) <= 1e-8

def test_lunacek_bi_rastrigin_func_result():
    ndim = 2
    x = np.ones(ndim) * 2.5
    assert abs(lunacek_bi_rastrigin_gen_func(x) - 0) <= 1e-8


def test_compare_weight_calculation():
    delta = 20
    x = np.array(
        [61.77016424, -8.24035508, -21.51173549, 10.66292843, -28.85312768, 31.24207327, 71.83751743, -67.53976114,
         -79.1941269, 27.53276621])
    f10 = opfunu.cec_based.F102022(ndim=len(x))
    d = x-f10.x_global

    expected = calculate_weight_cec(d, delta)
    actual = calculate_weight(d, delta)
    assert expected == actual

def test_compare_weight_calculation():
    delta = 20
    d = np.zeros(10)

    expected = calculate_weight_cec(d, delta)
    actual = calculate_weight(d, delta)
    assert expected == actual
