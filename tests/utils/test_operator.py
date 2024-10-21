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


def test_elliptic_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the elliptic function
    c_result = 1274605.1368484431877732276916504
    assert abs(operator.elliptic_func(x) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the elliptic function
    c_result = 12746051368.484432220458984375
    assert abs(operator.elliptic_func(x) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the elliptic function
    c_result = 12746051368.484432220458984375
    assert abs(operator.elliptic_func(x) - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    assert abs(operator.elliptic_func(x)) <= 1e-8


def test_bent_cigar_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the bent cigar function
    c_result = 9000001.0
    assert abs(operator.bent_cigar_func(x) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the bent cigar function
    c_result = 90000010000.0
    assert abs(operator.bent_cigar_func(x) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the bent cigar function
    c_result = 90000010000.0
    assert abs(operator.bent_cigar_func(x) - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    assert abs(operator.bent_cigar_func(x)) <= 1e-8


def test_discus_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the discus function
    c_result = 1000009.0
    assert abs(operator.discus_func(x) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the discus function
    c_result = 10000090000.0
    assert abs(operator.discus_func(x) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the discus function
    c_result = 10000090000.0
    assert abs(operator.discus_func(x) - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    assert abs(operator.discus_func(x)) <= 1e-8


def test_rosenbrock_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the rosenbrock function
    c_result = 3609.0
    assert abs(operator.rosenbrock_func(x, 1) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the rosenbrock function
    c_result = 91809090000.0
    assert abs(operator.rosenbrock_func(x, 1) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the rosenbrock function
    c_result = 88209090000.0
    assert abs(operator.rosenbrock_func(x, 1) - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    assert abs(operator.rosenbrock_func(x, 1)) <= 1e-8


def test_ackley_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the ackley function
    c_result = 3.625384938440362692091412100126
    assert abs(operator.ackley_func(x) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the ackley function
    c_result = 19.999999958776928821180263184942
    assert abs(operator.ackley_func(x) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the ackley function
    c_result = 19.999999958776928821180263184942
    assert abs(operator.ackley_func(x) - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    assert abs(operator.ackley_func(x)) <= 1e-8


def test_rastrigin_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the rastrigin function
    c_result = 10.0
    assert abs(operator.rastrigin_func(x) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the rastrigin function
    c_result = 100000.0
    assert abs(operator.rastrigin_func(x) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the rastrigin function
    c_result = 100000.0
    assert abs(operator.rastrigin_func(x) - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    assert abs(operator.rastrigin_func(x)) <= 1e-8


def test_greiwank_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the greiwank function
    c_result = 0.80675915472361392488664932898246
    assert abs(operator.griewank_func(x) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the greiwank function
    c_result = 25.998676315064038533364509930834
    assert abs(operator.griewank_func(x) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the greiwank function
    c_result = 25.998676315064038533364509930834
    assert abs(operator.griewank_func(x) - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    assert abs(operator.griewank_func(x)) <= 1e-8


def test_modified_schwefel_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the schwefel function
    c_result = 1.26225708028960070805624127388
    assert abs(operator.modified_schwefel_func(x) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the schwefel function
    c_result = 3690.834427433859673328697681427
    assert abs(operator.modified_schwefel_func(x) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the schwefel function
    c_result = 6770.3478366975014068884775042534
    assert abs(operator.modified_schwefel_func(x) - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    assert abs(operator.modified_schwefel_func(x)) <= 1e-8


def test_lunacek_bi_rastrigin_cec_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the lunacek bi rastrigin function
    c_result = 69.498300562505249899913906119764
    assert abs(operator.lunacek_bi_rastrigin_cec_func(x) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the lunacek bi rastrigin function
    c_result = 4000.0
    assert abs(operator.lunacek_bi_rastrigin_cec_func(x) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the lunacek bi rastrigin function
    c_result = 1373.1325657635586594551568850875
    assert abs(operator.lunacek_bi_rastrigin_cec_func(x) - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    assert abs(operator.lunacek_bi_rastrigin_cec_func(x)) <= 1e-8


def test_lunacek_bi_rastrigin_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the lunacek bi rastrigin function
    c_result = 69.498300562505249899913906119764
    py_result = operator.lunacek_bi_rastrigin_cec_func(x)
    assert abs(py_result - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the lunacek bi rastrigin function
    c_result = 4000.0
    py_result = operator.lunacek_bi_rastrigin_cec_func(x)
    assert abs(py_result - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the lunacek bi rastrigin function
    c_result = 1373.1325657635586594551568850875
    py_result = operator.lunacek_bi_rastrigin_cec_func(x)
    assert abs(py_result - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    py_result = operator.lunacek_bi_rastrigin_cec_func(x)
    assert abs(py_result) <= 1e-8


def test_grie_rosen_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the grie_rosen function
    c_result = 407.68044871484966051866649650037
    assert abs(operator.grie_rosen_cec_func(x) - c_result) <= 1e-8

    # Max range 100 and -100
    # The result from CEC2021 C version of the grie_rosen function will NOT match due
    # to the naive summation loop used, the python version uses numpy summation which
    # is more accurate, tests were performed with a naive summation loop and the results
    # matched. The python version is more accurate so was left in place.

    # Global optimum all zeros
    x = np.zeros(10)
    py_result = operator.grie_rosen_cec_func(x)
    assert abs(py_result) <= 1e-8


def test_escaffer6_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the escaffer6 function
    c_result = 9.7378453080159417254435538779944
    py_result = operator.expanded_schaffer_f6_func(x)
    assert abs(py_result - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the escaffer6 function
    c_result = 4.9887180668865669375122706696857
    py_result = operator.expanded_schaffer_f6_func(x)
    assert abs(py_result - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the escaffer6 function
    c_result = 4.9887180668865669375122706696857
    py_result = operator.expanded_schaffer_f6_func(x)
    assert abs(py_result - c_result) <= 1e-8

    # Global optimum all zeros
    x = np.zeros(10)
    py_result = operator.expanded_schaffer_f6_func(x)
    assert abs(py_result) <= 1e-8


def test_happy_cat_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the happy_cat function
    c_result = 2.0
    assert abs(operator.happy_cat_func(x) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the happy_cat function
    c_result = 5118.2823495138645739643834531307
    assert abs(operator.happy_cat_func(x) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the happy_cat function
    c_result = 4918.2823495138645739643834531307
    assert abs(operator.happy_cat_func(x) - c_result) <= 1e-8

    # Global optimum all -1
    x = np.ones(10) * -1
    assert abs(operator.happy_cat_func(x)) <= 1e-8


def test_hgbat_func_result_matches_cec():
    x = np.ones(10)
    # The result from CEC2021 C version of the hgbat function
    c_result = 2.0
    assert abs(operator.hgbat_func(x) - c_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    # The result from CEC2021 C version of the hgbat function
    c_result = 105095.4998749937512911856174469
    assert abs(operator.hgbat_func(x) - c_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    # The result from CEC2021 C version of the hgbat function
    c_result = 104895.4998749937512911856174469
    assert abs(operator.hgbat_func(x) - c_result) <= 1e-8

    # Global optimum all -1
    x = np.ones(10) * -1
    assert abs(operator.hgbat_func(x)) <= 1e-8


def test_schaffer_f7_func():
    x = np.ones(10)
    # The result from CEC2021 py version of the schaffer_F7 function
    py_result = 1.5079726648501366
    assert abs(operator.schaffer_f7_func(x) - py_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    py_result = 208.11565843799102
    assert abs(operator.schaffer_f7_func(x) - py_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    py_result = 208.11565843799102
    assert abs(operator.schaffer_f7_func(x) - py_result) <= 1e-8

    # Global optimum all 0
    x = np.zeros(10)
    assert abs(operator.schaffer_f7_func(x)) <= 1e-8


def test_rastrigin_func():
    x = np.ones(10) * 5.12 / 100  # Scale to range -5.12 to 5.12
    py_result = 5.156257201616086
    assert abs(operator.rastrigin_func(x) - py_result) <= 1e-8

    # All max range 5.12
    x = np.ones(10) * 5.12
    py_result = 289.247137257859
    assert abs(operator.rastrigin_func(x) - py_result) <= 1e-8

    # All min range -5.12
    x = np.ones(10) * -5.12
    py_result = 289.247137257859
    assert abs(operator.rastrigin_func(x) - py_result) <= 1e-8

    # Global optimum all 0
    x = np.zeros(10)
    assert abs(operator.rastrigin_func(x)) <= 1e-8


def test_levy_func():
    x = np.ones(10)
    # The result from CEC2022 py version of the levy function
    # Note: The CEC2022 implementation has a shift of 1 hard coded in the implementation
    py_result = 6.557399012947231
    assert abs(operator.levy_func(x, 1.0) - py_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    py_result = 46079.12977788857
    assert abs(operator.levy_func(x, 1.0) - py_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    py_result = 46079.12977788854
    assert abs(operator.levy_func(x, 1.0) - py_result) <= 1e-8

    # Global optimum all 0
    x = np.zeros(10)
    assert abs(operator.levy_func(x, 1.0)) <= 1e-8


def test_zakharov_func():
    x = np.ones(10)
    # The result from CEC2022 py version of the zakharov function
    py_result = 572680.3125
    assert abs(operator.zakharov_func(x) - py_result) <= 1e-8

    # All max range 100
    x = np.ones(10) * 100
    py_result = 57191413912500.0
    assert abs(operator.zakharov_func(x) - py_result) <= 1e-8

    # All min range -100
    x = np.ones(10) * -100
    py_result = 57191413912500.0
    assert abs(operator.zakharov_func(x) - py_result) <= 1e-8

    # Global optimum all 0
    x = np.zeros(10)
    assert abs(operator.zakharov_func(x)) <= 1e-8


def test_katsuura_func():
    """
    The katsuura function has 10^D global minima, testing non-zero values
    """
    x = np.ones(10) * 1.1
    # The result from CEC2022 py version of the katsuura function
    py_result = 21.22146061081653
    assert abs(operator.katsuura_func(x) - py_result) <= 1e-8

    x = np.ones(10) * 50.25
    py_result = 17.013342726722964
    assert abs(operator.katsuura_func(x) - py_result) <= 1e-8

    x = np.ones(10) * -99.1234
    py_result = 18.786692863814668
    # the CEC2022 version doesn't use np.sum so is slightly less accurate
    assert abs(operator.katsuura_func(x) - py_result) <= 1.1e-8

    # Global optimum all 0
    x = np.zeros(10)
    assert abs(operator.katsuura_func(x)) <= 1e-8
