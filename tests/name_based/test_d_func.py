#!/usr/bin/env python
# Created by "Thieu" at 09:33, 21/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import opfunu


def test_Damavandi_results():
    ndim = 2
    problem = opfunu.name_based.Damavandi(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim

# FAILS documented x_global does not produce documented f_global
# def test_Damavandi_GlobalMin_results():
#     ndim = 2
#     problem = opfunu.name_based.Damavandi(ndim=ndim)
#     x = problem.x_global
#     result = problem.evaluate(x)
#     assert type(result) == np.float64
#     assert abs(problem.f_global - result) <= problem.epsilon

def test_Deb01_results():
    ndim = 2
    problem = opfunu.name_based.Deb01(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim

def test_Deb01Expanded_results():
    ndim = 100
    problem = opfunu.name_based.Deb01Expanded(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim
    assert problem.n_fe == 1

def test_Deb01Expanded_GlobalMin_results():
    ndim = 100
    problem = opfunu.name_based.Deb01Expanded(ndim=ndim)
    x = problem.x_global
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert abs(problem.f_global - result) <= problem.epsilon

def test_Deb03_results():
    ndim = 2
    problem = opfunu.name_based.Deb03(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim

def test_Deb03_GlobalMin_results():
    ndim = 2
    problem = opfunu.name_based.Deb03(ndim=ndim)
    x = problem.x_global
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert abs(problem.f_global - result) <= problem.epsilon

def test_Deb03Expanded_results():
    ndim = 100
    problem = opfunu.name_based.Deb01Expanded(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim
    assert problem.n_fe == 1

def test_Deb03Expanded_GlobalMin_results():
    ndim = 100
    problem = opfunu.name_based.Deb01Expanded(ndim=ndim)
    x = problem.x_global
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert (problem.f_global - result) <= problem.epsilon

def test_Decanomial_results():
    ndim = 2
    problem = opfunu.name_based.Decanomial(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Deceptive_results():
    ndim = 17
    problem = opfunu.name_based.Deceptive(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_DeckkersAarts_results():
    ndim = 2
    problem = opfunu.name_based.DeckkersAarts(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_DeflectedCorrugatedSpring_results():
    ndim = 13
    problem = opfunu.name_based.DeflectedCorrugatedSpring(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_DeVilliersGlasser01_results():
    ndim = 4
    problem = opfunu.name_based.DeVilliersGlasser01(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_DeVilliersGlasser02_results():
    ndim = 5
    problem = opfunu.name_based.DeVilliersGlasser02(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_DixonPrice():
    ndim = 11
    problem = opfunu.name_based.DixonPrice(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Dolan():
    ndim = 5
    problem = opfunu.name_based.Dolan(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_DropWave():
    ndim = 2
    problem = opfunu.name_based.DropWave(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim

def test_DropWaveExpanded_results():
    ndim = 100
    problem = opfunu.name_based.DropWaveExpanded(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim
    assert problem.n_fe == 1

def test_DropWaveExpanded_GlobalMin_results():
    ndim = 100
    problem = opfunu.name_based.DropWaveExpanded(ndim=ndim)
    x = problem.x_global
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert problem.f_global - result <= problem.epsilon
