#!/usr/bin/env python
# Created by "Thieu" at 10:00, 13/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import opfunu


def test_F12022_results():
    ndim = 10
    problem = opfunu.cec_based.F12022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F22022_results():
    ndim = 10
    problem = opfunu.cec_based.F22022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F32022_results():
    ndim = 10
    problem = opfunu.cec_based.F32022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F42022_results():
    ndim = 10
    problem = opfunu.cec_based.F42022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F52022_results():
    ndim = 10
    problem = opfunu.cec_based.F52022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F62022_results():
    ndim = 10
    problem = opfunu.cec_based.F62022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F72022_results():
    ndim = 10
    problem = opfunu.cec_based.F72022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F82022_results():
    ndim = 10
    problem = opfunu.cec_based.F82022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F92022_results():
    ndim = 10
    problem = opfunu.cec_based.F92022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim

def test_F92022_optimal_results():
    ndim = 10
    problem = opfunu.cec_based.F92022(ndim=ndim)
    x = problem.x_global
    result = problem.evaluate(x)
    # Original python version of CEC2022 produces optimal 2300.00000008 using x_global
    assert abs(result - problem.f_global) <= problem.epsilon

def test_F102022_results():
    ndim = 10
    problem = opfunu.cec_based.F102022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F112022_results():
    ndim = 10
    problem = opfunu.cec_based.F112022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F122022_results():
    ndim = 10
    problem = opfunu.cec_based.F122022(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim
