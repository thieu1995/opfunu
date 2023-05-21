#!/usr/bin/env python
# Created by "Thieu" at 09:25, 13/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import opfunu


def test_F12021_results():
    ndim = 10
    problem = opfunu.cec_based.F12021(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == float
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F22021_results():
    ndim = 10
    problem = opfunu.cec_based.F22021(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == float
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F32021_results():
    ndim = 10
    problem = opfunu.cec_based.F32021(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == float
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F42021_results():
    ndim = 10
    problem = opfunu.cec_based.F42021(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == float
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F52021_results():
    ndim = 10
    problem = opfunu.cec_based.F52021(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == float
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F62021_results():
    ndim = 10
    problem = opfunu.cec_based.F62021(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == float
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F72021_results():
    ndim = 10
    problem = opfunu.cec_based.F72021(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == float
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F82021_results():
    ndim = 10
    problem = opfunu.cec_based.F82021(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == float
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F92021_results():
    ndim = 10
    problem = opfunu.cec_based.F92021(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == float
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F102021_results():
    ndim = 10
    problem = opfunu.cec_based.F102021(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == float
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim
