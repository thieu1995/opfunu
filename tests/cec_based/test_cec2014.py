#!/usr/bin/env python
# Created by "Thieu" at 15:46, 04/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import opfunu
import pytest


def test_F12014_results():
    ndim = 50
    problem = opfunu.cec_based.F12014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F22014_results():
    ndim = 50
    problem = opfunu.cec_based.F22014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F32014_results():
    ndim = 50
    problem = opfunu.cec_based.F32014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F42014_results():
    ndim = 50
    problem = opfunu.cec_based.F42014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F52014_results():
    ndim = 50
    problem = opfunu.cec_based.F52014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F62014_results():
    ndim = 50
    problem = opfunu.cec_based.F62014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F72014_results():
    ndim = 50
    problem = opfunu.cec_based.F72014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F82014_results():
    ndim = 50
    problem = opfunu.cec_based.F82014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F92014_results():
    ndim = 50
    problem = opfunu.cec_based.F92014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F102014_results():
    ndim = 50
    problem = opfunu.cec_based.F102014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F112014_results():
    ndim = 50
    problem = opfunu.cec_based.F112014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F122014_results():
    ndim = 50
    problem = opfunu.cec_based.F122014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F132014_results():
    ndim = 50
    problem = opfunu.cec_based.F132014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F142014_results():
    ndim = 50
    problem = opfunu.cec_based.F142014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F152014_results():
    ndim = 50
    problem = opfunu.cec_based.F152014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F162014_results():
    ndim = 50
    problem = opfunu.cec_based.F162014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F172014_results():
    ndim = 50
    problem = opfunu.cec_based.F172014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F182014_results():
    ndim = 50
    problem = opfunu.cec_based.F182014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F192014_results():
    ndim = 50
    problem = opfunu.cec_based.F192014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F202014_results():
    ndim = 50
    problem = opfunu.cec_based.F202014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F212014_results():
    ndim = 50
    problem = opfunu.cec_based.F212014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F222014_results():
    ndim = 50
    problem = opfunu.cec_based.F222014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F232014_results():
    ndim = 50
    problem = opfunu.cec_based.F232014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F242014_results():
    ndim = 50
    problem = opfunu.cec_based.F242014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F252014_results():
    ndim = 50
    problem = opfunu.cec_based.F252014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_F262014_results():
    ndim = 50
    problem = opfunu.cec_based.F262014(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert isinstance(problem, opfunu.cec_based.CecBenchmark)
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim



