#!/usr/bin/env python
# Created by "Thieu" at 20:19, 19/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import opfunu


def test_CamelThreeHump_results():
    ndim = 2
    problem = opfunu.name_based.CamelThreeHump(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_CamelSixHump_results():
    ndim = 2
    problem = opfunu.name_based.CamelSixHump(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_ChenBird_results():
    ndim = 2
    problem = opfunu.name_based.ChenBird(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_ChenV_results():
    ndim = 2
    problem = opfunu.name_based.ChenV(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Chichinadze_results():
    ndim = 2
    problem = opfunu.name_based.Chichinadze(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_ChungReynolds_results():
    ndim = 9
    problem = opfunu.name_based.ChungReynolds(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Cigar_results():
    ndim = 9
    problem = opfunu.name_based.Cigar(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Cola_results():
    ndim = 17
    problem = opfunu.name_based.Cola(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Colville_results():
    ndim = 4
    problem = opfunu.name_based.Colville(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Corana_results():
    ndim = 4
    problem = opfunu.name_based.Corana(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_CosineMixture_results():
    ndim = 11
    problem = opfunu.name_based.CosineMixture(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_CrossInTray_results():
    ndim = 2
    problem = opfunu.name_based.CrossInTray(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_CrossLegTable_results():
    ndim = 2
    problem = opfunu.name_based.CrossLegTable(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_CrownedCross_results():
    ndim = 2
    problem = opfunu.name_based.CrownedCross(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Cube_results():
    ndim = 2
    problem = opfunu.name_based.Cube(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim
