#!/usr/bin/env python
# Created by "Thieu" at 17:27, 22/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import opfunu


def test_Giunta_results():
    ndim = 2
    problem = opfunu.name_based.Giunta(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim

def test_GiuntaExpanded_results():
    ndim = 100
    problem = opfunu.name_based.GiuntaExpanded(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim
    assert problem.n_fe == 1

# TODO: Consider how a global min and global best x can be found then enable following test
# def test_GiuntaExpanded_GlobalMin_results():
#     ndim = 100
#     problem = opfunu.name_based.GiuntaExpanded(ndim=ndim)
#     x = problem.x_global
#     result = problem.evaluate(x)
#     assert type(result) == np.float64
#     assert problem.f_global - result <= problem.epsilon

def test_GoldsteinPrice_results():
    ndim = 2
    problem = opfunu.name_based.GoldsteinPrice(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Griewank_results():
    ndim = 17
    problem = opfunu.name_based.Griewank(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Gulf_results():
    ndim = 3
    problem = opfunu.name_based.Gulf(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Gear_results():
    ndim = 4
    problem = opfunu.name_based.Gear(ndim=ndim)
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim
