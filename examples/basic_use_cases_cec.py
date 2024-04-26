#!/usr/bin/env python
# Created by "Thieu" at 10:46, 30/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np


if __name__ == '__main__':
    # # get all the available functions accepting ANY dimension
    # any_dim_cec = opfunu.get_cec_based_functions(None)
    # print(any_dim_cec)
    #
    # # get all the available separable functions accepting 2D
    # separable_2d_cec = opfunu.get_cec_based_functions(
    #     ndim=2,  # dimension
    #     separable=True,
    # )
    # print(separable_2d_cec)
    #
    # # Import specific function
    # f12005 = opfunu.cec_based.F12005()
    # print(f12005.evaluate(np.array([5, 4, 5])))      # get results
    # print(f12005.bounds)
    # print(f12005.lb)
    # print(f12005.ub)
    #
    # # f12005 = opfunu.cec_based.F12005(ndim=1010)
    # # print(f12005.lb)
    #
    # lb = [-10, ] * 15
    # ub = [10, ] * 15
    # bounds = [lb, ub]
    # f12005 = opfunu.cec_based.F12005(bounds=bounds)
    # print(f12005.lb)
    #
    # # bounds = [[-10,] * 101, [10, ] * 101]
    # # f12005 = opfunu.cec_based.F12005(bounds=bounds)
    # # print(f12005.lb)
    #
    #
    # # Plot 2d or plot 3d contours
    # # Warning ! Only working on 2d functions objects !
    # # Warning 2! change n_space to reduce the computing time
    # f22005 = opfunu.cec_based.F22005(ndim=2)
    # # opfunu.plot_2d(f22005, n_space=1000, ax=None)
    # # opfunu.plot_3d(f22005, n_space=1000, ax=None)
    #
    # # Access/change the parameters of parametrics functions
    # print(f22005.get_paras())
    #
    # # Get the global minimum for a specific dimension
    # print(f22005.f_global)
    # print(f22005.x_global)
    #
    # # Acces/plot the latex formulas
    # latex = f22005.latex_formula
    # latex = f22005.latex_formula_dimension
    # latex = f22005.latex_formula_bounds
    # latex = f22005.latex_formula_global_optimum
    # print(latex)  # --> f(\mathbf{x}) = exp(-\sum_{i=1}^{d}(x_i / \beta)^{2m}) - 2exp(-\prod_{i=1}^{d}x_i^2) \prod_{i=1}^{d}cos^ 2(x_i)
    # opfunu.plot_latex_formula(latex)

    ## Test read file matrix
    f32005 = opfunu.cec_based.F32005(ndim=10)
    x = np.ones(10)
    print(f32005.evaluate(x))
    print(f32005.f_matrix)
    print(f32005.x_global)

    problem = opfunu.cec_based.F212005(ndim=10)
    x = np.ones(10)
    print(problem.evaluate(x))
    print(problem.x_global)
    print(problem.is_succeed(problem.x_global))

    # # get all the available separable functions accepting 2D
    my_list = opfunu.get_cec_based_functions(
        ndim=2,  # dimension
        rotated=True
    )
    print(my_list)  # --> 41

    ## Get all noise function
    my_list = opfunu.get_cec_based_functions(
        randomized_term=True
    )
    print(my_list)
