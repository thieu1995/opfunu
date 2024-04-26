#!/usr/bin/env python
# Created by "Thieu" at 20:59, 29/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import opfunu
import numpy as np


if __name__ == '__main__':
    # get all the available functions accepting ANY dimension
    any_dim_functions = opfunu.get_name_based_functions(None)
    print(any_dim_functions)

    # get all the available differentiable functions accepting 2D
    differentiable_2d_functions = opfunu.get_name_based_functions(
        ndim=2,  # dimension
        differentiable=True,
    )
    print(differentiable_2d_functions)  # --> 41

    # Import specific function
    ackley03 = opfunu.name_based.Ackley03()
    print(ackley03.evaluate(np.array([ 5, 4])))      # get results

    # Plot 2d or plot 3d contours
    # Warning ! Only working on 2d functions objects !
    # Warning 2! change n_space to reduce the computing time
    ackley02 = opfunu.name_based.Ackley02()
    # opfunu.plot_2d(ackley02, n_space=1000, ax=None)
    # opfunu.plot_3d(ackley02, n_space=1000, ax=None)

    # Access/change the parameters of parametrics functions
    print(ackley02.get_paras())

    # Get the global minimum for a specific dimension
    print(ackley02.f_global)
    print(ackley02.x_global)

    # Acces/plot the latex formulas
    latex = ackley02.latex_formula
    # latex = ackley02.latex_formula_dimension
    # latex = ackley02.latex_formula_bounds
    # latex = ackley02.latex_formula_global_optimum
    print(latex)  # --> f(\mathbf{x}) = exp(-\sum_{i=1}^{d}(x_i / \beta)^{2m}) - 2exp(-\prod_{i=1}^{d}x_i^2) \prod_{i=1}^{d}cos^ 2(x_i)
    opfunu.plot_latex_formula(latex)
