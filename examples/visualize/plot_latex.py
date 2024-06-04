#!/usr/bin/env python
# Created by "Thieu" at 21:17, 04/06/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.cec_based import F12010
from opfunu.name_based import Ackley02
from opfunu.utils.visualize import draw_latex

# Visualize opfunu function using method in object
f0 = F12010()
f1 = Ackley02()

## Plot using function inside the object
f0.plot_latex(f0.latex_formula, title="Latex equation")
f1.plot_latex(f1.latex_formula_global_optimum, title="Global optimum")

## Plot using module
draw_latex(f0.latex_formula_bounds, title="Boundary for Function")
draw_latex(f1.latex_formula_dimension, title=None)
