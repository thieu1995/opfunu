#!/usr/bin/env python
# Created by "Thieu" at 17:03, 24/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu import draw_2d, draw_3d

## Define a custom function, for example. I will use mealpy problem as an example
from mealpy import Problem, FloatVar
import numpy as np

# Our custom problem class
class Squared(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, solution):
        x = self.decode_solution(solution)["my_var"]
        return np.sum(x ** 2)

bound = FloatVar(lb=(-10., )*20, ub=(10., )*20, name="my_var")
custom_squared = Squared(bounds=bound, minmax="min", data="Amazing", name="Squared")

## Visualize function using utility function
draw_2d(custom_squared.obj_func, custom_squared.lb, custom_squared.ub, selected_dims=(2, 3), n_points=300)
draw_3d(custom_squared.obj_func, custom_squared.lb, custom_squared.ub, selected_dims=(2, 3), n_points=300)
