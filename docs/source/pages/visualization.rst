Visualization
=============

Inside Function
---------------

You can use our visualization module to draw our function.

Code::

	from opfunu.cec_based import F12010

	# Visualize opfunu function using method in object
	f0 = F12010()

	f0.plot_2d(selected_dims=(2, 3), n_points=300, ct_cmap="viridis", ct_levels=30, ct_alpha=0.7,
	           fixed_strategy="mean", fixed_values=None, title="Contour map of the F1 CEC 2010 function",
	           x_label=None, y_label=None, figsize=(10, 8), filename="2d-f12010", exts=(".png", ".pdf"), verbose=True)

	f0.plot_3d(selected_dims=(1, 6), n_points=500, ct_cmap="viridis", ct_levels=30, ct_alpha=0.7,
	           fixed_strategy="mean", fixed_values=None, title="3D visualization of the F1 CEC 2010 function",
	           x_label=None, y_label=None, figsize=(10, 8), filename="3d-f12010", exts=(".png", ".pdf"), verbose=True)

	## Visualize opfunu function using utility function
	from opfunu import draw_2d, draw_3d

	draw_2d(f0.evaluate, f0.lb, f0.ub, selected_dims=(2, 3), n_points=300)
	draw_3d(f0.evaluate, f0.lb, f0.ub, selected_dims=(2, 3), n_points=300)


Custom Function
---------------

You can also use our visualization module to draw your custom function.

Code::

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


.. toctree::
   :maxdepth: 4


.. toctree::
   :maxdepth: 4


.. toctree::
   :maxdepth: 4
