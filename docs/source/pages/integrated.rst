Collaborative Libraries
=======================

In this section, we will guide you how to integrate our library into other Optimization frameworks.


Mealpy Library
--------------

For example::

	from opfunu.cec_based import cec2017
	f3 = cec2017.F32017(ndim=30)

	from mealpy import GA, FloatVar

	problem = {
	    "obj_func": f3.evaluate,
	    "bounds": FloatVar(lb=f3.lb, ub=f3.ub),
	    "minmax": "min",
	}
	model = GA.BaseGA(epoch=100, pop_size=50)
	gbest = model.solve(problem_dict1)
	print(f"Solution: {gbest.solution}, Fit: {gbest.target.fitness}")



ScikitOpt Library
-----------------

For example::

	from opfunu.cec_based import cec2015
	f10 = cec2015.F102015(ndim=30)

	from sko.DE import DE

	de = DE(func=f10.evaluate, lb=f10.lb, ub=f10.ub,
					size_pop=50, max_iter=800)
	best_x, best_y = de.run()
	print(f"best_x: {best_x}, best_y: {best_y}")



Opytimizer Library
------------------

For example::

	from opfunu.cec_based import cec2022
	f5 = cec2022.F52022(ndim=30)

	from opytimizer import Opytimizer
	from opytimizer.core import Function
	from opytimizer.optimizers.swarm import PSO
	from opytimizer.spaces import SearchSpace

	space = SearchSpace(n_agents=20, n_variables=f5.ndim,
	                lower_bound=f5.lb, upper_bound=f5.ub)
	optimizer = PSO()
	function = Function(f5.evaluate)

	opt = Opytimizer(space, optimizer, function)
	opt.start(n_iterations=1000)



.. toctree::
   :maxdepth: 4


.. toctree::
   :maxdepth: 4


.. toctree::
   :maxdepth: 4
