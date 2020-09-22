Introduction
############

.. image:: https://img.shields.io/badge/release-0.8.0-yellow.svg?style=svg
    :target: https://github.com/thieu1995/opfunu

.. image:: https://img.shields.io/pypi/wheel/gensim.svg?style=svg
    :target: https://pypi.python.org/pypi/opfunu

.. image:: https://badge.fury.io/py/opfunu.svg?style=svg
    :target: https://badge.fury.io/py/opfunu

.. image:: https://img.shields.io/packagist/l/doctrine/orm.svg?style=svg
    :target: https://github.com/thieu1995/opfunu/blob/master/LICENSE


This is my first open-source library written in python for optimization benchmark functions.

It contains 4 sub-packages include:
	1. dimenison_based package: first developed based on the number of dimension of functions such as 2, 3, n dimension
	2. type_baesd: second developed based on the type of functions such as uni-modal, multi-modal
	3. cec_basic: third developed based on CEC competition but no real shift value and no real rotate value
	4. cec: final developed based CEC competition


If you see my code and data useful and use it, please cites my works here::

	@software{thieu_nguyen_2020_3711682,
	  author       = {Thieu Nguyen},
	  title        = {A framework of Optimization Functions using Numpy (OpFuNu) for optimization problems},
	  year         = 2020,
	  publisher    = {Zenodo},
	  url          = {https://doi.org/10.5281/zenodo.3620960}
	}

	@article{nguyen2019efficient,
	  title={Efficient Time-Series Forecasting Using Neural Network and Opposition-Based Coral Reefs Optimization},
	  author={Nguyen, Thieu and Nguyen, Tu and Nguyen, Binh Minh and Nguyen, Giang},
	  journal={International Journal of Computational Intelligence Systems},
	  volume={12},
	  number={2},
	  pages={1144--1161},
	  year={2019},
	  publisher={Atlantis Press}
	}


Setup
#####

Install the [current PyPI release](https://pypi.python.org/pypi/opfunu):

This is a simple example::

	pip install opfunu

Or install the development version from GitHub::

	pip install git+https://github.com/thieu1995/opfunu


Examples
########

+ All you need to do is: (Make sure your solution is a numpy 1-D array).

.. code-block:: python
	:linenos:

	## For dimension_based

	from opfunu.dimension_based.benchmark2d import Functions        # import 2-d benchmark functions
	import numpy as np

	solution2d = np.array([-0.1, 1.5])                              # Solution for 2-d benchmark
	func2d = Functions()                                            # create an object

	print(func2d._bartels_conn__(solution2d))                       # using function in above object
	print(func2d._bird__(solution2d))

	## For type_based (same as dimension_based)

	from opfunu.type_based.multi_modal import Functions             # import 2-d benchmark functions
	import numpy as np


	## For CEC

	from opfunu.cec.cec2014 import Functions                        # import cec2014 functions
	import numpy as np

	cec_sol = np.array([-0.1, 1.5])                              # Solution for 2-d benchmark
	cec_func = Functions()                                            # create an object

	print(cec_func.C1(cec_sol))                                  # using function in above object from C1, ..., C30
	print(cec_func.C30(cec_sol))


	## CEC-2005 or CEC-2008

	import numpy as np
	from opfunu.cec.cec2005.F1 import Model as f1
	from opfunu.cec.cec2008.F7 import Model as f7

	solution = np.array([0.5, 1, 1.5, 2, 3, 0.9, 1.2, 2, 1, 5])

	t1 = f1()
	result = t1._main__(temp)
	print(result)

	t2 = f7()
	result = t2._main__(temp)
	print(result)



	## CEC-2010

	import numpy as np
	from opfunu.cec.cec2010.function import F1, F2, ..., F12,..

	solution = np.random.uniform(0, 1, 1000)
	result = F12(temp)
	print(result)


	## CEC-2013 (2 ways to use depend on your purpose)

	import numpy as np
	from opfunu.cec.cec2013.unconstraint import Model as M13
	from opfunu.cec.cec2014.unconstraint2 import Model as MD2

	problem_size = 10
	solution = np.random.uniform(0, 1, problem_size)


	obj = MD2(problem_size)             # Object style solve different problems with different functions
	print(obj.F1(solution))
	print(obj.F2(solution))

	obj = M13(solution)                 # Object style solve same problem with every functions
	print(obj.F1())
	print(obj.F2())


	## CEC-2014 (3 ways to use depend on your purpose)

	import numpy as np
	from opfunu.cec.cec2014.function import F1, F2, ...
	from opfunu.cec.cec2014.unconstraint2 import Model as MD2
	from opfunu.cec.cec2014.unconstraint import Model as MD

	problem_size = 10
	solution = np.random.uniform(0, 1, problem_size)


	print(F1(solution))             # Function style

	func = MD(problem_size)         # Object style solve different problems with different functions
	print(func.F1(solution))
	print(func.F2(solution))

	obj = MD2(solution)             # Object style solve same problem with every functions
	print(obj.F1())
	print(obj.F2())


	## CEC-2015
	import numpy as np
	from opfunu.cec.cec2015.function import F1, F2,...

	temp = np.random.uniform(0, 1, 10)

	result = F1(temp)
	print(result)


	## CEC basic
	import numpy as np
	from opfunu.cec_basic.cec2014 import *

	problem_size = 20
	sol = np.random.uniform(0, 1, 20)

	print(F30(sol))

	## CEC 2020 engineering problems
	from opfunu.cec.cec2020.constant import benchmark_function as BF
	from opfunu.cec.cec2020 import engineering
	from numpy.random import uniform

	for i in range(1, 26):
	    out = BF(i)         # Get object contain information about problems
	    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]

	    solution = uniform(xmin, xmax)                              ## Create solution based on information above
	    problem = "p" + str(i)                                      ## Choice the problem
	    fx, gx, hx = getattr(engineering, problem)(solution)        ## Fitness function, constraint
	    print("\n==============" + problem + "=================")
	    print("fx:", fx)
	    print("gx:", gx)
	    print("hx:", hx)
	...

References
##########

References::

	1. dimension_based references
		1. http://benchmarkfcns.xyz/fcns
		2. https://en.wikipedia.org/wiki/Test_functions_for_optimization
		3. https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/
		4. http://www.sfu.ca/~ssurjano/optimization.html

	2. type_based
		A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

	3. cec
		1. Problem Definitions and Evaluation Criteria for the CEC 2014
		2. Special Session and Competition on Single Objective Real-Parameter Numerical Optimization


This project related to my another projects which are "meta-heuristics" and "neural-network", check it here::

	1. https://github.com/thieu1995/metaheuristics
	2. https://github.com/chasebk

