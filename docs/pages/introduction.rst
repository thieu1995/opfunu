Introduction
############

.. image:: https://img.shields.io/badge/release-1.0.1-yellow.svg?style=svg
    :target: https://github.com/thieu1995/opfunu

.. image:: https://img.shields.io/pypi/wheel/gensim.svg?style=svg
    :target: https://pypi.python.org/pypi/opfunu

.. image:: https://badge.fury.io/py/opfunu.svg?style=svg
    :target: https://badge.fury.io/py/opfunu

.. image:: https://readthedocs.org/projects/opfunu/badge/?version=latest
   :target: https://opfunu.readthedocs.io/en/latest/?badge=latest

.. image:: https://pepy.tech/badge/opfunu
   :target: https://pepy.tech/project/opfunu

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://www.gnu.org/licenses/gpl-3.0


OPFUNU is the largest python library for cutting-edge optimization problems (benchmark, mathematical, engineering, real-world). Contains all CEC competition functions from 2005, 2008, 2010, 2013, 2014, 2015, 2017, 2019, 2020, 2021, 2022. Besides, more than 300 traditional functions with different dimensions are implemented.

The current version 1.0.1 has 3 sub-packages including:

   1. name_based package: All functions sorted as order of the alphabet
   2. cec_based package: All CEC competition functions in years (2005, 2008, 2010, 2013, 2014, 2015, 2017, 2019, 2020, 2021, 2022)
   3. engineering_based package: All functions from some papers.


If you see my code and data useful and use it, please cites my works here::

	@software{thieu_nguyen_2020_3711682,
	  author       = {Nguyen Van Thieu},
	  title        = {Opfunu: A Python Library For Optimization Functions in Numpy},
	  year         = 2020,
	  publisher    = {Zenodo},
	  url          = {https://doi.org/10.5281/zenodo.3620960}
	}


Setup
#####

Install the [current PyPI release](https://pypi.python.org/pypi/opfunu):

This is a simple example::

	pip install opfunu==1.0.1

Or install the development version from GitHub::

	pip install git+https://github.com/thieu1995/opfunu


Examples
########

**How to get the function and use it**

.. code-block:: python
    :linenos:

	from opfunu.cec_based.cec2014 import F12014

	# * 1st way

	func = F12014(ndim=30)
	func.evaluate(func.create_solution())

	## or

	from opfunu.cec_based import F12014

	func = F102014(ndim=50)
	func.evaluate(func.create_solution())
	```


	# * 2nd way

	import opfunu

	funcs = opfunu.get_functions_by_classname("F12014")
	func = funcs[0](ndim=10)
	func.evaluate(func.create_solution())

	## or

	all_funcs_2014 = opfunu.get_functions_based_classname("2014")
	print(all_funcs_2014)



References
##########

References::

    1. http://benchmarkfcns.xyz/fcns
    2. https://en.wikipedia.org/wiki/Test_functions_for_optimization
    3. https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/
    4. http://www.sfu.ca/~ssurjano/optimization.html
    5. A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)
    6. Problem Definitions and Evaluation Criteria for the CEC 2014Special Session and Competition on Single Objective Real-Parameter Numerical Optimization

