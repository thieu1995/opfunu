
Quick Start
===========


Installation
------------

Install the `current PyPI release <https://pypi.python.org/pypi/opfunu/>`_::

   $ pip install opfunu==1.0.4


Or install the development version from GitHub::

   $ pip install git+https://github.com/thieu1995/opfunu


Install directly from source code::

   $ git clone https://github.com/thieu1995/opfunu.git
   $ cd opfunu
   $ python setup.py install


Lib's structure
---------------

Current Structure::

    docs
    examples
    opfunu
        cec_based
            cec.py
            cec2005.py
            cec2008.py
            ...
            cec2021.py
            cec2022.py
        name_based
            a_func.py
            b_func.py
            ...
            y_func.py
            z_func.py
        utils
            operator.py
            visualize.py
        __init__.py
        benchmark.py
    README.md
    setup.py


Usage
-----

After installation, you can import Opfunu as any other Python module::

   $ python
   >>> import opfunu
   >>> opfunu.__version__


Let's go through some examples.


Examples
--------

How to get the function and use it

**1st way**::

    from opfunu.cec_based.cec2014 import F12014

    func = F12014(ndim=30)
    func.evaluate(func.create_solution())

    ## or

    from opfunu.cec_based import F102014

    func = F102014(ndim=50)
    func.evaluate(func.create_solution())


**2nd way**::

    import opfunu

    funcs = opfunu.get_functions_by_classname("F12014")
    func = funcs[0](ndim=10)
    func.evaluate(func.create_solution())

    ## or

    all_funcs_2014 = opfunu.get_functions_based_classname("2014")
    print(all_funcs_2014)


For more usage examples please look at [examples](/examples) folder.


.. toctree::
   :maxdepth: 4


.. toctree::
   :maxdepth: 4


.. toctree::
   :maxdepth: 4
