
<p align="center"><img src=".github/img/logo.png" alt="OPFUNU" title="OPFUNU"/></p>

---


[![GitHub release](https://img.shields.io/badge/release-1.0.2-yellow.svg)](https://github.com/thieu1995/opfunu/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/opfunu) 
[![PyPI version](https://badge.fury.io/py/opfunu.svg)](https://badge.fury.io/py/opfunu)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/opfunu.svg)
![PyPI - Status](https://img.shields.io/pypi/status/opfunu.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/opfunu.svg)
[![Downloads](https://pepy.tech/badge/opfunu)](https://pepy.tech/project/opfunu)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/opfunu/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/opfunu/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/opfunu.svg)
[![Documentation Status](https://readthedocs.org/projects/opfunu/badge/?version=latest)](https://opfunu.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/thieu1995/opfunu.svg)](http://isitmaintained.com/project/thieu1995/opfunu "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/thieu1995/opfunu.svg)](http://isitmaintained.com/project/thieu1995/opfunu "Percentage of issues still open")
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/opfunu.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3620960.svg)](https://doi.org/10.5281/zenodo.3620960)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


OPFUNU (OPtimization benchmark FUnctions in NUmpy) is the largest python library for cutting-edge numerical 
optimization benchmark functions. Contains all CEC competition functions from 2005, 2008, 2010, 2013, 2014, 2015, 
2017, 2019, 2020, 2021, 2022. Besides, more than 300 traditional functions with different dimensions are implemented.

* **Free software:** GNU General Public License (GPL) V3 license
* **Total problems**: > 500 problems
* **Documentation:** https://opfunu.readthedocs.io
* **Python versions:** >= 3.7.x
* **Dependencies:** numpy, matplotlib


# Installation and Usage

### Install with pip

Install the [current PyPI release](https://pypi.python.org/pypi/opfunu):
```sh
$ pip install opfunu
```

After installation, you can import and check version of Opfunu:

```sh
$ python
>>> import opfunu
>>> opfunu.__version__

>>> dir(opfunu)
>>> help(opfunu)

>>> opfunu.FUNC_DATABASE      # List all name_based functions
>>> opfunu.CEC_DATABASE       # List all cec_based functions
>>> opfunu.ALL_DATABASE       # List all functions in this library

>>> opfunu.get_functions_by_classname("CEC2014")
>>> opfunu.get_functions_based_classname("2015")
>>> opfunu.get_functions_by_ndim(30)
>>> opfunu.get_functions_based_ndim(2)
>>> opfunu.get_all_named_functions()
>>> opfunu.get_all_cec_functions()
>>> opfunu.get_functions()
>>> opfunu.get_cecs()
```

### Lib's structure

```code

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
        validator.py
        visualize.py
    __init__.py
    benchmark.py
README.md
setup.py
```

Let's go through some examples.


### Examples

How to get the function and use it

#### 1st way

```python
from opfunu.cec_based.cec2014 import F12014

func = F12014(ndim=30)
func.evaluate(func.create_solution())

## or

from opfunu.cec_based import F102014

func = F102014(ndim=50)
func.evaluate(func.create_solution())
```


#### 2nd way

```python

import opfunu

funcs = opfunu.get_functions_by_classname("F12014")
func = funcs[0](ndim=10)
func.evaluate(func.create_solution())

## or

all_funcs_2014 = opfunu.get_functions_based_classname("2014")
print(all_funcs_2014)

```

For more usage examples please look at [examples](/examples) folder.



# Get helps (questions, problems)

* Official source code repo: https://github.com/thieu1995/opfunu
* Official document: https://opfunu.readthedocs.io/
* Download releases: https://pypi.org/project/opfunu/
* Issue tracker: https://github.com/thieu1995/opfunu/issues
* Notable changes log: https://github.com/thieu1995/opfunu/blob/master/ChangeLog.md
* Examples with different version: https://github.com/thieu1995/opfunu/blob/master/examples.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

* This project also related to our another projects which are optimization and machine learning. Check it here:
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/mafese
    * https://github.com/thieu1995/pfevaluator
    * https://github.com/thieu1995/MetaCluster
    * https://github.com/thieu1995/enoppy
    * https://github.com/thieu1995/permetrics
    * https://github.com/aiir-team


## Cite Us

If you are using opfunu in your project, we would appreciate citations:

```code
@software{thieu_nguyen_2020_3711682,
  author       = {Nguyen Van Thieu},
  title        = {Opfunu: An Open-source Python Library for Optimization Benchmark Functions},
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3620960},
  url          = {https://doi.org/10.5281/zenodo.3620960.}
}
```


## References 

```code
1. http://benchmarkfcns.xyz/fcns
2. https://en.wikipedia.org/wiki/Test_functions_for_optimization
3. https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/
4. http://www.sfu.ca/~ssurjano/optimization.html
5. A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)
6. Problem Definitions and Evaluation Criteria for the CEC 2014 Special Session and Competition on Single Objective Real-Parameter Numerical Optimization 
```
