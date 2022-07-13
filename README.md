# Optimization Function using Numpy (OpFuNu)
[![GitHub release](https://img.shields.io/badge/release-1.0.0-yellow.svg)]()
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/opfunu) 
[![PyPI version](https://badge.fury.io/py/opfunu.svg)](https://badge.fury.io/py/opfunu)
[![DOI version](https://zenodo.org/badge/DOI/10.5281/zenodo.3620960.svg)](https://badge.fury.io/py/opfunu)
[![Downloads](https://pepy.tech/badge/opfunu)](https://pepy.tech/project/opfunu)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Quick Notification

* The structure of version 1.0.0 is based on Object-Oriented Programming, which is entirely different from the previous version (<= 0.8.0). 
* All CEC functions from 2005, 2008, 2010, 2013, 2014, 2015, 2017, 2019, 2020, 2021, 2022 are implemented. This version is well-organized, faster and has no more bugs.
* All old code-based functions from previous version <= 0.8.0 will be removed in version 1.0.1



## Installation

Install the [current PyPI release](https://pypi.python.org/pypi/opfunu):

```bash
pip install opfunu==1.0.0
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/thieu1995/opfunu
```

## Lib's structure

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

## Examples

### How to get the function and use it

* 1st way

```python 
from opfunu.cec_based.cec2014 import F12014

func = F12014(ndim=30)
func.evaluate(func.create_solution())

## or

from opfunu.cec_based import F12014

func = F102014(ndim=50)
func.evaluate(func.create_solution())
```


* 2nd way

```python

import opfunu

funcs = opfunu.get_functions_by_classname("F12014")
func = funcs[0](ndim=10)
func.evaluate(func.create_solution())

## or

all_funcs_2014 = opfunu.get_functions_based_classname("2014")
print(all_funcs_2014)

```


## References

#### Publications

+ If you see my code and data useful and use it, please cite my works here

```code 
@software{thieu_nguyen_2020_3711682,
  author       = {Thieu Nguyen},
  title        = {A framework of Optimization Functions using Numpy (OpFuNu) for optimization problems},
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3620960},
  url          = {https://doi.org/10.5281/zenodo.3620960.}
}

```

#### Documentation 
```code 
1. dimension_based references
    1. http://benchmarkfcns.xyz/fcns
    2. https://en.wikipedia.org/wiki/Test_functions_for_optimization
    3. https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/
    4. http://www.sfu.ca/~ssurjano/optimization.html

2. type_based
    A Literature Survey of Benchmark Functions For Global Optimization Problems (2013)

3. cec
    Problem Definitions and Evaluation Criteria for the CEC 2014 
Special Session and Competition on Single Objective Real-Parameter Numerical Optimization 

```
