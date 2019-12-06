# Optimization Function in Numpy (OpFuNu)
[![PyPI version](https://badge.fury.io/py/opfunu.svg)](https://badge.fury.io/py/opfunu)

## Installation

Install the [current PyPI release](https://pypi.python.org/pypi/opfunu):

```bash
pip install opfunu
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/thieunguyen5991/opfunu
```


## Example
+ All you need to do is: (Make sure your solution is a numpy 1-D array)
```python 
from opfunu.benchmark import Functions      # import our library
import numpy as np

solution = np.array([1, 2, 3, 4])          # create a solution 
func = Functions()                          # create an object

print(func._ackley__(solution))             # using function in above object
print(func._bird__(solution))
```

### Publications
+ If you see my code and data useful and use it, please cites my works here
    + Nguyen, T., Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019). Efficient Time-Series Forecasting Using Neural Network and Opposition-Based Coral Reefs Optimization. International Journal of Computational Intelligence Systems, 12(2), 1144-1161.
    
    + Nguyen, T., Tran, N., Nguyen, B. M., & Nguyen, G. (2018, November). A Resource Usage Prediction System Using Functional-Link and Genetic Algorithm Neural Network for Multivariate Cloud Metrics. In 2018 IEEE 11th Conference on Service-Oriented Computing and Applications (SOCA) (pp. 49-56). IEEE.

    + Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019, April). Building Resource Auto-scaler with Functional-Link Neural Network and Adaptive Bacterial Foraging Optimization. In International Conference on Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.

+ This project related to my another project "meta-heuristics" and "neural-network", check it here
    + https://github.com/thieunguyen5991/metaheuristics
    + https://github.com/chasebk

### Documentation 
```code 
1. http://benchmarkfcns.xyz/fcns
2. https://en.wikipedia.org/wiki/Test_functions_for_optimization
3. https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/
4. http://www.sfu.ca/~ssurjano/optimization.html
```
