

## Example for version <= 0.8.0

+ All you need to do is: (Make sure your solution is a numpy 1-D array)

```python 
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

### CEC 2020 - engineering problem 

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

-- The problem 1-23 and 25 is DONE, the problem 24th is not DONE yet.
...


```
