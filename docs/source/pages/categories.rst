Function Categories
===================


In general, unconstrained problems can be classified into two categories: test functions and real-world problems::

	1. Test functions are artificial problems, and can be used to evaluate the behavior of an algorithm in sometimes diverse and difficult situations.
	Artificial problems may include single global minimum, single or multiple global minima in the presence of
	many local minima, long narrow valleys, null-space effects and flat surfaces. These problems can be easily manipulated and modified to test the algorithms in
	diverse scenarios.

	2.On the other hand, real-world problems originate from different fields such as physics, chemistry, engineering, mathematics etc. These problems are
	hard to manipulate and may contain complicated algebraic or differential expressions and may require a significant amount of data to compile.


The objective functions could be characterized as:

1. continuous, discontinuous
2. linear, non-linear
3. convex, non-conxex
4. unimodal, multimodal,
5. separable and non-separable.


Before solving an optimization problem. Need to ask question::

	1. What aspects of the function landscape make the optimization process difficult?
	2. What type of a priori knowledge is most effective for searching particular types of function landscape?

==>  **To answer these questions, benchmark functions can be classified with features like modality, basins, valleys, separability and dimensionality.**

**1) Modality**
The number of ambiguous peaks in the function landscape corresponds to the modality of a function. If algorithms encounters these peaks during a  search
process, there is a tendency that the algorithm may be trapped in one of such peaks. This will have a negative impact on the search process,  as this can
direct the search away from the true optimal solutions.

**2) Basins**
A relatively steep decline surrounding a large area is called a basin. Optimization algorithms can be easily attracted to such regions.  Once in these
regions, the search process of an algorithm is severely hampered. This is due to lack of information to direct the search process towards the minimum.
A basin corresponds to the plateau for a maximization problem, and a problem can have multiple plateaus.

**3) Valleys**
A valley occurs when a narrow area of little change is surrounded by regions of steep descent. As with the basins, minimizers are  initially attracted
to this region. The progress of a search process of an algorithm may be slowed down considerably on the floor of the valley

**4) Separability**
The separability is a measure of difficulty of different benchmark functions
In general, separable functions are relatively easy to solve, when compared with their inseperable counterpart,
because each variable of a function is independent of the other variables.
If all the parameters or variables are independent, then a sequence of n independent optimization processes can be performed. In other words, a function of p
variables is called separable, if it can written as a sum of p functions of just one variable
On the other hand, a function is called non-separable, if its variables show inter-relation among themselves or are not independent If the objective function
variables are independent of each other, then the objective functions can be decomposed into sub-objective function.
Then, each of these sub-objectives involves only one decision variable, while treating all the others as constant.

**5) Dimensionality**
The difficulty of a problem generally increases with its dimensionality. When the number of parameters or dimension increases, the search space also increases
exponentially. For highly nonlinear problems, this dimensionality may be a significant barrier for almost all optimization algorithms.

+ Multimodal: A function with more than one local optimum. The one has many local minima are among the most difficult class of problems for many algorithms.

+ Functions with flat surfaces pose a difficulty for the algorithms, since the flatness of the function does not give the algorithm any information to direct
the search process towards the minima.

.. toctree::
   :maxdepth: 4


.. toctree::
   :maxdepth: 4


.. toctree::
   :maxdepth: 4
