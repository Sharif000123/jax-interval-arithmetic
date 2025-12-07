# jax-interval-arithmetic
A JAX interpreter that replaces numeric execution with interval arithmetic. The NN's are transformed into JAXPRs for interval-valued computations, allowing users to track uncertainty, and keep true values in bounds. This also helps performing safe analyses on neural network behavior for input alternation.


Other repositories needed:
C++ Boost Interval Arithmetic Library "boost_1_88_0",
Nanobind (C++ binding for C++ Boost usage)


Main program is make_jaxpr.py in the formalax folder.