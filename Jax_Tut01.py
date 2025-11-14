import jax.numpy as jnp

def selu(x, alpha=1.67, lmbda=1.05): # SELU activation function, lambda is the scaling factor, in simple terms, it is the slope of the function, alpha is the negative slope, and x is the input
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha) # if x is greater than 0, return x, else return alpha * e^x - alpha

x = jnp.arange(5.1) # create an array from 0 to upper(x) - 1
# print(x)
print(selu(x))

from jax import random
import timeit

key = random.key(1701) # creating a random key
x = random.normal(key, (1_000_000,)) # creating a random normal distribution with 1,000,000 values. A random normal distribution is a distribution where the values are centered around 0 and the standard deviation is 1
time_taken = timeit.timeit(lambda: selu(x).block_until_ready(), number=1) # wait for the computation to finish before printing the time it took to compute the values
# time_taken = timeit.timeit(selu(x).block_until_ready()) # wait for the computation to finish before printing the time it took to compute the values
print(f"Time taken: {time_taken} seconds")

from jax import jit
# jit, or Just-In-Time compilation, is a technique that compiles the function when it is called, rather than when the program is run. This can make the function run faster, as it is compiled to machine code when it is called, rather than interpreted by the Python interpreter.

selu_jit = jit(selu) # compile the function using XLA (Accelerated Linear Algebra) to make it faster
_ = selu_jit(x)
time_taken2 = timeit.timeit(lambda: selu_jit(x).block_until_ready(), number=1) # the number indicates how many times the function will be run
print(f"Time taken with JIT (second func.): {time_taken2} seconds")

from jax import grad

def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x))) # this is the logistic function 

print("Logistic function",sum_logistic(0.0)) # print the logistic function at 0.0
x_small = jnp.arange(3.) # create an array from 0. to 2.
derivative_fn = grad(sum_logistic) # this is the derivative of the logistic function, a derivative is a measure of how a function changes as its input changes
print(derivative_fn(x_small)) # print the derivative of the logistic function at x_small

def first_finite_differences(f,x,eps=1E-3):
    return jnp.array([(f(x+eps*v) - f(x - eps * v)) / (2 * eps) for v in jnp.eye(len(x))])

print(first_finite_differences(sum_logistic, x_small)) # this is the first finite difference of the logistic function at x_small, the difference is basically the derivative of the function, and the derivative is a measure of how the function changes as its input changes

print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0)) # this is the third derivative of the logistic function at 1.0, this basically means that the function is being differentiated three times, and the result is printed

from jax import jacobian
print(jacobian(sum_logistic)(x_small)) # this is the jacobian of the logistic function at x_small, the jacobian is a matrix of all the first-order partial derivatives of a function

from jax import jacfwd, jacrev
def hessian(fun):
    return jit(jacfwd(jacrev(fun))) # this is the hessian of the function, the hessian is the matrix of all the second-order partial derivatives of a function, simply the hessian matrix is the jacobian of the jacobian of the function, so the jacobian of the jacobian of the function is the hessian of the function. So values in the hessian matrix are the second-order partial derivatives of the function

print(hessian(sum_logistic)(x_small)) # this is the hessian of the logistic function at x_small


key1, key2 = random.split(key) # split the key into two keys
mat = random.normal(key1, (150 ,100)) # create a random normal distribution with 150 rows and 100 columns, a distribution is a set of values that are centered around a mean value, and the standard deviation is 1
batched_x = random.normal(key2, (10, 100)) # create a random normal distribution with 10 rows and 100 columns

def apply_matrix(x):
    return jnp.dot(mat, x) # this is the dot product of the matrix and the input, and the dot product is a measure of how much two vectors are similar

def natively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched]) # this is the natively batched apply matrix, this is the stack of the apply matrix of the batched input, the stack is a function that takes a sequence of arrays and stacks them into a single array. A native batched apply matrix is a function that takes a batch of inputs and applies the matrix to each input in the batch, and a batch is a set of inputs that are processed together, in this case, the batch is a set of inputs that are processed together by the matrix and the input.

print('Natively batched')
time_taken3 = timeit.timeit(lambda: natively_batched_apply_matrix(batched_x).block_until_ready(), number=1) # the number indicates how many times the function will be run
print(f"Time taken with JIT (second func.): {time_taken3} seconds")

import numpy as np
@jit
def batched_apply_matrix(batched_x):
    return jnp.dot(batched_x, mat.T)

print('Manually batched')
time_taken4 = timeit.timeit(lambda: batched_apply_matrix(batched_x).block_until_ready(), number=1) # the number indicates how many times the function will be run
print(f"Time taken with JIT (second func.): {time_taken4} seconds")

from jax import vmap

@jit
def vmap_batched_apply_matrix(batched_x):
    return vmap(apply_matrix)(batched_x)

np.testing.assert_allclose(natively_batched_apply_matrix(batched_x),
                           vmap_batched_apply_matrix(batched_x), atol=1E-4,rtol=1E-4)
print('Auto-vectorized with vmap')
time_taken5 = timeit.timeit(lambda: vmap_batched_apply_matrix(batched_x).block_until_ready(), number=1) # the number indicates how many times the function will be run
print(f"Time taken with JIT (second func.): {time_taken5} seconds")


