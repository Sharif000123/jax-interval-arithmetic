import jax, timeit

import jax.numpy as jnp


global_list = []

def log2(x):
    global_list.append(x)
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2

# print(jax.make_jaxpr(log2)(3.0))

# print("---------------------")

def log2_with_print(x):
  print("printed x:", x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

# print(jax.make_jaxpr(log2_with_print)(3.))


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)

time_taken1 = timeit.timeit(lambda: selu(x).block_until_ready(), number=1)
print(f"Time taken: {time_taken1} seconds")


seluJit = jax.jit(selu)
seluJit(x).block_until_ready()
time_taken2 = timeit.timeit(lambda: seluJit(x).block_until_ready(), number=1)
print(f"Time taken: {time_taken2} seconds")

# Condition on value of x.

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

# jax.jit(f)(10)  # Raises an error
f(10)  # Works fine



# While loop conditioned on x and n.

def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

# jax.jit(g)(10, 20)  # Raises an error
g(10, 20)  # Works fine


# While loop conditioned on x and n with a jitted body.

@jax.jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

g_inner_jitted(10, 20)

#print(jax.make_jaxpr(g_inner_jitted)(10, 20)) # jaxpr doesnt work, because the function is not fully defined yet, so it cannot be printed, but it can be run, and it will work fine
print(jax.make_jaxpr(loop_body)(0)) # this works fine, because the function is fully defined, and it can be printed, and it will work fine

fJitCorrect = jax.jit(f, static_argnums=0)  # Correctly jits the function
print(fJitCorrect(10))  # Works fine

gJitCorrect = jax.jit(g, static_argnames=['n'])  # Correctly jits the function
print(gJitCorrect(10, 20))  # Works fine

from functools import partial


def unjittedLoopBody(prev_i):
    return prev_i + 1

def gInnerJittedPartial(x, n):
    i = 0
    while i < n:
        i = jax.jit(partial(unjittedLoopBody))(i)  # Not recommended, because it will recompile the function every time it is called, since each time the function is called, it will be recompiled, as the partial function returns a function with different hash
    return x + i

def gInnerJittedLambda(x, n):
    i = 0
    while i < n:
        i = jax.jit(lambda x: unjittedLoopBody(x))(i)  # Not recommended, because it will recompile the function every time it is called, since each time the function is called, it will be recompiled, as the lambda function returns a function with different hash
    return x + i

def gUnjittedNormal(x, n):
    i = 0
    while i < n:
        i = jax.jit(unjittedLoopBody)(i) # ok, since function is cached
    return x + i

print("jit called in a loop with partials:")
time_taken3 = timeit.timeit(lambda: gInnerJittedPartial(10, 20).block_until_ready(), number=1) # the number indicates how many times the function will be run
print(f"Time taken with JIT (second func.): {time_taken3} seconds")

print("jit called in a loop with lambda:")
time_taken4 = timeit.timeit(lambda: gInnerJittedLambda(10, 20).block_until_ready(), number=1) # the number indicates how many times the function will be run
print(f"Time taken with JIT (second func.): {time_taken4} seconds")

print("jit called in a loop with normal function:")
time_taken5 = timeit.timeit(lambda: gUnjittedNormal(10, 20).block_until_ready(), number=1) # the number indicates how many times the function will be run
print(f"Time taken with JIT (second func.): {time_taken5} seconds")