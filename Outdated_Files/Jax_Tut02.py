import jax
import jax.numpy as jnp

x = jnp.arange(5)
# print("x:",x, "\n jax.Array", jax.Array,"\n isinstance", isinstance(x, jax.Array))

# print(x.devices())

# shard means to spread out data across multiple devices

# print(x.sharding)

# @jax.jit
# def selu(x, alpha=1.67, lmbda=1.05):
#     return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# selu_jit = jax.jit(selu)

# print(selu_jit(1.0))
# print(selu(1.0))

# @jax.jit
# def f(x):
#     print(x)
#     return x + 1

# x = jnp.arange(5)
# result = f(x)
# basically, the function is being compiled to machine code when it is called, rather than when the program is run. This can make the function run faster, as it is compiled to machine code when it is called, rather than interpreted by the Python interpreter.

x = jnp.arange(5.0)
# print(jax.make_jaxpr(selu)(x))

# nested list of parameters
params = [1,2,(jnp.arange(3), jnp.arange(2))]
print("tree.structure \n",jax.tree.structure(params))
print("tree.leaves \n",jax.tree.leaves(params))

print('\nDictionary of parameters')
params = {'n': 5, 'W': jnp.ones((2, 2)), 'b': jnp.zeros(2)}

print("tree.structure \n",jax.tree.structure(params))
print("tree.leaves \n",jax.tree.leaves(params))

# Named tuple of parameters
from typing import NamedTuple

class Params(NamedTuple):
    a: int
    b: float

params = Params(1, 1.5)
print("NamedTuple\n",jax.tree.structure(params))
print("tree.structure \n",jax.tree.structure(params))
print("tree.leaves \n",jax.tree.leaves(params))

from jax import random 
key = random.key(43)
print(key)

print(random.normal(key))
print(random.normal(key))

for i in range(3):
    new_key, subkey = random.split(key)
    del key # this is to ensure that the key is not used again, del does not delete the key, it just removes the reference to the key
    
    val = random.normal(subkey)
    del subkey # this is to ensure that the key is not used again
    
    print(f"draw {i}: {val}")
    key = new_key # this is to ensure that the key is not used again