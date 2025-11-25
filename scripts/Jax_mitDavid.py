import jax
from jax import nn
from jax import numpy as jnp
w = jnp.arange(100).reshape(10, 10)

b = jnp.arange(10)

print(w)

print(b)


def net(x):
    x = w @ x + b
    return jnp.max(x, 0)


net(jnp.ones((10,)))

print(jax.make_jaxpr(net))
# <function make_jaxpr(net) at 0x7c20283beca0>

jax.make_jaxpr(net)(jnp.ones((10,)))


# { lambda a:i32[10,10] b:f32[10]; c:f32[10]. let
#     d:f32[10] = dot_general[
#       dimension_numbers=(([1], [0]), ([], []))
#       preferred_element_type=float32
#     ] a c
#     e:f32[10] = add d b
#     f:f32[] = reduce_max[axes=(0,)] e
#   in (f,) }
jaxpr = jax.make_jaxpr(net)(jnp.ones((10,)))
print(jaxpr)

# { lambda a:i32[10,10] b:f32[10]; c:f32[10]. let
#     d:f32[10] = dot_general[
#       dimension_numbers=(([1], [0]), ([], []))
#       preferred_element_type=float32
#     ] a c
#     e:f32[10] = add d b
#     f:f32[] = reduce_max[axes=(0,)] e
#   in (f,) }

print(jaxpr.eqns)