import os
os.environ['JAX_ENABLE_X64'] = 'True'
import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp

from formalax import Box, ibp
from tests.nets.acasxu import get_acasxu_network


network = get_acasxu_network(1, 1)
compute_bounds = ibp(network)
in_bounds = Box(lower_bound=jnp.zeros((10, 5)), upper_bound=jnp.ones((10, 5)))

jaxpr = jax.make_jaxpr(compute_bounds)(in_bounds)
print(jaxpr)

import sys
import os

sys.path.append(r"D:\Work\jax-interval-arithmetic")
import jax_new_interpreter

in_bounds_unfold = jax.tree.flatten(in_bounds)[0]

print(in_bounds_unfold)

interval_Output_JaxPR, interval_Output_JaxPR_2 = jax_new_interpreter.eval_jaxpr(jaxpr.jaxpr, jaxpr.literals, *in_bounds_unfold)

out1, out2 = compute_bounds(in_bounds)

print("out1 :", out1)

print("interval_Output_JaxPR :", interval_Output_JaxPR)

print("out2 :", out2)

print("interval_Output_JaxPR_2 :", interval_Output_JaxPR_2)


sys.path.append(r"D:\Work\jax-interval-arithmetic\build")
import ffi_module  # type: ignore

out1 = jax_new_interpreter.to_interval_matrix(out1)

# ffi_module.checkValid(out1,interval_Output_JaxPR)
