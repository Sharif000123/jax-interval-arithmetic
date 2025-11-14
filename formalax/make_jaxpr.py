import jax
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

# D:\Work\ML Stuff\Jax_MNIST Intervall Integration.py
sys.path.append(r"D:\Work\ML Stuff")
import "Jax_MNIST Intervall Integration"  # type: ignore 

