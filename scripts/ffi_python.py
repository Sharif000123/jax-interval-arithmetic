# import jax
# import jax.numpy as jnp
import sys
sys.path.append(r"D:\Work\ML Stuff\my_extension\build")

import ffi_module

ffi_module.printing("Hello from nanobind!", 42)
print(ffi_module.mult(3, 7))





# def rms_norm_ref(x, eps=1e-5):
#   scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
#   return x / scale
