# import jax
# import jax.numpy as jnp
import sys
import os

# Add the folder containing runtime dependencies first
os.add_dll_directory(r"C:\msys64\ucrt64\bin") # needed for Windows
sys.path.append(os.path.dirname(__file__))

# for direct access to the compiled module in build folder
sys.path.append(r"D:\Work\jax-interval-arithmetic\build")
import ffi_module  # type: ignore 

ffi_module.printing("From Python", 123)
# a = [1.0, 2.0]
# b = [3.0, 4.0]
print("test",type(ffi_module.Interval(1.0, 2.0)))

a = ffi_module.Interval(-1.0, 2.0)
b = ffi_module.Interval(3.0, 4.0)
# f = ffi_module.IntervalMatrix([1.0,2.0],[2.0,3.0])
x = ffi_module.IntervalMatrix([
    [ffi_module.Interval(1.0, 2.0),ffi_module.Interval(1.0, 2.0)],
    [ffi_module.Interval(1.0, 2.0),ffi_module.Interval(1.0, 2.0)]
    ])
y = ffi_module.IntervalMatrix([
    [ffi_module.Interval(1.0, 2.0),ffi_module.Interval(1.0, 2.0)],
    [ffi_module.Interval(1.0, 2.0),ffi_module.Interval(1.0, 2.0)]
    ])

print("Interval a:", a)
print(type(x))
print("Interval b:", b)
c = ffi_module.add(a, b)
print("Interval c (a + b):", c)
print("mult:", ffi_module.mult(a, b))
print("relu:", ffi_module.relu(a))
print("relu:", ffi_module.matrixMult(x,y))






# def rms_norm_ref(x, eps=1e-5):
#   scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
#   return x / scale
