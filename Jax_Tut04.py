import jax
import jax.numpy as jnp


x = jnp.arange(5)
w = jnp.array([2.,3.,4.])
y = jnp.array([2.,3.,4.,5.])

def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i - 1: i + 2], w))
    return jnp.array(output)

# print(convolve(x, w))
# print(jax.make_jaxpr(convolve)(x, w))

xs = jnp.stack([x,x]) #[[0 1 2 3 4] [0 1 2 3 4]]
ws = jnp.stack([w,w]) # [[2. 3. 4.] [2. 3. 4.]]

def manuallyBatchedConvolve(xs, ws):
    output = []
    for i in range(xs.shape[0]):
        output.append(convolve(xs[i], ws[i]))
    return jnp.array(output)

# print('Manually Batched Convolve:\n',manuallyBatchedConvolve(xs, ws))

def manuallyVectorizedConvolve(xs, ws):
    output = []
    for i in range(1, xs.shape[1] - 1):
        output.append(jnp.sum(xs[:, i - 1: i + 2] * ws, axis=1))
    return jnp.stack(output, axis=1)

print('Manually Vectorized Convolve:\n',manuallyVectorizedConvolve(xs, ws))

#Automatically vectorization

autoBatchConvolve = jax.vmap(convolve)
print('Automatically Batched Convolve:\n',autoBatchConvolve(xs, ws))

autoBatchConvolve2 = jax.vmap(convolve, in_axes=1, out_axes=1)
xst = jnp.transpose(xs) # transpose is a function that takes a matrix and flips it over its diagonal, so the rows become columns and the columns become rows
wst = jnp.transpose(ws)
print('Auto Batch Convolve 2:\n ',autoBatchConvolve2(xst, wst))

batchConvolveV3 = jax.vmap(convolve, in_axes=[0, None])

print('Batch Convolve V3:\n',batchConvolveV3(xs, w))

jittedBatchConvolve = jax.jit(autoBatchConvolve)

print('Jitted Batch Convolve:\n',jittedBatchConvolve(xs, ws))