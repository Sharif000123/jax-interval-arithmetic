import jax.experimental
from mpmath import iv
from functools import wraps

from jax import lax
from jax.extend import core
from jax._src.util import safe_map

from tqdm import tqdm

import jax.numpy as jnp

import jax
# from jax import jit
# import numpy as np

# Configurations mpmath
iv.dps = 15
iv.pretty = True

def f(x):
    
    return x @ x

# def interval_exp(x, **kwargs):
    
#     print("Interval Exp:", x)
#     return iv.exp(jnp.float32(x))


def interval_add(x, y, **kwargs):
    print("interval_add:", type(x), len(x), type(y))

    return x + y

# @jax.jit
def interval_mult_elementwise(x, y, **kwargs):
    print("elementwise mult:")
    print(f"x dimensions: {x.rows}x{x.cols}, y dimensions: {y.rows}x{y.cols}")
    if x.rows != y.rows or x.cols != y.cols:
        raise ValueError(f"Dimension mismatch.")
    newMatrix = iv.matrix(x.rows, x.cols) # creating new same size matrix
    for i in range(x.rows):
        for j in range(x.cols):
            newMatrix [i,j] = x[i,j] * y[i,j]
    return newMatrix

def interval_matrix_mult(x, y, **kwargs):
    # print("interval_matrix_mult:\nx:")#, x,"\ny:", y)
    # print("---")
    if x.cols != y.rows:
        raise ValueError(f"Matrix multiplication requires both matrices to have compatible dimensions.\nDimensions: x = {x.rows}x{x.cols}, y = {y.rows}x{y.cols}")
    return x * y

def interval_pow(x, y, **kwargs):
    print("interval_pow:")#, x, y)
    return x**y

# def interval_matrix_inverse(x, **kwargs):
#     return x**-1

def interval_transpose(x,**kwargs):
    print("interval_transpose:")#, x)
    return x.T

def interval_min(x,**kwargs):
    print("interval_min:")#, x)
    return min(x)

def interval_max(x,**kwargs):
    print("interval_min:")#, x)
    return max(x)

def interval_relu(x, y, **kwargs):
    print("relu x:")#, x)
    print("y:")#, y)
    newMatrix = iv.matrix(x.rows, x.cols)
    for i in range(x.rows):
        for j in range(x.cols):
            newMatrix [i,j] = max(x[i,j],y[i,j])
    return newMatrix

def interval_pjit(*args, **kwargs):
    return args[0]  # TODO

interval_primitives = {
    # lax.exp_p: interval_exp,
    lax.add_p: interval_add,
    lax.mul_p: interval_mult_elementwise,
    lax.dot_general_p: interval_matrix_mult,
    lax.pow_p: interval_pow,
    lax.transpose_p: interval_transpose,
    lax.reduce_min_p: interval_min,
    lax.reduce_max_p: interval_max,
    lax.max_p: interval_relu,
    # implement pjit, as recursive call (weil pjit verschachtelt weitere Jaxprs)
    # jax.experimental.pjit.pjit_p: interval_pjit,
    # lax.erf_inv_p: interval_matrix_inverse
    
}

# closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(5))
# print(closed_jaxpr.jaxpr)
# print(closed_jaxpr.literals)


# this is a simple Jaxpr interpreter that evaluates a Jaxpr by interpreting it directly.
def eval_jaxpr(jaxpr, consts, *args):

    env = {}
    
    # retrieving either literals or computed results
    def read(var):
        # Literals are values baked into jaxpr
        if type(var) is core.Literal: # check if var is a literal, a literal is constant value
            return iv.mpf(var.val.item())
        return env[var]
    
    #stores computed results in environment
    def write(var, val): # write to the environment
        env[var] = val
        
    
    # Convert inputs to interval matrices
    interval_args = [iv.matrix(arg) for arg in args]
    if len(interval_args) != len(jaxpr.invars):
        raise ValueError(f"Number of inputs ({len(interval_args)}) does not match number of JAXPR invars ({len(jaxpr.invars)})")

    # binding args and consts to environment
    safe_map(write, jaxpr.invars, interval_args) # safe_map applies write func to each element in jaxpr.invars & args, jaxpr.invars are input variables, and args. The difference is that args are the values of the input variables, while jaxpr.invars are the variables that hold the input variables.
    interval_consts = [iv.mpf(const) for const in consts]
    safe_map(write, jaxpr.constvars, interval_consts) # safe_map applies write func to each jaxpr.constvars & consts, jaxpr.constvars are constant variables, and consts are the constants. The difference is that consts are the values of the constants, while jaxpr.constvars are the variables that hold the constants.
    
    # Loop through equations and eval primitives using `bind`
    for eqn in tqdm(jaxpr.eqns): # looping through the equations in the jaxpr

        # # From Copilot code: If the primitive is "pjit", skip its evaluation.
        # if hasattr(eqn.primitive, 'name') and eqn.primitive.name == "pjit":
        #     print("Skipping primitive: pjit (copying invars to outvars)")
        #     invals = safe_map(read, eqn.invars)
        #     # Slice invals if there are more values than outvars.
        #     vals_to_write = invals[:len(eqn.outvars)]
        #     safe_map(write, eqn.outvars, vals_to_write)
        #     continue
        invars = safe_map(read, eqn.invars) # apply read for eachinput var in eqn.invars
        #`bind` is how a primitive is called
        if eqn.primitive in interval_primitives:
            outvals = interval_primitives[eqn.primitive](*invars, **eqn.params)
        else:
            raise NotImplementedError(f"Primitive not in interval_primitives: {eqn.primitive}")
            # print("Primitive not in interval_primitives:", eqn.primitive)
        
        #Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results: # check if the primitive returns multiple results
            #if not, wrap the result in a list to make it easier to handle, if it does, we will just use the result as is
            outvals = [outvals]
        
        #write results of primitive into env.
        safe_map(write, eqn.outvars, outvals)
    
    # read final result of Jaxpr from env.
    return safe_map(read, jaxpr.outvars) # read the output variables from the environment and return them as a list

#func call
closed_jaxpr = jax.make_jaxpr(f)(jnp.ones((2,2)))
# print("closed_jaxpr:")
# print(closed_jaxpr)
# print("closed_jaxpr.jaxpr:")
# print(closed_jaxpr.jaxpr)
# print("closed_jaxpr.literals:")
# print(closed_jaxpr.literals)
evalJaxpr = eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, jnp.array([[3.,2.],[2.,3.,]]))


print("evalJaxpr:")
print(evalJaxpr)
print("evalJaxpr[0]:")
print(evalJaxpr[0])

# print(eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, jnp.ones(5)))
'''
inverse_registry = {}

inverse_registry[lax.exp_p] = jnp.log
inverse_registry[lax.tanh_p] = jnp.arctanh



def inverse(fun):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        #since we assume unary functions (functions with one arg only), we wont worry about flattening and unflattening args
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        out = inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        return out[0]
    return wrapped


def inverse_jaxpr(jaxpr, consts, *args):
    env = {}
    
    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]
    
    def write(var, val):
        env[var] = val
    #args now correspond to Jaxpr outvars
    safe_map(write, jaxpr.outvars, args)
    safe_map(write, jaxpr.constvars, consts)
    
    # loop backwards
    for eqn in jaxpr.eqns[::-1]:
        #outvars are now invars
        invals = safe_map(read, eqn.outvars)
        if eqn.primitive not in inverse_registry:
            raise NotImplementedError(f"{eqn.primitive} does not have registered inverse.")
        #assuming a unary function
        outval = inverse_registry[eqn.primitive](*invals)
        safe_map(write, eqn.invars, [outval])
    return safe_map(read, jaxpr.invars)



# f_inv = inverse(f)
# assert jnp.allclose(f_inv(f(1.0)), 1.0)
# assert jnp.allclose(f_inv(f(1.0)), 1.0, rtol=1e-5, atol=1e-6)

# print(f_inv(f(1.0)))
# print("Diff:", f_inv(f(1.0)) - 1.0)


# print(jax.make_jaxpr(inverse(f))(f(1.)))

'''