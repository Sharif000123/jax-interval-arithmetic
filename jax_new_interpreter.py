import jax.numpy as jnp
from tqdm import tqdm
from jax._src.util import safe_map
from jax.extend import core
from jax import lax
from functools import wraps
from mpmath import iv
import jax.experimental
import numpy as np
import sys
from jax import config
import jax
import os
os.environ['JAX_ENABLE_X64'] = 'True'
config.update("jax_enable_x64", True)

os.add_dll_directory(r"C:\msys64\ucrt64\bin")  # needed for Windows


sys.path.append(r"D:\Work\jax-interval-arithmetic\build")
import ffi_module  # type: ignore


# Add the folder containing runtime dependencies first
sys.path.append(os.path.dirname(__file__))


# Configurations mpmath
iv.dps = 15
iv.pretty = True


def f(x):

    return x @ x


def interval_add(x, y):

    # print("interval_add:", type(x), len(x), type(y))
    # if y has more than one rows, use elementwise add function

    return ffi_module.matrixAddBias(x, y)


def interval_martix_sub(x, y):

    # print("interval_sub:", type(x), len(x), type(y))
    # print(f"x dimensions: {ffi_module.rows(x)}x{ffi_module.cols(x)}, y dimensions: {ffi_module.rows(y)}x{ffi_module.cols(y)}")

    return ffi_module.matrixMatrixSub(x, y)


def integer_interval_pow(x, y):
    # print("interval_add x :", x , " y ", y)
    return ffi_module.matrixPow(x, y)


def interval_abs(x):
    # print("interval_abs x :", x)
    return ffi_module.matrixAbs(x)


def interval_mult_elementwise(x, y, **kwargs):
    # print("interval_matrix_div:")  # , x, y)
    return ffi_module.matrixMultElementwise(x, y)


def interval_matrix_mult(x, y, **kwargs):

    x_cols = ffi_module.cols(x)
    y_rows = ffi_module.rows(y)
    if x_cols != y_rows:
        raise ValueError(
            f"Matrix multiplication requires both matrices to have compatible dimensions.\nDimensions: x = {ffi_module.rows(x)}x{x_cols}, y = {y_rows}x{ffi_module.cols(y)}")
    return ffi_module.matrixMult(x, y)


def interval_matrix_div(x, y, **kwargs):
    # print("interval_matrix_div:")  # , x, y)
    return ffi_module.matrixDiv(x, y)


def interval_pow(x, y, **kwargs):
    # print("interval_pow:")  # , x, y)
    return x**y


def interval_min(x, **kwargs):
    # print("interval_min:")  # , x)
    return min(x)


def interval_reduce_max(x, **kwargs):
    # print("interval_min:")  # , x)
    return max(x)


def interval_max(x, y, **kwargs):  # this function is called by the Jaxpr interpreter
    # if y is scalar, convert to 1x1 IntervalMatrix
    if hasattr(y, "lower") and hasattr(y, "upper"):
        scalar_matrix = ffi_module.IntervalMatrix(
            [[y]])  # creating 1x1 matrix, containing y
        return ffi_module.max(x, scalar_matrix)
    else:
        # print(f"x dimensions: {ffi_module.rows(x)}x{ffi_module.cols(x)}, y dimensions: {ffi_module.rows(y)}x{ffi_module.cols(y)}")
        return ffi_module.max(x, y)


def interval_min(x, y, **kwargs):  # this function is called by the Jaxpr interpreter
    # if y is scalar, convert to 1x1 IntervalMatrix
    if hasattr(y, "lower") and hasattr(y, "upper"):
        scalar_matrix = ffi_module.IntervalMatrix(
            [[y]])  # creating 1x1 matrix, containing y
        return ffi_module.min(x, scalar_matrix)
    else:
        # print(f"x dimensions: {ffi_module.rows(x)}x{ffi_module.cols(x)}, y dimensions: {ffi_module.rows(y)}x{ffi_module.cols(y)}")
        return ffi_module.min(x, y)


def interval_pjit(*args, **kwargs):
    return args[0]  # TODO


def to_interval_matrix(array, half_range: float = 0.0):

    # Convert 2D array-like to ffi_module.IntervalMatrix of intervals [x-half_range, x+half_range]
    arr = np.asarray(array)
    if arr.ndim == 1:
        # treat 1D as single-row matrix
        arr = arr[np.newaxis, :]
    rows = []
    # Makes sure that intervals are same size
    for row in arr:
        hr = float(half_range)
        rows.append([ffi_module.Interval(
            float(x) - hr - 1e-15 * max(abs(float(x)), 1.0),
            float(x) + hr + 1e-15 * max(abs(float(x)), 1.0)
        ) for x in row])
    return ffi_module.IntervalMatrix(rows)


def scalar_to_interval(x, half_range: float = 0.0):
    # defaults to 0.0 if half_range too small
    if half_range < 1e-15:
        return ffi_module.scalarToInterval(x)
    return ffi_module.scalarToInterval(x, half_range)


def elementwise_add(A, B, **kwargs):
    return ffi_module.elementwiseAdd(A, B)


def transpose_interval_matrix(M, **kwargs):
    return ffi_module.transposeIntervalMatrix(M)


def p_jitting(*args, jaxpr, **kwargs):
    return eval_jaxpr(jaxpr.jaxpr, jaxpr.literals, *args)


def broadcast_in_dim(operandArr, *, shape, broadcast_dimensions, **kwargs):

    output_shape = tuple(int(s) for s in shape)
    if len(output_shape) == 1:
        output_rows, output_cols = 1, output_shape[0]
    elif len(output_shape) == 2:
        output_rows, output_cols = 1, output_shape
    else:
        raise NotImplementedError(
            "broadcast_in_dim only supports 1D or 2D output shapes.")

    # If operand = scalar
    if hasattr(operandArr, "lower") and hasattr(operandArr, "upper"):
        rows = [[operandArr for _ in range(output_cols)]
                for _ in range(output_rows)]
        return ffi_module.IntervalMatrix(rows)
    return operandArr


def custom_jvp_call(*args, **kwargs):

    # assigns jaxpr_obj to the inner jaxpr
    jaxpr_obj = kwargs.get("jaxpr") or kwargs.get(
        "call_jaxpr") or kwargs.get("call_jaxpr_jaxpr")

    # checks if jaxpr_obj is a closedJaxpr object, for which .jaxpr and .literals are available
    if jaxpr_obj is None:
        raise NotImplementedError(
            "custom_jvp_call: missing inner jaxpr in params")

    # If a closedJaxpr-like is provided, it carries a .jaxpr and .literals
    if hasattr(jaxpr_obj, "jaxpr") and hasattr(jaxpr_obj, "literals"):
        inner_jaxpr = jaxpr_obj.jaxpr
        inner_consts = jaxpr_obj.literals
    else:
        # raw jaxpr (no literals)
        inner_jaxpr = jaxpr_obj
        inner_consts = ()
    # recurse into interpreter to evaluate inner jaxpr with same args (eval_jaxpr)
    return eval_jaxpr(inner_jaxpr, inner_consts, *args)


interval_primitives = {
    # lax.exp_p: interval_exp,
    lax.add_p: interval_add,
    lax.mul_p: interval_mult_elementwise,
    lax.dot_general_p: interval_matrix_mult,
    lax.div_p: interval_matrix_div,
    lax.pow_p: interval_pow,
    lax.transpose_p: transpose_interval_matrix,
    lax.reduce_min_p: interval_min,
    lax.reduce_max_p: interval_reduce_max,
    lax.max_p: interval_max,
    lax.min_p: interval_min,
    lax.sub_p: interval_martix_sub,
    lax.integer_pow_p: integer_interval_pow,
    lax.abs_p: interval_abs,
    # jax.experimental.pjit.pjit_p: p_jitting,
    # jax.jit: p_jitting,
    jax.extend.core.primitives.broadcast_in_dim_p: broadcast_in_dim,
    jax.extend.core.primitives.custom_jvp_call_p: custom_jvp_call,
    # implement pjit, as recursive call (weil pjit verschachtelt weitere Jaxprs)
    # jax.experimental.pjit.pjit_p: interval_pjit,
    # lax.erf_inv_p: interval_matrix_inverse

}

# Try to add pjit_p; fall back to jax.jit if wrong jax version installed
try:
    # old jax version
    interval_primitives[jax.experimental.pjit.pjit_p] = p_jitting
except (AttributeError, ImportError):
    try:
        interval_primitives[jax.jit] = p_jitting  # new jax version
    except (AttributeError, ImportError):
        Exception("Could not add pjit to interval_primitives.")


# this is a simple Jaxpr interpreter that evaluates a Jaxpr by interpreting it directly.
def eval_jaxpr(jaxpr, consts, *args, half_range: float = 0.0):

    env = {}

    def convert(val):
        numpyArray = np.asarray(val)
        if numpyArray.ndim == 0:
            return scalar_to_interval(float(numpyArray), half_range)
        else:
            return to_interval_matrix(numpyArray, half_range)

    # retrieving either literals or computed results
    def read(var):
        # Literals are values baked into jaxpr
        if type(var) is core.Literal:  # check if var is a literal, a literal is constant value
            v = var.val
            return convert(v)
        return env[var]

    # stores computed results in environment

    def write(var, val):  # write to the environment
        if isinstance(val, jax.Array):
            val = convert(val)
        env[var] = val

    # safe_map applies write func to each element in jaxpr.invars & args, jaxpr.invars are input variables, and args. The difference is that args are the values of the input variables, while jaxpr.invars are the variables that hold the input variables.
    safe_map(write, jaxpr.invars, args)
    # interval_consts = [convert(const) for const in consts]

    # safe_map applies write func to each jaxpr.constvars & consts, jaxpr.constvars are constant variables, and consts are the constants. The difference is that consts are the values of the constants, while jaxpr.constvars are the variables that hold the constants.
    safe_map(write, jaxpr.constvars, consts)

    # Loop through equations and eval primitives using `bind`
    for eqn in tqdm(jaxpr.eqns):  # looping through the equations in the jaxpr

        # apply read for eachinput var in eqn.invars
        invars = safe_map(read, eqn.invars)
        # `bind` is how a primitive is called
        if eqn.primitive in interval_primitives:
            # print("prim :", eqn.primitive, "invars: ", invars)
            outvals = interval_primitives[eqn.primitive](*invars, **eqn.params)
        else:
            raise NotImplementedError(
                f"Primitive not in interval_primitives: {eqn.primitive}")
            # print("Primitive not in interval_primitives:", eqn.primitive)

        # Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results:  # check if the primitive returns multiple results
            # if not, wrap the result in a list to make it easier to handle, if it does, we will just use the result as is
            outvals = [outvals]

        # write results of primitive into env.
        safe_map(write, eqn.outvars, outvals)

    # read final result of Jaxpr from env.
    # read the output variables from the environment and return them as a list
    return safe_map(read, jaxpr.outvars)
