import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

@jax.tree_util.register_pytree_node_class
class IntervalArray:
    """
    JAX-compatible array representing an interval [lower, upper] for each element.
    Stores lower and upper as jnp.ndarray leaves.
    Works with jax.tree_map / optax because of pytree node.
    Supports basic arithmetic operations (elementwise add/mul) and matrix multiplications (matmul).
    """

    def __init__(self, lower: jnp.ndarray, upper: jnp.ndarray):
        lower = jnp.asarray(lower)
        upper = jnp.asarray(upper)
        if lower.shape != upper.shape:
            raise ValueError("Lower and upper bound must have same shape.")
        self.lower = lower
        self.upper = upper

    # This does: jax.tree_map(lambda x: x, interval_array)
    @classmethod
    def from_scalar(cls, x:float):
        a = jnp.asarray(x)
        return cls(a, a)

    @classmethod
    def from_array(cls, array: jnp.ndarray, half_range: float = 0.0):
        """
        Create IntervalArray from numeric array.
        If half_range == 0, extract intervals [x,x].
        If half_range > 0, [x - half_range, x + half_range].
        """
        array = jnp.asarray(array)
        if half_range == 0.0:
            return cls(array, array)
        half_range = jnp.asarray(half_range, dtype=array.dtype)
        return cls(array - half_range, array + half_range)

    def __shape__(self) -> Tuple[int,...]:
        return self.lower.shape

    def __repr__(self): # pretty print
        return f"IntervalArray(upper: {self.lower}, lower: {self.upper})"

    def tree_flatten(self):
        # children are two arrays; no aux data needed (aux is empty tuple)
        childen = (self.lower, self.upper)
        aux = None
        return childen, aux
    @classmethod
    def tree_unflatten(cls, aux, children):
        upper, lower = children
        return cls(upper, lower)

    # Arithmetic operations
    # Elementwise addition
    def __add__(self, other):
        if isinstance(other, IntervalArray):
            return IntervalArray(self.lower + other.lower, self.upper + other.upper)
        else:
            other = jnp.asarray(other)
            return IntervalArray(self.lower + other, self.upper + other)

    __radd__ = __add__ # Makes sure a + b == b + a

    # Elementwise subtraction
    def __sub__(self, other):
        if isinstance(other, IntervalArray):
            return IntervalArray(self.lower - other.lower, self.upper - other.upper)
        else:
            other = jnp.asarray(other)
            return IntervalArray(self.lower + other, self.upper + other)

    # Elementwise multiplication
    def __mul__(self, other):
        if isinstance(other, IntervalArray):
            ll = self.lower * other.lower
            lu = self.lower * other.upper
            ul = self.upper * other.lower
            uu = self.upper * other.upper
            lower = jnp.minimum(jnp.minimum(ll, lu), jnp.minimum(ul, uu))
            upper = jnp.maximum(jnp.maximum(ll, lu), jnp.maximum(ul, uu))
            return IntervalArray(lower, upper)
        else:
            other = jnp.asarray(other)
            ll = self.lower * other
            uu = self.upper * other
            lower = jnp.minimum(ll, uu)
            upper = jnp.maximum(ll, uu)
            return IntervalArray(lower, upper)

    __rmul__ = __mul__ # Makes sure a * b == b * a

    # Elementwise division
    def __truediv__(self, other):
        if isinstance(other, IntervalArray):
            # Division by Interval: using multiplication with reciprocal (assuming 0.0 not in interval)
            if jnp.any((other.lower <= 0.0) & (other.upper >= 0.0)):
                raise ValueError("Division by interval containing zero is undefined/not supported.")
            return self * IntervalArray(1.0 / other.upper, 1.0 / other.lower)
        else:
            other = jnp.asarray(other)
            return IntervalArray(self.lower / other, self.upper / other)

    # Matrix multiplication
    def __matmul__(self, other):
        if not isinstance(other, IntervalArray):
            raise ValueError("matmul only supported with two IntervalArray objects.")
        if self.lower.ndim  != 2 or other.lower.ndim != 2:
            raise ValueError("matmul only supports 2D arrays (matrices).")
        if self.lower.shape[1] != other.lower.shape[0]:
            raise ValueError("Shapes not aligned for matmul.")
        #Compute all combinations of prods
        ll = jnp.einsum('ik,kj->ij', self.lower, other.lower)
        lu = jnp.einsum('ik,kj->ij', self.lower, other.upper)
        ul = jnp.einsum('ik,kj->ij', self.upper, other.lower)
        uu = jnp.einsum('ik,kj->ij', self.upper, other.upper)

        per_k_min = jnp.minimum(jnp.minimum(ll, lu), jnp.minimum(ul, uu))
        per_k_max = jnp.maximum(jnp.maximum(ll, lu), jnp.maximum(ul, uu))

        #not sure if axis is 2 or 1
        lower = jnp.sum(per_k_min, axis=2)
        upper = jnp.sum(per_k_max, axis=2)
        return IntervalArray(lower, upper)

    # convenience operator support
    def __matmul__(self, other):
        return self.__matmul__(other)


    # Activation functions
    def relu(self):
        return IntervalArray(jnp.maximum(0.0, self.lower), jnp.maximum(0.0, self.upper))
    def sigmoid(self):
        return IntervalArray(jax.nn.sigmoid(self.lower), jax.nn.sigmoid(self.upper))
    def tanh(self):
        return IntervalArray(jnp.tanh(self.lower), jnp.tanh(self.upper))
    def exp(self):
        return IntervalArray()

    # Properties
    @property
    def shape(self):
        return self.lower.shape
    def dtype(self):
        return self.lower.dtype
    def size(self):
        return self.lower.size

    def mean(self):
        return (self.lower + self.upper) / 2.0

def to_interval_array(array, half_range: float = 0.0):
    array = jnp.asarray(array)
    return IntervalArray.from_array(array, half_range)

def to_interval_matrix(matrix):
    matrix = jnp.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Input must be 2D array (matrix).")
    return IntervalArray.from_array(matrix, half_range=0.0)

def to_interval_matrix_range(matrix, half_range: float):
    matrix = jnp.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Input must be 2D array (matrix).")
    return IntervalArray.from_array(matrix, half_range=half_range)

# test
# @jax.jit
def example_func(a_float, b_float):
    A = to_interval_array(a_float)
    B = to_interval_array(b_float)
    C = A * B
    # C = A + B
    print("A: ", A)
    print("A: ", A.lower)
    print("B: ", B)
    print("C: ", C)
    return C.lower, C.upper

example_func(jnp.array([1.0, 2.0]), jnp.array([5.0, 6.0]))


