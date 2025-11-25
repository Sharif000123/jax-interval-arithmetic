#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Real

from .protocols import MinMaxSplittable

__all__ = ("Reals", "NonNegative", "NonPositive")


@jax.tree_util.register_dataclass
@dataclass(eq=False, frozen=True, slots=True)
class Reals[T: Real[Array, " shape"]]:
    """The unbounded set of all real numbers.

    Represents and array of a particular shape that can contain arbitrary
    real numbers.
    """

    # --------------------------------------------------------------------------
    # Array Attributes
    # --------------------------------------------------------------------------
    shape: tuple[int, ...]
    dtype: jnp.dtype = float

    # --------------------------------------------------------------------------
    # MinMaxSplittable Implementation
    # --------------------------------------------------------------------------

    def __iter__(self):
        yield self.lower_bound
        yield self.upper_bound

    # --------------------------------------------------------------------------
    # Bounds Implementation
    # --------------------------------------------------------------------------

    @property
    def lower_bound(self) -> T:
        return jnp.full(self.shape, -jnp.inf, self.dtype)

    @property
    def upper_bound(self) -> T:
        return jnp.full(self.shape, jnp.inf, self.dtype)

    @property
    def concrete(self) -> MinMaxSplittable[T]:
        return self

    # --------------------------------------------------------------------------
    # HasProjection Implementation
    # --------------------------------------------------------------------------

    def project(self, x: T) -> T:
        return x

    # --------------------------------------------------------------------------
    # HasRandomSample Implementation
    # --------------------------------------------------------------------------

    def random_sample(self, key: jax.Array, n: int = 1) -> T:
        """Draws a random sample from a standard normal distribution.

        Args:
            key: A jax pseudo random number generator key.
            n: The number of samples to draw.

        Returns:
            An array of samples with a leading batch dimension of size `n`.
        """
        return jax.random.normal(key, (n, *self.shape), self.dtype)

    # --------------------------------------------------------------------------
    # Comparison
    # --------------------------------------------------------------------------

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.shape == other.shape
            and self.dtype == other.dtype
        )


@jax.tree_util.register_dataclass
@dataclass(eq=False, frozen=True, slots=True)
class NonNegative[T: Real[Array, " shape"]](Reals[T]):
    """The set of all non-negative real numbers.

    Represents and array of a particular shape that contains only non-negative real
    numbers.
    """

    # --------------------------------------------------------------------------
    # Bounds Implementation
    # --------------------------------------------------------------------------

    @property
    def lower_bound(self) -> T:
        return jnp.zeros(self.shape, self.dtype)

    # --------------------------------------------------------------------------
    # HasProjection Implementation
    # --------------------------------------------------------------------------

    def project(self, x: T) -> T:
        return jnp.clip(x, min=0.0)

    # --------------------------------------------------------------------------
    # HasRandomSample Implementation
    # --------------------------------------------------------------------------

    def random_sample(self, key: jax.Array, n: int = 1) -> T:
        """Randomly samples a non-negative real number.

        Draws a random sample from a standard normal distribution
        and returns then the absolute value of this sample.

        Args:
            key: A jax pseudo random number generator key.
            n: The number of samples to draw.

        Returns:
            An array of samples with a leading batch dimension of size `n`.
        """
        x = jax.random.normal(key, (n, *self.shape), self.dtype)
        return jnp.abs(x)


@jax.tree_util.register_dataclass
@dataclass(eq=False, frozen=True, slots=True)
class NonPositive[T: Real[Array, " shape"]](Reals[T]):
    """The set of all non-positive real numbers.

    Represents and array of a particular shape that contains only non-positive real
    numbers.
    """

    # --------------------------------------------------------------------------
    # Bounds Implementation
    # --------------------------------------------------------------------------

    @property
    def upper_bound(self) -> T:
        return jnp.zeros(self.shape, self.dtype)

    # --------------------------------------------------------------------------
    # HasProjection Implementation
    # --------------------------------------------------------------------------

    def project(self, x: T) -> T:
        return jnp.clip(x, max=0.0)

    # --------------------------------------------------------------------------
    # HasRandomSample Implementation
    # --------------------------------------------------------------------------

    def random_sample(self, key: jax.Array, n: int = 1) -> T:
        """Randomly samples a non-positive real number.

        Draws a random sample from a standard normal distribution
        and returns then the negative absolute value of this sample.

        Args:
            key: A jax pseudo random number generator key.
            n: The number of samples to draw.

        Returns:
            An array of samples with a leading batch dimension of size `n`.
        """
        x = jax.random.normal(key, (n, *self.shape), self.dtype)
        return -jnp.abs(x)
