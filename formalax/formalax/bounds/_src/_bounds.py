#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import itertools as it
from typing import Any, Protocol, Sequence, TypeGuard

from jaxtyping import Array, Real, Shaped

from ...sets.box import Box
from ...sets.protocols import MinMaxSplittable
from ...sets.singleton import Singleton
from ...utils.zip import strict_zip

__all__ = (
    "Bounds",
    "is_bounds",
    "all_as_bounds",
    "example_args",
    "flatten_bounds",
    "collect_bounds",
    "duplicate_for_bounds",
)


class Bounds[T: Real[Array, "..."]](Protocol):
    """A pair of a lower and an upper bound on a set of ``Arrays``.

    The term "concrete bounds" refers to constant bounds in contrast to, for
    example, linear bounds.
    A ``Bounds`` instance does not need to store concrete bounds but may compute
    it from other forms of bounds.
    Consider caching computed concrete bounds.
    """

    @property
    def lower_bound(self) -> T:
        """The concrete lower bound, the first element of ``self.concrete``."""
        ...

    @property
    def upper_bound(self) -> T:
        """The concrete upper bound, the second element of ``self.concrete``."""
        ...

    @property
    def concrete(self) -> MinMaxSplittable[T]:
        """The concrete bounds underlying this ``Bounds`` instance."""
        ...


def is_bounds(x: Any) -> TypeGuard[Bounds]:
    """Whether ``x`` provides the properties of ``Bounds``."""
    return (
        hasattr(x, "lower_bound")
        and hasattr(x, "upper_bound")
        and hasattr(x, "concrete")
    )


# ======================================================================================
# Utils
# ======================================================================================


def all_as_bounds[T](*xs: T | Bounds[T]) -> tuple[Bounds[T], ...]:
    """Convert all arguments ``x`` which are not ``Bounds`` to ``Box(x, x)``.

    Args:
        *xs: The arguments to convert. Arguments that are already ``Bounds``
            are left as they are.

    Returns:
        A tuple of ``Bounds`` instances.
    """
    return tuple(x if is_bounds(x) else Singleton(x) for x in xs)


def example_args[T: Shaped](args_flat: Sequence[T | Bounds[T]]) -> list[T]:
    """Replace ``Bounds`` by the lower bound to have array-only arguments."""
    assert not is_bounds(args_flat)
    return [a.lower_bound if is_bounds(a) else a for a in args_flat]


def flatten_bounds[T](
    values: Sequence[Bounds[T] | T],
) -> tuple[tuple[T, ...], tuple[bool, ...]]:
    """Flattens a Sequence of ``Bounds`` instances and concrete values.

    Args:
        values: The sequence to flatten.

    Returns:
        - The flattened ``Bounds`` and single values.
        - A tuple of the same length as ``values`` that indicates if an
            element of ``values`` was a ``Bounds`` instance.
    """
    has_bounds = tuple(is_bounds(val) for val in values)
    flat_values = sum(
        (
            tuple(val.concrete) if bounded else (val,)
            for val, bounded in strict_zip(values, has_bounds)
        ),
        start=(),
    )
    return flat_values, has_bounds


def collect_bounds[T](
    has_bounds: Sequence[bool], flat_values: Sequence[T]
) -> list[Bounds[T] | T]:
    """Groups lower and upper bounds in a flat sequence of values.

    In ``flat_values``, lower and upper bounds follow sequentially at the positions
    indicated in ``has_bounds``.
    In the output, lower and upper bounds are paired in ``Bounds`` instances.
    The output has the same length as ``has_bounds``.
    """
    assert sum(2 if bounded else 1 for bounded in has_bounds) == len(flat_values)
    vi = iter(flat_values)
    return [Box(next(vi), next(vi)) if bounded else next(vi) for bounded in has_bounds]


def duplicate_for_bounds[T](
    has_bounds: Sequence[bool], base_values: Sequence[T]
) -> tuple[T, ...]:
    """Duplicates values in ``base_values`` whenever ``has_bounds`` contains true."""
    return tuple(
        it.chain.from_iterable(
            (val, val) if bounded else (val,)
            for val, bounded in strict_zip(base_values, has_bounds)
        )
    )
