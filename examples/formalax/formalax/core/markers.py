#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import jax

__all__ = ("Marker", "relu_marker", "detect_jax_nn_relu", "markup_primitive")


class Marker:
    """Marker for predefined computations.

    Markers are used to indicate a certain computation that could be
    decomposed into further Jax primitives but should be handled as is.

    The typical example is ReLU, which is not a JAX primitive but is treated
    as a primitive operation in many bound propagation algorithms.
    """

    __slots__ = ("name", "multiple_results")

    def __init__(self, name, multiple_results=False):
        self.name = name
        self.multiple_results = multiple_results


relu_marker = Marker("relu")


def detect_jax_nn_relu(eqn: jax.core.JaxprEqn) -> bool:
    """Checks whether ``eqn`` is a ``jax.nn.relu`` call."""
    return (
        eqn.primitive == jax.custom_derivatives.custom_jvp_call_p
        and len(sub_eqns := eqn.params["call_jaxpr"].jaxpr.eqns) == 1
        and (sub_eqn := sub_eqns[0]).primitive == jax.experimental.pjit.pjit_p
        and "name" in (sub_params := sub_eqn.params)
        and sub_params["name"] == "relu"
    )


def markup_primitive(eqn: jax.core.JaxprEqn) -> jax.core.Primitive | Marker:
    """Marks up special operations, such as ``jax.nn.relu`` and otherwise returns
    the primitive of ``eqn``.
    """
    if detect_jax_nn_relu(eqn):
        return relu_marker
    else:
        return eqn.primitive
