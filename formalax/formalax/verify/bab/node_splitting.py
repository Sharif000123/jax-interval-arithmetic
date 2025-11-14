#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
import functools
import itertools as it
import math
import typing
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Real

from ...bounds._src._bounds import Bounds, is_bounds
from ...bounds._src._bwcrown import crown, crown_ibp
from ...bounds._src._ibp import ibp_jaxpr
from ...sets.box import Box
from ...utils.jaxpr import jaxpr_as_fun
from ...utils.zip import strict_zip
from .bab import (
    BAB_TARGET,
    ArrayRef,
    BaBCallable,
    BranchSelection,
    BranchStore,
    ComputeBoundsBaB,
    MakeForBaB,
    SelectBranches,
    SimpleBranchStore,
    SplitBranches,
    bab,
    fun_to_jaxpr_for_bab,
)
from .select import select_worst

__all__ = [
    "NodeSplit",
]

# ==============================================================================
# Branch Data
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclass(eq=False, frozen=True, slots=True)
class NodeSplit:
    """A branching that partitions the variables (nodes) in a jaxpr into disjoint boxes.

    Due to being stored in a branch store, the arrays of a `NodeSplit` have a
    leading batch/branches axis.
    The only exception to this is the root branch, which is created without
    a leading branch axis.

    The `split_lbs` and `split_ubs` define which nodes are split.
    If a node is not split, `split_lbs` is `-inf` and `split_ubs` is `inf`.
    If a node is split, `split_lbs`, `split_ubs`, or both are set to finite value.

    Args:
        split_lbs: The lower bounds on each node's value in the split.
        split_ubs: The upper bounds of each node's value in the split.
    """

    split_lbs: dict[jax.core.Var, Real[Array, "b ..."]]
    split_ubs: dict[jax.core.Var, Real[Array, "b ..."]]

    @classmethod
    def root_branch(
        cls,
        jaxpr: jax.core.ClosedJaxpr,
        *args: Bounds | Real[Array, "..."],
        target: BAB_TARGET,
    ) -> "NodeSplit":
        """Creates the branch data of the root branch.

        Since the root branch has no nodes split, all `lbs` are `-inf`
        and all `ubs` are `inf`.
        """
        # TODO: implement correctly
        lbs = {var: jnp.full_like(var.aval.shape, -jnp.inf) for var in jaxpr.vars}
        ubs = {var: jnp.full_like(var.aval.shape, jnp.inf) for var in jaxpr.vars}
        return cls(lbs, ubs)


# ==============================================================================
# Compute Bounds
# ==============================================================================

