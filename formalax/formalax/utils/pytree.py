#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from typing import Any

import jax


def flatten_keep_none(tree: Any) -> tuple[list, jax.tree_util.PyTreeDef]:
    """Flattens a pytree while keeping ``None`` leaves."""
    return jax.tree.flatten(tree, is_leaf=lambda x: x is None)
