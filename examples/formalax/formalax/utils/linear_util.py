#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import jax
import jax.extend.linear_util as lu
import jax.tree


@lu.transformation_with_aux
def flatten_fun(in_tree, is_leaf, *args_flat):
    """Recreate flattened arguments and flatten the output."""
    # Based on the flatten_fun_for_vmap function from
    # https://github.com/google/jax/blob/main/jax/_src/interpreters/batching.py
    py_args, py_kwargs = jax.tree.unflatten(in_tree, args_flat)
    ans = yield py_args, py_kwargs
    yield jax.tree.flatten(ans, is_leaf)


@lu.transformation_with_aux
def flatten_output(is_leaf, *args):
    """Flatten the output of a function."""
    ans = yield args, {}
    yield jax.tree.flatten(ans, is_leaf)


@lu.transformation
def args_to_kwargs(to_key: tuple[str | None, ...], *args):
    """Convert positional arguments to keyword arguments.

    Args:
        to_key: A tuple of strings or ``None``.
            Each positional argument is converted to a keyword argument with the
            corresponding key from ``to_key?? as the keyword, unless the key
            is ``None``.
            In that case, the positional argument remains a positional argument.
    """
    args = tuple(args)

    kwargs = {
        key: arg for key, arg in zip(to_key, args, strict=False) if key is not None
    }
    args = (
        tuple(arg for key, arg in zip(to_key, args, strict=False) if key is None)
        + args[len(to_key) :]
    )
    ans = yield args, kwargs
    yield ans
