#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Literal, Protocol, Sequence

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import optax
from frozendict import frozendict
from jax import lax
from jax.core import JaxprEqn, Var
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Real

from ...bounds._src._affinebounds import AffineBounds, Zero
from ...bounds._src._bounds import Bounds, all_as_bounds, is_bounds
from ...bounds._src._bwcrown import (
    BACKWARDS_CROWN_PARAM_STRATEGIES,
    PARAM_DOMAIN,
    BackwardsCROWNRule,
    CROWNCallable,
    _ibp_bounds,
    _primitive_backward_crown_rules,
    _resolve_param_strategies,
    backwards_crown_relu_rule,
    crown,
    crown_ibp,
    is_param_domain,
)
from ...bounds._src._bwlirpa import (
    COMPUTE_BOUNDS,
    BackwardsLiRPARule,
    _reshape_as_output,
    _restore_weights_shape,
    _single_output_asserts,
    backwards_lirpa,
    fun_to_jaxpr_for_lirpa,
    nonlinear_backwards_lirpa_rule,
)
from ...bounds._src._ibp import ibp_jaxpr
from ...bounds._src._lirpabounds import (
    LiRPABounds,
    LiRPAWeights,
    incorporate_batch_axes,
)
from ...core.batch_axes import BatchAxisMapping
from ...core.markers import Marker, markup_primitive, relu_marker
from ...sets.box import Box
from ...sets.reals import NonNegative, Reals
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
    "BETA_CROWN_PARAM_STRATEGIES",
    "beta_crown",
]


BETA_CROWN_PARAM_STRATEGIES = Literal["full", "shared", "default"]
"""The parameter selection strategies of beta-CROWN rules.

 - The ``"full"`` strategy means that each entry in an array that is
    computed by an elementwise primitive gets its own Lagrange multiplier (beta).
 - Under the ``"shared"`` strategy, all entries in an array computed by an elementwise
    primitive share the same Lagrange multiplier (beta).
    This means, for example, that all ReLU nodes on a ReLU layer share one
    Lagrange multiplier.
 - The ``"default"`` strategy uses either the ``"full"`` or ``"shared"`` strategy,
    depending on the primitive.
    Which option is determined by the beta-CROWN rules for the different primitives.
"""


class BetaCROWNRule(BackwardsLiRPARule, Protocol):
    """Type signature of a beta-CROWN rule.

    This class differs from ``BackwardsCROWNRule`` in that it defines a particular
    parameter shape and adds a beta parameter selection strategy to the
    `parameter_domains` method.

    The parameters of `__call__` are `((split_bounds, betas), alphas)`,
    where `split_bounds` are the input bounds of the node under splitting,
    `betas` are the Lagrange multipliers for the split constraints,
    and `alphas` are the relaxation parameters of a backwards CROWN rule.
    """

    def __call__(
        self,
        params: tuple[
            None
            | tuple[Sequence[Bounds[Real[Array, "..."]]], PyTree[Real[Array, "..."]]],
            PyTree[Real[Array, "..."] | BACKWARDS_CROWN_PARAM_STRATEGIES],
        ],
        out_weights: Sequence[LiRPAWeights[Real[Array, "..."] | Zero]],
        in_bounds: Sequence[Bounds | jax.Array],
        *,
        in_batch_axes: Sequence[tuple[int, ...]],
        out_batch_axes: Sequence[tuple[int, ...]],
        batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
        backwards_lirpa: Callable[
            [
                jax.core.ClosedJaxpr,
                Sequence[LiRPAWeights[Real[Array, "..."] | Zero]],
                Bounds | jax.Array,
                ...,
            ],
            tuple[LiRPABounds[Real[Array, "..."] | Zero], ...],
        ],
        **kwargs,
    ) -> (
        tuple[LiRPABounds[Real[Array, "..."] | Zero], ...]
        | LiRPABounds[Real[Array, "..."] | Zero]
    ): ...

    def parameter_domains(
        self,
        in_shapes: Sequence[tuple[int, ...]],
        out_shapes: Sequence[tuple[int, ...]],
        strategies: Sequence[BACKWARDS_CROWN_PARAM_STRATEGIES],
        beta_strategies: Sequence[BETA_CROWN_PARAM_STRATEGIES],
        **kwargs,
    ) -> tuple[
        tuple[Sequence[PARAM_DOMAIN], PARAM_DOMAIN] | None,
        tuple[PARAM_DOMAIN, ...] | None,
    ]:
        """Declares the domains of the parameters of this rule.

        See `BackwardsCROWNRule.parameter_domains` for more details.

        Args:
            in_shapes: The shapes of the inputs to the primitive call.
            out_shapes: The shapes of the outputs to the primitive call.
            strategies: See `BackwardsCROWNRule.parameter_domains`.
            beta_strategies: The parameter selection strategies for
                the beta parameters for each *input* of the primitive.
                Raise a ``ValueError`` if your method does not support
                one of the given strategies.
            **kwargs: Any keyword arguments of the primitive.

        Raises:
            ValueError: If one or several of the given parameter strategies are
                not supported.

        Returns:
            The domains of the parameters of this rule.
            These domains are `((split_bounds_domains, beta_domains), alpha_domains)`.
        """
        ...


def beta_crown(
    jaxpr: jax.core.ClosedJaxpr,  # called by bab => always a jaxpr
    optim: optax.GradientTransformation | None = None,
    steps: int = 10,
    default_rng_key: int | PRNGKeyArray = 0,
    compute_bounds: COMPUTE_BOUNDS = _ibp_bounds,
    split_param_strategies: (
        BETA_CROWN_PARAM_STRATEGIES | dict[Var, BETA_CROWN_PARAM_STRATEGIES]
    ) = "full",
    crown_param_strategies: (
        BACKWARDS_CROWN_PARAM_STRATEGIES | dict[Var, BACKWARDS_CROWN_PARAM_STRATEGIES]
    ) = "external-full",
    in_batch_axes: int | None | tuple[int | None, ...] = ...,
    out_batch_axes: int | None | tuple[int | None, ...] = ...,
    beta_crown_rules: frozendict[jax.core.Primitive | Marker, BetaCROWNRule]
    | None = None,
    backwards_crown_rules: frozendict[jax.core.Primitive | Marker, BackwardsCROWNRule]
    | None = None,
) -> Callable[[Bounds | jax.Array, ...], tuple[Bounds | jax.Array, ...]]:
    """Creates a function that performs beta-CROWN on ``jaxpr``.

    This function implements both beta-CROWN without optimized relaxation parameters
    and alpha-beta-CROWN with optimized relaxation parameters.
    In any case, node splits are handled by optimizing Lagrange multipliers
    for the split constraints (``beta``).

    The returned function computes CROWN bounds under node splits [Wang et al., 2021],
    optionally also optimizing relaxation parameters [Xu et al., 2021].
    For this, it first computes fixed bounds using ``compute_bounds`` and then repeatedly
    applies ``backwards_crown``, while optimizing Lagrange multiplies of split
    constraints and relaxation parameters of the ``backwards_crown`` relaxation using
    gradient descent/ascent.
    It uses specialized CROWN rules that take split constraints into account.
    See [Wang et al., 2021] and ``backwards_crown`` for more details.

    [Wang et al., 2021]: Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana,
        Cho-Jui Hsieh, J. Zico Kolter: Beta-CROWN: Efficient Bound Propagation with
        Per-neuron Split Constraints for Neural Network Robustness Verification.
        NeurIPS 2021: 29909-29921
    [Xu et al., 2021]: Kaidi Xu, Huan Zhang, Shiqi Wang, Yihan Wang, Suman Jana,
        Xue Lin, Cho-Jui Hsieh: Fast and Complete: Enabling Complete Neural Network
        Verification with Rapid and Massively Parallel Incomplete Verifiers. ICLR 2021

    Args:
        jaxpr: The jaxpr to compute bounds on.
        optim: The optimizer to use for optimization.
            Uses ``optax.adam(1e-3)`` by default.
        steps: The number of gradient descent steps (gradient updates) to perform.
        default_rng_key: The default random number generator key to use for sampling
            initial parameters.
            This value can be overriden using the ``rng_key`` argument of the returned
            function.
        compute_bounds: The function to compute bounds on the values in ``jaxpr``.
            Uses ``ibp`` by default.
        split_param_strategies: The parameter selection strategies for the
            split constraint Lagrange multipliers (beta parameters).
            This can either be a single parameter selection
            strategy for every variable in the jaxpr, or a dictionary
            mapping variables in the jaxpr to a parameter selection strategies.
            By default, uses ``"full"`` for all parameters.
            See ``BETA_CROWN_PARAM_STRATEGIES`` for more details.
        crown_param_strategies: The parameter selection strategies
            for the ``backwards_crown`` relaxation parameters.
            This can either be a single parameter selection
            strategy for every variable in the jaxpr, or a dictionary
            mapping variables in the jaxpr to a parameter selection strategies.
            By default, uses ``"external-full"`` for all parameters.
            See ``BACKWARDS_CROWN_PARAM_STRATEGIES`` for more details.
        in_batch_axes: Batch axes of the inputs.
            See ``backwards_crown`` for more details.
        out_batch_axes: Batch axes of the outputs.
            See ``backwards_crown`` for more details.
        beta_crown_rules: The beta-CROWN rules to use.
            If ``None``, the default beta-CROWN rules are used.
        backwards_crown_rules: The backwards CROWN rules to use for all primitives
            not in ``beta_crown_rules``.
            If ``None``, the default backwards CROWN rules are used.

    Returns:
        A function ``fun_beta_crown`` that computes output bounds on ``jaxpr`` given
        input bounds on some of the arguments of ``jaxpr`` and node splits.
        When calling ``fun_beta_crown``, pass ``Bounds`` instances for the arguments with
        input bounds while passing regular ``jax.Arrays`` for the arguments without
        input bounds.
        The output of ``fun_beta_crown`` has the same structure as the output of ``jaxpr``,
        except that all outputs which depends on arguments with input bounds are
        ``Bounds`` instances.
        Use the ``rng_key`` argument of ``fun_beta_crown`` to specify the random number
        generator key to use for sampling initial parameters.
    """
    bwcrown, param_domains = _beta_crown_propagate(
        jaxpr,
        compute_bounds,
        split_param_strategies,
        crown_param_strategies,
        in_batch_axes,
        out_batch_axes,
        beta_crown_rules,
        backwards_crown_rules,
    )


def _beta_crown_optimize(optim: optax.GradientTransformation | None):
    if optim is None:
        optim = optax.adam(1e-3)


def _beta_crown_propagate(
    jaxpr: jax.core.ClosedJaxpr,
    compute_bounds: COMPUTE_BOUNDS,
    split_param_strategies: (
        BETA_CROWN_PARAM_STRATEGIES | dict[Var, BETA_CROWN_PARAM_STRATEGIES]
    ) = "full",
    crown_param_strategies: (
        BACKWARDS_CROWN_PARAM_STRATEGIES | dict[Var, BACKWARDS_CROWN_PARAM_STRATEGIES]
    ) = "external-full",
    in_batch_axes: int | None | tuple[int | None, ...] = ...,
    out_batch_axes: int | None | tuple[int | None, ...] = ...,
    beta_crown_rules: frozendict[jax.core.Primitive | Marker, BetaCROWNRule]
    | None = None,
    backwards_crown_rules: frozendict[jax.core.Primitive | Marker, BackwardsCROWNRule]
    | None = None,
) -> tuple[
    Callable[[Bounds | jax.Array, ...], tuple[Bounds | jax.Array, ...]],
    tuple[PARAM_DOMAIN, ...],
]:
    """Create the bound propagation function and parameter domains for beta-CROWN."""
    if beta_crown_rules is None:
        beta_crown_rules = _primitive_backward_beta_crown_rules
    if backwards_crown_rules is None:
        backwards_crown_rules = _primitive_backward_crown_rules

    if isinstance(split_param_strategies, str):
        split_param_strategies = defaultdict(lambda: split_param_strategies)
    else:
        split_param_strategies = defaultdict(lambda: "default", split_param_strategies)
    crown_param_strategies, _ = _resolve_param_strategies(crown_param_strategies)

    split_bounds_domains, split_param_domains, relax_param_domains = (
        _collect_parameter_domains(
            jaxpr, split_param_strategies, crown_param_strategies, beta_crown_rules
        )
    )

    # We keep split bounds, optimizable split parameters (betas), and
    # relaxation parameters (alphas) separate.
    # The parameter pytree is a triple of dictionaries.
    # Each dictionary maps a jaxpr variable to a parameter.
    # The first dictionary maps the first output variable of an equation to
    # the split bounds on the inputs of that equation.
    # The second dictionary similarly maps the first output variable to
    # the optimizable split parameters.
    # The third dictionary maps variables to relaxation parameters.
    # These are not stored at the first output variable but at all output variables.

    def get_params(
        eqn: JaxprEqn,
        params: tuple[
            dict[Var, PyTree[Real[Array, "..."]]],
            dict[Var, PyTree[Real[Array, "..."]]],
            dict[Var, PyTree[Real[Array, "..."]]],
        ],
    ) -> (
        tuple[
            tuple[PyTree[Real[Array, "..."]], PyTree[Real[Array, "..."]]],
            tuple[PyTree[Real[Array, "..."]], ...],
        ]
        | tuple[PyTree[Real[Array, "..."]], ...]
    ):
        if eqn.primitive == jax.experimental.pjit.pjit_p:
            # Pass on all parameters for nested jaxprs.
            return params

        split_bounds, split_params, relaxation_params = params

        def project(params, domains):
            params = jax.tree.map(lambda p, dom: dom.project(p), [params], [domains])
            return params[0]

        def get_split_params(eqn: JaxprEqn):
            var = eqn.outvars[0]
            try:
                bounds, betas = split_bounds[var], split_params[var]
                # Project beta variables to guarantee soundness
                betas = project(betas, split_param_domains[var])
                return (bounds, betas)
            except KeyError as ex:
                raise ValueError(
                    f"No parameter value for variable {var} in {params}."
                ) from ex

        def get_relaxation_params(var: Var):
            if var in relax_param_domains:
                try:
                    # We project the parameters here to guarantee soundness
                    alphas = relaxation_params[var]
                    alphas = project(alphas, relax_param_domains[var])
                    return alphas
                except KeyError as ex:
                    raise ValueError(
                        f"No parameter value for variable {var} in {params}."
                    ) from ex
            else:
                return crown_param_strategies[var]

        if eqn.outvars[0] in split_bounds_domains:
            relax_params = [get_relaxation_params(var) for var in eqn.outvars]
            return (get_split_params(eqn), relax_params)
        else:
            return [get_relaxation_params(var) for var in eqn.outvars]

    bwlirpa = backwards_lirpa(
        jaxpr,
        backwards_crown_rules,
        compute_bounds,
        get_params,
        in_batch_axes=in_batch_axes,
        out_batch_axes=out_batch_axes,
    )

    def wrapper_fun(
        split_bounds: tuple[PyTree[Real[Array, "..."]], ...],
        split_params: tuple[PyTree[Real[Array, "..."]], ...],
        relaxation_params: tuple[PyTree[Real[Array, "..."]], ...],
        *args: Bounds | jax.Array,
    ) -> tuple[Bounds | jax.Array, ...]:
        return bwlirpa((split_bounds, split_params, relaxation_params), *args)

    wrapper_fun.__doc__ = CROWNCallable.__doc__
    return wrapper_fun, (split_bounds_domains, split_param_domains, relax_param_domains)


def _collect_parameter_domains(
    jaxpr: jax.core.ClosedJaxpr,
    split_param_strategies: dict[Var, BETA_CROWN_PARAM_STRATEGIES],
    crown_param_strategies: dict[Var, BACKWARDS_CROWN_PARAM_STRATEGIES],
    beta_crown_rules: frozendict[jax.core.Primitive | Marker, BetaCROWNRule],
    backwards_crown_rules: frozendict[jax.core.Primitive | Marker, BackwardsCROWNRule],
) -> tuple[dict[Var, PARAM_DOMAIN], dict[Var, PARAM_DOMAIN], dict[Var, PARAM_DOMAIN]]:
    """Collects the parameter domains of the beta-CROWN rules used for ``jaxpr``.

    All parameters of an equation are associated with the first output variable
    of the equation.
    """
    jaxpr = jaxpr.jaxpr

    split_bounds_domains = {}
    split_param_domains = {}
    relax_param_domains = {}
    for eqn in jaxpr.eqns:
        prim = markup_primitive(eqn)
        in_shapes = [var.aval.shape for var in eqn.invars]
        out_shapes = [var.aval.shape for var in eqn.outvars]
        if prim in beta_crown_rules:
            rule = beta_crown_rules[prim]

            crown_strategies = [crown_param_strategies[var] for var in eqn.outvars]
            split_strategies = [split_param_strategies[var] for var in eqn.invars]
            (split_domains, beta_domains), alpha_domains = rule.parameter_domains(
                in_shapes, out_shapes, crown_strategies, split_strategies, **eqn.params
            )

            var = eqn.outvars[0]
            assert var not in split_bounds_domains
            assert var not in split_param_domains
            assert var not in relax_param_domains
            split_bounds_domains[var] = split_domains
            split_param_domains[var] = beta_domains
            relax_param_domains[var] = alpha_domains
        elif prim in backwards_crown_rules:
            rule = beta_crown_rules[prim]

            crown_strategies = [crown_param_strategies[var] for var in eqn.outvars]
            alpha_domains = rule.parameter_domains(
                in_shapes, out_shapes, crown_strategies, **eqn.params
            )
            var = eqn.outvars[0]
            assert var not in relax_param_domains
            relax_param_domains[var] = alpha_domains
        elif prim == jax.experimental.pjit.pjit_p:
            # Jaxpr variables are globally unique, so we can put all parameters
            # in one domains dictionary.
            split_domains, beta_domains, alpha_domains = _collect_parameter_domains(
                eqn.params["jaxpr"],
                split_param_strategies,
                crown_param_strategies,
                beta_crown_rules,
                backwards_crown_rules,
            )
            split_bounds_domains |= split_domains
            split_param_domains |= beta_domains
            relax_param_domains |= alpha_domains
    return split_bounds_domains, split_param_domains, relax_param_domains


# -----------------------------------------------------------------------------
# Beta-CROWN Rules
# -----------------------------------------------------------------------------

# this dict needs to be hashable for caching in backwards_lirpa
_primitive_backward_beta_crown_rules: frozendict[
    jax.core.Primitive | Marker, BackwardsCROWNRule
] = frozendict()


def register_beta_crown_rule(
    primitive: jax.core.Primitive | Marker,
    rule: BackwardsCROWNRule,
):
    """Registers a new beta-CROWN rule.

    Args:
        primitive: The primitive (or marker) that is handled by this rule.
        rule: The beta-CROWN rule for ``primitive``.
    """
    global _primitive_backward_beta_crown_rules
    _primitive_backward_beta_crown_rules = _primitive_backward_beta_crown_rules | {
        primitive: rule
    }


# ==============================================================================
# Beta-CROWN Primitive Rules
# ==============================================================================


class BaseBetaCROWNRule(BetaCROWNRule, ABC):
    """A base class for beta-CROWN rules.

    Delegates to a backwards CROWN rule but adds Lagrange terms (beta)
    to the LiRPA bounds.

    Implement ``_add_lagrange_terms`` and ``_beta_domains`` when subclassing.
    While ``_beta_domains`` defines the number and shape of the beta parameters,
    ``_add_lagrange_terms`` adds the Lagrange terms to the LiRPA bounds.
    """

    def __init__(self, crown_rule: BackwardsCROWNRule):
        self._crown_rule = crown_rule

    def __call__(
        self,
        params: tuple[
            None
            | tuple[Sequence[Bounds[Real[Array, "..."]]], PyTree[Real[Array, "..."]]],
            PyTree[Real[Array, "..."] | BACKWARDS_CROWN_PARAM_STRATEGIES],
        ],
        out_weights: Sequence[LiRPAWeights[Real[Array, "..."] | Zero]],
        in_bounds: Sequence[Bounds | jax.Array],
        **kwargs,
    ) -> (
        tuple[LiRPABounds[Real[Array, "..."] | Zero], ...]
        | LiRPABounds[Real[Array, "..."] | Zero]
    ):
        split_params, relaxation_params = params
        if split_params is None:
            return self._crown_rule(relaxation_params, out_weights, in_bounds, **kwargs)

        split_bounds, betas = split_params
        # clip the input bounds by the split bounds
        in_bounds = tuple(
            self._clip_in_bounds(val, split)
            for val, split in strict_zip(in_bounds, split_bounds)
        )

        in_lirpa_bounds = self._crown_rule(
            relaxation_params, out_weights, in_bounds, **kwargs
        )

        single = isinstance(in_lirpa_bounds, LiRPABounds)
        if single:
            in_lirpa_bounds = (in_lirpa_bounds,)
        in_lirpa_bounds = tuple(
            self._add_lagrange_terms(lirpa_bounds, split_bounds, betas)
            for lirpa_bounds in in_lirpa_bounds
        )
        if single:
            in_lirpa_bounds = in_lirpa_bounds[0]
        return in_lirpa_bounds

    def _clip_in_bounds(
        self,
        val: Bounds | jax.Array,
        split_bounds: Bounds[Real[Array, "..."]],
    ) -> Bounds | jax.Array:
        if is_bounds(val):
            return Box(
                jnp.maximum(val.lower_bound, split_bounds.lower_bound),
                jnp.minimum(val.upper_bound, split_bounds.upper_bound),
            )
        else:
            return val

    def parameter_domains(
        self,
        in_shapes: Sequence[tuple[int, ...]],
        out_shapes: Sequence[tuple[int, ...]],
        strategies: Sequence[BACKWARDS_CROWN_PARAM_STRATEGIES],
        beta_strategies: Sequence[BETA_CROWN_PARAM_STRATEGIES],
        **kwargs,
    ) -> tuple[PARAM_DOMAIN | None, ...]:
        # node bounds
        split_bounds_domains = tuple(
            tuple(Reals(shape), Reals(shape)) for shape in in_shapes
        )
        beta_domains = self._beta_domains(
            in_shapes, out_shapes, beta_strategies, **kwargs
        )
        alpha_domains = self._crown_rule.parameter_domains(
            in_shapes, out_shapes, strategies, **kwargs
        )
        # Split parameters are handled externally
        return ((split_bounds_domains, beta_domains), alpha_domains)

    @abstractmethod
    def _add_lagrange_terms(
        self,
        lirpa_bounds: LiRPABounds,
        split_bounds: Sequence[tuple[Real[Array, " *shape"], Real[Array, " *shape"]]],
        betas: PyTree[Real[Array, "..."]],
    ) -> LiRPABounds:
        """Adds the Lagrange terms (beta terms) to the LiRPA bounds for one output.

        Args:
            lirpa_bounds: The LiRPA bounds to add the Lagrange terms to.
            split_bounds: The bounds on the node's input in the branch.
            betas: The beta parameters.
        Returns:
            The LiRPA bounds with the Lagrange terms added.
        """
        raise NotImplementedError()

    @abstractmethod
    def _beta_domains(
        self,
        in_shapes: Sequence[tuple[int, ...]],
        out_shapes: Sequence[tuple[int, ...]],
        beta_strategies: Sequence[BETA_CROWN_PARAM_STRATEGIES],
        **kwargs,
    ) -> tuple[PARAM_DOMAIN | None, ...]:
        """See parameter_domains."""
        raise NotImplementedError()


class ReLUBetaCROWNRule(BaseBetaCROWNRule):
    """The beta-CROWN rule for ReLU splitting.

    This rule implements the beta-CROWN rule for ReLU splitting, as described in
    [Wang et al., 2021].

    This rule only allows one split at zero per ReLU.
    It uses fewer parameters than `ContinuousSplitBetaCROWNRule`.

    [Wang et al., 2021]: Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana,
        Cho-Jui Hsieh, J. Zico Kolter: Beta-CROWN: Efficient Bound Propagation with
        Per-neuron Split Constraints for Neural Network Robustness Verification.
        NeurIPS 2021: 29909-29921

    Args:
        crown_rule: The backwards CROWN rule for ReLU to wrap.
    """

    def _add_lagrange_terms(
        self,
        lirpa_bounds: LiRPABounds,
        split_bounds: Sequence[tuple[Real[Array, " *shape"], Real[Array, " *shape"]]],
        betas: Real[Array, " *shape"],
    ):
        "Adds Lagrange (beta) terms to LiRPA bounds."
        assert len(lirpa_bounds.lb_weights) == 1, "ReLU has multiple inputs."
        assert len(lirpa_bounds.ub_weights) == 1, "ReLU has multiple inputs."

        split_lbs, split_ubs = split_bounds
        # zero out betas when entries are not split
        lb_split = jnp.isfinite(split_lbs)
        ub_split = jnp.isfinite(split_ubs)
        # multiply by bold S matrix in [Wang et al. 2021]
        betas = jnp.where(
            lb_split, -betas, jnp.where(ub_split, betas, jnp.zeros_like(betas))
        )

        lb_weight = lirpa_bounds.lb_weights[0] + betas
        ub_weight = lirpa_bounds.ub_weights[0] - betas
        return LiRPABounds.from_arrays(
            lb_weight,
            ub_weight,
            lirpa_bounds.lb_bias,
            lirpa_bounds.ub_bias,
            lirpa_bounds.domain,
            lirpa_bounds.full_in_shapes,
            lirpa_bounds.full_out_shape,
            lirpa_bounds.batch_axis_mappings,
        )

    def _beta_domains(
        self,
        in_shapes: Sequence[tuple[int, ...]],
        out_shapes: Sequence[tuple[int, ...]],
        beta_strategies: Sequence[BETA_CROWN_PARAM_STRATEGIES],
        **kwargs,
    ) -> tuple[PARAM_DOMAIN | None, ...]:
        assert len(in_shapes) == 1, "ReLU has multiple inputs."
        assert len(beta_strategies) == 1, "ReLU got multiple beta strategies."
        in_shape = in_shapes[0]
        strategy = beta_strategies[0]
        # ReLU has one input which has one beta parameter
        match strategy:
            case "full" | "default":
                return NonNegative(in_shape)
            case "shared":
                return NonNegative((1,) * len(in_shape))
            case _:
                raise ValueError(f"Unsupported beta strategy: {strategy}")


class ContinuousSplitBetaCROWNRule(BaseBetaCROWNRule):
    """A generic beta-CROWN rule that wraps a backwards CROWN rule.

    This rule implements arbitrary splits in the continuous input space of
    a node (a jaxpr equation).
    With this split, a node can be split arbitrarily often and a split may set
    both a lower bound and an upper bound on the node's input.

    Because of this, this rule has separate Lagrange multipliers
    (beta parameters) for the lower and the upper bound.
    This means this rule introduces twice as many parameters as a ReLU
    beta-CROWN rule.

    This rule can augment the parameters of any backwards CROWN rule with node
    bounds and Lagrange multipliers (beta).

    Args:
        crown_rule: The backwards CROWN rule to wrap.
    """

    def _add_lagrange_terms(
        self,
        lirpa_bounds: LiRPABounds,
        split_bounds: Sequence[tuple[Real[Array, " *shape"], Real[Array, " *shape"]]],
        betas: Sequence[tuple[Real[Array, " *shape"], Real[Array, " *shape"]]],
    ):
        "Adds Lagrange (beta) terms to LiRPA bounds."

        # Add Lagrange terms (betas are multipliers)
        # ------------------------------------------
        # For the lower bound, we want to optimise:
        #      min_x y(x)
        #      s.t.  z_i >= l_i  (1)
        #            z_j <= u_j  (2)
        # (y: output, x: input, z: intermediate variable) which we rewrite as
        #      min_x y(x)
        #      s.t.  s_k (z_k + c_k) <= 0
        # where s_k = -1 for constraint 1, s_k = 1 for constraint 2
        # c_k = l_i for constraint 1, and c_k = u_j for constraint 2.
        # Using weak duality, we obtain:
        #   max_{beta} min_{x} y(x) + sum_{k} beta_k s_k (z_k + c_k)
        # = max_{beta} min_{x} y(x) + sum_{k} beta_k s_k z_k
        #                           + sum_{k} beta_k s_k c_k.
        # The terms of the first sum are added to the LiRPA weights, since they
        # are multiplied to the intermediate variables z_k.
        # The terms of the second sum go into the LiRPA bias.
        #
        # For the upper bound, we want to optimise:
        #      max_x y(x)          <- note the max
        #      s.t.  z_i >= l_i
        #            z_j <= u_j
        # for which we obtain:
        # min_{beta} max_{x} y(x) - sum_{k} beta_k s_k z_k
        #                         - sum_{k} beta_k s_k c_k
        # with min/max and signs of the sums flipped compared to the lower bound.

        split_lbs, split_ubs = split_bounds
        lbs_betas, ubs_betas = betas

        # Zero out dimensions that are not split and multiply by s_k
        lbs_betas = tuple(
            -jnp.isfinite(lbs).float() * lbs_beta  # leading "-" is s_k
            for lbs, lbs_beta in strict_zip(split_lbs, lbs_betas)
        )
        ubs_betas = tuple(
            jnp.isfinite(ubs).float() * ubs_beta
            for ubs, ubs_beta in strict_zip(split_ubs, ubs_betas)
        )

        lb_weights = tuple(
            lb_w + lbs_beta + ubs_beta
            for lb_w, lbs_beta, ubs_beta in strict_zip(
                lirpa_bounds.lb_weights, lbs_betas, ubs_betas
            )
        )
        ub_weights = tuple(
            ub_w - lbs_beta - ubs_beta
            for ub_w, lbs_beta, ubs_beta in strict_zip(
                lirpa_bounds.ub_weights, lbs_betas, ubs_betas
            )
        )

        beta_sum = sum(
            lbs * lbs_beta + ubs * ubs_beta
            for lbs, ubs, lbs_beta, ubs_beta in strict_zip(
                split_lbs, split_ubs, lbs_betas, ubs_betas
            )
        ).sum()
        lb_bias = lirpa_bounds.lb_bias + beta_sum
        ub_bias = lirpa_bounds.ub_bias - beta_sum

        return LiRPABounds.from_arrays(
            lb_weights,
            ub_weights,
            lb_bias,
            ub_bias,
            lirpa_bounds.domain,
            lirpa_bounds.full_in_shapes,
            lirpa_bounds.full_out_shape,
            lirpa_bounds.batch_axis_mappings,
        )

    def _beta_domains(
        self,
        in_shapes: Sequence[tuple[int, ...]],
        out_shapes: Sequence[tuple[int, ...]],
        beta_strategies: Sequence[BETA_CROWN_PARAM_STRATEGIES],
        **kwargs,
    ) -> tuple[PARAM_DOMAIN | None, ...]:
        def param_shape(shape, strategy):
            match strategy:
                case "full" | "default":
                    return shape
                case "shared":
                    return (1,) * len(shape)
                case _:
                    raise ValueError(f"Unsupported beta strategy: {strategy}")

        param_shapes = tuple(
            param_shape(shape, strategy)
            for shape, strategy in strict_zip(in_shapes, beta_strategies)
        )
        return tuple(NonNegative(shape) for shape in param_shapes)


beta_crown_relu_rule = ReLUBetaCROWNRule(backwards_crown_relu_rule)
"""The beta-CROWN rule for ReLU.

Implements the beta-CROWN rule for ReLU from [Wang et al., 2021].

[Wang et al., 2021]: Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana,
    Cho-Jui Hsieh, J. Zico Kolter: Beta-CROWN: Efficient Bound Propagation with
    Per-neuron Split Constraints for Neural Network Robustness Verification.
    NeurIPS 2021: 29909-29921
"""
register_beta_crown_rule(relu_marker, beta_crown_relu_rule)

# TODO: rules for min, max, abs


def _register_continuous_split(primitive):
    crown_rule = _primitive_backward_crown_rules[primitive]
    register_beta_crown_rule(
        primitive,
        ContinuousSplitBetaCROWNRule(crown_rule),
    )


# TODO: implement improved rules for abs, min, max, and reduce_window_max.

_register_continuous_split(lax.abs_p)
_register_continuous_split(lax.acosh_p)
_register_continuous_split(lax.asinh_p)
_register_continuous_split(lax.atan_p)
_register_continuous_split(lax.cbrt_p)
_register_continuous_split(lax.cosh_p)
_register_continuous_split(lax.erf_p)
_register_continuous_split(lax.exp_p)
_register_continuous_split(lax.expm1_p)
_register_continuous_split(lax.integer_pow_p)
_register_continuous_split(lax.log_p)
_register_continuous_split(lax.log1p_p)
_register_continuous_split(lax.logistic_p)
_register_continuous_split(lax.max_p)
_register_continuous_split(lax.min_p)
_register_continuous_split(lax.reduce_window_max_p)
_register_continuous_split(lax.rsqrt_p)
_register_continuous_split(lax.sqrt_p)
_register_continuous_split(lax.square_p)
_register_continuous_split(lax.tanh_p)
