#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import pytest

from ..module_cases import test_multiple_cases, test_once_cases
from .test_ibp import TestIBP

# Initialize Test class
IBP_test_instance = TestIBP()


@pytest.mark.benchmark(
    group="Bound Propagation Benchmarks",
    max_time=15,
    min_rounds=50,
    disable_gc=False,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.parametrize("case", test_once_cases)
def test_bounds_once_benchmark(case, request, benchmark):
    benchmark(IBP_test_instance.test_bounds_once, case, request)


@pytest.mark.parametrize("case", test_multiple_cases)
def test_bounds_repeat5_benchmark(case, argument_seeds5, request, benchmark):
    benchmark(IBP_test_instance.test_bounds_repeat5, case, argument_seeds5, request)


def test_intermediate_bounds_benchmark(benchmark):
    benchmark(IBP_test_instance.test_intermediate_bounds)


@pytest.mark.parametrize(
    "case",
    [
        "simple_relu_case",
        "jax_relu_case",
    ],
)
def test_grad_once_benchmark(case, request, benchmark):
    benchmark(IBP_test_instance.test_grad_once, case, request)


@pytest.mark.parametrize(
    "case",
    [
        "simple_relu_case",
        "jax_relu_case",
        "affine_layer_case2",
        "shallow_nn_case",
        "shallow_nn_jit_case",
        "shallow_vmapped_case",
        "shallow_vmapped_jit_case",
    ],
)
def test_grad_repeat5_benchmark(case, argument_seeds5, request, benchmark):
    benchmark(IBP_test_instance.test_grad_repeat5, case, argument_seeds5, request)


@pytest.mark.parametrize(
    "rng_seed", [pytest.param(f"smears-{i}", id=f"seed{i}") for i in range(5)]
)
@pytest.mark.parametrize(
    "case",
    ["shallow_nn_case"],
)
def test_smears_benchmark(case, rng_seed, request, benchmark):
    benchmark(IBP_test_instance.test_smears, case, rng_seed, request)


def test_relu_smears_benchmark(benchmark):
    benchmark(IBP_test_instance.test_relu_smears)


@pytest.mark.parametrize(
    "rng_seed", [pytest.param(f"smears-affine-{i}", id=f"seed{i}") for i in range(5)]
)
@pytest.mark.parametrize(
    "case",
    ["affine_layer_case2"],
)
def test_affine_smears_benchmark(case, rng_seed, request, benchmark):
    benchmark(IBP_test_instance.test_affine_smears, case, rng_seed, request)
