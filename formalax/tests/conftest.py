#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import random
from pathlib import Path

import numpy as np
import pytest

# make fixtures available to all tests
from .module_cases import (  # noqa: F401
    argument_seeds3,
    argument_seeds5,
    argument_seeds10,
)
from .nets import (  # noqa: F401
    acasxu_network,
    emnist_flax_conv,
    mnist_equinox_conv_batchnorm,
    mnist_flax_mlp,
    mnist_ibp_training_flax_conv,
    mnist_onnx_conv,
    mnist_onnx_fully_connected,
)


@pytest.fixture(autouse=True)
def seed_rngs(base_seed=0):  # reproducibility backup (seed should be reset still)
    random.seed(base_seed + 1)
    np.random.seed(base_seed + 2)


@pytest.fixture
def resource_dir():
    preferred_path = Path("resources")
    if preferred_path.exists():
        return preferred_path
    else:
        return Path("tests/resources")


def pytest_addoption(parser):
    # don't run benchmarks by default, enable with command line option
    # See:
    # https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="Run benchmarks."
    )


def pytest_collection_modifyitems(config, items):
    # Follow-up for turning off benchmarks by default
    # https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    if config.getoption("--benchmark"):
        # with --benchmark option, run all tests
        return
    skip_benchmark = pytest.mark.skip(
        reason="Skipping benchmarks. Run with --benchmark to execute benchmarks."
    )
    for item in items:
        # the benchmark mark comes from the pytest-benchmark plugin
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
