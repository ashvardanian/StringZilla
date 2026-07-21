"""Shared engine and device scaffolding for the StringZillas test family."""

from typing import Literal

import numpy as np
import stringzillas as szs

from test.sz_helpers import capability_sweep


DeviceName = Literal["default", "cpu_cores", "gpu_device"]
DEVICE_NAMES = ["default", "cpu_cores", "gpu_device"] if "cuda" in szs.__capabilities__ else ["default", "cpu_cores"]


def device_scope_and_capabilities(device: DeviceName):
    """Create a DeviceScope based on the specified device type."""
    if device == "default":
        return szs.DeviceScope(), ("serial",)
    elif device == "cpu_cores":
        return szs.DeviceScope(cpu_cores=2), ("serial", "parallel")
    elif device == "gpu_device":
        return szs.DeviceScope(gpu_device=0), ("cuda",)
    else:
        raise ValueError(f"Unknown device type: {device}")


InputSizeConfig = Literal["one-large", "few-big", "many-small"]
INPUT_SIZE_CONFIGS = ["one-large", "few-big", "many-small"]


def generate_string_batches(config: InputSizeConfig):
    """Generate string batches based on the specified configuration.

    Returns:
        tuple: (batch_size, min_length, max_length) parameters for generating test strings
    """
    if config == "one-large":
        return 1, 50, 1024  # Single pair of long strings
    elif config == "few-big":
        return 7, 30, 128  # Few pairs of medium strings
    elif config == "many-small":
        return 1000, 10, 30  # Many pairs of short strings
    else:
        raise ValueError(f"Unknown input size config: {config}")


def run_across_engines(make_engine, queries, candidates=None, device=None):
    """Build a fresh engine for every `capability_sweep()` config and run the same inputs through
    each, returning one result matrix per config in sweep order. `device` defaults to a plain
    `szs.DeviceScope()`; callers sweep `DeviceScope(cpu_cores=2)` too via parametrization."""
    if device is None:
        device = szs.DeviceScope()
    matrices = []
    for capabilities in capability_sweep():
        engine = make_engine(capabilities=capabilities)
        if candidates is not None:
            matrices.append(engine(queries, candidates, device=device))
        else:
            matrices.append(engine(queries, device=device))
    return matrices


def assert_engines_agree(matrices, oracle_matrix=None, mask=None, context=""):
    """Every backend in `matrices` (one per `capability_sweep()` config) must agree with the first
    (kernel parity) and, if given, with an independently computed oracle matrix (correctness). `mask`
    optionally restricts the comparison to a boolean-indexed subset of cells, carving out cells
    excluded for a reason documented at the call site without weakening every other cell."""
    reference_matrix = matrices[0]
    for matrix in matrices[1:]:
        compared_matrix = matrix[mask] if mask is not None else matrix
        compared_reference = reference_matrix[mask] if mask is not None else reference_matrix
        assert np.array_equal(compared_matrix, compared_reference), f"Backend divergence across capability sweep{context}"
    if oracle_matrix is not None:
        compared_reference = reference_matrix[mask] if mask is not None else reference_matrix
        compared_oracle = oracle_matrix[mask] if mask is not None else oracle_matrix
        assert np.array_equal(compared_reference, compared_oracle), f"Backend(s) disagree with the oracle{context}"
