#!/usr/bin/env python3
"""Fingerprints: MinHash fingerprinting via szs.Fingerprints construction and computation.

Mirrors the C++ test/fingerprints.cuh translation unit.

Covers: engine construction with explicit `capabilities` or an inferred `device_scope` across
`ndim`, hash and count output shape and `uint32` dtype for empty and non-empty batches, identical
strings producing identical fingerprints versus different strings producing different hashes,
determinism of repeated computation over random batches, and degenerate batch shapes such as an
empty batch, a single-character batch, and an all-same-character batch.
Compares against: self-consistency across `device_name` and `capabilities_mode` pairings and
across repeated calls on the same batch; `Fingerprints` has no NEON dispatch on Arm, so there is
no `capability_sweep()` differential oracle to compare against here.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat affine-gaps
    SZ_TARGET=stringzillas-cpus uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/fingerprints.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/fingerprints.py -q
    SZ_TARGET=stringzillas-cuda uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/fingerprints.py -q
"""

from random import randint

import pytest
import numpy as np

import stringzillas as szs
from stringzilla import Strs

from test.sz_helpers import SEED_VALUES, seed_random_generators, get_random_string
from test.szs_helpers import DeviceName, DEVICE_NAMES, device_scope_and_capabilities


# region Unit


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("ndim", [1, 7, 64, 1024])
def test_fingerprints(capabilities_mode: str, device_name: str, ndim: int):
    """Constructs `Fingerprints` with explicit `capabilities` or an inferred `device_scope`, across
    `ndim`; hashes and counts have the right shape and `uint32` dtype for an empty batch and a real
    batch, identical strings produce identical fingerprints, and different strings produce
    different hashes."""

    # Create engine with smaller dimensions to avoid memory issues
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.Fingerprints(ndim=ndim, capabilities=base_caps if capabilities_mode == "base" else device_scope)

    # Empty input should return empty arrays
    hashes, counts = engine(Strs([]), device=device_scope)
    assert hashes.shape == (0, ndim)
    assert counts.shape == (0, ndim)
    assert hashes.dtype == np.uint32
    assert counts.dtype == np.uint32

    test_strings = Strs(["hello", "world", "hello"])
    hashes, counts = engine(test_strings, device=device_scope)

    # Check output shape and types
    assert hashes.shape == (3, ndim), f"Expected (3, {ndim}), got {hashes.shape}"
    assert counts.shape == (3, ndim), f"Expected (3, {ndim}), got {counts.shape}"
    assert hashes.dtype == np.uint32
    assert counts.dtype == np.uint32

    # Identical strings should produce identical fingerprints
    assert np.array_equal(hashes[0], hashes[2]), "Identical strings should produce identical hashes"
    assert np.array_equal(counts[0], counts[2]), "Identical strings should produce identical counts"

    # Different strings should produce different fingerprints, but we can't always expect
    # different counts on very short inputs
    assert not np.array_equal(hashes[0], hashes[1]), "Different strings should produce different hashes"


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("ndim", [1, 7, 64, 1024])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_fingerprints_random(batch_size: int, capabilities_mode: str, device_name: str, ndim: int, seed_value: int):
    """Fingerprints over randomly generated batches, seeded via `SEED_VALUES` for reproducibility,
    match the requested shape and reproduce identical hashes and counts when computed twice on the
    same batch."""

    seed_random_generators(seed_value)
    batch = [get_random_string(length=randint(5, 50)) for _ in range(batch_size)]

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.Fingerprints(ndim=ndim, capabilities=base_caps if capabilities_mode == "base" else device_scope)

    strs = Strs(batch)
    hashes, counts = engine(strs, device=device_scope)
    assert hashes.shape == (batch_size, ndim)
    assert counts.shape == (batch_size, ndim)

    # Verify consistency
    hashes_repeated, counts_repeated = engine(strs, device=device_scope)
    assert np.array_equal(hashes, hashes_repeated), "Same input should produce same hashes"
    assert np.array_equal(counts, counts_repeated), "Same input should produce same counts"


# endregion Unit


# region Corner cases


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("ndim", [1, 7, 64, 1024])
def test_fingerprints_degenerate_cases(capabilities_mode: str, device_name: DeviceName, ndim: int):
    """Degenerate-shape coverage for `Fingerprints`: an empty batch, a single-character batch, an
    all-same-character batch, and determinism, where scoring the same batch twice must produce
    byte-identical hashes and counts. `Fingerprints` has no NEON dispatch on Arm, so this is plain
    serial-backend functional coverage rather than a `capability_sweep` differential."""

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.Fingerprints(ndim=ndim, capabilities=base_caps if capabilities_mode == "base" else device_scope)

    # Empty batch.
    empty_hashes, empty_counts = engine(Strs([]), device=device_scope)
    assert empty_hashes.shape == (0, ndim)
    assert empty_counts.shape == (0, ndim)

    # Single-character batch.
    single_character_hashes, single_character_counts = engine(Strs(["a"]), device=device_scope)
    assert single_character_hashes.shape == (1, ndim)
    assert single_character_counts.shape == (1, ndim)

    # All-same-character batch: every row must fingerprint identically.
    all_same_character_strings = ["a" * 5, "a" * 5, "a" * 5]
    all_same_character_hashes, all_same_character_counts = engine(Strs(all_same_character_strings), device=device_scope)
    assert np.array_equal(all_same_character_hashes[0], all_same_character_hashes[1])
    assert np.array_equal(all_same_character_hashes[1], all_same_character_hashes[2])
    assert np.array_equal(all_same_character_counts[0], all_same_character_counts[1])
    assert np.array_equal(all_same_character_counts[1], all_same_character_counts[2])

    # Determinism: scoring the same batch twice must be byte-identical.
    mixed_strings = ["a", "a" * 40, "b", ""]
    mixed_strs = Strs(mixed_strings)
    first_hashes, first_counts = engine(mixed_strs, device=device_scope)
    second_hashes, second_counts = engine(mixed_strs, device=device_scope)
    assert first_hashes.shape == (len(mixed_strings), ndim)
    assert np.array_equal(first_hashes, second_hashes), "Same input should produce identical hashes"
    assert np.array_equal(first_counts, second_counts), "Same input should produce identical counts"


# endregion Corner cases


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-x", "-s", __file__]))
