"""
Shared pytest configuration for the StringZilla per-family test modules.

Hosts the session-wide environment banner and the QEMU capability mask so every split test file
(test_string.py, test_find.py, test_utf8_wordbreaks.py, …) inherits them without importing anything. The
seeded-RNG helpers and `SEED_VALUES` live in `test_helpers` and are imported by each module directly.
"""

import os
import platform

import pytest

from test.sz_helpers import (
    ITERATIONS_MULTIPLIER,
    SEED_VALUES,
    UnicodeDataDownloadError,
    _random_seed_for_run,
    get_combining_classes,
    get_grapheme_break_properties,
    get_line_break_properties,
    get_sentence_break_properties,
    get_uncased_folding_rules,
    get_word_break_properties,
    numpy_available,
    pyarrow_available,
)

import stringzilla as sz

if numpy_available:
    import numpy as np
if pyarrow_available:
    import pyarrow as pa


@pytest.fixture(scope="session", autouse=True)
def log_test_environment():
    """Automatically log environment info before running any tests."""

    print()  # New line for better readability
    print("=== StringZilla Test Environment ===")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"StringZilla version: {sz.__version__}")
    print(f"StringZilla capabilities: {sorted(sz.__capabilities__)}")
    print(f"NumPy available: {numpy_available}")
    if numpy_available:
        print(f"NumPy version: {np.__version__}")
    print(f"PyArrow available: {pyarrow_available}")
    if pyarrow_available:
        print(f"PyArrow version: {pa.__version__}")
    print(f"Test seeds: {SEED_VALUES}")
    if _random_seed_for_run in SEED_VALUES:
        print(f"  (random seed for this run: {_random_seed_for_run})")
    if ITERATIONS_MULTIPLIER != 1.0:
        print(f"Iterations multiplier: {ITERATIONS_MULTIPLIER:.2f}x")

    # If QEMU is indicated via env (e.g., set by pyproject), mask out SVE/SVE2 to avoid emulation flakiness.
    is_qemu = os.environ.get("SZ_IS_QEMU_", "").lower() in ("1", "true", "yes", "on")
    if is_qemu:
        sve_like = {"sve", "sve2", "sve2aes"}
        current = list(getattr(sz, "__capabilities__", ()))
        desired = tuple(c for c in current if c.lower() not in sve_like)
        if len(desired) != len(current):
            print(f"QEMU env detected; disabling {sve_like} for stability")
            sz.reset_capabilities(desired)

    print("=" * 40)
    print()  # New line for better readability


# Unicode property tables shared across the segmentation families. Session-scoped so each table is
# downloaded and parsed once, and every consumer skips uniformly when the data is unreachable.
@pytest.fixture(scope="session")
def grapheme_break_props():
    try:
        return get_grapheme_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode grapheme-break data unavailable")


@pytest.fixture(scope="session")
def word_break_props():
    try:
        return get_word_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode word-break data unavailable")


@pytest.fixture(scope="session")
def sentence_break_props():
    try:
        return get_sentence_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode sentence-break data unavailable")


@pytest.fixture(scope="session")
def line_break_props():
    try:
        return get_line_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode line-break data unavailable")


@pytest.fixture(scope="session")
def combining_classes():
    try:
        return get_combining_classes()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode combining-class data unavailable")


@pytest.fixture(scope="session")
def unicode_folds():
    try:
        return get_uncased_folding_rules()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode case-folding data unavailable")
