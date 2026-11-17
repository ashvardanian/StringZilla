"""
Shared pytest configuration for the StringZilla per-family test modules.

Hosts the settings header, the case cap, the `STRINGZILLA_FILTER` hook and the `seed`/`rng` fixtures,
so every split test file (string_types.py, find.py, utf8_wordbreaks.py, …) inherits them
without importing anything. The settings themselves are parsed once into `SETTINGS` in `base`.

File: test/conftest.py
Author: Ash Vardanian
Date: August 30, 2025
"""

from __future__ import annotations

import collections
import ctypes
import ctypes.util
import enum
import platform
import random
import sys

import pytest
from base import (
    BACKEND_AXES,
    MAX_CASES_PER_TEST,
    SETTINGS,
    UnicodeDataDownloadError,
    capability_sweep,
    get_combining_classes,
    get_extended_pictographic,
    get_grapheme_break_properties,
    get_grapheme_break_test_cases,
    get_indic_conjunct_break_properties,
    get_line_break_properties,
    get_line_break_test_cases,
    get_normalization_test_cases,
    get_sentence_break_properties,
    get_sentence_break_test_cases,
    get_uncased_folding_rules,
    get_word_break_properties,
    get_word_break_test_cases,
    numpy_available,
    pyarrow_available,
    stream_key,
)

import stringzilla as sz

if numpy_available:
    import numpy as np
if pyarrow_available:
    import pyarrow as pa


def pytest_report_header() -> list[str]:
    """What this run exercises, printed where pytest prints its own header, which shows without `-s`."""
    return [
        f"StringZilla {sz.__version__} from {sz.__file__}",
        f"- Platform: {platform.platform()}",
        f"- Architecture: {platform.machine()}",
        f"- Capabilities: {sz.capabilities_enabled_cpu()!r}",
        f"- NumPy: {np.__version__ if numpy_available else 'none'}",
        f"- PyArrow: {pa.__version__ if pyarrow_available else 'none'}",
        f"- In QEMU: {str(SETTINGS.in_qemu).lower()}",
        f"- Seed: {SETTINGS.seed}, scale: {SETTINGS.scale:g}, filter: {SETTINGS.filter.pattern or 'none'}",
    ]


# region Shared Pytest
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Refuses a test whose cases outgrow MAX_CASES_PER_TEST, then deselects what the filter setting leaves out."""
    variants: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for item in items:
        callspec = getattr(item, "callspec", None)
        parameters = dict(callspec.params) if callspec is not None else {}
        parameters.pop("__pytest_repeat_step_number", None)
        kept = {key: value for key, value in parameters.items() if key not in BACKEND_AXES}
        variants[(item.module.__name__, getattr(item, "originalname", item.name))].add(repr(kept))
    oversized = [
        f"{module}::{name} has {len(cases)}"
        for (module, name), cases in variants.items()
        if len(cases) > MAX_CASES_PER_TEST
    ]
    if oversized:
        raise pytest.UsageError(f"over MAX_CASES_PER_TEST={MAX_CASES_PER_TEST}: " + "; ".join(oversized))
    deselected = [item for item in items if not SETTINGS.filter.selects(item.nodeid)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if SETTINGS.filter.selects(item.nodeid)]


def pytest_make_parametrize_id(config: pytest.Config, val: object, argname: str) -> str | None:
    """Names enum parameters as the code spells them, `Backend.LINUX_URING618`, so an id greps to its source."""
    return f"{type(val).__name__}.{val.name}" if isinstance(val, enum.Enum) else None


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> int:
    """Share inputs across backends while keeping other parameters and repeat steps distinct."""
    callspec = getattr(request.node, "callspec", None)
    parameters = dict(callspec.params) if callspec is not None else {}
    repeat_step = parameters.pop("__pytest_repeat_step_number", 0)
    name = request.node.originalname or request.node.name
    name += repr({key: value for key, value in parameters.items() if key not in BACKEND_AXES})
    return stream_key(SETTINGS.seed, name, repeat_step)


@pytest.fixture
def rng(seed: int) -> random.Random:
    """A generator private to this test, so a neighbour's draws cannot shift this one's."""
    return random.Random(seed)


@pytest.fixture
def unraisable(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Collects what the interpreter reports as unraisable, like a finalizer's exception, for the test to check."""
    seen: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    return seen


# endregion Shared Pytest


BASELINE_BY_MACHINE = {
    ("x86_64", "amd64", "x64"): sz.Capability.HASWELL,
    ("arm64", "aarch64"): sz.Capability.NEON,
}
"""The capability every supported toolchain emits for a given 64-bit architecture. A machine that
detects one of these but did not compile it in has a broken probe, not a slow CPU.
"""


def baseline_for_this_machine():
    """The capability this machine is expected to carry, or None where none is guaranteed."""
    if not sys.maxsize > 2**32:
        return None  # 32-bit targets (i686, armv7) have no guaranteed baseline
    machine = platform.machine().lower()
    for names, baseline in BASELINE_BY_MACHINE.items():
        if machine in names:
            return baseline
    return None


@pytest.fixture
def detected_baseline() -> sz.Capability:
    """The baseline of this machine that its CPU reports, skipping where there is no SIMD to expect."""
    baseline = baseline_for_this_machine()
    if baseline is None:
        pytest.skip(f"no SIMD baseline is guaranteed on {platform.machine()}")
    if baseline not in sz.capabilities_detected_cpu():
        pytest.skip(f"this CPU does not report {baseline.name}")
    return baseline


@pytest.fixture
def cudart() -> ctypes.CDLL:
    """The CUDA runtime library, to allocate managed memory with, skipping the test where it does not load."""
    library = ctypes.util.find_library("cudart") or "libcudart.so"
    try:
        return ctypes.CDLL(library)
    except OSError:
        pytest.skip(f"needs the CUDA runtime library {library}")


def pytest_itemcollected(item: pytest.Item) -> None:
    """Runs a test drawing from `rng` or `np_rng` on one thread, as `--parallel-threads` would share its generator."""
    if {"rng", "np_rng"} & set(getattr(item, "fixturenames", ())):
        item.add_marker(pytest.mark.thread_unsafe(reason="draws from a generator its threads would share"))


@pytest.fixture
def np_rng(seed: int) -> np.random.Generator:
    """A NumPy generator private to this test, so a neighbour's draws cannot shift this one's."""
    numpy = pytest.importorskip("numpy", reason="NumPy is not installed")
    generator: np.random.Generator = numpy.random.default_rng(seed)
    return generator


# Unicode property tables and conformance corpora shared across the segmentation families,
# session-scoped so each file is downloaded and parsed once. Every consumer skips uniformly when the
# data is unreachable, and no test body touches the network from a `pytest-run-parallel` thread.
@pytest.fixture(scope="session")
def grapheme_break_props():
    """Grapheme_Cluster_Break of every code point, from GraphemeBreakProperty.txt."""
    try:
        return get_grapheme_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode grapheme-break data unavailable")


@pytest.fixture(scope="session")
def word_break_props():
    """Word_Break of every code point, from WordBreakProperty.txt."""
    try:
        return get_word_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode word-break data unavailable")


@pytest.fixture(scope="session")
def sentence_break_props():
    """Sentence_Break of every code point, from SentenceBreakProperty.txt."""
    try:
        return get_sentence_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode sentence-break data unavailable")


@pytest.fixture(scope="session")
def line_break_props():
    """Line_Break of every code point, from LineBreak.txt."""
    try:
        return get_line_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode line-break data unavailable")


@pytest.fixture(scope="session")
def combining_classes():
    """Canonical_Combining_Class of every code point that is not 0, for normalization reordering."""
    try:
        return get_combining_classes()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode combining-class data unavailable")


@pytest.fixture(scope="session")
def unicode_folds():
    """Full case folding of every code point that folds, from CaseFolding.txt."""
    try:
        return get_uncased_folding_rules()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode case-folding data unavailable")


@pytest.fixture(scope="session")
def indic_conjunct_breaks():
    """Indic_Conjunct_Break of every code point that has one, from DerivedCoreProperties.txt."""
    try:
        return get_indic_conjunct_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode Indic-conjunct-break data unavailable")


@pytest.fixture(scope="session")
def extended_pictographic():
    """Every Extended_Pictographic code point, from emoji-data.txt."""
    try:
        return get_extended_pictographic()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode Extended_Pictographic data unavailable")


@pytest.fixture(scope="session")
def grapheme_break_cases():
    """Texts and their expected boundaries from the official GraphemeBreakTest.txt."""
    try:
        return get_grapheme_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode GraphemeBreakTest data unavailable")


@pytest.fixture(scope="session")
def word_break_cases():
    """Texts and their expected boundaries from the official WordBreakTest.txt."""
    try:
        return get_word_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode WordBreakTest data unavailable")


@pytest.fixture(scope="session")
def sentence_break_cases():
    """Texts and their expected boundaries from the official SentenceBreakTest.txt."""
    try:
        return get_sentence_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode SentenceBreakTest data unavailable")


@pytest.fixture(scope="session")
def line_break_cases():
    """Texts and their expected boundaries from the official LineBreakTest.txt."""
    try:
        return get_line_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode LineBreakTest data unavailable")


@pytest.fixture(scope="session")
def normalization_cases():
    """Source, NFC, NFD, NFKC and NFKD rows from the official NormalizationTest.txt."""
    try:
        return get_normalization_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode NormalizationTest data unavailable")


@pytest.fixture(params=capability_sweep(), ids=lambda mask: mask.name.lower().replace("|", "+"))
def capabilities(request: pytest.FixtureRequest) -> sz.Capability:
    """One CPU mask from the differential correctness sweep, including serial fallback."""
    return request.param


@pytest.fixture
def backend_results(capabilities):
    """Compare this item's CPU mask with the serial reference on identical inputs."""

    def run(operation):
        masks = dict.fromkeys((sz.Capability.SERIAL, capabilities))
        return {mask: operation(mask) for mask in masks}

    return run
