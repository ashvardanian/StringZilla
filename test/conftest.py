"""
Shared pytest configuration for the StringZilla per-family test modules.

Hosts the settings header, the `STRINGZILLA_FILTER` hook and the `seed`/`rng` fixtures,
so every split test file (string_types.py, find.py, utf8_wordbreaks.py, …) inherits them
without importing anything. The settings themselves are parsed once into `SETTINGS` in `base`.

File: test/conftest.py
Author: Ash Vardanian
Date: August 30, 2025
"""

from __future__ import annotations

import platform
import random

import pytest
from base import (
    SETTINGS,
    capability_sweep,
    StreamKey,
    UnicodeDataDownloadError,
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
        f"- Platform: {platform.platform()}",
        f"- Architecture: {platform.machine()}",
        f"- Python: {platform.python_version()}",
        f"- StringZilla: {sz.__version__}",
        f"- Capabilities: {sz.cpu_capabilities_enabled()!r}",
        f"- NumPy: {np.__version__ if numpy_available else 'none'}",
        f"- PyArrow: {pa.__version__ if pyarrow_available else 'none'}",
        f"- Seed: {SETTINGS.seed}",
        f"- Filter: {SETTINGS.filter or 'none'}",
        f"- Scale: {SETTINGS.scale}",
        f"- In QEMU: {str(SETTINGS.in_qemu).lower()}",
    ]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keeps only the tests whose node id `STRINGZILLA_FILTER` selects, on top of any `-k`.

    Tests drawing from `rng` or `np_rng` run on one thread, as `--parallel-threads` hands every thread the
    same generator, and interleaved draws would not replay from the seed.
    """
    deselected = [item for item in items if not SETTINGS.selects(item.nodeid)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if SETTINGS.selects(item.nodeid)]
    for item in items:
        if {"rng", "np_rng"} & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.thread_unsafe(reason="draws from a generator its threads would share"))


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> StreamKey:
    """Share inputs across CPU masks while keeping other parameters and repeat steps distinct."""
    name = request.node.name
    callspec = getattr(request.node, "callspec", None)
    if callspec is not None and {"capability", "capabilities"} & callspec.params.keys():
        parameters = {key: value for key, value in callspec.params.items() if key not in {"capability", "capabilities"}}
        name = (request.node.originalname or request.node.name) + repr(parameters)
    return stream_key(SETTINGS.seed, name)


@pytest.fixture
def rng(seed: StreamKey) -> random.Random:
    """A generator private to this test, so a neighbour's draws cannot shift this one's."""
    return random.Random(seed)


@pytest.fixture
def np_rng(seed: StreamKey) -> np.random.Generator:
    """A NumPy generator private to this test, so a neighbour's draws cannot shift this one's."""
    numpy = pytest.importorskip("numpy")
    generator: np.random.Generator = numpy.random.default_rng(seed)
    return generator


# Unicode property tables and conformance corpora shared across the segmentation families,
# session-scoped so each file is downloaded and parsed once. Every consumer skips uniformly when the
# data is unreachable, and no test body touches the network from a `pytest-run-parallel` thread.
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


@pytest.fixture(scope="session")
def indic_conjunct_breaks():
    try:
        return get_indic_conjunct_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode Indic-conjunct-break data unavailable")


@pytest.fixture(scope="session")
def extended_pictographic():
    try:
        return get_extended_pictographic()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode Extended_Pictographic data unavailable")


@pytest.fixture(scope="session")
def grapheme_break_cases():
    try:
        return get_grapheme_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode GraphemeBreakTest data unavailable")


@pytest.fixture(scope="session")
def word_break_cases():
    try:
        return get_word_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode WordBreakTest data unavailable")


@pytest.fixture(scope="session")
def sentence_break_cases():
    try:
        return get_sentence_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode SentenceBreakTest data unavailable")


@pytest.fixture(scope="session")
def line_break_cases():
    try:
        return get_line_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Unicode LineBreakTest data unavailable")


@pytest.fixture(scope="session")
def normalization_cases():
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
