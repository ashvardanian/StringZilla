#!/usr/bin/env python3
"""UTF-8 codepoint counting and iteration: sz.utf8_count and sz.utf8_codepoints.

Mirrors the C++ scripts/test_utf8_codepoints.cpp translation unit.

Covers: utf8_count and utf8_codepoints on ASCII, multi-byte, and mixed-script text, str/Str/bytes
input parity, round-trip fidelity against random valid UTF-8 corpora, malformed-byte safety
including lossy U+FFFD decoding, and cross-backend agreement on curated, malformed, and random
corpora.
Compares against: CPython's own codepoint iteration for valid UTF-8, and cross-backend
self-consistency across every SIMD backend.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/utf8_codepoints.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/utf8_codepoints.py -q
"""

from random import Random

import pytest

import stringzilla as sz

from test.sz_helpers import (
    vector_width_bracketing_strings,
    SEED_VALUES,
    scale_iterations,
    seed_random_generators,
    run_across_backends,
    assert_backends_agree,
    get_random_string,
    malformed_utf8_corpus,
)
from test.utf8_helpers import adversarial_utf8_inputs, random_segmentation_corpus


# region Unit


def test_unit_utf8_count():
    """utf8_count returns the Unicode scalar count, not the byte length, across ASCII, multi-byte, and
    mixed-script text, the Str method form, and bytes input."""
    # ASCII strings: len == utf8_count
    assert sz.utf8_count("hello") == 5
    assert sz.utf8_count("") == 0
    assert sz.utf8_count("a") == 1

    # Multi-byte UTF-8: character count != byte count
    assert sz.utf8_count("café") == 4  # é is 2 bytes but 1 char
    assert sz.utf8_count("日本語") == 3  # Each CJK char is 3 bytes
    assert sz.utf8_count("🎉") == 1  # Emoji is 4 bytes but 1 char
    assert sz.utf8_count("👨‍👩‍👧") == 5  # Family emoji: 3 people + 2 ZWJ

    # Mixed ASCII and multi-byte
    assert sz.utf8_count("hello世界") == 7  # 5 ASCII + 2 CJK
    assert sz.utf8_count("naïve") == 5  # ï is 2 bytes

    # Method form on Str
    assert sz.Str("café").utf8_count() == 4

    # Bytes input
    assert sz.utf8_count(b"caf\xc3\xa9") == 4  # café in UTF-8 bytes

    # Various multi-byte sequences
    assert sz.utf8_count("αβγδ") == 4  # Greek letters (2 bytes each)
    assert sz.utf8_count("АБВГ") == 4  # Cyrillic letters (2 bytes each)
    assert sz.utf8_count("𐍈") == 1  # Gothic letter (4 bytes)


def test_unit_utf8_codepoints():
    """utf8_codepoints yields one int per Unicode scalar value, matching Python's own iteration."""
    assert list(sz.utf8_codepoints("")) == []
    assert list(sz.utf8_codepoints("AB")) == [65, 66]
    for text in ["café", "日本語", "🎉", "αβγδ", "АБВГ", "𐍈", "hello世界"]:
        assert list(sz.utf8_codepoints(text)) == [ord(c) for c in text]
    # str, Str, and bytes inputs agree.
    assert list(sz.utf8_codepoints(sz.Str("café"))) == [ord(c) for c in "café"]
    assert list(sz.utf8_codepoints("café".encode())) == [ord(c) for c in "café"]


# endregion Unit


# region Corner cases


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_count_roundtrip(seed_value: int):
    """On valid UTF-8 corpora, utf8_count equals Python's codepoint count for both str and bytes inputs."""
    rng = Random(seed_value)
    for _ in range(scale_iterations(200)):
        raw = random_segmentation_corpus(rng.randint(1, 400), "valid", None, rng)
        text = raw.decode("utf-8")
        assert sz.utf8_count(text) == len(text)
        assert sz.utf8_count(raw) == len(text)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_codepoints_roundtrip(seed_value: int):
    """On valid UTF-8 corpora, utf8_codepoints reproduces Python's codepoint sequence for str and bytes inputs."""
    rng = Random(seed_value)
    for _ in range(scale_iterations(200)):
        raw = random_segmentation_corpus(rng.randint(1, 400), "valid", None, rng)
        text = raw.decode("utf-8")
        expected = [ord(c) for c in text]
        assert list(sz.utf8_codepoints(text)) == expected
        assert list(sz.utf8_codepoints(raw)) == expected


def test_utf8_count_malformed():
    """Malformed-byte safety: utf8_count must survive the adversarial battery and stay within [0, byte length]."""
    rng = Random(0)
    for raw in adversarial_utf8_inputs(rng, random_input_count=scale_iterations(500)):
        count = sz.utf8_count(raw)
        assert 0 <= count <= len(raw)


def test_utf8_codepoints_malformed():
    """Lossy + total: malformed bytes decode to U+FFFD, iteration never raises, and only scalar values are emitted."""
    assert list(sz.utf8_codepoints(b"a\xffb")) == [0x61, 0xFFFD, 0x62]
    rng = Random(0)
    for raw in adversarial_utf8_inputs(rng, random_input_count=scale_iterations(500)):
        for codepoint in sz.utf8_codepoints(raw):
            assert 0 <= codepoint <= 0x10FFFF and not (0xD800 <= codepoint <= 0xDFFF)


# endregion Corner cases


# region Backend differential


UTF8_CODEPOINTS_BACKEND_DIFFERENTIAL_TEXTS = [
    "hello",
    "café",
    "日本語",
    "🎉",
    "αβγδ",
    "АБВГ",
    "𐍈",
    "hello世界",
    "👨‍👩‍👧",
] + vector_width_bracketing_strings()


@pytest.mark.parametrize("text", UTF8_CODEPOINTS_BACKEND_DIFFERENTIAL_TEXTS)
def test_utf8_count_backend_differential(text):
    """utf8_count agrees across every capability_sweep() backend and equals Python's own codepoint count
    for CJK, emoji, ZWJ, and vector-width-boundary text; a divergence is a kernel bug, not a binding bug."""
    results = run_across_backends(lambda: sz.utf8_count(text))
    assert_backends_agree(results, oracle=len(text), format_inputs=lambda: repr(text))


@pytest.mark.parametrize("text", UTF8_CODEPOINTS_BACKEND_DIFFERENTIAL_TEXTS)
def test_utf8_codepoints_backend_differential(text):
    """utf8_codepoints agrees across every capability_sweep() backend on the same curated text as the
    utf8_count sweep; a divergence is a kernel bug, not a binding bug."""
    results = run_across_backends(lambda: list(sz.utf8_codepoints(text)))
    assert_backends_agree(results, oracle=[ord(c) for c in text], format_inputs=lambda: repr(text))


@pytest.mark.parametrize("raw", malformed_utf8_corpus())
def test_utf8_count_backend_differential_malformed(raw):
    """utf8_count agrees across every capability_sweep() backend on the malformed-byte corpus and never
    crashes; a divergence is a kernel bug, not a binding bug."""
    results = run_across_backends(lambda: sz.utf8_count(raw))
    assert_backends_agree(results, format_inputs=lambda: raw.hex())
    assert 0 <= next(iter(results.values())) <= len(raw)


@pytest.mark.parametrize("raw", malformed_utf8_corpus())
def test_utf8_codepoints_backend_differential_malformed(raw):
    """utf8_codepoints agrees across every capability_sweep() backend on the malformed-byte corpus, never
    crashes, and only emits valid, non-surrogate Unicode scalar values; a divergence is a kernel bug, not
    a binding bug."""
    results = run_across_backends(lambda: list(sz.utf8_codepoints(raw)))
    assert_backends_agree(results, format_inputs=lambda: raw.hex())
    for codepoint in next(iter(results.values())):
        assert 0 <= codepoint <= 0x10FFFF and not (0xD800 <= codepoint <= 0xDFFF)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_count_backend_differential_random(seed_value):
    """utf8_count agrees across every capability_sweep() backend and with Python's own codepoint count
    for random ASCII corpora; a divergence is a kernel bug, not a binding bug."""
    seed_random_generators(seed_value)
    text = get_random_string()
    results = run_across_backends(lambda: sz.utf8_count(text))
    assert_backends_agree(results, oracle=len(text), format_inputs=lambda: repr(text))


# endregion Backend differential
