#!/usr/bin/env python3
"""UTF-8 delimiter splitting: utf8_split_delimiters segments text on any Unicode punctuation, symbol, or
separator/whitespace codepoint, a superset of the utf8_tokens whitespace splitter, plus malformed-byte safety.

Mirrors the C++ test/utf8_tokens.cpp delimiter coverage.

Covers: the delimiter-splitting iterator and its skip_empty flag, module function and Str.utf8_split_delimiters
method parity, adversarial-byte safety against malformed UTF-8, and cross-backend agreement on realistic,
malformed, and lane-straddling inputs.
Compares against: cross-backend self-consistency across every capability_sweep() backend; utf8_split_delimiters
has no external oracle.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/utf8_delimiters.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/utf8_delimiters.py -q
"""

from random import Random

import pytest

import stringzilla as sz
from stringzilla import Str

from test.sz_helpers import (
    SEED_VALUES,
    assert_backends_agree,
    malformed_utf8_corpus,
    run_across_backends,
    vector_width_bracketing_strings,
)
from test.utf8_helpers import adversarial_utf8_inputs


# region Unit


def test_utf8_delimiters():
    """utf8_split_delimiters segments on any Unicode punctuation, symbol, or separator/whitespace
    codepoint; skip_empty drops the run between adjacent delimiters, and the Str method matches the
    module function."""
    # Splits on ',' and ' ': "Hi, world" -> "Hi", "" (between ',' and ' '), "world".
    assert list(sz.utf8_split_delimiters("Hi, world")) == [Str("Hi"), Str(""), Str("world")]

    # skip_empty drops the empty between adjacent delimiters.
    assert list(sz.utf8_split_delimiters("Hi, world", skip_empty=True)) == [Str("Hi"), Str("world")]

    # A single delimiter yields two segments, no empty.
    assert list(sz.utf8_split_delimiters("a.b")) == [Str("a"), Str("b")]

    # Empty string yields one empty Str (mirrors utf8_tokens).
    assert list(sz.utf8_split_delimiters("")) == [Str("")]

    # No delimiters: the whole string is one segment.
    assert list(sz.utf8_split_delimiters("hello")) == [Str("hello")]

    # Unicode punctuation: U+2014 EM DASH (Pd) is a delimiter.
    assert list(sz.utf8_split_delimiters("a—b", skip_empty=True)) == [Str("a"), Str("b")]

    # Unicode punctuation: CJK U+3002 IDEOGRAPHIC FULL STOP (Po).
    assert list(sz.utf8_split_delimiters("日本。語", skip_empty=True)) == [Str("日本"), Str("語")]

    # Emoji symbol (So) is a delimiter.
    assert list(sz.utf8_split_delimiters("a\U0001f600b", skip_empty=True)) == [Str("a"), Str("b")]

    # Method form on Str matches the module function.
    assert list(sz.Str("Hi, world").utf8_split_delimiters(skip_empty=True)) == [Str("Hi"), Str("world")]

    # Letters and digits are not delimiters.
    assert list(sz.utf8_split_delimiters("abc123")) == [Str("abc123")]


# endregion Unit

# region Corner cases


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_delimiters_safety(seed_value: int):
    """Adversarial-byte safety: no crash, and the kept segment bytes, with delimiters dropped, never
    exceed the input length nor escape its bounds."""
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        segments = list(sz.utf8_split_delimiters(raw))
        assert sum(len(bytes(segment)) for segment in segments) <= len(raw)


# endregion Corner cases

# region Backend differential

DELIMITER_REALISTIC_TEXTS = [
    "Hi, world",
    "a.b,c",
    "日本。語",
    "a—b",
    "abc123",
]


@pytest.mark.parametrize(
    "text", DELIMITER_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_delimiters_backend_differential(text):
    """utf8_split_delimiters must split identically across every SIMD backend on realistic, malformed, and
    lane-straddling inputs, compared as raw segment bytes so malformed UTF-8 sweeps without raising."""
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_split_delimiters(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")


# endregion Backend differential
