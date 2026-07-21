#!/usr/bin/env python3
"""Scheme-C mode tests for the UTF-8 splitters: the bare name yields the separators, ``split_*`` yields the
content between them, and ``with_separators=True`` interleaves both losslessly.

Mirrors the Rust ``utf8_split_modes`` test and the C++ ``test/utf8_tokens.cpp`` separator/round-trip coverage.

Covers: the separator-versus-split accessor pairing for whitespace, newlines, and delimiters, the
with_separators round-trip reconstruction for delimiters and newlines, Str method parity against the free
functions, full UAX-29 tiling by utf8_wordbreaks, empty-input and skip_empty behavior, and the regression
distinguishing zero-width format characters from hair-space whitespace.
Compares against: round-trip self-consistency, concatenated segments reproduce the input, and cross-backend
self-consistency across every capability_sweep() backend.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/utf8_split_modes.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/utf8_split_modes.py -q
"""

import pytest

import stringzilla as sz
from stringzilla import Str

from test.sz_helpers import (
    assert_backends_agree,
    malformed_utf8_corpus,
    run_across_backends,
    vector_width_bracketing_strings,
)


# region Unit


def test_between_vs_separators():
    """The bare name yields the separators; ``split_`` yields the content between."""
    text = "a b  c"
    assert list(sz.utf8_split_whitespaces(text)) == [Str("a"), Str("b"), Str(""), Str("c")]
    assert list(sz.utf8_whitespaces(text)) == [Str(" "), Str(" "), Str(" ")]
    # Newlines and delimiters expose the same pair of accessors.
    assert list(sz.utf8_split_newlines("a\nb")) == [Str("a"), Str("b")]
    assert list(sz.utf8_newlines("a\nb")) == [Str("\n")]
    assert list(sz.utf8_delimiters("a,b")) == [Str(",")]


def test_with_separators_round_trip():
    """``with_separators=True`` interleaves segments and separators; concatenation reproduces the input."""
    for text in ["a b  c", "  x  ", "", "abc", "a\r\nb", ",,a,,"]:
        joined = "".join(str(s) for s in sz.utf8_split_delimiters(text, with_separators=True))
        assert joined == text
        joined_nl = "".join(str(s) for s in sz.utf8_split_newlines(text, with_separators=True))
        assert joined_nl == text


def test_member_form_matches_free_function():
    """``Str(text).utf8_*`` agrees with ``sz.utf8_*(text)``."""
    text = "one two  three"
    assert list(sz.Str(text).utf8_split_whitespaces(skip_empty=True)) == [Str("one"), Str("two"), Str("three")]
    assert list(sz.Str(text).utf8_whitespaces()) == list(sz.utf8_whitespaces(text))


def test_wordbreaks_tiles():
    """``utf8_wordbreaks`` tiles into all UAX-29 segments, both words and the separators between them."""
    segments = list(sz.utf8_wordbreaks("Hello, world!"))
    assert "".join(str(s) for s in segments) == "Hello, world!"
    assert len(segments) == 5


# endregion Unit


# region Corner cases


def test_empty_and_skip_empty():
    """Empty input yields one empty segment; ``skip_empty`` drops zero-length spans."""
    assert list(sz.utf8_split_whitespaces("")) == [Str("")]
    assert list(sz.utf8_split_whitespaces("a  b")) == [Str("a"), Str(""), Str("b")]
    assert list(sz.utf8_split_whitespaces("a  b", skip_empty=True)) == [Str("a"), Str("b")]


def test_zero_width_format_chars_are_not_whitespace():
    """Regression: U+200B/200C/200D (ZWSP/ZWNJ/ZWJ) are Format chars with White_Space=No and must not be
    whitespace separators, while U+200A HAIR SPACE, the last codepoint of the U+2000 space block, must split.
    The kernel overran the E2 80 block to 0x8D, splitting on the zero-width joiners and shattering ZWJ emoji."""
    assert list(sz.utf8_split_whitespaces("a b")) == [Str("a"), Str("b")]  # U+200A HAIR SPACE → splits
    for cp in ("​", "‌", "‍"):  # ZWSP / ZWNJ / ZWJ → not whitespace, no split
        assert list(sz.utf8_split_whitespaces(f"a{cp}b")) == [Str(f"a{cp}b")]
    # A ZWJ emoji sequence (man-ZWJ-woman) stays one segment instead of being split on the joiner.
    assert list(sz.utf8_split_whitespaces("\U0001f468‍\U0001f469")) == [Str("\U0001f468‍\U0001f469")]


# endregion Corner cases


# region Backend differential


SPLIT_MODE_REALISTIC_TEXTS = [
    "a b  c",
    "  x  ",
    "one two  three",
    "日本 語",
    "привет мир",
]


@pytest.mark.parametrize(
    "text", SPLIT_MODE_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_split_modes_backend_differential(text):
    """utf8_split_whitespaces must split identically across every SIMD backend on realistic, malformed, and
    lane-straddling inputs, compared as raw segment bytes so malformed UTF-8 sweeps without raising."""
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_split_whitespaces(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")


# endregion Backend differential
