#!/usr/bin/env python3
"""UTF-8 delimiter splitting: sz.utf8_split_newlines and sz.utf8_split_whitespaces, plus malformed-byte
safety.

Mirrors the C++ test/utf8_tokens.cpp translation unit.

Covers: utf8_split_newlines across ASCII, CRLF, and the Unicode line terminators NEL, line separator,
and paragraph separator, plus vertical tab and form feed; utf8_split_whitespaces across ASCII and the
Unicode whitespace code points NBSP, ideographic space, and en/em space; the with_separators lossless
reconstruction and skip_empty pruning on both splitters; the Str method forms; and malformed-byte
safety across both splitters.
Compares against: CPython str.splitlines() and str.split() under skip_empty, and cross-backend
self-consistency across every capability_sweep() backend.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/utf8_tokens.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/utf8_tokens.py -q
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


def test_utf8_lines():
    """Splits on every Unicode line terminator, including CRLF as a single delimiter, agreeing with
    `str.splitlines()` except for its empty-string and trailing-newline results.
    `with_separators=True` reconstructs the input losslessly, and `skip_empty=True` drops every empty
    segment including middle ones, unlike Python's `splitlines()`."""
    # Basic newline characters
    assert list(sz.utf8_split_newlines("a\nb\nc")) == [Str("a"), Str("b"), Str("c")]
    assert list(sz.utf8_split_newlines("a\rb\rc")) == [Str("a"), Str("b"), Str("c")]
    assert list(sz.utf8_split_newlines("a\r\nb\r\nc")) == [Str("a"), Str("b"), Str("c")]

    # CRLF treated as single delimiter
    result = list(sz.utf8_split_newlines("line1\r\nline2"))
    assert len(result) == 2
    assert result[0] == Str("line1")
    assert result[1] == Str("line2")

    # Empty string: yields one empty Str; Python's splitlines() returns [] instead.
    assert list(sz.utf8_split_newlines("")) == [Str("")]

    # No newlines
    assert list(sz.utf8_split_newlines("hello")) == [Str("hello")]

    # Consecutive newlines create empty lines
    result = list(sz.utf8_split_newlines("a\n\nb"))
    assert len(result) == 3
    assert result[1] == Str("")  # Empty line between

    # Trailing newline: includes trailing empty; Python omits it.
    result = list(sz.utf8_split_newlines("a\nb\n"))
    assert len(result) == 3
    assert result[2] == Str("")

    # There is no `keepends` parameter; `with_separators=True` is the lossless alternative, interleaving
    # the newline runs as their own segments so concatenation reproduces the input.
    result = list(sz.utf8_split_newlines("a\nb\nc", with_separators=True))
    assert "".join(str(s) for s in result) == "a\nb\nc"
    assert result == [Str("a"), Str("\n"), Str("b"), Str("\n"), Str("c")]

    # CRLF stays a single separator segment under with_separators.
    result = list(sz.utf8_split_newlines("a\r\nb", with_separators=True))
    assert result == [Str("a"), Str("\r\n"), Str("b")]

    # Unicode newlines: vertical tab, form feed
    assert len(list(sz.utf8_split_newlines("a\vb\fc"))) == 3

    # Unicode newlines: NEL (U+0085), Line Separator (U+2028), Paragraph Separator (U+2029)
    nel = "\u0085"
    line_sep = "\u2028"
    para_sep = "\u2029"
    assert len(list(sz.utf8_split_newlines(f"a{nel}b"))) == 2
    assert len(list(sz.utf8_split_newlines(f"a{line_sep}b"))) == 2
    assert len(list(sz.utf8_split_newlines(f"a{para_sep}b"))) == 2

    # Method form on Str
    result = list(sz.Str("a\nb").utf8_split_newlines())
    assert result == [Str("a"), Str("b")]

    # skip_empty=True skips every empty segment, leading, middle, and trailing.
    # Python's splitlines() keeps middle empty lines.
    assert list(sz.utf8_split_newlines("", skip_empty=True)) == []
    assert list(sz.utf8_split_newlines("a\nb\n", skip_empty=True)) == [Str("a"), Str("b")]
    assert list(sz.utf8_split_newlines("a\n\nb", skip_empty=True)) == [Str("a"), Str("b")]  # skips middle empty
    assert list(sz.utf8_split_newlines("\na\n", skip_empty=True)) == [Str("a")]  # skips leading & trailing


def test_utf8_tokens():
    """Splits on ASCII and Unicode whitespace, including NBSP, ideographic space, and en/em space.
    Unlike `str.split()`, consecutive delimiters and an empty or whitespace-only input yield explicit
    empty segments instead of being collapsed away."""
    # Basic whitespace: space, tab, newline
    result = list(sz.utf8_split_whitespaces("hello world"))
    assert result == [Str("hello"), Str("world")]

    result = list(sz.utf8_split_whitespaces("a\tb\nc"))
    assert result == [Str("a"), Str("b"), Str("c")]

    # Empty string: yields one empty Str; Python's split() returns [] instead.
    assert list(sz.utf8_split_whitespaces("")) == [Str("")]

    # String with only whitespace: yields empty strings between delimiters; Python's split() returns
    # [] for whitespace-only strings.
    result = list(sz.utf8_split_whitespaces("   "))
    assert all(str(s) == "" for s in result)

    # Consecutive whitespace creates empty strings between each delimiter; Python's split() treats
    # consecutive whitespace as a single delimiter.
    result = list(sz.utf8_split_whitespaces("a   b"))
    assert len(result) == 4  # a, '', '', b
    assert str(result[0]) == "a"
    assert str(result[-1]) == "b"

    # Leading/trailing whitespace creates empty strings
    result = list(sz.utf8_split_whitespaces("  hello  "))
    assert "hello" in [str(s) for s in result]

    # No whitespace
    assert list(sz.utf8_split_whitespaces("hello")) == [Str("hello")]

    # Unicode whitespace: NO-BREAK SPACE (U+00A0)
    nbsp = "\u00a0"
    result = list(sz.utf8_split_whitespaces(f"a{nbsp}b"))
    assert len(result) == 2
    assert str(result[0]) == "a"
    assert str(result[1]) == "b"

    # Unicode whitespace: IDEOGRAPHIC SPACE (U+3000)
    ideo_space = "\u3000"
    result = list(sz.utf8_split_whitespaces(f"日本{ideo_space}語"))
    assert len(result) == 2
    assert str(result[0]) == "日本"
    assert str(result[1]) == "語"

    # Unicode whitespace: EN SPACE (U+2002), EM SPACE (U+2003)
    en_space = "\u2002"
    em_space = "\u2003"
    result = list(sz.utf8_split_whitespaces(f"a{en_space}b{em_space}c"))
    assert len(result) == 3
    assert str(result[0]) == "a"
    assert str(result[1]) == "b"
    assert str(result[2]) == "c"

    # Method form on Str
    result = list(sz.Str("hello world").utf8_split_whitespaces())
    assert len(result) == 2
    assert str(result[0]) == "hello"
    assert str(result[1]) == "world"


# endregion Unit


# region Corner cases


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_tokens_safety(seed_value: int):
    """Adversarial-byte safety for the delimiter splitters. Unlike the boundary segmenters these drop their
    separators, so they do not tile, and the invariant is: no crash, and the kept segment bytes never exceed
    the input length nor escape its bounds."""
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        for splitter in (sz.utf8_split_whitespaces, sz.utf8_split_newlines):
            segments = list(splitter(raw))
            assert sum(len(bytes(segment)) for segment in segments) <= len(raw)


# endregion Corner cases


# region Conformance


@pytest.mark.parametrize(
    "text",
    ["hello\nworld", "a\r\nb\nc", "no newline", "trailing\n"],
)
def test_utf8_lines_matches_python(text):
    """`utf8_split_newlines(skip_empty=True)` matches `str.splitlines()` exactly on ASCII, CRLF,
    no-newline, and trailing-newline inputs, none of which have middle empty lines."""
    py_result = text.splitlines()
    sz_result = [str(s) for s in sz.utf8_split_newlines(text, skip_empty=True)]
    assert sz_result == py_result


@pytest.mark.parametrize(
    "text",
    ["hello world", "  spaced  ", "a\tb\nc", "no-spaces", "   ", ""],
)
def test_utf8_tokens_matches_python(text):
    """`utf8_split_whitespaces(skip_empty=True)` matches `str.split()` exactly across spaced, tabbed,
    whitespace-only, and empty inputs."""
    py_result = text.split()
    sz_result = [str(s) for s in sz.utf8_split_whitespaces(text, skip_empty=True)]
    assert sz_result == py_result


# endregion Conformance


# region Backend differential


TOKEN_REALISTIC_TEXTS = [
    "hello world",
    "a\tb\nc",
    "日本 語",
    "  hello  ",
    "привет мир",
]


@pytest.mark.parametrize(
    "text", TOKEN_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_tokens_backend_differential(text):
    """utf8_split_whitespaces must split identically across every SIMD backend on realistic, malformed, and
    lane-straddling inputs, compared as raw segment bytes so malformed UTF-8 sweeps without raising."""
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_split_whitespaces(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")


# endregion Backend differential
