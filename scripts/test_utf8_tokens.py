#!/usr/bin/env python3
"""UTF-8 delimiter-splitter tests: utf8_lines (newline boundaries) and utf8_tokens (whitespace), plus
malformed-byte safety. These splitters drop their separators, so they are checked for no-crash and
in-bounds segments rather than the exact tiling required of the boundary segmenters.

Mirrors the C++ scripts/test_utf8_tokens.cpp translation unit.
"""

from random import Random

import pytest

import stringzilla as sz
from stringzilla import Str

from test_helpers import SEED_VALUES
from test_utf8_helpers import adversarial_utf8_inputs


def test_utf8_lines():
    """Test UTF-8 line splitting iterator."""
    # Basic newline characters
    assert list(sz.utf8_lines("a\nb\nc")) == [Str("a"), Str("b"), Str("c")]
    assert list(sz.utf8_lines("a\rb\rc")) == [Str("a"), Str("b"), Str("c")]
    assert list(sz.utf8_lines("a\r\nb\r\nc")) == [Str("a"), Str("b"), Str("c")]

    # CRLF treated as single delimiter
    result = list(sz.utf8_lines("line1\r\nline2"))
    assert len(result) == 2
    assert result[0] == Str("line1")
    assert result[1] == Str("line2")

    # Empty string: yields one empty Str (differs from Python's splitlines which returns [])
    assert list(sz.utf8_lines("")) == [Str("")]

    # No newlines
    assert list(sz.utf8_lines("hello")) == [Str("hello")]

    # Consecutive newlines create empty lines
    result = list(sz.utf8_lines("a\n\nb"))
    assert len(result) == 3
    assert result[1] == Str("")  # Empty line between

    # Trailing newline: includes trailing empty (differs from Python which omits it)
    result = list(sz.utf8_lines("a\nb\n"))
    assert len(result) == 3
    assert result[2] == Str("")

    # keepends=True
    result = list(sz.utf8_lines("a\nb\nc", keepends=True))
    assert result[0] == Str("a\n")
    assert result[1] == Str("b\n")
    assert result[2] == Str("c")  # Last line has no newline

    # keepends=True with CRLF
    result = list(sz.utf8_lines("a\r\nb", keepends=True))
    assert result[0] == Str("a\r\n")
    assert result[1] == Str("b")

    # Unicode newlines: vertical tab, form feed
    assert len(list(sz.utf8_lines("a\vb\fc"))) == 3

    # Unicode newlines: NEL (U+0085), Line Separator (U+2028), Paragraph Separator (U+2029)
    nel = "\u0085"
    line_sep = "\u2028"
    para_sep = "\u2029"
    assert len(list(sz.utf8_lines(f"a{nel}b"))) == 2
    assert len(list(sz.utf8_lines(f"a{line_sep}b"))) == 2
    assert len(list(sz.utf8_lines(f"a{para_sep}b"))) == 2

    # Method form on Str
    result = list(sz.Str("a\nb").utf8_lines())
    assert result == [Str("a"), Str("b")]

    # skip_empty=True skips ALL empty segments (leading, middle, trailing)
    # This differs from Python's splitlines() which keeps middle empty lines
    assert list(sz.utf8_lines("", skip_empty=True)) == []
    assert list(sz.utf8_lines("a\nb\n", skip_empty=True)) == [Str("a"), Str("b")]
    assert list(sz.utf8_lines("a\n\nb", skip_empty=True)) == [Str("a"), Str("b")]  # skips middle empty
    assert list(sz.utf8_lines("\na\n", skip_empty=True)) == [Str("a")]  # skips leading & trailing


@pytest.mark.parametrize(
    "text",
    ["hello\nworld", "a\r\nb\nc", "no newline", "trailing\n"],
)
def test_utf8_lines_matches_python(text):
    """Verify skip_empty=True matches Python's splitlines() for cases without middle empty lines."""
    py_result = text.splitlines()
    sz_result = [str(s) for s in sz.utf8_lines(text, skip_empty=True)]
    assert sz_result == py_result


def test_utf8_tokens():
    """Test UTF-8 whitespace splitting iterator."""
    # Basic whitespace: space, tab, newline
    result = list(sz.utf8_tokens("hello world"))
    assert result == [Str("hello"), Str("world")]

    result = list(sz.utf8_tokens("a\tb\nc"))
    assert result == [Str("a"), Str("b"), Str("c")]

    # Empty string: yields one empty Str (differs from Python's split which returns [])
    assert list(sz.utf8_tokens("")) == [Str("")]

    # String with only whitespace: yields empty strings between delimiters
    # (differs from Python's split() which returns [] for whitespace-only strings)
    result = list(sz.utf8_tokens("   "))
    assert all(str(s) == "" for s in result)

    # Consecutive whitespace creates empty strings between each delimiter
    # (differs from Python's split() which treats consecutive whitespace as single delimiter)
    result = list(sz.utf8_tokens("a   b"))
    assert len(result) == 4  # a, '', '', b
    assert str(result[0]) == "a"
    assert str(result[-1]) == "b"

    # Leading/trailing whitespace creates empty strings
    result = list(sz.utf8_tokens("  hello  "))
    assert "hello" in [str(s) for s in result]

    # No whitespace
    assert list(sz.utf8_tokens("hello")) == [Str("hello")]

    # Unicode whitespace: NO-BREAK SPACE (U+00A0)
    nbsp = "\u00a0"
    result = list(sz.utf8_tokens(f"a{nbsp}b"))
    assert len(result) == 2
    assert str(result[0]) == "a"
    assert str(result[1]) == "b"

    # Unicode whitespace: IDEOGRAPHIC SPACE (U+3000)
    ideo_space = "\u3000"
    result = list(sz.utf8_tokens(f"日本{ideo_space}語"))
    assert len(result) == 2
    assert str(result[0]) == "日本"
    assert str(result[1]) == "語"

    # Unicode whitespace: EN SPACE (U+2002), EM SPACE (U+2003)
    en_space = "\u2002"
    em_space = "\u2003"
    result = list(sz.utf8_tokens(f"a{en_space}b{em_space}c"))
    assert len(result) == 3
    assert str(result[0]) == "a"
    assert str(result[1]) == "b"
    assert str(result[2]) == "c"

    # Method form on Str
    result = list(sz.Str("hello world").utf8_tokens())
    assert len(result) == 2
    assert str(result[0]) == "hello"
    assert str(result[1]) == "world"


@pytest.mark.parametrize(
    "text",
    ["hello world", "  spaced  ", "a\tb\nc", "no-spaces", "   ", ""],
)
def test_utf8_tokens_matches_python(text):
    """Verify skip_empty=True matches Python's split() behavior."""
    py_result = text.split()
    sz_result = [str(s) for s in sz.utf8_tokens(text, skip_empty=True)]
    assert sz_result == py_result


#  region Synthetic corner cases (malformed safety)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_tokens_safety(seed_value: int):
    """Adversarial-byte safety for the delimiter splitters. Unlike the boundary segmenters these drop their
    separators (so they do not tile), so the invariant is: no crash, and the kept segment bytes never exceed
    the input length nor escape its bounds."""
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        for splitter in (sz.utf8_tokens, sz.utf8_lines):
            segments = list(splitter(raw))
            assert sum(len(bytes(segment)) for segment in segments) <= len(raw)


#  endregion Synthetic corner cases (malformed safety)
