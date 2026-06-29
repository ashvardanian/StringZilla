#!/usr/bin/env python3
"""Scheme-C mode tests for the UTF-8 splitters: the bare name yields the separators, ``split_*`` yields the
content between them, and ``with_separators=True`` interleaves both losslessly. Mirrors the Rust
``utf8_split_modes`` test and the C++ ``test/utf8_tokens.cpp`` separator/round-trip coverage.
"""

import stringzilla as sz
from stringzilla import Str


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


def test_empty_and_skip_empty():
    """Empty input yields one empty segment; ``skip_empty`` drops zero-length spans."""
    assert list(sz.utf8_split_whitespaces("")) == [Str("")]
    assert list(sz.utf8_split_whitespaces("a  b")) == [Str("a"), Str(""), Str("b")]
    assert list(sz.utf8_split_whitespaces("a  b", skip_empty=True)) == [Str("a"), Str("b")]


def test_member_form_matches_free_function():
    """``Str(text).utf8_*`` agrees with ``sz.utf8_*(text)``."""
    text = "one two  three"
    assert list(sz.Str(text).utf8_split_whitespaces(skip_empty=True)) == [Str("one"), Str("two"), Str("three")]
    assert list(sz.Str(text).utf8_whitespaces()) == list(sz.utf8_whitespaces(text))


def test_wordbreaks_tiles():
    """``utf8_wordbreaks`` tiles into all UAX-29 segments (words and the separators between them)."""
    segments = list(sz.utf8_wordbreaks("Hello, world!"))
    assert "".join(str(s) for s in segments) == "Hello, world!"
    assert len(segments) == 5
