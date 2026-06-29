#!/usr/bin/env python3
"""UTF-8 delimiter-splitter tests: utf8_delimiters splits on any Unicode punctuation, symbol, or
separator/whitespace codepoint (the superset of utf8_tokens), plus malformed-byte safety.

Mirrors the C++ test/utf8_tokens.cpp delimiter coverage.
"""

from random import Random

import pytest

import stringzilla as sz
from stringzilla import Str

from test.helpers import SEED_VALUES
from test.utf8_helpers import adversarial_utf8_inputs


def test_utf8_delimiters():
    """Test UTF-8 delimiter splitting iterator."""
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

    # Letters and digits are NOT delimiters.
    assert list(sz.utf8_split_delimiters("abc123")) == [Str("abc123")]


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_delimiters_safety(seed_value: int):
    """Adversarial-byte safety: no crash, and the kept (delimiter-dropped) segment bytes never exceed the
    input length nor escape its bounds."""
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        segments = list(sz.utf8_split_delimiters(raw))
        assert sum(len(bytes(segment)) for segment in segments) <= len(raw)
