from typing import Union, Optional
from random import choice, randint
from string import ascii_lowercase

import pytest

import stringzilla as sz
from stringzilla import Str, Strs


def test_unit_construct():
    native = "aaaaa"
    big = Str(native)
    assert len(big) == len(native)


def test_unit_indexing():
    native = "abcdef"
    big = Str(native)
    for i in range(len(native)):
        assert big[i] == native[i]


def test_unit_count():
    native = "aaaaa"
    big = Str(native)
    assert big.count("a") == 5
    assert big.count("aa") == 2
    assert big.count("aa", allowoverlap=True) == 4


def test_unit_contains():
    big = Str("abcdef")
    assert "a" in big
    assert "ab" in big
    assert "xxx" not in big


def test_unit_rich_comparisons():
    assert Str("aa") == "aa"
    assert Str("aa") < "b"
    assert Str("abb")[1:] == "bb"


def test_unit_buffer_protocol():
    import numpy as np

    my_str = Str("hello")
    arr = np.array(my_str)
    assert arr.dtype == np.dtype("c")
    assert arr.shape == (len("hello"),)
    assert "".join([c.decode("utf-8") for c in arr.tolist()]) == "hello"


def test_unit_split():
    native = "token1\ntoken2\ntoken3"
    big = Str(native)
    assert native.splitlines() == list(big.splitlines())
    assert native.splitlines(True) == list(big.splitlines(keeplinebreaks=True))
    assert native.split("token3") == list(big.split("token3"))

    words = sz.split(big, "\n")
    assert len(words) == 3
    assert str(words[0]) == "token1"
    assert str(words[2]) == "token3"

    parts = sz.split(big, "\n", keepseparator=True)
    assert len(parts) == 3
    assert str(parts[0]) == "token1\n"
    assert str(parts[2]) == "token3"


def test_unit_sequence():
    native = "p3\np2\np1"
    big = Str(native)

    lines = big.splitlines()
    assert [2, 1, 0] == list(lines.order())

    lines.sort()
    assert [0, 1, 2] == list(lines.order())
    assert ["p1", "p2", "p3"] == list(lines)

    # Reverse order
    assert [2, 1, 0] == list(lines.order(reverse=True))
    lines.sort(reverse=True)
    assert ["p3", "p2", "p1"] == list(lines)


def test_unit_globals():
    """Validates that the previously unit-tested member methods are also visible as global functions."""

    assert sz.find("abcdef", "bcdef") == 1
    assert sz.find("abcdef", "x") == -1

    assert sz.count("abcdef", "x") == 0
    assert sz.count("aaaaa", "a") == 5
    assert sz.count("aaaaa", "aa") == 2
    assert sz.count("aaaaa", "aa", allowoverlap=True) == 4

    assert sz.levenstein("aaa", "aaa") == 0
    assert sz.levenstein("aaa", "bbb") == 3
    assert sz.levenstein("abababab", "aaaaaaaa") == 4
    assert sz.levenstein("abababab", "aaaaaaaa", 2) == 2
    assert sz.levenstein("abababab", "aaaaaaaa", bound=2) == 2
