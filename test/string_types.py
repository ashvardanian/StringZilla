#!/usr/bin/env python3
"""String and Strings containers: construction, indexing, slicing, comparisons, and the buffer protocol.

Mirrors the C++ test/string.cpp translation unit.

Covers: String/Strings construction from native str, bytes, and bytearray, indexing, slicing, and rich
comparisons, the read-only buffer protocol, split, rsplit, strip, and splitlines in both eager and
lazy iterator form, translate, fill_random, and UTF-8 decoding, degenerate and out-of-ssize_t corner
cases, the pyarrow/numpy interop surface for String and Strings, and the collection verbs Strings.translate,
Strings.escape, Strings.unescape, Strings.to_json and Strings.utf8_count, String.join, with the sz.to_json and
sz.to_csv column dumpers.
Compares against: CPython str and bytes methods, the json, csv, html and urllib.parse modules,
pandas when installed, and cross-backend agreement via capability_sweep().

Run:
    uv pip install numpy pyarrow pytest pytest-repeat
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/string_types.py -q
    STRINGZILLA_SEED=42 STRINGZILLA_SCALE=10 uv run --no-project python -m pytest test/string_types.py -q

File: test/string_types.py
Author: Ash Vardanian
Date: September 16, 2023
"""

import csv
import html
import io
import json
import math
import os
import sys
import tempfile
import urllib.parse
from random import Random
from string import ascii_lowercase
from typing import Sequence

import pytest
from base import (
    DEGENERATE_HAYSTACKS,
    VECTOR_WIDTH_LENGTHS,
    StreamKey,
    assert_backends_agree,
    differential_bodies,
    get_random_string,
    numpy_available,
    pyarrow_available,
    scale_iterations,
    unaligned_views,
)

import stringzilla as sz
from stringzilla import String, Strings

# NumPy and PyArrow are optional; the naked `except` also catches PyPy's non-ImportError on import.
try:
    import numpy as np
except:  # noqa: E722
    pass

try:
    import pyarrow as pa
except:  # noqa: E722
    pass


# region Unit

SPLIT_PARITY_MAXSPLITS = [-1, 0, 1, 2, 99]
"""`maxsplit` values spanning "no limit" (-1), "no splits" (0), a couple of partial splits, and a limit
far beyond the number of separators, reused to check lazy/eager split parity."""


SPLITLINES_CASES = [
    "",
    "a",
    "a\n",
    "\n",
    "\n\n",
    "a\nb",
    "a\n\nb",
    "a\nb\n",
    "a\r\nb",
    "a\r\r\nb",
    "a\r",
    "Åsa\nBob",
    "a\x85b",
    "a b c",
    "a\x1cb\x1dc\x1ed",
    "é—…z",
    " a b",
    "a\x0bb\x0cc",
    "\r\na\r\n",
]


def test_library_properties():
    """`sz.__version__` is a three-part semantic version."""
    assert len(sz.__version__.split(".")) == 3, "Semantic versioning must be preserved"


@pytest.mark.parametrize("native_type", [str, bytes, bytearray])
def test_unit_construct(native_type):
    """`String` construction accepts native `str`, `bytes`, and `bytearray` input and reports the same
    length as the source."""
    native = "aaaaa"
    if native_type is bytes or native_type is bytearray:
        native = native_type(native, "utf-8")
    big = String(native)
    assert len(big) == len(native)


def test_string_repr():
    """`repr(String)` renders as `sz.String(<repr of the native string>)`, and `str(String)` round-trips to
    the original native string."""
    native = "abcdef"
    big = String(native)
    assert repr(big) == f"sz.String({native!r})"
    assert str(big) == native


def test_unit_indexing():
    """Indexing a `String` by position returns the same character as indexing the native string, for
    every position."""
    native = "abcdef"
    big = String(native)
    for i in range(len(native)):
        assert big[i] == native[i]


def test_unit_string_rich_comparisons():
    """`String` supports every rich comparison, equality, ordering, and slicing-then-comparison, against
    both native strings and other `String` instances, matching CPython `str` ordering."""
    # Equality
    assert String("aa") == "aa"
    assert String("a") != "b"
    assert String("abc") == String("abc")
    assert String("abc") != String("abd")

    # Less than and less than or equal to
    assert String("aa") < "b"
    assert String("ab") <= "ab"
    assert String("a") < String("b")
    assert String("abc") <= String("abcd")

    # Greater than and greater than or equal to
    assert String("b") > "aa"
    assert String("ab") >= "ab"
    assert String("b") > String("a")
    assert String("abcd") >= String("abc")

    # Slicing and comparisons
    s2 = String("abb")
    assert s2[1:] == "bb"
    assert s2[:-1] == "ab"
    assert s2[-1:] == "b"
    assert s2[1:] != "abb"
    assert s2[:-2] == "a"
    assert s2[-2:] == "bb"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_unit_buffer_protocol():
    """The buffer protocol round-trips `String` through a NumPy array via `memoryview`, including
    multi-dimensional contiguous arrays, and raises on non-contiguous or non-buffer input."""

    # Convert from StringZilla string to NumPy array through buffer protocol
    sz_string = String("hello")
    np_array = np.array(sz_string)
    assert np_array.dtype == np.dtype("c")
    assert np_array.shape == (len("hello"),)
    assert "".join([c.decode("utf-8") for c in np_array.tolist()]) == "hello"

    # Convert from NumPy array to StringZilla string through buffer protocol
    np_array = np.arange(ord("a"), ord("z"), dtype=np.uint8)
    sz_string = sz.String(memoryview(np_array))
    assert len(np_array) == len(sz_string)

    # Make sure multi-dimensional contiguous arrays are supported for image processing
    np_array = np.arange(ord("a"), ord("a") + 8, dtype=np.uint8).reshape((2, 2, 2))
    sz_string = sz.String(memoryview(np_array))
    assert np_array.size == len(sz_string)

    # Expect errors if the data is not contiguous
    np_array = np.arange(ord("a"), ord("z"), dtype=np.uint8)[::2]
    with pytest.raises(ValueError):
        sz.String(memoryview(np_array))

    # Expect errors if the data is not passed through a `memoryview`
    with pytest.raises(TypeError):
        sz.String(np.array())
    # Expect errors if the data is not string-like
    with pytest.raises(TypeError):
        sz.String({})


def test_string_write_to():
    """`String.write_to` writes the full native content to a file path, byte for byte."""
    native = "line1\nline2\nline3"
    big = String(native)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        temp_filename = tmpfile.name  # Store the name for later use

    try:
        big.write_to(temp_filename)
        with open(temp_filename, "r") as file:
            content = file.read()
            assert content == native, "The content of the file does not match the expected output"
    finally:
        os.remove(temp_filename)


def test_unit_split():
    """`sz.split`/`sz.rsplit` and their byteset and `keepseparator` variants match CPython
    `str.split`/`str.rsplit` across every `maxsplit` value; unlike CPython, an empty separator leaves the
    text whole instead of raising."""
    native = "line1\nline2\nline3"
    big = String(native)

    # Splitting using a string
    lines = sz.split(big, "\n")
    assert lines == ["line1", "line2", "line3"]

    lines = sz.rsplit(big, "\n")
    assert lines == ["line1", "line2", "line3"]

    lines = sz.split(big, "\n", keepseparator=True)
    assert lines == ["line1\n", "line2\n", "line3"]

    letters = sz.split("a b c d")
    assert letters == ["a", "b", "c", "d"]

    # Splitting using character sets
    letters = sz.split_byteset("a b_c d", " _")
    assert letters == ["a", "b", "c", "d"]

    letters = sz.rsplit_byteset("a b_c d", " _")
    assert letters == ["a", "b", "c", "d"]

    # Check for equivalence with native Python strings for newline separators
    assert native.splitlines() == list(big.splitlines())
    assert native.splitlines(True) == list(big.splitlines(keeplinebreaks=True))

    # Check for equivalence with native Python strings, including SWAR boundary conditions
    assert native.split("l") == list(big.split("l"))
    assert native.split("li") == list(big.split("li"))
    assert native.split("lin") == list(big.split("lin"))
    assert native.split("line") == list(big.split("line"))
    assert native.split("line1") == list(big.split("line1"))
    assert native.split("line3") == list(big.split("line3"))
    assert native.split("\n", maxsplit=0) == list(big.split("\n", maxsplit=0))
    assert native.split("\n", maxsplit=1) == list(big.split("\n", maxsplit=1))
    assert native.split("\n", maxsplit=2) == list(big.split("\n", maxsplit=2))
    assert native.split("\n", maxsplit=3) == list(big.split("\n", maxsplit=3))
    assert native.split("\n", maxsplit=4) == list(big.split("\n", maxsplit=4))

    # Check for equivalence with native Python strings in reverse order, including boundary conditions
    assert native.rsplit("line1") == list(big.rsplit("line1"))
    assert native.rsplit("line3") == list(big.rsplit("line3"))
    assert native.rsplit("\n", maxsplit=0) == list(big.rsplit("\n", maxsplit=0))
    assert native.rsplit("\n", maxsplit=1) == list(big.rsplit("\n", maxsplit=1))
    assert native.rsplit("\n", maxsplit=2) == list(big.rsplit("\n", maxsplit=2))
    assert native.rsplit("\n", maxsplit=3) == list(big.rsplit("\n", maxsplit=3))
    assert native.rsplit("\n", maxsplit=4) == list(big.rsplit("\n", maxsplit=4))

    # An empty separator never matches, whatever the `maxsplit`
    for split in (sz.split, sz.rsplit, sz.split_byteset, sz.rsplit_byteset):
        assert split(big, "") == [native]
        assert split(big, "", maxsplit=-1) == [native]
        assert split("", "") == [""]


def test_unit_split_skip_empty():
    """`skip_empty=True` drops zero-length segments from byteset/substring splits; the default keeps them."""

    def strings(x):
        return [str(s) for s in x]

    # Substring separator: leading/middle/trailing empties are dropped.
    assert strings(String("a,,b,").split(",", skip_empty=True)) == ["a", "b"]
    assert strings(String(",a,,b,").split(",", skip_empty=True)) == ["a", "b"]
    assert strings(String("a,,b,").split(",")) == ["a", "", "b", ""]  # default: keep empties

    # Reverse eager split reports source order; it reverses internally.
    assert strings(String("a,,b,").rsplit(",", skip_empty=True)) == ["a", "b"]
    assert strings(String("a,,b,").rsplit(",")) == ["a", "", "b", ""]

    # Byteset variants.
    assert strings(String("a,;b;").split_byteset(",;", skip_empty=True)) == ["a", "b"]
    assert strings(String("a,;b;").rsplit_byteset(",;", skip_empty=True)) == ["a", "b"]
    assert strings(String("a,;b;").split_byteset(",;")) == ["a", "", "b", ""]

    # Lazy iterator variants; rsplit_iter yields reverse order.
    assert strings(sz.split_iter("a,,b,", ",", skip_empty=True)) == ["a", "b"]
    assert strings(sz.rsplit_iter("a,,b,", ",", skip_empty=True)) == ["b", "a"]
    assert strings(sz.split_byteset_iter("a,;b;", ",;", skip_empty=True)) == ["a", "b"]
    assert strings(sz.rsplit_byteset_iter("a,;b;", ",;", skip_empty=True)) == ["b", "a"]

    # `skip_empty` composes with `maxsplit`: only non-empty leading segments consume the budget.
    assert strings(String("a,,b,c").split(",", maxsplit=1, skip_empty=True)) == ["a", ",b,c"]

    # An all-separator string collapses to nothing when skipping empties.
    assert strings(String(",,,").split(",", skip_empty=True)) == []
    assert strings(sz.split_iter(",,,", ",", skip_empty=True)) == []

    # A kept separator is content, so its part is never empty, eagerly or lazily.
    assert strings(String(",a,").split(",", keepseparator=True, skip_empty=True)) == [",", "a,"]
    assert strings(sz.split_iter(",a,", ",", keepseparator=True, skip_empty=True)) == [",", "a,"]


def test_unit_split_iterators():
    """`split_iter`/`rsplit_iter` and their byteset variants preserve `split`/`rsplit` semantics while
    yielding lazily; `rsplit_iter` yields elements in reverse order, unlike the eager `rsplit`."""
    native = "line1\nline2\nline3"
    big = String(native)

    # Splitting using a string
    lines = list(sz.split_iter(big, "\n"))
    assert lines == ["line1", "line2", "line3"]

    lines = list(sz.rsplit_iter(big, "\n"))
    assert lines == ["line3", "line2", "line1"]

    lines = list(sz.split_iter(big, "\n", keepseparator=True))
    assert lines == ["line1\n", "line2\n", "line3"]

    lines = list(sz.rsplit_iter(big, "\n", keepseparator=True))
    assert lines == ["\nline3", "\nline2", "line1"]

    letters = list(sz.split_iter("a b c d"))
    assert letters == ["a", "b", "c", "d"]

    # Splitting using character sets
    letters = list(sz.split_byteset_iter("a-b_c-d", "-_"))
    assert letters == ["a", "b", "c", "d"]

    letters = list(sz.rsplit_byteset_iter("a-b_c-d", "-_"))
    assert letters == ["d", "c", "b", "a"]

    # Check for equivalence with native Python strings, including boundary conditions
    assert native.split("line1") == list(big.split_iter("line1"))
    assert native.split("line3") == list(big.split_iter("line3"))
    assert native.split("\n", maxsplit=0) == list(big.split_iter("\n", maxsplit=0))
    assert native.split("\n", maxsplit=1) == list(big.split_iter("\n", maxsplit=1))
    assert native.split("\n", maxsplit=2) == list(big.split_iter("\n", maxsplit=2))
    assert native.split("\n", maxsplit=3) == list(big.split_iter("\n", maxsplit=3))
    assert native.split("\n", maxsplit=4) == list(big.split_iter("\n", maxsplit=4))

    def rlist(seq):
        seq = list(seq)
        seq.reverse()
        return seq

    # Check for equivalence with native Python strings in reverse order, including boundary conditions
    assert native.rsplit("line1") == rlist(big.rsplit_iter("line1"))
    assert native.rsplit("line3") == rlist(big.rsplit_iter("line3"))
    assert native.rsplit("\n", maxsplit=0) == rlist(big.rsplit_iter("\n", maxsplit=0))
    assert native.rsplit("\n", maxsplit=1) == rlist(big.rsplit_iter("\n", maxsplit=1))
    assert native.rsplit("\n", maxsplit=2) == rlist(big.rsplit_iter("\n", maxsplit=2))
    assert native.rsplit("\n", maxsplit=3) == rlist(big.rsplit_iter("\n", maxsplit=3))
    assert native.rsplit("\n", maxsplit=4) == rlist(big.rsplit_iter("\n", maxsplit=4))

    # An empty separator never matches, and an exhausted iterator stays exhausted
    for split_iter in (sz.split_iter, sz.rsplit_iter, sz.split_byteset_iter, sz.rsplit_byteset_iter):
        parts = split_iter(big, "")
        assert list(parts) == [native]
        assert next(parts, None) is None


def test_unit_strip():
    """`sz.lstrip`/`sz.rstrip`/`sz.strip` and the `String` methods match CPython `str.lstrip`/`rstrip`/
    `strip`, with and without a custom character set, on `str` and `bytes` input."""
    # Test with whitespace (default behavior)
    native_whitespace = "  \t\n hello world \r\f\v  "
    big_whitespace = String(native_whitespace)

    # Test lstrip
    assert native_whitespace.lstrip() == str(sz.lstrip(big_whitespace))
    assert native_whitespace.lstrip() == str(big_whitespace.lstrip())

    # Test rstrip
    assert native_whitespace.rstrip() == str(sz.rstrip(big_whitespace))
    assert native_whitespace.rstrip() == str(big_whitespace.rstrip())

    # Test strip
    assert native_whitespace.strip() == str(sz.strip(big_whitespace))
    assert native_whitespace.strip() == str(big_whitespace.strip())

    # Test with custom character set
    native_custom = "aaabbbhello worldcccaaa"
    big_custom = String(native_custom)
    chars = "abc"

    # Test lstrip with custom chars
    assert native_custom.lstrip(chars) == str(sz.lstrip(big_custom, chars))
    assert native_custom.lstrip(chars) == str(big_custom.lstrip(chars))

    # Test rstrip with custom chars
    assert native_custom.rstrip(chars) == str(sz.rstrip(big_custom, chars))
    assert native_custom.rstrip(chars) == str(big_custom.rstrip(chars))

    # Test strip with custom chars
    assert native_custom.strip(chars) == str(sz.strip(big_custom, chars))
    assert native_custom.strip(chars) == str(big_custom.strip(chars))

    # Test edge cases
    # Empty string
    empty = ""
    big_empty = String(empty)
    assert empty.strip() == str(sz.strip(big_empty))
    assert empty.strip() == str(big_empty.strip())

    # String with only whitespace
    only_whitespace = " \t\n\r\f\v "
    big_only_whitespace = String(only_whitespace)
    assert only_whitespace.strip() == str(sz.strip(big_only_whitespace))
    assert only_whitespace.strip() == str(big_only_whitespace.strip())

    # String with only custom chars
    only_custom = "aaabbbccc"
    big_only_custom = String(only_custom)
    assert only_custom.strip("abc") == str(sz.strip(big_only_custom, "abc"))
    assert only_custom.strip("abc") == str(big_only_custom.strip("abc"))

    # String with no chars to strip
    no_strip = "hello world"
    big_no_strip = String(no_strip)
    assert no_strip.strip() == str(sz.strip(big_no_strip))
    assert no_strip.strip() == str(big_no_strip.strip())

    # Test with bytes
    native_bytes = b"  hello world  "
    big_bytes = String(native_bytes)
    assert native_bytes.strip() == bytes(sz.strip(big_bytes))
    assert native_bytes.strip() == bytes(big_bytes.strip())

    # Test asymmetric stripping
    native_asymmetric = "aaahello worldbbb"
    big_asymmetric = String(native_asymmetric)

    # Only strip 'a' from left
    assert native_asymmetric.lstrip("a") == str(sz.lstrip(big_asymmetric, "a"))
    assert native_asymmetric.lstrip("a") == str(big_asymmetric.lstrip("a"))

    # Only strip 'b' from right
    assert native_asymmetric.rstrip("b") == str(sz.rstrip(big_asymmetric, "b"))
    assert native_asymmetric.rstrip("b") == str(big_asymmetric.rstrip("b"))

    # Test with special characters
    native_special = "!!!###hello world***!!!"
    big_special = String(native_special)
    special_chars = "!#*"

    assert native_special.strip(special_chars) == str(sz.strip(big_special, special_chars))
    assert native_special.strip(special_chars) == str(big_special.strip(special_chars))

    # Test with single character
    native_single = "aaa"
    big_single = String(native_single)
    assert native_single.strip("a") == str(sz.strip(big_single, "a"))
    assert native_single.strip("a") == str(big_single.strip("a"))

    # Test with no matching characters
    native_no_match = "hello world"
    big_no_match = String(native_no_match)
    assert native_no_match.strip("xyz") == str(sz.strip(big_no_match, "xyz"))
    assert native_no_match.strip("xyz") == str(big_no_match.strip("xyz"))


def test_unit_slicing():
    """`String` slicing with positive and negative start/stop indices matches native string slicing."""
    native = "abcdef"
    big = String(native)
    assert big[1:3] == "bc"
    assert big[1:] == "bcdef"
    assert big[:3] == "abc"
    assert big[-1:] == "f"
    assert big[:-1] == "abcde"
    assert big[-3:] == "def"
    assert big[:-3] == "abc"


@pytest.mark.parametrize("native", SPLITLINES_CASES)
def test_unit_splitlines_parity(native):
    """splitlines must match CPython: CR LF, NEL, LS and PS as single breaks, no break inside a
    multi-byte character, no trailing empty line after a final break, and [] for ''."""
    assert [str(s) for s in String(native).splitlines()] == native.splitlines()
    assert [str(s) for s in String(native).splitlines(keeplinebreaks=True)] == native.splitlines(keepends=True)


@pytest.mark.parametrize("maxsplit", SPLIT_PARITY_MAXSPLITS)
def test_unit_split_iter_parity(maxsplit):
    """Lazy `split_iter` must match eager `split` for every maxsplit, including negative, a
    regression guard for the negative-maxsplit fix."""
    haystack = String("a,b,c,d,e")
    lazy = list(map(str, haystack.split_iter(",", maxsplit)))
    assert lazy == haystack.split(",", maxsplit)


@pytest.mark.parametrize("maxsplit", SPLIT_PARITY_MAXSPLITS)
def test_unit_rsplit_iter_parity(maxsplit):
    """`rsplit_iter` yields elements in reverse order relative to `rsplit`; reversing the lazy
    stream must reproduce the eager `rsplit` result for every maxsplit, including negative, a
    regression guard for the negative-maxsplit fix."""
    haystack = String("a,b,c,d,e")
    lazy_reversed = list(reversed(list(map(str, haystack.rsplit_iter(",", maxsplit)))))
    assert lazy_reversed == haystack.rsplit(",", maxsplit)


@pytest.mark.parametrize("maxsplit", SPLIT_PARITY_MAXSPLITS)
def test_unit_split_byteset_iter_parity(maxsplit):
    """Lazy `split_byteset_iter` must match eager `split_byteset` for every maxsplit, including
    negative, a regression guard for the negative-maxsplit fix."""
    haystack = String("a,b;c,d;e")
    lazy = list(map(str, haystack.split_byteset_iter(",;", maxsplit)))
    assert lazy == haystack.split_byteset(",;", maxsplit)


@pytest.mark.parametrize("maxsplit", SPLIT_PARITY_MAXSPLITS)
def test_unit_rsplit_byteset_iter_parity(maxsplit):
    """`rsplit_byteset_iter` yields elements in reverse order relative to `rsplit_byteset`;
    reversing the lazy stream must reproduce the eager result for every maxsplit, including
    negative, a regression guard for the negative-maxsplit fix."""
    haystack = String("a,b;c,d;e")
    lazy_reversed = list(reversed(list(map(str, haystack.rsplit_byteset_iter(",;", maxsplit)))))
    assert lazy_reversed == haystack.rsplit_byteset(",;", maxsplit)


def test_unit_string_memoryview_is_readonly():
    """String exposes a read-only buffer; the buffer protocol must mark it non-writable."""
    assert memoryview(String("abc")).readonly is True


def test_unit_strings_sequence_slicing():
    """Slicing a `Strings` sequence, including negative indices, steps, nested slices, and empty or
    single-element results, matches indexing the equivalent native list."""
    native = "1, 2, 3, 4, 5, 6"
    big = String(native)
    big_sequence = big.split(", ")

    def to_str(seq):
        return "".join([str(x) for x in seq])

    assert big_sequence[1:3] == ["2", "3"], to_str(big_sequence[1:3])
    assert big_sequence[1:] == ["2", "3", "4", "5", "6"], to_str(big_sequence[1:])
    assert big_sequence[:3] == ["1", "2", "3"], to_str(big_sequence[:3])

    # Use negative indices to slice from the end
    assert big_sequence[-1:] == ["6"], to_str(big_sequence[-1:])
    assert big_sequence[:-1] == ["1", "2", "3", "4", "5"], to_str(big_sequence[:-1])
    assert big_sequence[-3:] == ["4", "5", "6"], to_str(big_sequence[-3:])
    assert big_sequence[:-3] == ["1", "2", "3"], to_str(big_sequence[:-3])

    # Introduce a step to skip some values
    assert big_sequence[::2] == ["1", "3", "5"], to_str(big_sequence[::2])
    assert big_sequence[::-1] == ["6", "5", "4", "3", "2", "1"], to_str(big_sequence[::-1])

    # Let's go harder with nested slicing
    assert big_sequence[1:][::-1] == ["6", "5", "4", "3", "2"]
    assert big_sequence[1:][::-2] == ["6", "4", "2"]
    assert big_sequence[1:][::-3] == ["6", "3"]

    # Edge cases: slices of length 1 and 0
    # Test single element slices (length 1)
    assert big_sequence[0:1] == ["1"], "Single element slice [0:1] failed"
    assert big_sequence[1:2] == ["2"], "Single element slice [1:2] failed"
    assert big_sequence[-1:] == ["6"], "Single element slice [-1:] failed"
    assert big_sequence[-2:-1] == ["5"], "Single element slice [-2:-1] failed"

    # Test empty slices (length 0)
    assert big_sequence[0:0] == [], "Empty slice [0:0] failed"
    assert big_sequence[1:1] == [], "Empty slice [1:1] failed"
    assert big_sequence[5:5] == [], "Empty slice [5:5] failed"
    assert big_sequence[10:10] == [], "Empty slice [10:10] out of bounds failed"
    assert big_sequence[1:0] == [], "Empty slice [1:0] (start > end) failed"
    assert big_sequence[3:1] == [], "Empty slice [3:1] (start > end) failed"

    # Test edge cases with pre-constructed Strings objects passed as slices
    single_slice = big_sequence[0:1]  # Length 1 slice
    empty_slice = big_sequence[0:0]  # Length 0 slice

    # Verify the slices work correctly when passed around
    assert len(single_slice) == 1, "Single slice length incorrect"
    assert len(empty_slice) == 0, "Empty slice length incorrect"
    assert str(single_slice[0]) == "1", "Single slice element access failed"

    # Test iteration over edge case slices
    single_items = [str(item) for item in single_slice]
    assert single_items == ["1"], "Single slice iteration failed"

    empty_items = [str(item) for item in empty_slice]
    assert empty_items == [], "Empty slice iteration failed"

    # Test repr and str don't crash on edge cases
    assert isinstance(repr(single_slice), str), "Single slice repr failed"
    assert isinstance(repr(empty_slice), str), "Empty slice repr failed"
    assert isinstance(str(single_slice), str), "Single slice str failed"
    assert isinstance(str(empty_slice), str), "Empty slice str failed"


def test_unit_strings_rich_comparisons():
    """`Strings` equality and ordering compare correctly against another `Strings`, a native list, a
    tuple, and a generator of unknown length."""
    arr: Strings = String("a b c d e f g h").split()

    # Test against another Strings object
    identical_arr: Strings = String("a b c d e f g h").split()
    different_arr: Strings = String("a b c d e f g i").split()
    shorter_arr: Strings = String("a b c d e").split()
    longer_arr: Strings = String("a b c d e f g h i j").split()

    assert arr == identical_arr
    assert arr != different_arr
    assert arr != shorter_arr
    assert arr != longer_arr
    assert shorter_arr < arr
    assert longer_arr > arr

    # Test against a Python list and a tuple
    list_equal = ["a", "b", "c", "d", "e", "f", "g", "h"]
    list_different = ["a", "b", "c", "d", "x", "f", "g", "h"]
    tuple_equal = ("a", "b", "c", "d", "e", "f", "g", "h")
    tuple_different = ("a", "b", "c", "d", "e", "f", "g", "i")

    assert arr == list_equal
    assert arr != list_different
    assert arr == tuple_equal
    assert arr != tuple_different

    # Test against a generator of unknown length
    generator_equal = (x for x in "a b c d e f g h".split())
    generator_different = (x for x in "a b c d e f g i".split())
    generator_shorter = (x for x in "a b c d e".split())
    generator_longer = (x for x in "a b c d e f g h i j".split())

    assert arr == generator_equal
    assert arr != generator_different
    assert arr != generator_shorter
    assert arr != generator_longer


def test_string_lengths():
    """`len(String)` counts bytes, not codepoints, so a multi-byte UTF-8 string reports its byte length."""
    assert 4 == len(sz.String("abcd"))
    assert 8 == len(sz.String("αβγδ"))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("length", range(1, 300))
def test_translations(length: int, rng: Random):
    """`sz.translate` on `str` and `bytes` input matches `bytes.translate` for the identity,
    inversion, and threshold lookup tables, both copying and in place on a `bytearray`."""
    map_identity = np.arange(256, dtype=np.uint8)
    map_invert = np.arange(255, -1, -1, dtype=np.uint8)
    map_threshold = np.where(np.arange(256) > 127, 255, 0).astype(np.uint8)
    view_identity = memoryview(map_identity)
    view_invert = memoryview(map_invert)
    view_threshold = memoryview(map_threshold)

    body = get_random_string(rng, length=length)
    body_bytes = body.encode("utf-8")

    # Check mapping strings and byte-strings into new strings
    assert sz.translate(body, view_identity) == body
    assert sz.translate(body_bytes, view_identity) == body_bytes
    assert sz.translate(body_bytes, view_identity) == body_bytes.translate(view_identity)
    assert sz.translate(body_bytes, view_invert) == body_bytes.translate(view_invert)
    assert sz.translate(body_bytes, view_threshold) == body_bytes.translate(view_threshold)

    # Check in-place translations on mutable byte-arrays - all of them return nothing
    # Compare against the byte-level oracle `bytes.translate`; a str `.translate` would re-encode any
    # byte >= 128 as a multi-byte UTF-8 codepoint, which is not what an in-place byte translation produces.
    after_identity = bytearray(body_bytes)
    assert sz.translate(after_identity, view_identity, inplace=True) is None
    assert sz.equal(after_identity, body_bytes.translate(view_identity))
    after_invert = bytearray(body_bytes)
    assert sz.translate(after_invert, view_invert, inplace=True) is None
    assert sz.equal(after_invert, body_bytes.translate(view_invert))
    after_threshold = bytearray(body_bytes)
    assert sz.translate(after_threshold, view_threshold, inplace=True) is None
    assert sz.equal(after_threshold, body_bytes.translate(view_threshold))


@pytest.mark.parametrize("length", list(range(0, 300)) + [1024, 4096, 100000])
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_translations_random(length: int, rng: Random, np_rng: "np.random.Generator"):
    """`sz.translate` matches a scalar lookup-table oracle across a random 256-byte permutation,
    over a range of input lengths."""
    body = get_random_string(rng, length=length).encode()
    lut = np_rng.integers(0, 256, size=256, dtype=np.uint8)
    assert sz.translate(body, memoryview(lut)) == lookup_table_oracle(body, bytes(lut))


def test_fill_random_slice(seed: StreamKey):
    """`sz.fill_random` with a `[start, end)` window fills only that window and leaves the rest of
    the buffer untouched."""
    # Prepare a zeroed buffer and keep a copy for comparison
    original = bytearray(64)
    updated_in_slices = bytearray(original)

    # Fill only a slice [start:end) deterministically
    start, end = 10, 30
    sz.fill_random(updated_in_slices, nonce=seed, start=start, end=end)

    # Unchanged prefix and suffix
    assert bytes(updated_in_slices[:start]) == bytes(original[:start])
    assert bytes(updated_in_slices[end:]) == bytes(original[end:])

    # Changed inner region
    assert bytes(updated_in_slices[start:end]) != bytes(original[start:end])


def test_fill_random_different_nonces():
    """`sz.fill_random` with different nonces produces different output for the same buffer size."""
    first_buffer = bytearray(64)
    second_buffer = bytearray(64)
    sz.fill_random(first_buffer, nonce=1)
    sz.fill_random(second_buffer, nonce=2)
    assert bytes(first_buffer) != bytes(second_buffer)


@pytest.mark.parametrize("length", [0, 1, 7, 64])
def test_fill_random_alphabet(length: int, seed: StreamKey):
    """`sz.random` is deterministic for a fixed nonce, and every generated byte belongs to a
    supplied alphabet when one is given."""

    # Same nonce should produce the same result
    random_string = sz.random(length, nonce=seed)
    same_nonce_random_string = sz.random(length, nonce=seed)
    assert isinstance(random_string, (bytes, bytearray))
    assert len(random_string) == length
    assert random_string == same_nonce_random_string

    # With alphabet: all bytes must belong to alphabet
    alphabet = b"0123456789"
    random_digits = sz.random(128, nonce=seed, alphabet=alphabet)
    assert set(random_digits).issubset(set(alphabet))


@pytest.mark.parametrize(
    "byte_string, encoding, expected",
    [
        (b"hello world", "utf-8", "hello world"),
        (b"\xf0\x9f\x98\x81", "utf-8", "😁"),  # Emoji
        (b"hello world", "ascii", "hello world"),
        (b"\xf0hello world", "latin-1", "ðhello world"),
        (b"", "utf-8", ""),  # Empty string case
    ],
)
def test_decoding_valid_strings(byte_string, encoding, expected):
    """`String.decode` matches `bytes.decode` for valid input across `utf-8`, `ascii`, and `latin-1`,
    including an empty string and a multi-byte emoji."""
    assert byte_string.decode(encoding) == expected
    assert sz.String(byte_string).decode(encoding) == expected


# endregion Unit


# region Corner cases

DEGENERATE_WINDOWS = [(3, 1), (100, 200), (-1, -3), (2, 2)]
"""Degenerate [start, end) windows: start > end, far out of range, an inverted negative pair, and a
non-empty-but-zero-length window. Shared by every slicing-style degenerate test."""


OUT_OF_SSIZE_T_INDICES = [2**63, 2**64, -(2**63) - 1]
"""Indices that overflow a signed/unsigned `ssize_t`/`size_t`, used to confirm huge integers raise a
catchable Python exception instead of leaking a `SystemError`."""


OUT_OF_RANGE_SEEDS = [-1, 2**70]
"""Seeds that overflow the C `unsigned long`/`unsigned int` nonce parameters accepted by
`sample`/`shuffled`, which must raise a catchable exception rather than abort the interpreter."""


STARTSWITH_ENDSWITH_BOUNDS = [-10, -2, 0, 2, 5, 6, 10**7, -(10**7)]


TRANSLATE_DEGENERATE_STARTS = [0, 3, 100, -3, 10**6]


@pytest.mark.parametrize("start, stop", DEGENERATE_WINDOWS)
def test_unit_slicing_degenerate(start, stop):
    """Degenerate windows must mirror `str`, returning empty rather than crashing on a negative length."""
    native = "abcdef"
    big = String(native)
    assert big[start:stop] == native[start:stop]


@pytest.mark.parametrize("container", [String("hello"), String("hello").split("l")], ids=["str", "strings"])
@pytest.mark.parametrize("bad_index", OUT_OF_SSIZE_T_INDICES)
def test_unit_index_out_of_ssize_t(container, bad_index):
    """A huge integer index raises IndexError/OverflowError, not a leaked SystemError."""
    with pytest.raises((IndexError, OverflowError)):
        container[bad_index]


@pytest.mark.parametrize("haystack", DEGENERATE_HAYSTACKS)
def test_unit_contains_empty_needle(haystack):
    """The empty string is contained in every string, including the empty one, matching CPython."""
    assert ("" in String(haystack)) == ("" in haystack) == True  # noqa: E712


@pytest.mark.parametrize("start", STARTSWITH_ENDSWITH_BOUNDS)
def test_unit_startswith_endswith_degenerate(start):
    """startswith/endswith mirror CPython across negative/huge/inverted windows."""
    haystack = "hello"
    assert sz.startswith(haystack, "lo", start) == haystack.startswith("lo", start)
    assert sz.endswith(haystack, "lo", start) == haystack.endswith("lo", start)


@pytest.mark.parametrize(
    "sz_func, native_method, haystack, needle",
    [
        (sz.startswith, "startswith", b"\x00x", b"\x00y"),
        (sz.endswith, "endswith", b"ab\x00cd", b"\x00ZZ"),
    ],
    ids=["startswith-embedded_nul", "endswith-embedded_nul"],
)
def test_unit_startswith_endswith_binary_safe(sz_func, native_method, haystack, needle):
    """Comparison must not stop at an embedded NUL byte."""
    assert sz_func(haystack, needle) == getattr(haystack, native_method)(needle) == False  # noqa: E712


@pytest.mark.parametrize("start", TRANSLATE_DEGENERATE_STARTS)
def test_unit_translate_degenerate_window(start):
    """translate with an out-of-range / inverted / negative window must not read or write out of bounds."""
    table = {"a": "A"}
    mutable = bytearray(b"abcabc")
    sz.translate(mutable, table, True, start)  # in-place: must not corrupt or crash
    assert len(mutable) == 6
    sz.translate("abcdef", table, False, start)  # copy: must not leak heap bytes


def test_unit_translate_non_ascii():
    """A `str` translates as its UTF-8 bytes and decodes back, and a dict entry names one byte below U+0100."""
    assert sz.translate("café ∑", {"a": "A"}) == "cAfé ∑"
    assert sz.translate(b"caf\xe9", {"\xe9": "e"}) == b"cafe"
    with pytest.raises(UnicodeDecodeError):
        sz.translate("abc", {"a": "\xe9"})
    with pytest.raises(ValueError):
        sz.translate("abc", {"∑": "a"})
    with pytest.raises(ValueError):
        sz.translate("abc", {"\ud800": "a"})


@pytest.mark.parametrize(
    "operation, expected",
    [
        (lambda strings: list(strings.shuffled()), []),
        (lambda strings: list(strings.sorted()), []),
        (lambda strings: tuple(strings.argsort()), ()),
        (lambda strings: list(strings.sample(5)), []),
    ],
    ids=["shuffled", "sorted", "argsort", "sample"],
)
def test_unit_strings_reorder_empty(operation, expected):
    """sample/shuffled/sorted on an empty Strings return empty, consistent with argsort()."""
    empty = String("").split(",", skip_empty=True)
    assert len(empty) == 0
    assert operation(empty) == expected


def test_unit_strings_sample_zero():
    """sample(0) on a non-empty Strings returns an empty result."""
    assert list(String("a,b,c").split(",").sample(0)) == []


@pytest.mark.parametrize(
    "needle, haystack",
    [
        (String("zzz"), String("hello world")),
        (String(""), String("hello world")),
        (String("hello world"), String("zzz")),
    ],
    ids=["disjoint", "empty_needle_separate_allocation", "haystack_shorter_than_needle"],
)
def test_unit_offset_within_not_contained(needle, haystack):
    """offset_within returns -1 whenever the needle is not a pointer-derived slice of the haystack,
    even when its contents match a substring from a separate allocation."""
    assert needle.offset_within(haystack) == -1


@pytest.mark.parametrize(
    "path, expected_exception",
    [
        (os.path.join("no", "such", "dir", "f.bin"), OSError),
        ("with\x00null", ValueError),
    ],
    ids=["missing_directory", "embedded_nul_in_path"],
)
def test_unit_write_to_errors_are_catchable(path, expected_exception):
    """A failing write_to raises a catchable exception instead of aborting the interpreter."""
    with pytest.raises(expected_exception):
        String("payload").write_to(path)


@pytest.mark.parametrize("seed_value", OUT_OF_RANGE_SEEDS)
def test_unit_strings_sample_bad_seed_is_catchable(seed_value):
    """An out-of-range seed raises a catchable exception instead of leaking a SystemError."""
    strings = String("a,b,c").split(",")
    with pytest.raises((OverflowError, ValueError, TypeError)):
        strings.sample(2, seed=seed_value)


@pytest.mark.parametrize("seed_value", OUT_OF_RANGE_SEEDS)
def test_unit_strings_shuffled_bad_seed_is_catchable(seed_value):
    """An out-of-range seed raises a catchable exception instead of leaking a SystemError."""
    strings = String("a,b,c").split(",")
    with pytest.raises((OverflowError, ValueError, TypeError)):
        strings.shuffled(seed=seed_value)


def test_unit_string_reinit_rebinds():
    """Calling `__init__` again rebinds the object to the new content."""
    s = String("abc")
    s.__init__("xyz")
    assert str(s) == "xyz"


def test_unit_strings_reinit_rebinds():
    """Calling `__init__` again rebinds a Strings container to the new content."""
    ss = String("a,b").split(",")
    ss.__init__(("p", "q"))
    assert [str(x) for x in ss] == ["p", "q"]


def test_unit_file_reinit_rebinds():
    """Calling `File.__init__` again, even on the same path, must not corrupt the mapping or raise."""
    import gc

    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(b"hello reinit")
        temp_filename = tmpfile.name
    file = None
    try:
        file = sz.File(temp_filename)
        file.__init__(temp_filename)
        file.__init__(temp_filename)
        assert bytes(sz.String(file)) == b"hello reinit"
    finally:
        # Windows refuses to delete a file with a live mapping, so drop it before `os.remove`.
        file = None
        gc.collect()
        os.remove(temp_filename)


@pytest.mark.parametrize(
    "byte_string, encoding",
    [
        # Use `bytes.fromhex()` to avoid putting binary literals in source code
        # This prevents PyTest's source parsing from encountering invalid UTF-8
        (bytes.fromhex("ff"), "utf-8"),  # Invalid UTF-8 byte
        (bytes.fromhex("80") + b"hello", "ascii"),  # Non-ASCII byte in ASCII string
    ],
)
def test_decoding_exceptions(byte_string, encoding):
    """`String.decode` raises `UnicodeDecodeError` on invalid UTF-8 or a non-ASCII byte in an ASCII
    decode, matching `bytes.decode`."""
    with pytest.raises(UnicodeDecodeError):
        byte_string.decode(encoding)
    with pytest.raises(UnicodeDecodeError):
        sz.String(byte_string).decode(encoding)


def test_invalid_utf8_handling():
    """`repr()` and `str()` on a `Strings` holding invalid UTF-8, mixed with valid strings, never raise
    and always return non-empty strings."""

    # Test arrays with invalid UTF-8 sequences
    test_arrays = [
        # Use `bytes.fromhex()` to avoid putting binary literals in source code
        # This prevents PyTest's source parsing from encountering invalid UTF-8
        [b"hello", bytes.fromhex("80") + b"world", b"valid"],  # Mixed valid/invalid
        [bytes.fromhex("fffe"), bytes.fromhex("80"), bytes.fromhex("f4908080")],  # All invalid
        [b"normal", b"string with " + bytes.fromhex("80") + b" bytes"],  # Partial invalid
    ]

    for test_array in test_arrays:
        strings_obj = Strings(test_array)

        # These should never raise exceptions
        repr_result = repr(strings_obj)
        str_result = str(strings_obj)

        # Basic assertions
        assert isinstance(repr_result, str)
        assert isinstance(str_result, str)
        assert len(repr_result) > 0
        assert len(str_result) > 0


# endregion Corner cases


# region Interop


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_string_to_pyarrow_conversion():
    """`String.address`/`String.nbytes` expose a valid buffer for `pa.foreign_buffer`, which reconstructs
    the original bytes."""
    native = "hello"
    big = String(native)
    assert isinstance(big.address, int) and big.address != 0
    assert isinstance(big.nbytes, int) and big.nbytes == len(native)

    arrow_buffer = pa.foreign_buffer(big.address, big.nbytes, big)
    assert arrow_buffer.to_pybytes() == native.encode("utf-8")


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strings_to_pyarrow_conversion():
    """`Strings.tape_address/offsets_address/tape_nbytes/offsets_nbytes/offsets_are_large` describe a valid
    tape layout whose sizes match the concatenated string lengths and N+1 offset count, and wrapping
    those buffers in a `pa.Array` reconstructs the original list exactly."""
    native_list = ["hello", "world", "test", "", "python"]
    strings = Strings(native_list)

    # Check that all properties return valid values
    assert isinstance(strings.tape_address, int) and strings.tape_address != 0, "tape_address should be non-zero"
    assert isinstance(strings.offsets_address, int) and strings.offsets_address != 0, (
        "offsets_address should be non-zero"
    )
    assert isinstance(strings.tape_nbytes, int) and strings.tape_nbytes >= 0, "tape_nbytes should be non-negative"
    assert isinstance(strings.offsets_nbytes, int) and strings.offsets_nbytes > 0, "offsets_nbytes should be positive"
    assert isinstance(strings.offsets_are_large, bool), "offsets_are_large should be a boolean"

    # Calculate expected tape size (sum of all string lengths)
    expected_tape_nbytes = sum(len(s) for s in native_list)
    assert strings.tape_nbytes == expected_tape_nbytes, (
        f"Expected tape_nbytes={expected_tape_nbytes}, got {strings.tape_nbytes}"
    )

    # For 5 strings, we should have 6 offsets (N+1 format)
    # Offsets should be either 4 bytes (u32) or 8 bytes (u64) each
    expected_offsets_count = len(native_list) + 1
    if strings.offsets_are_large:
        expected_offsets_nbytes = expected_offsets_count * 8
    else:
        expected_offsets_nbytes = expected_offsets_count * 4
    assert strings.offsets_nbytes == expected_offsets_nbytes, (
        f"Expected offsets_nbytes={expected_offsets_nbytes}, got {strings.offsets_nbytes}"
    )

    # Create PyArrow buffers from the properties
    tape_buffer = pa.foreign_buffer(strings.tape_address, strings.tape_nbytes, strings)
    offsets_buffer = pa.foreign_buffer(strings.offsets_address, strings.offsets_nbytes, strings)

    # Verify the tape contains the concatenated strings
    concatenated = "".join(native_list)
    assert tape_buffer.to_pybytes() == concatenated.encode("utf-8"), "Tape should contain concatenated strings"

    # Create an Arrow array from the buffers
    if strings.offsets_are_large:
        arrow_array = pa.Array.from_buffers(pa.large_string(), len(native_list), [None, offsets_buffer, tape_buffer])
    else:
        arrow_array = pa.Array.from_buffers(pa.string(), len(native_list), [None, offsets_buffer, tape_buffer])

    # Verify the Arrow array matches the original data
    assert arrow_array.to_pylist() == native_list, "Arrow array should match original list"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strings_pyarrow_empty():
    """An empty `Strings` still holds a tape: zero tape bytes, one u32 offset, and `offsets_are_large` false."""
    strings = Strings([])

    tape_addr = strings.tape_address
    assert tape_addr is not None and tape_addr >= 0

    offsets_addr = strings.offsets_address
    assert offsets_addr is not None and offsets_addr >= 0

    tape_bytes = strings.tape_nbytes
    assert tape_bytes == 0  # No strings = no tape content

    offsets_bytes = strings.offsets_nbytes
    assert offsets_bytes == 4  # Arrow uses N+1 offsets, so 1 offset for 0 strings (u32 = 4 bytes)

    are_large = strings.offsets_are_large
    assert are_large == False  # Default to u32 offsets


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strings_pyarrow_large_strings():
    """`Strings.offsets_nbytes` sizes correctly for whichever offset width `offsets_are_large` selects as
    string sizes approach the u32/u64 threshold, and the resulting Arrow array round-trips the list."""
    # Create strings with total size that could trigger u64 layout
    large_string = "x" * 10000
    native_list = [large_string, "small", large_string, ""]
    strings = Strings(native_list)

    # Check properties work correctly
    assert isinstance(strings.offsets_are_large, bool), "offsets_are_large should be boolean"
    assert strings.tape_nbytes == sum(len(s) for s in native_list), "tape_nbytes should match total length"

    expected_offsets_count = len(native_list) + 1
    offset_size = 8 if strings.offsets_are_large else 4
    assert strings.offsets_nbytes == expected_offsets_count * offset_size, "offsets_nbytes should be correct"

    # Create PyArrow array and verify
    tape_buffer = pa.foreign_buffer(strings.tape_address, strings.tape_nbytes, strings)
    offsets_buffer = pa.foreign_buffer(strings.offsets_address, strings.offsets_nbytes, strings)

    if strings.offsets_are_large:
        arrow_array = pa.Array.from_buffers(pa.large_string(), len(native_list), [None, offsets_buffer, tape_buffer])
    else:
        arrow_array = pa.Array.from_buffers(pa.string(), len(native_list), [None, offsets_buffer, tape_buffer])

    assert arrow_array.to_pylist() == native_list, "Arrow array should match original list"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strings_pyarrow_gathered_and_borrowed():
    """`split` and `shuffled` gather onto tapes of their own, while a contiguous slice borrows the bytes and
    offsets of its source, its first offset past zero; Arrow reads all three back through the tape properties."""
    parts = String("apple banana cherry date").split(" ")
    shuffled = parts.shuffled(seed=42)
    borrowed = parts[1:3]
    assert not parts.offsets_are_large
    assert shuffled.tape_address != parts.tape_address
    assert borrowed.tape_address == parts.tape_address
    assert borrowed.offsets_address == parts.offsets_address + 4
    assert borrowed.tape_nbytes == len("applebananacherry")

    for strings in (parts, shuffled, borrowed):
        tape_buffer = pa.foreign_buffer(strings.tape_address, strings.tape_nbytes, strings)
        offsets_buffer = pa.foreign_buffer(strings.offsets_address, strings.offsets_nbytes, strings)
        arrow_array = pa.Array.from_buffers(pa.string(), len(strings), [None, offsets_buffer, tape_buffer])
        assert arrow_array.to_pylist() == [str(part) for part in strings]


@pytest.mark.parametrize("container_class", [tuple, list, iter])
def test_strings_from_python_basic(container_class: type):
    """`Strings` built from a tuple, list, or iterator preserves item order and count."""
    base_items = ["hello", "world", "test", " ", "from", " ", "container", ""]
    container = container_class(base_items)

    strings = Strings(container)

    assert len(strings) == len(base_items)
    assert strings[0] == "hello"
    assert strings[1] == "world"
    assert strings[2] == "test"
    assert strings[3] == " "
    assert strings[4] == "from"
    assert strings[5] == " "
    assert strings[6] == "container"
    assert strings[7] == ""


@pytest.mark.skipif(sys.maxsize <= 2**32, reason="64-bit system required for 4GB+ test")
def test_strings_from_4gb_list():
    """Growing a `Strings` past 4 GB of total content forces its offset table from u32 to u64, and indexing
    still returns the correct first and last strings afterward.
    This will require over 8 GB of memory. To stress-test the behavior, limit memory per process. For 9 and 13 GB:

    ulimit -v 9437184 && uv run --no-project python -m pytest test/string_types.py -s -x -k 4gb_list
    ulimit -v 13631488 && uv run --no-project python -m pytest test/string_types.py -s -x -k 4gb_list
    """

    try:
        import ctypes
        import gc
    except ImportError:
        pytest.skip("ctypes & gc not available (e.g., PyPy)")

    # Each individual string won't be very large, but many of them will be used
    part_size = 64 * 1024 * 1024
    parts_count = math.ceil(UINT32_MAX / part_size) + 1  # Ensures we exceed UINT32_MAX by a small margin
    try:
        parts_pythonic = [
            long_repeated_string(ctypes, ascii_lowercase[part_index % len(ascii_lowercase)], part_size)
            for part_index in range(parts_count)
        ]
        parts_stringzilla = Strings(parts_pythonic)

        # Basic verification
        last_used_char = ascii_lowercase[(parts_count - 1) % len(ascii_lowercase)]
        assert len(parts_stringzilla) == parts_count
        assert parts_stringzilla[0] == long_repeated_string(ctypes, "a", part_size)
        assert parts_stringzilla[-1] == long_repeated_string(ctypes, last_used_char, part_size)

        del parts_pythonic
        del parts_stringzilla
    except (MemoryError, OSError):
        pytest.skip("Memory allocation failed")
    finally:
        gc.collect()


@pytest.mark.skipif(sys.maxsize <= 2**32, reason="64-bit system required for 4GB+ test")
def test_strings_from_4gb_generator():
    """Streaming a generator of strings past 4 GB of total content forces the offset table from u32 to
    u64, and indexing still returns the correct first and last strings afterward.
    This will require over 8 GB of memory. To stress-test the behavior, limit memory per process. For 5 and 13 GB:

    ulimit -v 5242880 && uv run --no-project python -m pytest test/string_types.py -s -x -k 4gb_generator
    ulimit -v 13631488 && uv run --no-project python -m pytest test/string_types.py -s -x -k 4gb_generator
    """

    try:
        import ctypes
        import gc
    except ImportError:
        pytest.skip("ctypes & gc not available (e.g., PyPy)")

    # Each individual string won't be very large, but many of them will be used
    part_size = 64 * 1024 * 1024
    parts_count = math.ceil(UINT32_MAX / part_size) + 1  # Ensures we exceed UINT32_MAX by a small margin
    try:
        parts_stringzilla = Strings(
            long_repeated_string(ctypes, ascii_lowercase[part_index % len(ascii_lowercase)], part_size)
            for part_index in range(parts_count)
        )

        # Basic verification
        last_used_char = ascii_lowercase[(parts_count - 1) % len(ascii_lowercase)]
        assert len(parts_stringzilla) == parts_count
        assert parts_stringzilla[0] == long_repeated_string(ctypes, "a", part_size)
        assert parts_stringzilla[-1] == long_repeated_string(ctypes, last_used_char, part_size)

        del parts_stringzilla
    except (MemoryError, OSError):
        pytest.skip("Memory allocation failed")
    finally:
        gc.collect()


@pytest.mark.parametrize("container_class", [tuple, list])
def test_strings_reference_counting(container_class: type):
    """`Strings` copies its source, so once constructed it pins neither the container nor any of its items."""

    # CPython-only: PyPy and other interpreters may not expose refcounts or use a different GC model
    if not hasattr(sys, "getrefcount"):
        pytest.skip("Reference counting semantics are not available")

    # Round-tripped through bytes, as string literals are immortal and their refcounts never move
    base_items = [word.encode().decode() for word in ("ref", "count", "test")]
    container = container_class(base_items)
    initial_refcounts = [sys.getrefcount(container)] + [sys.getrefcount(item) for item in base_items]

    strings = Strings(container)
    assert [sys.getrefcount(container)] + [sys.getrefcount(item) for item in base_items] == initial_refcounts
    assert [str(part) for part in strings] == ["ref", "count", "test"]


@pytest.mark.parametrize("mutation", [list.clear, list.reverse, lambda source: source.__delitem__(slice(1, None))])
def test_strings_survives_source_mutation(mutation):
    """`Strings(list)` copies its source, so clearing, reversing or truncating that list afterwards leaves
    every element readable."""
    # Built twice rather than copied, so the source list holds the only reference to its own items and
    # mutating it really does free them - a shared list would keep them alive and prove nothing
    expected = ["prefix-" + "x" * 200 + f"-{index}" for index in range(8)]
    source = ["prefix-" + "x" * 200 + f"-{index}" for index in range(8)]
    strings = Strings(source)

    mutation(source)
    # Churn same-size-class allocations so a freed item is handed straight back out
    churn = [("z" * 208).encode() for _ in range(scale_iterations(5000))]

    assert [str(part) for part in strings] == expected, f"spans went stale across {len(churn)} reuses"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strings_from_arrow_basic():
    """`Strings` borrows the offsets and bytes of a PyArrow string array in place, honors the offset of a sliced
    array, and keeps the buffers alive once the array itself is gone."""
    arrow_array = pa.array(["hello", "world", "test", "arrow"])
    strings = Strings(arrow_array)
    _, offsets_buffer, data_buffer = arrow_array.buffers()
    assert strings.tape_address == data_buffer.address
    assert strings.offsets_address == offsets_buffer.address
    assert [str(part) for part in Strings(arrow_array.slice(1, 2))] == ["world", "test"]
    del arrow_array, offsets_buffer, data_buffer

    assert len(strings) == 4
    assert strings[0] == "hello"
    assert strings[1] == "world"
    assert strings[2] == "test"
    assert strings[3] == "arrow"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strings_from_arrow_binary_array():
    """`Strings` built from a PyArrow binary array preserves each element's raw bytes exactly, whether
    returned as `str` or as a byte-like object."""
    binary_data = [b"hello", b"world", b"binary", b"data"]
    arrow_array = pa.array(binary_data, type=pa.binary())

    strings = Strings(arrow_array)

    assert len(strings) == 4
    # Strings should handle binary data properly - compare as bytes
    for i, expected in enumerate(binary_data):
        str_bytes = strings[i].encode("latin-1") if isinstance(strings[i], str) else bytes(strings[i])
        assert str_bytes == expected


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strings_from_arrow_large_strings():
    """`Strings` built from a PyArrow `large_string` array preserves length and element order exactly."""
    arrow_array = pa.array(["hello", "world", "large", "strings"], type=pa.large_string())

    strings = Strings(arrow_array)

    assert len(strings) == 4
    assert strings[0] == "hello"
    assert strings[1] == "world"
    assert strings[2] == "large"
    assert strings[3] == "strings"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strings_from_arrow_error_cases():
    """`Strings` raises `TypeError` for non-iterable inputs like an int or `None`, and `TypeError` or
    `ValueError` for a non-string PyArrow array."""
    # Test with non-iterable object
    with pytest.raises(TypeError):
        Strings(123)  # Integer is not iterable

    with pytest.raises(TypeError):
        Strings(None)  # None is not iterable

    # Test with non-string Arrow array
    int_array = pa.array([1, 2, 3, 4])
    with pytest.raises((TypeError, ValueError)):
        Strings(int_array)


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strings_from_arrow_c_interface():
    """PyArrow's `__arrow_c_array__` capsules are valid `PyCapsule` objects, and `Strings` built from the
    array through that C Data Interface preserves length and element order."""
    arrow_array = pa.array(["test", "c", "interface"])

    # Check that Arrow array has the __arrow_c_array__ method
    assert hasattr(arrow_array, "__arrow_c_array__")

    # Get the C interface capsules
    schema_capsule, array_capsule = arrow_array.__arrow_c_array__()

    # Verify capsules are valid PyCapsule objects
    if sys.version_info >= (3, 1):
        assert str(type(schema_capsule)) == "<class 'PyCapsule'>"
        assert str(type(array_capsule)) == "<class 'PyCapsule'>"

    # Test actual conversion
    strings = Strings(arrow_array)
    assert len(strings) == 3
    assert strings[0] == "test"
    assert strings[1] == "c"
    assert strings[2] == "interface"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strings_from_arrow_with_nulls():
    """Arrow arrays with null values, including all-null and mixed null/empty-string arrays, convert to
    `Strings` with every null mapped to an empty string."""
    # Create an array with None values
    arrow_array = pa.array(["hello", None, "world", None, "test"])

    strings = Strings(arrow_array)

    assert len(strings) == 5
    assert strings[0] == "hello"
    assert strings[1] == ""  # None values should be converted to empty strings
    assert strings[2] == "world"
    assert strings[3] == ""  # None values should be converted to empty strings
    assert strings[4] == "test"

    # Test with all nulls
    all_nulls = pa.array([None, None, None], type=pa.string())
    strings_nulls = Strings(all_nulls)
    assert len(strings_nulls) == 3
    assert all(s == "" for s in strings_nulls)

    # Test with mixed None and empty strings
    mixed = pa.array(["", None, "hello", "", None, "world"])
    strings_mixed = Strings(mixed)
    assert len(strings_mixed) == 6
    assert strings_mixed[0] == ""
    assert strings_mixed[1] == ""
    assert strings_mixed[2] == "hello"
    assert strings_mixed[3] == ""
    assert strings_mixed[4] == ""
    assert strings_mixed[5] == "world"


# endregion Interop


# region Escaping and dumping

ESCAPE_KINDS = ["json", "json_ascii", "hex", "html", "url", "console", "csv"]

ESCAPE_CORPUS = [
    "",
    "plain ascii",
    'quote " and backslash \\',
    "controls \x00\x01\x08\x0c\x1f",
    "line breaks \n\r\t",
    "comma, separated",
    "slash / and tilde ~",
    "markup <b>&amp;</b> 'single'",
    "accents café ∑ 日本",
    "beyond the plane \U0001f600",
    "bidirectional \u202e override \u2066 isolate",
    "c1 control \u0085",
    "percent %41",
]
"""Strings holding the special bytes of every escape kind, multi-byte UTF-8, and a codepoint past U+FFFF."""


def csv_field(text: str) -> str:
    """One field as `csv.writer` quotes it, written beside another so an empty one stays unquoted."""
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\r\n").writerow([text, ""])
    return buffer.getvalue()[:-3]


def csv_unescaped(field: str) -> str:
    """One field as `csv.reader` reads it, where an empty line holds no fields at all."""
    row = next(csv.reader([field]))
    return row[0] if row else ""


STDLIB_ESCAPERS = {
    "json": lambda text: json.dumps(text, ensure_ascii=False)[1:-1],
    "json_ascii": lambda text: json.dumps(text)[1:-1],
    "hex": lambda text: text.encode().hex(),
    "html": lambda text: html.escape(text).replace("&#x27;", "&#39;"),
    "url": lambda text: urllib.parse.quote(text, safe=""),
    "csv": csv_field,
}
"""The standard library's writer for every kind that has one; `html.escape` alone spells `'` in hex."""

STDLIB_UNESCAPERS = {
    "json": lambda escaped: json.loads(f'"{escaped}"'),
    "json_ascii": lambda escaped: json.loads(f'"{escaped}"'),
    "hex": lambda escaped: bytes.fromhex(escaped).decode(),
    "html": html.unescape,
    "url": lambda escaped: urllib.parse.unquote_to_bytes(escaped).decode(),
    "csv": csv_unescaped,
}

DUMP_COLUMNS = {
    "name": ["Ada", 'Grace "Amazing" Hopper', "Linus\nTorvalds", "", "café \U0001f600"],
    "city, country": ["London, UK", "New York", "Helsinki\r\n", "日本", "<&>"],
}
"""Two columns, one named with a comma, whose cells need every JSON and CSV escape between them."""


@pytest.mark.parametrize("kind", sorted(STDLIB_ESCAPERS))
def test_strings_escape_matches_stdlib(kind: str):
    """`Strings.escape` writes the bytes the standard library's own writer does, for every kind that has one."""
    # Python 3.10's CSV module rejects NUL; the round-trip checks below still cover it.
    corpus = [text for text in ESCAPE_CORPUS if kind != "csv" or sys.version_info >= (3, 11) or "\0" not in text]
    escaped = Strings(corpus).escape(kind)
    assert [str(text) for text in escaped] == [STDLIB_ESCAPERS[kind](text) for text in corpus]


@pytest.mark.parametrize("kind", ESCAPE_KINDS)
def test_strings_unescape_round_trips(kind: str):
    """`Strings.unescape` inverts `Strings.escape` for every kind, as the standard library's own readers do, and every
    kind but `json_ascii`, which writes an ill-formed byte as U+FFFD, carries ill-formed UTF-8 through too."""
    corpus = ESCAPE_CORPUS + ["delete \x7f"]
    escaped = Strings(corpus).escape(kind=kind)
    assert [str(text) for text in escaped.unescape(kind)] == corpus
    if kind in STDLIB_UNESCAPERS:
        # Python 3.10's CSV reader rejects NUL; StringZilla already round-tripped it above.
        supported = [text for text in corpus if kind != "csv" or sys.version_info >= (3, 11) or "\0" not in text]
        assert [STDLIB_UNESCAPERS[kind](str(text)) for text in Strings(supported).escape(kind)] == supported
    if kind != "json_ascii":
        ill_formed = b"ill-formed \xc3\x28 \xff"
        assert bytes(Strings([ill_formed]).escape(kind).unescape(kind)[0]) == ill_formed


@pytest.mark.parametrize(
    "kind, malformed",
    [
        ("json", "lone \\ud800 surrogate"),
        ("json_ascii", "unknown \\q escape"),
        ("hex", "abc"),
        ("url", "truncated %4"),
        ("console", "truncated \\x4"),
        ("csv", '"unbalanced'),
    ],
)
def test_strings_unescape_names_the_malformed_index(kind: str, malformed: str):
    """`Strings.unescape` raises a `ValueError` naming the index of the first malformed string."""
    well_formed = str(Strings(['ok, "fine"']).escape(kind)[0])
    with pytest.raises(ValueError, match="index 2 "):
        Strings([well_formed, well_formed, malformed, well_formed, malformed]).unescape(kind)


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_strings_to_json_matches_json_dumps(ensure_ascii: bool):
    """`Strings.to_json` writes the bytes of a compact `json.dumps`, over a tape, a borrowed slice of
    one, and a strided slice gathered from it."""

    def expected(texts):
        return json.dumps(texts, ensure_ascii=ensure_ascii, separators=(",", ":")).encode()

    tape = Strings(ESCAPE_CORPUS * 6)
    assert tape.to_json(ensure_ascii=ensure_ascii) == expected(ESCAPE_CORPUS * 6)
    assert tape[3:].to_json(ensure_ascii) == expected((ESCAPE_CORPUS * 6)[3:])
    assert tape[::-1].to_json(ensure_ascii) == expected((ESCAPE_CORPUS * 6)[::-1])
    assert Strings([]).to_json() == b"[]"


def test_strings_to_json_ascii_escapes_delete():
    """`json.dumps` writes DEL as `\\u007f` once it keeps to ASCII, so `ensure_ascii` must as well."""
    assert Strings(["delete \x7f"]).to_json() == json.dumps(["delete \x7f"], separators=(",", ":")).encode()


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_to_json_matches_json_dumps(ensure_ascii: bool):
    """`sz.to_json` writes the bytes of a compact `json.dumps` of the records, from `Strings` and lists alike."""
    tiled = {name: cells * 30 for name, cells in DUMP_COLUMNS.items()}
    records = [dict(zip(tiled, row)) for row in zip(*tiled.values())]
    columns = {"name": Strings(tiled["name"]), "city, country": tiled["city, country"]}
    expected = json.dumps(records, ensure_ascii=ensure_ascii, separators=(",", ":")).encode()
    assert sz.to_json(columns, ensure_ascii=ensure_ascii) == expected
    assert sz.to_json({}) == sz.to_json({"empty": []}) == b"[]"


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_to_json_matches_pandas(ensure_ascii: bool):
    """`sz.to_json` writes the bytes of `pandas.DataFrame.to_json(orient="records")` for string columns
    holding no `/`, which pandas alone writes as `\\/`."""
    pandas = pytest.importorskip("pandas")
    expected = pandas.DataFrame(DUMP_COLUMNS).to_json(orient="records", force_ascii=ensure_ascii).encode()
    assert sz.to_json(DUMP_COLUMNS, ensure_ascii=ensure_ascii) == expected


@pytest.mark.parametrize("lineterminator", ["\n", "\r\n"])
def test_to_csv_matches_csv_writer(lineterminator: str):
    """`sz.to_csv` writes the bytes of `csv.writer`: a header of the names, then a line per row, a lone empty
    field quoted as `""` so its line is not mistaken for one of no fields, over more lines than one block joins."""
    tiled = {name: cells * 30 for name, cells in DUMP_COLUMNS.items()}
    for columns in (DUMP_COLUMNS, tiled, {"": ["", "x", ""] * 50}, {"empty": []}, {}):
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator=lineterminator)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
        assert sz.to_csv(columns, lineterminator) == buffer.getvalue().encode()


def test_to_csv_matches_pandas():
    """`sz.to_csv` writes the bytes of `pandas.DataFrame.to_csv(index=False)` for string columns."""
    pandas = pytest.importorskip("pandas")
    for columns in (DUMP_COLUMNS, {"": ["", "x", ""]}):
        expected = pandas.DataFrame(columns).to_csv(index=False, lineterminator="\n").encode()
        assert sz.to_csv({name: Strings(column) for name, column in columns.items()}) == expected


def test_dumps_refuse_malformed_columns():
    """Columns of different lengths, a non-string cell, and a terminator other than LF or CRLF are refused."""
    ragged = {"a": ["x", "y"], "b": Strings(["z"])}
    with pytest.raises(ValueError, match="'b'"):
        sz.to_json(ragged)
    with pytest.raises(ValueError, match="'b'"):
        sz.to_csv(ragged)
    with pytest.raises(TypeError):
        sz.to_json({"a": ["x", None]})
    with pytest.raises(ValueError):
        sz.to_csv({"a": ["x"]}, lineterminator=";")


def test_strings_translate_matches_bytes_translate():
    """`Strings.translate` maps every string as `bytes.translate` does, over a tape and a slice of one,
    and takes the same dict tables as `String.translate`."""
    table = bytes(range(256)).translate(bytes.maketrans(b"abc", b"ABC"))
    texts = ["cabbage", "", "abc" * 50, "été"]
    expected = [text.encode().translate(table) for text in texts]
    for strings in (Strings(texts), Strings(["skipped"] + texts)[1:]):
        assert [bytes(text) for text in strings.translate(table)] == expected
    assert [str(text) for text in Strings(texts).translate({"a": "A"})] == [text.replace("a", "A") for text in texts]


COLUMN_TEXTS = ["hello", "", "help yourself", "élan", "yellow", "el", "ab", "cd", "x" * 300 + "el"] + [
    "e" * length for length in range(18)
]
"""Keys on both sides of the 16 bytes the hash kernels batch, a long one, a two-byte `é`, and the adjacent
"ab" and "cd" whose seam a search over the whole tape must not match as "bc"."""


def column_sources(texts: list) -> list:
    """`texts` as an owned tape, a slice whose offsets start past zero, a strided gather and, with PyArrow, a
    sliced Arrow array borrowed in place, each paired with the texts it holds."""
    sources = [
        (Strings(texts), texts),
        (Strings(["skipped"] + texts + ["tail"])[1:-1], texts),
        (Strings(texts)[::-2], texts[::-2]),
    ]
    if pyarrow_available:
        sources.append((Strings(pa.array(["skipped"] + texts).slice(1)), texts))
    return sources


@pytest.mark.parametrize("needle", ["el", "", "l", "élan", "bc", "zzz", "x" * 40])
def test_strings_find_matches_bytes_find(backend_results, needle: str):
    """`Strings.find` returns what `bytes.find` does in every string, -1 where the needle is absent and never
    a match across the seam of two strings, over every kind of tape, and writes the same into `out`."""
    for strings, texts in column_sources(COLUMN_TEXTS):
        expected = tuple(text.encode().find(needle.encode()) for text in texts)
        results = backend_results(lambda capabilities, strings=strings: strings.find(needle, capabilities=capabilities))
        assert_backends_agree(results, oracle=expected, format_inputs=lambda: f"needle={needle!r}")
        assert strings.find(needle.encode()) == strings.find(String(needle)) == expected
        if numpy_available:
            out = np.full(len(strings), 7, dtype=np.intp)
            assert strings.find(needle, out=out) is out and tuple(out.tolist()) == expected


@pytest.mark.parametrize("seed", [0, 7, 2**64 - 1])
def test_strings_hashes_match_hash(backend_results, seed: int):
    """`Strings.hashes(seed)` returns `sz.hash(text, seed)` for every string, over every kind of tape, and
    writes the same into `out`."""
    for strings, texts in column_sources(COLUMN_TEXTS):
        expected = tuple(sz.hash(text, seed) for text in texts)
        results = backend_results(lambda capabilities, strings=strings: strings.hashes(seed, capabilities=capabilities))
        assert_backends_agree(results, oracle=expected, format_inputs=lambda: f"seed={seed}")
        if numpy_available:
            out = np.zeros(len(strings), dtype=np.uint64)
            assert strings.hashes(seed=seed, out=out) is out and tuple(out.tolist()) == expected


SPLIT_TEXTS = ["", ",", ",,", " a  b ", ",a,,b,", "Åsa, Bob,,é ,", "\n", "a\n", "\r\n\r\n", "a\r"] + [
    "x" * offset + delimiter + "y" * 40 + delimiter
    for offset in (13, 14, 15, 16, 30, 31, 32, 62, 63, 64)
    for delimiter in (",", " ", "\r\n", "\x85", " ", " ")
]
"""Empty parts at either end and in the middle, multi-byte UTF-8 beside the delimiters, and every delimiter and
line break, multi-byte ones included, straddling the 16, 32 and 64-byte vector boundaries."""


def test_strings_split_matches_str_split(backend_results):
    """Splits at one byte or a byteset and `splitlines` match `str.split` and `str.splitlines` on every backend, with
    `skip_empty` dropping the empty parts, `rsplit` reporting source order, and a bare `split()` keeping empty parts
    between single spaces."""

    def split_all(capabilities):
        return [
            [
                [str(part) for part in String(text).split(",", capabilities=capabilities)],
                [str(part) for part in String(text).rsplit(",", capabilities=capabilities)],
                [str(part) for part in String(text).split(capabilities=capabilities)],
                [str(part) for part in String(text).split(" ", skip_empty=True, capabilities=capabilities)],
                [str(part) for part in String(text).split_byteset(", ", capabilities=capabilities)],
                [str(part) for part in String(text).splitlines(capabilities=capabilities)],
            ]
            for text in SPLIT_TEXTS
        ]

    oracle = [
        [
            text.split(","),
            text.rsplit(","),
            text.split(" "),
            [part for part in text.split(" ") if part],
            text.replace(" ", ",").split(","),
            text.splitlines(),
        ]
        for text in SPLIT_TEXTS
    ]
    assert_backends_agree(backend_results(split_all), oracle=oracle)


@pytest.mark.parametrize("separator", ["", ", ", "é"])
def test_string_join_matches_str_join(backend_results, separator: str):
    """`String.join` writes what `str.join` does, from every kind of tape read in place, past one 64-row block, and
    from lists of `str` or `bytes` and generators, refusing an item that is not a string."""
    texts = COLUMN_TEXTS * 3
    for strings, column in column_sources(texts):
        results = backend_results(
            lambda capabilities, strings=strings: str(String(separator).join(strings, capabilities=capabilities))
        )
        assert_backends_agree(results, oracle=separator.join(column))
    assert str(String(separator).join(texts)) == separator.join(texts)
    assert str(String(separator).join(text.encode() for text in texts)) == separator.join(texts)
    assert str(String(separator).join([])) == ""
    with pytest.raises(TypeError, match="item 1 "):
        String(separator).join(["a", None])


UTF8_ROWS = [text.encode() for text in COLUMN_TEXTS + ["日本語", "\U0001f600" * 20, "c1 control \u0085"]]


def test_strings_utf8_count_matches_decode(backend_results):
    """`Strings.utf8_count` returns `len(row.decode())` for every row, over every kind of tape, and writes the same
    into `out`."""
    for strings, rows in column_sources(UTF8_ROWS):
        expected = tuple(len(row.decode()) for row in rows)
        results = backend_results(lambda capabilities, strings=strings: strings.utf8_count(capabilities=capabilities))
        assert_backends_agree(results, oracle=expected)
        if numpy_available:
            out = np.zeros(len(strings), dtype=np.uintp)
            assert strings.utf8_count(out=out) is out and tuple(out.tolist()) == expected


@pytest.mark.parametrize(
    "rows, malformed_index",
    [
        ([b"abc", b"\xc3", b"\xa9"], 1),
        ([b"abc", b"\x80def"], 1),
        ([b"ok", b"ok", b"tail \xe2\x82"], 2),
        ([b"\xed\xa0\x80", b"\xff"], 0),
        ([b"x" * 64 + b"\xf4\x90\x80\x80"], 0),
    ],
    ids=["sequence_across_rows", "stray_continuation", "truncated_at_tape_end", "surrogate", "past_u10ffff"],
)
def test_strings_utf8_count_names_the_malformed_index(backend_results, rows: list, malformed_index: int):
    """`Strings.utf8_count` raises a `ValueError` naming the first row `bytes.decode` refuses, even where the bytes
    of two rows would decode together."""
    for row in rows[:malformed_index]:
        row.decode()
    with pytest.raises(UnicodeDecodeError):
        rows[malformed_index].decode()

    def message(capabilities):
        with pytest.raises(ValueError, match=f"index {malformed_index} ") as error:
            Strings(rows).utf8_count(capabilities=capabilities)
        return str(error.value)

    assert_backends_agree(backend_results(message))


# endregion Escaping and dumping


# region Backend differential


def long_repeated_string(ctypes, fill_char: str, string_size: int) -> str:
    buffer = ctypes.create_string_buffer(string_size)
    ctypes.memset(buffer, ord(fill_char), string_size)
    return buffer.value.decode("ascii")


def safe_windows(length: int) -> Sequence[Sequence[int]]:
    """A handful of [start, end) windows into a `length`-byte body, including overlapping and odd
    offsets, the empty window at both ends, and the full span, all guaranteed in-bounds for `length >= 0`."""
    windows = {
        (0, length),
        (0, 0),
        (length, length),
        (0, length // 2),
        (length // 2, length),
        (length // 3, (2 * length) // 3),
        (1, length - 1) if length >= 2 else (0, length),
    }
    return sorted(windows)


def lookup_table_oracle(data: bytes, table: bytes) -> bytes:
    """Reference byte-for-byte lookup, independent of `sz.translate`'s SIMD kernel."""
    return bytes(table[byte] for byte in data)


UINT32_MAX = 2**32 - 1  # ! Many of the 32/64-bit algo corner cases happen at this input size


_TRANSLATE_IDENTITY_TABLE = bytes(range(256))
"""Fixed 256-byte lookup tables reused by every translate differential test: the no-op identity, a
full inversion, and an arbitrary fixed bijection-ish permutation that scrambles every byte differently."""
_TRANSLATE_INVERT_TABLE = bytes(255 - value for value in range(256))
_TRANSLATE_SCRAMBLE_TABLE = bytes((value * 167 + 41) % 256 for value in range(256))
TRANSLATE_TABLES = (_TRANSLATE_IDENTITY_TABLE, _TRANSLATE_INVERT_TABLE, _TRANSLATE_SCRAMBLE_TABLE)


@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_backend_differential_equal(backend_results, length: int, rng: Random):
    """`sz.equal` (the compare kernel) must agree across every backend and match Python's byte-level
    `==`, as must `String.__eq__`, for random/tiled/all-same-char bodies at every SIMD-relevant length."""
    for body in differential_bodies(length, rng):
        matching = body
        mismatching = body[:-1] + ("z" if body[-1:] != "z" else "y") if length > 0 else "nonempty"

        assert String(body) == matching
        assert String(body) != mismatching
        assert_backends_agree(
            backend_results(
                lambda capabilities, body=body, matching=matching: sz.equal(body, matching, capabilities=capabilities)
            ),
            oracle=True,
        )
        assert_backends_agree(
            backend_results(
                lambda capabilities, body=body, mismatching=mismatching: sz.equal(
                    body, mismatching, capabilities=capabilities
                )
            ),
            oracle=False,
        )


def test_unit_equal_compares_both_operands():
    """`sz.equal` must compare its two operands, not the first against itself. Regression for a binding
    bug where the second argument was ignored and the result was always True."""
    assert sz.equal("abc", "abc") is True
    assert sz.equal("abc", "xyz") is False  # same length, different content
    assert sz.equal("abc", "ab") is False  # different length
    assert sz.equal(b"\x00\x01", b"\x00\x01") is True
    assert sz.equal(b"\x00\x01", b"\x02\x03") is False
    assert sz.equal("", "") is True
    assert sz.equal("", "a") is False


@pytest.mark.parametrize("offset", [0, 1, 3, 7, 15])
def test_unit_backend_differential_equal_unaligned(backend_results, offset):
    """Equality on a sliced (misaligned-base-pointer) view must still agree across every backend."""
    text = "ab" * 200
    for _offset, view in unaligned_views(text, offsets=(offset,)):
        matching = text[offset:]
        mismatching = "z" + text[offset + 1 :] if len(matching) > 0 else "nonempty"
        assert_backends_agree(
            backend_results(
                lambda capabilities, view=view, matching=matching: sz.equal(view, matching, capabilities=capabilities)
            ),
            oracle=True,
        )
        assert_backends_agree(
            backend_results(
                lambda capabilities, view=view, mismatching=mismatching: sz.equal(
                    view, mismatching, capabilities=capabilities
                )
            ),
            oracle=False,
        )


@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_backend_differential_contains(backend_results, length: int, rng: Random):
    """`sz.find`, the kernel behind `needle in String(haystack)`, must agree across every backend, for
    present and absent needles, over random/tiled/all-same-char needles at every SIMD-relevant length."""
    for needle in differential_bodies(length, rng):
        haystack = f"head_{needle}_tail"
        absent_needle = needle + "\x01"  # a byte that never appears in any of the three alphabets

        assert needle in String(haystack) and absent_needle not in String(haystack)
        assert_backends_agree(
            backend_results(
                lambda capabilities, needle=needle, haystack=haystack: sz.find(
                    haystack, needle, capabilities=capabilities
                )
            ),
            oracle=haystack.find(needle),
        )
        assert_backends_agree(
            backend_results(
                lambda capabilities, absent_needle=absent_needle: sz.find(
                    haystack, absent_needle, capabilities=capabilities
                )
            ),
            oracle=-1,
        )


@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_copy_slice(length: int, rng: Random):
    """`bytes(String)` and `String(s)[a:b]` (the copy kernel) must match Python's own byte-level slicing,
    including overlapping/odd windows, over random/tiled/all-same-char bodies."""
    for body in differential_bodies(length, rng):
        body_bytes = body.encode()
        assert bytes(String(body)) == body_bytes
        for start, stop in safe_windows(length):
            assert bytes(String(body)[start:stop]) == body_bytes[start:stop]


@pytest.mark.parametrize("offset", [0, 1, 3, 7, 15])
def test_unit_copy_unaligned(offset):
    """Copying out of a misaligned (sliced) view must match Python's own bytes."""
    text = "xy" * 200
    for _offset, view in unaligned_views(text, offsets=(offset,)):
        assert bytes(view) == text[offset:].encode()


@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_concat(length: int, rng: Random):
    """`String + String` (the concat/copy kernel) must match Python's `+`, over random/tiled/all-same-char
    left-hand operands at every SIMD-relevant length."""
    right = get_random_string(rng, length=37)
    for left in differential_bodies(length, rng):
        oracle = (left + right).encode()
        assert bytes(String(left) + String(right)) == oracle
        assert bytes(String(left) + right) == oracle


@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_backend_differential_translate(backend_results, length: int, rng: Random):
    """`sz.translate` (the lookup kernel) must agree across every backend and match a scalar oracle, for
    full-body and windowed (overlapping/odd `start`/`end`) translations, both copying and in-place."""
    for body in differential_bodies(length, rng):
        body_bytes = body.encode()
        for table in TRANSLATE_TABLES:
            full_oracle = lookup_table_oracle(body_bytes, table)
            assert_backends_agree(
                backend_results(
                    lambda capabilities, body_bytes=body_bytes, table=table: sz.translate(
                        body_bytes, table, capabilities=capabilities
                    )
                ),
                oracle=full_oracle,
            )

            for start, stop in safe_windows(length):
                # Copying mode returns only the translated [start:stop) slice; in-place mode mutates that
                # same window within the full-length buffer, leaving the rest untouched.
                windowed_slice_oracle = lookup_table_oracle(body_bytes[start:stop], table)
                windowed_full_oracle = bytearray(body_bytes)
                windowed_full_oracle[start:stop] = windowed_slice_oracle

                def windowed_copy(capabilities, body_bytes=body_bytes, table=table, start=start, stop=stop):
                    return sz.translate(body_bytes, table, False, start, stop, capabilities=capabilities)

                assert_backends_agree(backend_results(windowed_copy), oracle=windowed_slice_oracle)

                def windowed_inplace(capabilities, body_bytes=body_bytes, table=table, start=start, stop=stop):
                    mutable = bytearray(body_bytes)
                    sz.translate(mutable, table, True, start, stop, capabilities=capabilities)
                    return bytes(mutable)

                assert_backends_agree(backend_results(windowed_inplace), oracle=bytes(windowed_full_oracle))


@pytest.mark.parametrize("offset", [0, 1, 3, 7, 15])
def test_unit_backend_differential_translate_unaligned(backend_results, offset):
    """Translating a misaligned (sliced) view must agree across every backend."""
    text = "ab" * 200
    for _offset, view in unaligned_views(text, offsets=(offset,)):
        oracle = lookup_table_oracle(str(view).encode(), _TRANSLATE_INVERT_TABLE)
        assert_backends_agree(
            backend_results(
                lambda capabilities, view=view: sz.translate(view, _TRANSLATE_INVERT_TABLE, capabilities=capabilities)
            ),
            oracle=oracle,
        )


@pytest.mark.parametrize("kind", ESCAPE_KINDS)
def test_unit_backend_differential_escape(backend_results, kind: str):
    """`Strings.escape` and `Strings.unescape` must agree across every backend, whose byte-set searches skip the bytes
    needing no escape, with a special byte at either end and in the middle of every SIMD-relevant length."""
    texts = []
    for length in VECTOR_WIDTH_LENGTHS:
        body = "a" * length
        for special in '"\\,\n<%\x01\x7f\u202e':
            texts += [special + body, body + special, body[: length // 2] + special + body[length // 2 :]]
    strings = Strings(texts)

    def escape_and_back(capabilities):
        escaped = strings.escape(kind, capabilities=capabilities)
        unescaped = escaped.unescape(kind, capabilities=capabilities)
        return [bytes(text) for text in escaped], [str(text) for text in unescaped]

    results = backend_results(escape_and_back)
    assert_backends_agree(results)
    assert next(iter(results.values()))[1] == texts


@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
@pytest.mark.parametrize("nonce", [0, 1, 42, 314159])
def test_unit_backend_differential_fill_random(backend_results, length, nonce):
    """`sz.fill_random` (the fill+lookup kernel) must produce identical bytes from every backend for a
    fixed nonce; there is no Python oracle, so backend self-agreement is the only correctness signal."""

    def fill(capabilities):
        buffer = bytearray(length)
        sz.fill_random(buffer, nonce=nonce, capabilities=capabilities)
        return bytes(buffer)

    assert_backends_agree(backend_results(fill))


@pytest.mark.parametrize("start, end", [(1, 9), (3, 7), (0, 1), (15, 16), (7, 64)])
def test_unit_backend_differential_fill_random_windowed(backend_results, start, end):
    """`sz.fill_random` over an odd/overlapping [start, end) window inside a larger buffer must agree
    across every backend, both inside the filled window and on the untouched borders."""
    buffer_length = 96
    nonce = 2026

    def fill(capabilities):
        buffer = bytearray(buffer_length)
        sz.fill_random(buffer, nonce=nonce, start=start, end=end, capabilities=capabilities)
        return bytes(buffer)

    assert_backends_agree(backend_results(fill))


# endregion Backend differential
