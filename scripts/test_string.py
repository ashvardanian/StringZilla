#!/usr/bin/env python3
"""String class & container tests: Str/Strs construction, indexing, comparisons, slicing, the buffer
protocol, split/strip, translate, fill_random, decoding, invalid-UTF-8 handling, and PyArrow/4GB interop.

Mirrors the C++ scripts/test_string.cpp translation unit.
"""

import os
import sys
import math
import tempfile
from string import ascii_lowercase
from typing import Dict, Sequence

import pytest

import stringzilla as sz
from stringzilla import Str, Strs

from test_helpers import (
    SEED_VALUES,
    seed_random_generators,
    get_random_string,
    numpy_available,
    pyarrow_available,
)

# NumPy / PyArrow are optional; the defensive naked `except` mirrors the old monolith (PyPy raises non-ImportError).
try:
    import numpy as np
except:  # noqa: E722
    pass

try:
    import pyarrow as pa
except:  # noqa: E722
    pass


def test_library_properties():
    assert len(sz.__version__.split(".")) == 3, "Semantic versioning must be preserved"
    assert "serial" in sz.__capabilities__, "Serial backend must be present"
    assert isinstance(sz.__capabilities_str__, str) and len(sz.__capabilities_str__) > 0
    sz.reset_capabilities(sz.__capabilities__)  # Should not raise


@pytest.mark.parametrize("native_type", [str, bytes, bytearray])
def test_unit_construct(native_type):
    native = "aaaaa"
    if native_type is bytes or native_type is bytearray:
        native = native_type(native, "utf-8")
    big = Str(native)
    assert len(big) == len(native)


def test_str_repr():
    native = "abcdef"
    big = Str(native)
    assert repr(big) == f"sz.Str({repr(native)})"
    assert str(big) == native


def test_unit_indexing():
    native = "abcdef"
    big = Str(native)
    for i in range(len(native)):
        assert big[i] == native[i]


def test_unit_str_rich_comparisons():
    # Equality
    assert Str("aa") == "aa"
    assert Str("a") != "b"
    assert Str("abc") == Str("abc")
    assert Str("abc") != Str("abd")

    # Less than and less than or equal to
    assert Str("aa") < "b"
    assert Str("ab") <= "ab"
    assert Str("a") < Str("b")
    assert Str("abc") <= Str("abcd")

    # Greater than and greater than or equal to
    assert Str("b") > "aa"
    assert Str("ab") >= "ab"
    assert Str("b") > Str("a")
    assert Str("abcd") >= Str("abc")

    # Slicing and comparisons
    s2 = Str("abb")
    assert s2[1:] == "bb"
    assert s2[:-1] == "ab"
    assert s2[-1:] == "b"
    assert s2[1:] != "abb"
    assert s2[:-2] == "a"
    assert s2[-2:] == "bb"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_unit_buffer_protocol():
    """Tests weather conversion to and from the buffer protocol works as expected."""

    # Convert from StringZilla string to NumPy array through buffer protocol
    sz_str = Str("hello")
    np_array = np.array(sz_str)
    assert np_array.dtype == np.dtype("c")
    assert np_array.shape == (len("hello"),)
    assert "".join([c.decode("utf-8") for c in np_array.tolist()]) == "hello"

    # Convert from NumPy array to StringZilla string through buffer protocol
    np_array = np.arange(ord("a"), ord("z"), dtype=np.uint8)
    sz_str = sz.Str(memoryview(np_array))
    assert len(np_array) == len(sz_str)

    # Make sure multi-dimensional contiguous arrays are supported for image processing
    np_array = np.arange(ord("a"), ord("a") + 8, dtype=np.uint8).reshape((2, 2, 2))
    sz_str = sz.Str(memoryview(np_array))
    assert np_array.size == len(sz_str)

    # Expect errors if the data is not contiguous
    np_array = np.arange(ord("a"), ord("z"), dtype=np.uint8)[::2]
    with pytest.raises(ValueError):
        sz.Str(memoryview(np_array))

    # Expect errors if the data is not passed through a `memoryview`
    with pytest.raises(TypeError):
        sz.Str(np.array())
    # Expect errors if the data is not string-like
    with pytest.raises(TypeError):
        sz.Str(dict())


def test_str_write_to():
    native = "line1\nline2\nline3"
    big = Str(native)

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
    native = "line1\nline2\nline3"
    big = Str(native)

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

    # If the passed separator is an empty string, the library must raise a `ValueError`
    with pytest.raises(ValueError):
        sz.split(big, "")
    with pytest.raises(ValueError):
        sz.rsplit(big, "")
    with pytest.raises(ValueError):
        sz.split_byteset(big, "")
    with pytest.raises(ValueError):
        sz.rsplit_byteset(big, "")


def test_unit_split_skip_empty():
    """`skip_empty=True` drops zero-length segments from byteset/substring splits (default keeps them)."""

    def strs(x):
        return [str(s) for s in x]

    # Substring separator: leading/middle/trailing empties are dropped.
    assert strs(Str("a,,b,").split(",", skip_empty=True)) == ["a", "b"]
    assert strs(Str(",a,,b,").split(",", skip_empty=True)) == ["a", "b"]
    assert strs(Str("a,,b,").split(",")) == ["a", "", "b", ""]  # default: keep empties

    # Reverse eager split reports source order (reverses internally).
    assert strs(Str("a,,b,").rsplit(",", skip_empty=True)) == ["a", "b"]
    assert strs(Str("a,,b,").rsplit(",")) == ["a", "", "b", ""]

    # Byteset variants.
    assert strs(Str("a,;b;").split_byteset(",;", skip_empty=True)) == ["a", "b"]
    assert strs(Str("a,;b;").rsplit_byteset(",;", skip_empty=True)) == ["a", "b"]
    assert strs(Str("a,;b;").split_byteset(",;")) == ["a", "", "b", ""]

    # Lazy iterator variants (rsplit_iter yields reverse order).
    assert strs(sz.split_iter("a,,b,", ",", skip_empty=True)) == ["a", "b"]
    assert strs(sz.rsplit_iter("a,,b,", ",", skip_empty=True)) == ["b", "a"]
    assert strs(sz.split_byteset_iter("a,;b;", ",;", skip_empty=True)) == ["a", "b"]
    assert strs(sz.rsplit_byteset_iter("a,;b;", ",;", skip_empty=True)) == ["b", "a"]

    # `skip_empty` composes with `maxsplit`: only non-empty leading segments consume the budget.
    assert strs(Str("a,,b,c").split(",", maxsplit=1, skip_empty=True)) == ["a", ",b,c"]

    # An all-separator string collapses to nothing when skipping empties.
    assert strs(Str(",,,").split(",", skip_empty=True)) == []
    assert strs(sz.split_iter(",,,", ",", skip_empty=True)) == []


def test_unit_split_iterators():
    """
    Test the iterator-based split methods.
    This is slightly different from `split` and `rsplit` in that it returns an iterator instead of a list.
    Moreover, the native `rsplit` and even `rsplit_byteset` report results in the identical order to `split`
    and `split_byteset`. Here `rsplit_iter` reports elements in the reverse order, compared to `split_iter`.
    """
    native = "line1\nline2\nline3"
    big = Str(native)

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

    # If the passed separator is an empty string, the library must raise a `ValueError`
    with pytest.raises(ValueError):
        sz.split_iter(big, "")
    with pytest.raises(ValueError):
        sz.rsplit_iter(big, "")
    with pytest.raises(ValueError):
        sz.split_byteset_iter(big, "")
    with pytest.raises(ValueError):
        sz.rsplit_byteset_iter(big, "")


def test_unit_strip():
    # Test with whitespace (default behavior)
    native_whitespace = "  \t\n hello world \r\f\v  "
    big_whitespace = Str(native_whitespace)

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
    big_custom = Str(native_custom)
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
    big_empty = Str(empty)
    assert empty.strip() == str(sz.strip(big_empty))
    assert empty.strip() == str(big_empty.strip())

    # String with only whitespace
    only_whitespace = " \t\n\r\f\v "
    big_only_whitespace = Str(only_whitespace)
    assert only_whitespace.strip() == str(sz.strip(big_only_whitespace))
    assert only_whitespace.strip() == str(big_only_whitespace.strip())

    # String with only custom chars
    only_custom = "aaabbbccc"
    big_only_custom = Str(only_custom)
    assert only_custom.strip("abc") == str(sz.strip(big_only_custom, "abc"))
    assert only_custom.strip("abc") == str(big_only_custom.strip("abc"))

    # String with no chars to strip
    no_strip = "hello world"
    big_no_strip = Str(no_strip)
    assert no_strip.strip() == str(sz.strip(big_no_strip))
    assert no_strip.strip() == str(big_no_strip.strip())

    # Test with bytes
    native_bytes = b"  hello world  "
    big_bytes = Str(native_bytes)
    assert native_bytes.strip() == bytes(sz.strip(big_bytes))
    assert native_bytes.strip() == bytes(big_bytes.strip())

    # Test asymmetric stripping
    native_asymmetric = "aaahello worldbbb"
    big_asymmetric = Str(native_asymmetric)

    # Only strip 'a' from left
    assert native_asymmetric.lstrip("a") == str(sz.lstrip(big_asymmetric, "a"))
    assert native_asymmetric.lstrip("a") == str(big_asymmetric.lstrip("a"))

    # Only strip 'b' from right
    assert native_asymmetric.rstrip("b") == str(sz.rstrip(big_asymmetric, "b"))
    assert native_asymmetric.rstrip("b") == str(big_asymmetric.rstrip("b"))

    # Test with special characters
    native_special = "!!!###hello world***!!!"
    big_special = Str(native_special)
    special_chars = "!#*"

    assert native_special.strip(special_chars) == str(sz.strip(big_special, special_chars))
    assert native_special.strip(special_chars) == str(big_special.strip(special_chars))

    # Test with single character
    native_single = "aaa"
    big_single = Str(native_single)
    assert native_single.strip("a") == str(sz.strip(big_single, "a"))
    assert native_single.strip("a") == str(big_single.strip("a"))

    # Test with no matching characters
    native_no_match = "hello world"
    big_no_match = Str(native_no_match)
    assert native_no_match.strip("xyz") == str(sz.strip(big_no_match, "xyz"))
    assert native_no_match.strip("xyz") == str(big_no_match.strip("xyz"))


def test_unit_slicing():
    native = "abcdef"
    big = Str(native)
    assert big[1:3] == "bc"
    assert big[1:] == "bcdef"
    assert big[:3] == "abc"
    assert big[-1:] == "f"
    assert big[:-1] == "abcde"
    assert big[-3:] == "def"
    assert big[:-3] == "abc"


def test_unit_strs_rich_comparisons():
    arr: Strs = Str("a b c d e f g h").split()

    # Test against another Strs object
    identical_arr: Strs = Str("a b c d e f g h").split()
    different_arr: Strs = Str("a b c d e f g i").split()
    shorter_arr: Strs = Str("a b c d e").split()
    longer_arr: Strs = Str("a b c d e f g h i j").split()

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


def test_unit_strs_sequence_slicing():
    native = "1, 2, 3, 4, 5, 6"
    big = Str(native)
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

    # Test edge cases with pre-constructed Strs objects passed as slices
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


def test_string_lengths():
    assert 4 == len(sz.Str("abcd"))
    assert 8 == len(sz.Str("αβγδ"))


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
    assert byte_string.decode(encoding) == expected
    assert sz.Str(byte_string).decode(encoding) == expected


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
    with pytest.raises(UnicodeDecodeError):
        byte_string.decode(encoding)
    with pytest.raises(UnicodeDecodeError):
        sz.Str(byte_string).decode(encoding)


def baseline_translate(body: str, lut: Sequence) -> str:
    return "".join([chr(lut[ord(c)]) for c in body])


def translation_table_to_dict(lut: Sequence) -> Dict[int, str]:
    """Convert lookup table to translation dict for str.translate()"""
    return {i: chr(lut[i]) for i in range(256)}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("length", range(1, 300))
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_translations(length: int, seed_value: int):
    seed_random_generators(seed_value)

    map_identity = np.arange(256, dtype=np.uint8)
    map_invert = np.arange(255, -1, -1, dtype=np.uint8)
    map_threshold = np.where(np.arange(256) > 127, 255, 0).astype(np.uint8)
    dict_identity = translation_table_to_dict(map_identity)
    dict_invert = translation_table_to_dict(map_invert)
    dict_threshold = translation_table_to_dict(map_threshold)
    view_identity = memoryview(map_identity)
    view_invert = memoryview(map_invert)
    view_threshold = memoryview(map_threshold)

    body = get_random_string(length=length)
    body_bytes = body.encode("utf-8")

    # Check mapping strings and byte-strings into new strings
    assert sz.translate(body, view_identity) == body
    assert sz.translate(body_bytes, view_identity) == body_bytes
    assert sz.translate(body_bytes, view_identity) == body_bytes.translate(view_identity)
    assert sz.translate(body_bytes, view_invert) == body_bytes.translate(view_invert)
    assert sz.translate(body_bytes, view_threshold) == body_bytes.translate(view_threshold)

    # Check in-place translations on mutable byte-arrays - all of them return nothing
    after_identity = bytearray(body_bytes)
    assert sz.translate(after_identity, view_identity, inplace=True) is None
    assert sz.equal(after_identity, body.translate(dict_identity))
    after_invert = bytearray(body_bytes)
    assert sz.translate(after_invert, view_invert, inplace=True) is None
    assert sz.equal(after_invert, body.translate(dict_invert))
    after_threshold = bytearray(body_bytes)
    assert sz.translate(after_threshold, view_threshold, inplace=True) is None
    assert sz.equal(after_threshold, body.translate(dict_threshold))


@pytest.mark.parametrize("length", list(range(0, 300)) + [1024, 4096, 100000])
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_translations_random(length: int, seed_value: int):
    seed_random_generators(seed_value)
    body = get_random_string(length=length)
    lut = np.random.randint(0, 256, size=256, dtype=np.uint8)
    assert sz.translate(body, memoryview(lut)) == baseline_translate(body, lut)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_fill_random_slice(seed_value: int):
    # Prepare a zeroed buffer and keep a copy for comparison
    original = bytearray(64)
    updated_in_slices = bytearray(original)

    # Fill only a slice [start:end) deterministically
    start, end = 10, 30
    sz.fill_random(updated_in_slices, nonce=seed_value, start=start, end=end)

    # Unchanged prefix and suffix
    assert bytes(updated_in_slices[:start]) == bytes(original[:start])
    assert bytes(updated_in_slices[end:]) == bytes(original[end:])

    # Changed inner region
    assert bytes(updated_in_slices[start:end]) != bytes(original[start:end])


def test_fill_random_different_nonces():
    first_buffer = bytearray(64)
    second_buffer = bytearray(64)
    sz.fill_random(first_buffer, nonce=1)
    sz.fill_random(second_buffer, nonce=2)
    assert bytes(first_buffer) != bytes(second_buffer)


@pytest.mark.parametrize("length", [0, 1, 7, 64])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_fill_random_alphabet(length: int, seed_value: int):

    # Same nonce should produce the same result
    random_string = sz.random(length, nonce=seed_value)
    same_nonce_random_string = sz.random(length, nonce=seed_value)
    assert isinstance(random_string, (bytes, bytearray))
    assert len(random_string) == length
    assert random_string == same_nonce_random_string

    # With alphabet: all bytes must belong to alphabet
    alphabet = b"0123456789"
    random_digits = sz.random(128, nonce=seed_value, alphabet=alphabet)
    assert set(random_digits).issubset(set(alphabet))


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_str_to_pyarrow_conversion():
    native = "hello"
    big = Str(native)
    assert isinstance(big.address, int) and big.address != 0
    assert isinstance(big.nbytes, int) and big.nbytes == len(native)

    arrow_buffer = pa.foreign_buffer(big.address, big.nbytes, big)
    assert arrow_buffer.to_pybytes() == native.encode("utf-8")


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_to_pyarrow_conversion():
    """Test PyArrow property getters for Strs with tape-based layouts."""
    # Test with a list of strings (should create U32_TAPE layout by default)
    native_list = ["hello", "world", "test", "", "python"]
    strs = Strs(native_list)

    # Check that all properties return valid values
    assert isinstance(strs.tape_address, int) and strs.tape_address != 0, "tape_address should be non-zero"
    assert isinstance(strs.offsets_address, int) and strs.offsets_address != 0, "offsets_address should be non-zero"
    assert isinstance(strs.tape_nbytes, int) and strs.tape_nbytes >= 0, "tape_nbytes should be non-negative"
    assert isinstance(strs.offsets_nbytes, int) and strs.offsets_nbytes > 0, "offsets_nbytes should be positive"
    assert isinstance(strs.offsets_are_large, bool), "offsets_are_large should be a boolean"

    # Calculate expected tape size (sum of all string lengths)
    expected_tape_nbytes = sum(len(s) for s in native_list)
    assert (
        strs.tape_nbytes == expected_tape_nbytes
    ), f"Expected tape_nbytes={expected_tape_nbytes}, got {strs.tape_nbytes}"

    # For 5 strings, we should have 6 offsets (N+1 format)
    # Offsets should be either 4 bytes (u32) or 8 bytes (u64) each
    expected_offsets_count = len(native_list) + 1
    if strs.offsets_are_large:
        expected_offsets_nbytes = expected_offsets_count * 8
    else:
        expected_offsets_nbytes = expected_offsets_count * 4
    assert (
        strs.offsets_nbytes == expected_offsets_nbytes
    ), f"Expected offsets_nbytes={expected_offsets_nbytes}, got {strs.offsets_nbytes}"

    # Create PyArrow buffers from the properties
    tape_buffer = pa.foreign_buffer(strs.tape_address, strs.tape_nbytes, strs)
    offsets_buffer = pa.foreign_buffer(strs.offsets_address, strs.offsets_nbytes, strs)

    # Verify the tape contains the concatenated strings
    concatenated = "".join(native_list)
    assert tape_buffer.to_pybytes() == concatenated.encode("utf-8"), "Tape should contain concatenated strings"

    # Create an Arrow array from the buffers
    if strs.offsets_are_large:
        arrow_array = pa.Array.from_buffers(pa.large_string(), len(native_list), [None, offsets_buffer, tape_buffer])
    else:
        arrow_array = pa.Array.from_buffers(pa.string(), len(native_list), [None, offsets_buffer, tape_buffer])

    # Verify the Arrow array matches the original data
    assert arrow_array.to_pylist() == native_list, "Arrow array should match original list"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_pyarrow_empty():
    """Test PyArrow properties with empty Strs - auto-converts FRAGMENTED to tape."""
    strs = Strs([])

    # Empty Strs starts as FRAGMENTED but tape accessors auto-convert to tape layout
    # For empty, we get valid values but with 0/empty content
    tape_addr = strs.tape_address
    assert tape_addr is not None and tape_addr >= 0

    offsets_addr = strs.offsets_address
    assert offsets_addr is not None and offsets_addr >= 0

    tape_bytes = strs.tape_nbytes
    assert tape_bytes == 0  # No strings = no tape content

    offsets_bytes = strs.offsets_nbytes
    assert offsets_bytes == 4  # Arrow uses N+1 offsets, so 1 offset for 0 strings (u32 = 4 bytes)

    are_large = strs.offsets_are_large
    assert are_large == False  # Default to u32 offsets


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_pyarrow_large_strings():
    """Test PyArrow properties with strings that might require u64 offsets."""
    # Create strings with total size that could trigger u64 layout
    large_string = "x" * 10000
    native_list = [large_string, "small", large_string, ""]
    strs = Strs(native_list)

    # Check properties work correctly
    assert isinstance(strs.offsets_are_large, bool), "offsets_are_large should be boolean"
    assert strs.tape_nbytes == sum(len(s) for s in native_list), "tape_nbytes should match total length"

    expected_offsets_count = len(native_list) + 1
    offset_size = 8 if strs.offsets_are_large else 4
    assert strs.offsets_nbytes == expected_offsets_count * offset_size, "offsets_nbytes should be correct"

    # Create PyArrow array and verify
    tape_buffer = pa.foreign_buffer(strs.tape_address, strs.tape_nbytes, strs)
    offsets_buffer = pa.foreign_buffer(strs.offsets_address, strs.offsets_nbytes, strs)

    if strs.offsets_are_large:
        arrow_array = pa.Array.from_buffers(pa.large_string(), len(native_list), [None, offsets_buffer, tape_buffer])
    else:
        arrow_array = pa.Array.from_buffers(pa.string(), len(native_list), [None, offsets_buffer, tape_buffer])

    assert arrow_array.to_pylist() == native_list, "Arrow array should match original list"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_pyarrow_fragmented_conversion():
    """Test that FRAGMENTED layout auto-converts to tape when accessing tape properties."""
    # Create a tape-based Strs from split
    text = Str("apple banana cherry date")
    tape_strs = text.split(" ")

    # shuffled() returns a FRAGMENTED layout
    fragmented_strs = tape_strs.shuffled(seed=42)

    # Verify we can access tape properties (triggers conversion)
    tape_addr = fragmented_strs.tape_address
    assert tape_addr > 0, "tape_address should be valid"

    offsets_addr = fragmented_strs.offsets_address
    assert offsets_addr > 0, "offsets_address should be valid"

    tape_bytes = fragmented_strs.tape_nbytes
    expected_bytes = sum(len(str(fragmented_strs[i])) for i in range(len(fragmented_strs)))
    assert tape_bytes == expected_bytes, f"tape_nbytes should be {expected_bytes}, got {tape_bytes}"

    offsets_bytes = fragmented_strs.offsets_nbytes
    expected_offsets = (len(fragmented_strs) + 1) * 4  # u32 offsets
    assert offsets_bytes == expected_offsets, f"offsets_nbytes should be {expected_offsets}, got {offsets_bytes}"

    are_large = fragmented_strs.offsets_are_large
    assert are_large == False, "Small strings should use u32 offsets"

    # Verify we can create a valid PyArrow array from the converted data
    tape_buffer = pa.foreign_buffer(fragmented_strs.tape_address, fragmented_strs.tape_nbytes, fragmented_strs)
    offsets_buffer = pa.foreign_buffer(fragmented_strs.offsets_address, fragmented_strs.offsets_nbytes, fragmented_strs)

    arrow_array = pa.Array.from_buffers(pa.string(), len(fragmented_strs), [None, offsets_buffer, tape_buffer])

    # The data should match (order is shuffled)
    arrow_list = arrow_array.to_pylist()
    original_list = [str(fragmented_strs[i]) for i in range(len(fragmented_strs))]
    assert arrow_list == original_list, "Arrow array should match shuffled strings"


@pytest.mark.parametrize("container_class", [tuple, list, iter])
@pytest.mark.parametrize("view", [False, True])
def test_strs_from_python_basic(container_class: type, view: bool):
    """Test basic conversion from Python containers to Strs."""
    base_items = ["hello", "world", "test", " ", "from", " ", "container", ""]
    container = container_class(base_items)

    # Skip iter+view combination as it's not supported
    if container_class == iter and view:
        with pytest.raises(ValueError, match="View mode.*not supported for iterators"):
            Strs(container, view=view)
        return

    strs = Strs(container, view=view)

    assert len(strs) == len(base_items)
    assert strs[0] == "hello"
    assert strs[1] == "world"
    assert strs[2] == "test"
    assert strs[3] == " "
    assert strs[4] == "from"
    assert strs[5] == " "
    assert strs[6] == "container"
    assert strs[7] == ""


UINT32_MAX = 2**32 - 1  # ! Many of the 32/64-bit algo corner cases happen at this input size


def long_repeated_string(ctypes, fill_char: str, string_size: int) -> str:
    buffer = ctypes.create_string_buffer(string_size)
    ctypes.memset(buffer, ord(fill_char), string_size)
    return buffer.value.decode("ascii")


@pytest.mark.skipif(sys.maxsize <= 2**32, reason="64-bit system required for 4GB+ test")
def test_strs_from_4gb_list():
    """Test Strs with >4GB array of strings to verify 32-bit to 64-bit layout transition.
    This will require over 8 GB of memory. To stress-test the behavior, limit memory per process. For 5 and 13 GB:

    ulimit -v 9437184 && uv run --no-project python -m pytest scripts/test_stringzilla.py -s -x -k 4gb_list
    ulimit -v 13631488 && uv run --no-project python -m pytest scripts/test_stringzilla.py -s -x -k 4gb_list
    """

    try:
        import gc
        import ctypes
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
        parts_stringzilla = Strs(parts_pythonic)

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
def test_strs_from_4gb_generator():
    """Test Strs with >4GB of strings streams to verify 32-bit to 64-bit layout transition.
    This will require over 8 GB of memory. To stress-test the behavior, limit memory per process. For 5 and 13 GB:

    ulimit -v 5242880 && uv run --no-project python -m pytest scripts/test_stringzilla.py -s -x -k 4gb_generator
    ulimit -v 13631488 && uv run --no-project python -m pytest scripts/test_stringzilla.py -s -x -k 4gb_generator
    """

    try:
        import gc
        import ctypes
    except ImportError:
        pytest.skip("ctypes & gc not available (e.g., PyPy)")

    # Each individual string won't be very large, but many of them will be used
    part_size = 64 * 1024 * 1024
    parts_count = math.ceil(UINT32_MAX / part_size) + 1  # Ensures we exceed UINT32_MAX by a small margin
    try:
        parts_stringzilla = Strs(
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


@pytest.mark.parametrize("container_class", [tuple, list, iter])
@pytest.mark.parametrize("view", [False, True])
def test_strs_reference_counting(container_class: type, view: bool):
    """Test reference counting to prevent memory leaks."""

    # CPython-only: PyPy and other interpreters may not expose refcounts or use a different GC model
    if not hasattr(sys, "getrefcount"):
        pytest.skip("Reference counting semantics are not available")

    import gc

    base_items = ["ref", "count", "test"]
    container = container_class(base_items)

    # Skip iter+view combination as it's not supported
    if container_class == iter and view:
        with pytest.raises(ValueError, match="View mode.*not supported for iterators"):
            Strs(container, view=view)
        return

    initial_refcount = sys.getrefcount(container)

    strs = Strs(container, view=view)
    during_refcount = sys.getrefcount(container)

    # For iterators, we can't check refcount behavior the same way since iter() creates a new object
    # and the iterator consumes the original container during iteration
    if container_class != iter:
        # View mode should increment refcount, copy mode should not
        if view:
            assert during_refcount == initial_refcount + 1, "View mode should increment refcount"
        else:
            assert during_refcount == initial_refcount, "Copy mode should not change refcount"

    # Verify functionality
    assert len(strs) == 3
    assert strs[0] == "ref"
    assert strs[1] == "count"
    assert strs[2] == "test"

    del strs
    gc.collect()

    if container_class != iter:
        final_refcount = sys.getrefcount(container)
        assert final_refcount == initial_refcount, "Refcount should return to initial value"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
@pytest.mark.parametrize("view", [False, True])
def test_strs_from_arrow_basic(view: bool):
    """Test basic conversion from Arrow string array to Strs."""
    arrow_array = pa.array(["hello", "world", "test", "arrow"])
    strs = Strs(arrow_array, view=view)

    assert len(strs) == 4
    assert strs[0] == "hello"
    assert strs[1] == "world"
    assert strs[2] == "test"
    assert strs[3] == "arrow"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_from_arrow_binary_array():
    """Test conversion from Arrow binary array."""
    binary_data = [b"hello", b"world", b"binary", b"data"]
    arrow_array = pa.array(binary_data, type=pa.binary())

    strs = Strs(arrow_array)

    assert len(strs) == 4
    # Strs should handle binary data properly - compare as bytes
    for i, expected in enumerate(binary_data):
        str_bytes = strs[i].encode("latin-1") if isinstance(strs[i], str) else bytes(strs[i])
        assert str_bytes == expected


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_from_arrow_large_strings():
    """Test conversion from Arrow large string array."""
    arrow_array = pa.array(["hello", "world", "large", "strings"], type=pa.large_string())

    strs = Strs(arrow_array)

    assert len(strs) == 4
    assert strs[0] == "hello"
    assert strs[1] == "world"
    assert strs[2] == "large"
    assert strs[3] == "strings"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_from_arrow_error_cases():
    """Test error handling for invalid inputs."""
    # Test with non-iterable object
    with pytest.raises(TypeError):
        Strs(123)  # Integer is not iterable

    with pytest.raises(TypeError):
        Strs(None)  # None is not iterable

    # Test with non-string Arrow array
    int_array = pa.array([1, 2, 3, 4])
    with pytest.raises((TypeError, ValueError)):
        Strs(int_array)


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_from_arrow_c_interface():
    """Test the low-level Arrow C Data Interface."""
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
    strs = Strs(arrow_array)
    assert len(strs) == 3
    assert strs[0] == "test"
    assert strs[1] == "c"
    assert strs[2] == "interface"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_strs_from_arrow_with_nulls():
    """Test conversion from Arrow array with null values (validity bits)."""
    # Create an array with None values
    arrow_array = pa.array(["hello", None, "world", None, "test"])

    strs = Strs(arrow_array)

    assert len(strs) == 5
    assert strs[0] == "hello"
    assert strs[1] == ""  # None values should be converted to empty strings
    assert strs[2] == "world"
    assert strs[3] == ""  # None values should be converted to empty strings
    assert strs[4] == "test"

    # Test with all nulls
    all_nulls = pa.array([None, None, None], type=pa.string())
    strs_nulls = Strs(all_nulls)
    assert len(strs_nulls) == 3
    assert all(s == "" for s in strs_nulls)

    # Test with mixed None and empty strings
    mixed = pa.array(["", None, "hello", "", None, "world"])
    strs_mixed = Strs(mixed)
    assert len(strs_mixed) == 6
    assert strs_mixed[0] == ""
    assert strs_mixed[1] == ""
    assert strs_mixed[2] == "hello"
    assert strs_mixed[3] == ""
    assert strs_mixed[4] == ""
    assert strs_mixed[5] == "world"


def test_invalid_utf8_handling():
    """Test that both Str and Strs handle invalid UTF-8 bytes gracefully."""

    # Test arrays with invalid UTF-8 sequences
    test_arrays = [
        # Use `bytes.fromhex()` to avoid putting binary literals in source code
        # This prevents PyTest's source parsing from encountering invalid UTF-8
        [b"hello", bytes.fromhex("80") + b"world", b"valid"],  # Mixed valid/invalid
        [bytes.fromhex("fffe"), bytes.fromhex("80"), bytes.fromhex("f4908080")],  # All invalid
        [b"normal", b"string with " + bytes.fromhex("80") + b" bytes"],  # Partial invalid
    ]

    for test_array in test_arrays:
        strs_obj = Strs(test_array)

        # These should never raise exceptions
        repr_result = repr(strs_obj)
        str_result = str(strs_obj)

        # Basic assertions
        assert isinstance(repr_result, str)
        assert isinstance(str_result, str)
        assert len(repr_result) > 0
        assert len(str_result) > 0
