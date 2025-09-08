#!/usr/bin/env python3
"""
Test suite for StringZilla package.
For full coverage, preinstall NumPy and PyArrow.
To run locally:

    uv pip install numpy pyarrow pytest pytest-repeat
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest scripts/test_stringzilla.py -s -x

Recommended flags for better diagnostics:

    -s                  show test output (no capture)
    -vv                 verbose output
    --maxfail=1         stop at first failure
    --full-trace        full Python tracebacks
    -k <pattern>        filter tests by substring
    -X faulthandler     to dump on fatal signals
    --verbose           enable verbose output

Example:

    uv pip install -e . --force-reinstall --no-build-isolation --verbose
    uv run --no-project python -X faulthandler -m pytest scripts/test_stringzilla.py -s -vv --maxfail=1 --full-trace
"""

import os
import sys
import math
import tempfile
import platform
from random import choice, randint, seed
from string import ascii_lowercase
from typing import Optional, Sequence, Dict

import pytest

import stringzilla as sz
from stringzilla import Str, Strs

# NumPy is available on most platforms and is required for most tests.
# ! When using PyPy on some platforms NumPy has internal issues, that will
# ! raise a weird error, not an `ImportError`. That's why we intentionally
# ! use a naked `except:`. Necessary evil!
try:
    import numpy as np

    numpy_available = True
except:  # noqa: E722
    # NumPy is not installed, most tests will be skipped
    numpy_available = False


# PyArrow is not available on most platforms.
# ! When using PyPy on some platforms PyArrow has internal issues, that will
# ! raise a weird error, not an `ImportError`. That's why we intentionally
# ! use a naked `except:`. Necessary evil!
try:
    import pyarrow as pa

    pyarrow_available = True
except:  # noqa: E722
    # PyArrow is not installed, most tests will be skipped
    pyarrow_available = False

# Reproducible test seeds for consistent CI runs (keep in sync with test_stringzillas.py)
SEED_VALUES = [
    42,  # Classic test seed
    0,  # Edge case: zero seed
    1,  # Minimal positive seed
    314159,  # Pi digits
]


@pytest.fixture(scope="session", autouse=True)
def log_test_environment():
    """Automatically log environment info before running any tests."""

    print()  # New line for better readability
    print("=== StringZilla Test Environment ===")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"StringZilla version: {sz.__version__}")
    print(f"StringZilla capabilities: {sorted(sz.__capabilities__)}")
    print(f"NumPy available: {numpy_available}")
    if numpy_available:
        print(f"NumPy version: {np.__version__}")
    print(f"PyArrow available: {pyarrow_available}")
    if pyarrow_available:
        print(f"PyArrow version: {pa.__version__}")

    # If QEMU is indicated via env (e.g., set by pyproject), mask out SVE/SVE2 to avoid emulation flakiness.
    is_qemu = os.environ.get("SZ_IS_QEMU_", "").lower() in ("1", "true", "yes", "on")
    if is_qemu:
        sve_like = {"sve", "sve2", "sve2+aes"}
        current = list(getattr(sz, "__capabilities__", ()))
        desired = tuple(c for c in current if c.lower() not in sve_like)
        if len(desired) != len(current):
            print(f"QEMU env detected; disabling {sve_like} for stability")
            sz.reset_capabilities(desired)

    print("=" * 40)
    print()  # New line for better readability


def seed_random_generators(seed_value: Optional[int] = None):
    """Seed Python and NumPy RNGs for reproducibility."""
    if seed_value is None:
        return
    seed(seed_value)
    # Try to seed NumPy's random number generator
    # This handles both NumPy 1.x and 2.x, and any import issues
    if numpy_available:
        try:
            np.random.seed(seed_value)
        except (ImportError, AttributeError, Exception):
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


def test_unit_count():
    native = "aaaaa"
    big = Str(native)
    assert big.count("a") == 5
    assert big.count("aa") == 2
    assert big.count("aa", allowoverlap=True) == 4


def test_unit_count_byteset():
    native = "abcdef"
    big = Str(native)

    assert big.count_byteset("abc") == 3  # a, b, c
    assert big.count_byteset("xyz") == 0  # no matches
    assert big.count_byteset("aeiou") == 2  # a and e

    # Empty inputs
    assert sz.count_byteset("", "abc") == 0
    assert sz.count_byteset("abc", "") == 0
    assert sz.count_byteset("", "") == 0

    # Single character set
    assert sz.count_byteset("hello", "l") == 2
    assert sz.count_byteset("hello", "x") == 0

    # Repeated patterns
    assert sz.count_byteset("mississippi", "si") == 8  # s:4, i:4 total
    assert sz.count_byteset("aaaaaa", "a") == 6

    # Test start/end bounds
    native = "abcdefghij"
    big = Str(native)

    assert big.count_byteset("abc", 0, 3) == 3  # "abc"
    assert big.count_byteset("abc", 1, 3) == 2  # "bc"
    assert big.count_byteset("abc", 3) == 0  # "defghij"
    assert big.count_byteset("hij", 7) == 3  # "hij"
    assert big.count_byteset("hij", -3) == 3  # last 3 chars "hij"
    assert big.count_byteset("abc", 0, -7) == 3  # first 3 chars "abc"

    # Test edge cases
    assert sz.count_byteset("a", "a", 0, 0) == 0  # empty slice
    assert sz.count_byteset("abc", "abc", 10, 20) == 0  # out of bounds
    assert sz.count_byteset("abc", "abc", -10, -5) == 0  # negative out of bounds


def test_unit_contains():
    big = Str("abcdef")
    assert "a" in big
    assert "ab" in big
    assert "xxx" not in big


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


def test_unit_strs_sequence():
    native = "p3\np2\np1"
    big = Str(native)

    lines = big.splitlines()
    assert [2, 1, 0] == list(lines.argsort())
    assert "p3" in lines
    assert "p4" not in lines

    assert repr(lines) == "sz.Strs(['p3', 'p2', 'p1'])"
    assert repr(Str("a" * 1_000_000).split()).endswith("... ])")

    assert str(lines) == "['p3', 'p2', 'p1']"
    assert str(Str("a" * 1_000_000).split()).startswith("['aaa")
    assert str(Str("a" * 1_000_000).split()).endswith("aaa']")

    lines_sorted = lines.sorted()
    assert [0, 1, 2] == list(lines_sorted.argsort())
    assert ["p1", "p2", "p3"] == list(lines_sorted)

    # Reverse order
    assert [2, 1, 0] == list(lines_sorted.argsort(reverse=True))
    lines_sorted_reverse = lines.sorted(reverse=True)
    assert ["p3", "p2", "p1"] == list(lines_sorted_reverse)

    # Sampling an array
    sampled = lines.sample(100, seed=42)
    assert "p3" in sampled
    assert "p4" not in sampled


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


def test_unit_globals():
    """Validates that the previously unit-tested member methods are also visible as global functions."""

    assert sz.find("abcdef", "bcdef") == 1
    assert sz.find("abcdef", "x") == -1
    assert sz.rfind("abcdef", "bcdef") == 1
    assert sz.rfind("abcdef", "x") == -1

    # Corner-cases for `find` and `rfind`, when we pass empty strings
    assert sz.find("abcdef", "") == "abcdef".find("")
    assert sz.rfind("abcdef", "") == "abcdef".rfind("")
    assert sz.find("abcdef", "", 1) == "abcdef".find("", 1)
    assert sz.rfind("abcdef", "", 1) == "abcdef".rfind("", 1)
    assert sz.find("abcdef", "", 1, 3) == "abcdef".find("", 1, 3)
    assert sz.rfind("abcdef", "", 1, 3) == "abcdef".rfind("", 1, 3)
    assert sz.find("", "abcdef") == "".find("abcdef")
    assert sz.rfind("", "abcdef") == "".rfind("abcdef")

    assert sz.find("Hello, world!", "world", 0, 11) == "Hello, world!".find("world", 0, 11)
    assert sz.rfind("Hello, world!", "world", 0, 11) == "Hello, world!".rfind("world", 0, 11)

    assert sz.find_first_of("abcdef", "cde") == 2
    assert sz.find_first_of("abcdef", "xyz") == -1
    assert sz.find_first_of("hello world", "aeiou") == 1

    assert sz.find_last_of("abcdef", "abc") == 2
    assert sz.find_last_of("abcdef", "xyz") == -1
    assert sz.find_last_of("hello world", "aeiou") == 7

    assert sz.find_first_not_of("aaabbbccc", "ab") == 6
    assert sz.find_first_not_of("abcdef", "abcdef") == -1
    assert sz.find_first_not_of("   hello", " ") == 3

    assert sz.find_last_not_of("aaabbbccc", "bc") == 2
    assert sz.find_last_not_of("abcdef", "abcdef") == -1
    assert sz.find_last_not_of("hello   ", " ") == 4

    # Test byteset counting
    assert sz.count_byteset("abcdef", "abc") == 3
    assert sz.count_byteset("abcdef", "xyz") == 0
    assert sz.count_byteset("hello world", "aeiou") == 3  # e, o, o
    assert sz.count_byteset("mississippi", "si") == 8  # s, i, s, s, i, s, s, i
    assert sz.count_byteset("", "abc") == 0
    assert sz.count_byteset("abc", "") == 0
    assert sz.count_byteset("abcdef", "abc", 1, 4) == 2  # bc in "bcd"
    assert sz.count_byteset("hello world", "aeiou", 2) == 2  # o, o
    assert sz.count_byteset("hello world", "aeiou", 0, 5) == 2  # e, o in "hello"

    # Compare partitioning functions
    assert sz.partition("abcdef", "c") == ("ab", "c", "def")
    assert sz.rpartition("abcdef", "c") == ("ab", "c", "def")

    with pytest.raises(ValueError):
        sz.partition("abcdef", "")
    with pytest.raises(ValueError):
        sz.rpartition("abcdef", "")

    assert sz.count("abcdef", "x") == 0
    assert sz.count("aaaaa", "a") == 5
    assert sz.count("aaaaa", "aa") == 2
    assert sz.count("aaaaa", "aa", allowoverlap=True) == 4

    assert sz.bytesum("hello") > 0
    assert sz.bytesum("hello") > sz.bytesum("hell")

    assert sz.translate("ABC", {"A": "X", "B": "Y", "C": "Z"}) == "XYZ"
    assert sz.translate("ABC", {"A": "X", "B": "Y"}) == "XYC"
    assert sz.translate("ABC", {"A": "X", "B": "Y"}, start=1, end=-1) == "YC"
    assert sz.translate("ABC", bytes(range(256))) == "ABC"
    with pytest.raises(TypeError):
        sz.translate("ABC", {"A": "X", "B": "Y"}, start=1, end=-1, inplace=True)

    mutable_buffer = bytearray(b"ABC")
    assert sz.fill_random(mutable_buffer) is None
    assert sz.fill_random(mutable_buffer, 42) is None

    assert sz.split("hello world test", " ") == ["hello", "world", "test"]
    assert sz.rsplit("hello world test", " ", 1) == ["hello world", "test"]


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


def test_slice_of_split():
    def impl(native_str: str):
        native_split = native_str.split()
        text = sz.Str(native_str)
        sz_split = text.split()
        for slice_idx in range(len(native_split)):
            assert str(sz_split[slice_idx]) == native_split[slice_idx]

    native_str = "Weebles wobble before they fall down, don't they?"
    impl(native_str)
    # ~5GB to overflow 32-bit sizes
    copies = int(len(native_str) / 5e9)
    # Eek. Cover 64-bit indices
    impl(native_str * copies)


def get_random_string(
    length: Optional[int] = None,
    variability: Optional[int] = None,
) -> str:
    if length is None:
        length = randint(3, 300)
    if variability is None:
        variability = len(ascii_lowercase)
    return "".join(choice(ascii_lowercase[:variability]) for _ in range(length))


def is_equal_strings(native_strings, big_strings):
    for native_slice, big_slice in zip(native_strings, big_strings):
        assert native_slice == big_slice, f"Mismatch between `{native_slice}` and `{str(big_slice)}`"


def check_identical(
    native: str,
    big: Str,
    needle: Optional[str] = None,
    check_iterators: bool = False,
):
    if needle is None:
        part_offset = randint(0, len(native) - 1)
        part_length = randint(1, len(native) - part_offset)
        needle = native[part_offset:part_length]

    present_in_native: bool = needle in native
    present_in_big = needle in big
    assert present_in_native == present_in_big
    assert native.find(needle) == big.find(needle)
    assert native.rfind(needle) == big.rfind(needle)
    assert native.count(needle) == big.count(needle)

    # Check that the `start` and `stop` positions are correctly inferred
    len_half = len(native) // 2
    len_quarter = len(native) // 4
    assert native.find(needle, len_half) == big.find(needle, len_half)
    assert native.find(needle, len_quarter, 3 * len_quarter) == big.find(needle, len_quarter, 3 * len_quarter)

    # Check splits and other sequence operations
    native_strings = native.split(needle)
    big_strings: Strs = big.split(needle)
    assert len(native_strings) == len(big_strings)

    if check_iterators:
        for i in range(len(native_strings)):
            assert len(native_strings[i]) == len(big_strings[i])
            assert (
                native_strings[i] == big_strings[i]
            ), f"Mismatch between `{native_strings[i]}` and `{str(big_strings[i])}`"
            assert [c for c in native_strings[i]] == [c for c in big_strings[i]]

    is_equal_strings(native_strings, big_strings)


@pytest.mark.parametrize("repetitions", range(1, 10))
def test_fuzzy_repetitions(repetitions: int):
    native = "abcd" * repetitions
    big = Str(native)

    check_identical(native, big, "a", True)
    check_identical(native, big, "ab", True)
    check_identical(native, big, "abc", True)
    check_identical(native, big, "abcd", True)
    check_identical(native, big, "abcde", True)  # Missing pattern


@pytest.mark.parametrize("pattern_length", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("haystack_length", range(1, 65))
@pytest.mark.parametrize("variability", range(1, 25))
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_fuzzy_substrings(pattern_length: int, haystack_length: int, variability: int, seed_value: int):
    seed_random_generators(seed_value)
    native = get_random_string(variability=variability, length=haystack_length)
    big = Str(native)
    pattern = get_random_string(variability=variability, length=pattern_length)
    assert (pattern in native) == big.contains(
        pattern
    ), f"Failed to check if {pattern} at offset {native.find(pattern)} is present in {native}"
    assert native.find(pattern) == big.find(
        pattern
    ), f"Failed to locate {pattern} at offset {native.find(pattern)} in {native}"


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


@pytest.mark.parametrize("body", ["", "hello", "world", "abcdefg", "a" * 32])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hash_basic_equivalence(body: str, seed_value: int):
    # TODO: Add streaming hashers and compare slices vs overall
    hash_seeded = sz.hash(body, seed=seed_value)
    hash_member = sz.Str(body).hash(seed=seed_value)
    assert hash_seeded == hash_member


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hasher_incremental_vs_one_shot(seed_value: int):
    data_full = b"hello world"
    data_prefix = b"hello "
    data_suffix = b"world"

    hasher = sz.Hasher(seed=seed_value)
    hasher.update(data_prefix)
    hasher.update(data_suffix)
    streamed_hash = hasher.digest()

    expected_hash = sz.hash(data_full, seed=seed_value)
    assert isinstance(streamed_hash, int)
    assert streamed_hash == expected_hash


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hasher_reset_and_hexdigest(seed_value: int):
    data = b"some test payload"
    hasher = sz.Hasher(seed=seed_value)
    hasher.update(data)
    streamed_hash = hasher.digest()
    streamed_hex = hasher.hexdigest()
    assert isinstance(streamed_hex, str) and len(streamed_hex) == 16 and streamed_hex == format(streamed_hash, "016x")

    hasher.reset()
    hasher.update(data)
    re_streamed_hash = hasher.digest()
    re_streamed_hex = hasher.hexdigest()
    assert streamed_hash == re_streamed_hash
    assert streamed_hex == re_streamed_hex


@pytest.mark.parametrize("length", list(range(0, 300)) + [1024, 4096, 100000])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_bytesum_random(length: int, seed_value: int):
    def sum_bytes(body: str) -> int:
        return sum([ord(c) for c in body])

    seed_random_generators(seed_value)
    body = get_random_string(length=length)
    assert sum_bytes(body) == sz.bytesum(body)


@pytest.mark.parametrize("list_length", [10, 20, 30, 40, 50])
@pytest.mark.parametrize("part_length", [5, 10])
@pytest.mark.parametrize("variability", [2, 3])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_fuzzy_sorting(list_length: int, part_length: int, variability: int, seed_value: int):
    seed_random_generators(seed_value)
    native_list = [get_random_string(variability=variability, length=part_length) for _ in range(list_length)]
    native_joined = ".".join(native_list)
    big_joined = Str(native_joined)
    big_list = big_joined.split(".")

    # Before testing sorting, validate pairwise comparator consistency
    def py_cmp(a: str, b: str) -> int:
        return -1 if a < b else (1 if a > b else 0)

    def sz_cmp(a: str, b: str) -> int:
        sa, sb = Str(a), Str(b)
        if sa < sb:
            return -1
        if sa > sb:
            return 1
        return 0

    # Check every consecutive pair a[i], a[i+1]
    for i in range(len(native_list) - 1):
        a, b = native_list[i], native_list[i + 1]
        assert py_cmp(a, b) == sz_cmp(a, b), f"Comparator mismatch at {i}: '{a}' vs '{b}'"

    native_ordered = sorted(native_list)
    native_order = big_list.argsort()
    for i in range(list_length):
        assert native_ordered[i] == native_list[native_order[i]], "Order is wrong"
        assert native_ordered[i] == str(big_list[int(native_order[i])]), "Split is wrong?!"

    native_list.sort()
    big_list = big_list.sorted()

    assert len(native_list) == len(big_list)
    for native_str, big_str in zip(native_list, big_list):
        assert native_str == str(big_str), "Order is wrong"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_str_to_pyarrow_conversion():
    native = "hello"
    big = Str(native)
    assert isinstance(big.address, int) and big.address != 0
    assert isinstance(big.nbytes, int) and big.nbytes == len(native)

    arrow_buffer = pa.foreign_buffer(big.address, big.nbytes, big)
    assert arrow_buffer.to_pybytes() == native.encode("utf-8")


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


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", "-s", __file__]))
