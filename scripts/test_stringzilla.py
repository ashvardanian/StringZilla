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
import hashlib
import hmac
import unicodedata
from random import choice, randint, seed
from string import ascii_lowercase
from typing import Optional, Sequence, Dict

import pytest

# Import shared Unicode data loading functions
from test_helpers import (
    UnicodeDataDownloadError,
    get_uncased_folding_rules,
    get_word_break_test_cases,
    get_grapheme_break_properties,
    get_grapheme_break_test_cases,
    baseline_grapheme_boundaries,
    get_sentence_break_properties,
    get_sentence_break_test_cases,
    baseline_sentence_boundaries,
    get_line_break_test_cases,
)

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

# Generate a random seed at module load time for this test run (keep in sync with test_stringzillas.py)
# Use SystemRandom for true randomness independent of the seeded RNG state
_random_seed_for_run = int.from_bytes(os.urandom(4), "little")

# Reproducible test seeds for consistent CI runs
SEED_VALUES = [
    42,  # Classic test seed
    0,  # Edge case: zero seed
    1,  # Minimal positive seed
    314159,  # Pi digits
    _random_seed_for_run,  # Random seed for this run (logged at startup)
]

# Override SEED_VALUES with environment variable if set (for reproducible CI fuzzing)
_env_seed = os.environ.get("SZ_TESTS_SEED")
if _env_seed:
    try:
        _parsed_seed = int(_env_seed)
        SEED_VALUES = [_parsed_seed]
        print(f"SZ_TESTS_SEED={_parsed_seed} (from environment, overriding default seeds)")
    except ValueError:
        pass  # Keep default SEED_VALUES if parsing fails


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
    print(f"Test seeds: {SEED_VALUES}")
    if _random_seed_for_run in SEED_VALUES:
        print(f"  (random seed for this run: {_random_seed_for_run})")

    # If QEMU is indicated via env (e.g., set by pyproject), mask out SVE/SVE2 to avoid emulation flakiness.
    is_qemu = os.environ.get("SZ_IS_QEMU_", "").lower() in ("1", "true", "yes", "on")
    if is_qemu:
        sve_like = {"sve", "sve2", "sve2aes"}
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

    # Top-K partial sort: only the leading `top` elements are returned.
    assert ["p1", "p2"] == list(lines.sorted(top=2))
    assert [0, 1] == list(lines.argsort(top=2, reverse=True))  # two largest: p3, p2
    assert list(lines.sorted()) == list(lines.sorted(top=999))  # `top` beyond the count sorts everything

    # Uncased, stable ordering (fold-equal strings keep input order).
    mixed = Str("Banana\napple\nBANANA\nApple").splitlines()
    assert ["apple", "Apple", "Banana", "BANANA"] == list(mixed.sorted(uncased=True))
    assert [1, 3, 0, 2] == list(mixed.argsort(uncased=True))
    assert ["Banana", "BANANA", "apple", "Apple"] == list(mixed.sorted(uncased=True, reverse=True))

    # Keyword-only: positional arguments are rejected.
    try:
        lines.sorted(True)
        assert False, "sorted() should reject positional arguments"
    except TypeError:
        pass

    # Sampling an array
    sampled = lines.sample(100, seed=42)
    assert "p3" in sampled
    assert "p4" not in sampled


def test_unit_strs_argsort_out():
    """`argsort(out=...)` writes the permutation into a caller `uint64` buffer and returns it."""
    import array

    strs = Str("banana\napple\ncherry\nApple\nBANANA").splitlines()
    n = len(strs)

    # `array('Q')` matches `sz_sorted_idx_t` (unsigned 64-bit); out is filled and returned.
    buf = array.array("Q", [0] * n)
    assert strs.argsort(uncased=True, out=buf) is buf
    assert tuple(buf) == strs.argsort(uncased=True)

    # Top-K writes exactly `top` indices and leaves the rest of a wider buffer untouched.
    wide = array.array("Q", [12345] * n)
    strs.argsort(top=2, out=wide)
    assert tuple(wide[:2]) == strs.argsort(top=2)
    assert all(x == 12345 for x in wide[2:]), "argsort(out=...) clobbered past `top`"

    # Rejections: wrong itemsize, undersized, read-only, and `sorted()` has no `out=`.
    with pytest.raises(TypeError):
        strs.argsort(out=array.array("I", [0] * n))
    with pytest.raises(ValueError):
        strs.argsort(out=array.array("Q", [0] * (n - 1)))
    with pytest.raises((TypeError, BufferError)):
        strs.argsort(out=bytes(8 * n))
    with pytest.raises(TypeError):
        strs.sorted(out=buf)


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


@pytest.mark.parametrize("body", ["", "x", "hello", "abcdefg", "a" * 17, "a" * 32, "a" * 64, "a" * 100])
def test_hash_multiseed_equivalence(body: str):
    from array import array

    seeds_list = [0, 1, 42, 314159, 7, 8, 9, 10, 11]  # > 4 to exercise the 4-wide tail handling
    seeds = array("Q", seeds_list)
    expected = tuple(sz.hash(body, s) for s in seeds_list)

    # Tuple-returning form, standalone and as a member of `Str`
    assert sz.hash_multiseed(body, seeds) == expected
    assert sz.Str(body).hash_multiseed(seeds) == expected

    # Output-buffer form fills in place and returns None
    out = array("Q", [0] * len(seeds))
    assert sz.hash_multiseed(body, seeds, out=out) is None
    assert tuple(out) == expected


def test_hash_multiseed_errors():
    from array import array

    seeds = array("Q", [1, 2, 3])
    with pytest.raises(TypeError):  # A plain list of ints is not a uint64 buffer
        sz.hash_multiseed("x", [1, 2, 3])
    with pytest.raises(TypeError):  # Wrong item size (32-bit) is rejected
        sz.hash_multiseed("x", array("I", [1, 2, 3]))
    with pytest.raises(ValueError):  # Output buffer too small for the seed count
        sz.hash_multiseed("x", seeds, out=array("Q", [0]))


@pytest.mark.parametrize("length", list(range(0, 300)) + [1024, 4096, 100000])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_bytesum_random(length: int, seed_value: int):
    def sum_bytes(body: str) -> int:
        return sum([ord(c) for c in body])

    seed_random_generators(seed_value)
    body = get_random_string(length=length)
    assert sum_bytes(body) == sz.bytesum(body)


@pytest.mark.parametrize("length", [0, 1, 3, 7, 15, 31, 63, 64, 65, 127, 128, 129, 255, 256, 1000, 4096, 10000])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_sha256(length: int, seed_value: int):

    seed_random_generators(seed_value)
    text = get_random_string(length=length)
    expected = hashlib.sha256(text.encode()).digest()

    # One-shot: standalone function
    assert sz.sha256(text) == expected

    # One-shot: Str method
    assert Str(text).sha256() == expected

    # One-shot: bytes input
    assert sz.sha256(text.encode()) == expected

    # Progressive: single update
    h = sz.Sha256()
    h.update(text)
    assert h.digest() == expected
    assert h.hexdigest() == expected.hex()

    # Progressive: chunked updates
    if length > 0:
        h = sz.Sha256()
        chunk_size = max(1, length // 3)
        for i in range(0, length, chunk_size):
            h.update(text[i : i + chunk_size])
        assert h.digest() == expected
        assert h.hexdigest() == expected.hex()

    # Reset
    h.reset().update(text)
    assert h.digest() == expected

    # Copy
    mid = length // 2
    h1 = sz.Sha256().update(text[:mid])
    h2 = h1.copy()
    h1.update(text[mid:])
    h2.update(text[mid:])
    assert h1.digest() == h2.digest() == expected


@pytest.mark.parametrize("key_length", [0, 1, 16, 32, 64, 65, 128])
@pytest.mark.parametrize("message_length", [0, 1, 63, 64, 65, 127, 128, 1000])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hmac_sha256(key_length: int, message_length: int, seed_value: int):

    seed_random_generators(seed_value)
    key = get_random_string(length=key_length).encode()
    message = get_random_string(length=message_length).encode()

    # Test against Python's hmac module
    expected = hmac.new(key, message, hashlib.sha256).digest()
    result = sz.hmac_sha256(key, message)
    assert result == expected

    # Test with string inputs
    key_str = key.decode("latin1")
    message_str = message.decode("latin1")
    result_str = sz.hmac_sha256(key_str, message_str)
    assert result_str == expected


def test_hmac_sha256_kwargs():
    """Test hmac_sha256 with keyword arguments"""
    key = b"secret"
    message = b"Hello, world!"

    # Test against Python's hmac module
    expected = hmac.new(key, message, hashlib.sha256).digest()

    # Test with positional arguments
    result_positional = sz.hmac_sha256(key, message)
    assert result_positional == expected

    # Test with keyword arguments (as shown in README line 483)
    result_kwargs = sz.hmac_sha256(key=key, message=message)
    assert result_kwargs == expected

    # Test with mixed arguments
    result_mixed = sz.hmac_sha256(key, message=message)
    assert result_mixed == expected

    # Test with reversed keyword arguments
    result_reversed = sz.hmac_sha256(message=message, key=key)
    assert result_reversed == expected

    # Missing argument
    with pytest.raises(TypeError, match="expects exactly 2 arguments"):
        sz.hmac_sha256(key=key)

    # Duplicate argument
    with pytest.raises(TypeError, match="key specified twice"):
        sz.hmac_sha256(key, key=key)

    # Unknown keyword argument (only detected when total args == 2)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        sz.hmac_sha256(key=key, unknown=b"test")

    # Too many arguments (3 args)
    with pytest.raises(TypeError, match="expects exactly 2 arguments"):
        sz.hmac_sha256(key=key, message=message, unknown=b"test")


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

    # The buffer-protocol `out=` path must produce the same permutation as the tuple path.
    if numpy_available:
        out = np.zeros(list_length, dtype=np.uintp)
        returned = big_list.argsort(out=out)
        assert returned is out
        assert out.tolist() == list(native_order)

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


def test_unit_utf8_uncased_fold():
    """Test basic case folding functionality."""
    # ASCII
    assert sz.utf8_uncased_fold("HELLO") == b"hello"
    assert sz.utf8_uncased_fold("Hello World") == b"hello world"
    assert sz.utf8_uncased_fold("") == b""
    assert sz.utf8_uncased_fold("already lowercase") == b"already lowercase"

    # German sharp S expansion (ß → ss)
    assert sz.utf8_uncased_fold("Straße") == b"strasse"
    assert sz.utf8_uncased_fold("GROSSE") == b"grosse"

    # Method form on Str
    assert sz.Str("HELLO").utf8_uncased_fold() == b"hello"

    # Bytes input
    assert sz.utf8_uncased_fold(b"HELLO") == b"hello"


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("A", b"a"),
        ("Z", b"z"),
        ("ß", b"ss"),  # German sharp S (U+00DF)
        ("ẞ", b"ss"),  # Capital sharp S (U+1E9E)
        ("ﬁ", b"fi"),  # fi ligature (U+FB01)
        ("ﬀ", b"ff"),  # ff ligature (U+FB00)
        ("ﬂ", b"fl"),  # fl ligature (U+FB02)
        ("ﬃ", b"ffi"),  # ffi ligature (U+FB03)
        ("ﬄ", b"ffl"),  # ffl ligature (U+FB04)
        ("Σ", "σ".encode("utf-8")),  # Greek Sigma
        ("Ω", "ω".encode("utf-8")),  # Greek Omega
        ("Ä", "ä".encode("utf-8")),  # German umlaut
        ("É", "é".encode("utf-8")),  # French accent
        ("Ñ", "ñ".encode("utf-8")),  # Spanish tilde
    ],
)
def test_utf8_uncased_fold_expansions(input_str, expected):
    """Test case folding with specific known transformations including expansions."""
    assert sz.utf8_uncased_fold(input_str) == expected


def test_unit_utf8_norm():
    """Test basic Unicode normalization functionality."""

    # ASCII is unchanged by all forms
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        assert sz.utf8_norm("hello", form) == b"hello"
        assert sz.utf8_norm("", form) == b""

    # NFC vs NFD differ for precomposed characters
    # "café" with precomposed é (U+00E9)
    cafe_nfc = "café"
    cafe_nfd = "café"  # e + combining acute accent
    # NFC of precomposed → precomposed bytes
    assert sz.utf8_norm(cafe_nfc, "NFC") == cafe_nfc.encode("utf-8")
    # NFD of precomposed → decomposed bytes
    assert sz.utf8_norm(cafe_nfc, "NFD") == cafe_nfd.encode("utf-8")
    # NFC of decomposed → precomposed bytes (composition)
    assert sz.utf8_norm(cafe_nfd, "NFC") == cafe_nfc.encode("utf-8")

    # NFKD of fi ligature (U+FB01) decomposes to "fi"
    assert sz.utf8_norm("ﬁ", "NFKD") == b"fi"
    assert sz.utf8_norm("ﬁ", "NFKC") == b"fi"

    # Idempotence: normalizing an already-normalized string gives the same result
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        once = sz.utf8_norm("café", form)
        twice = sz.utf8_norm(once, form)
        assert once == twice, f"Idempotence failed for form {form}"

    # Method form on Str
    assert sz.Str("café").utf8_norm("NFD") == "café".encode("utf-8")

    # Bytes input
    assert sz.utf8_norm("café".encode("utf-8"), "NFC") == "café".encode("utf-8")

    # Cross-check with Python unicodedata.normalize
    for sample in ("café", "Ångström", "ﬁle", "ẛ̣"):
        for form in ("NFC", "NFD", "NFKC", "NFKD"):
            expected = unicodedata.normalize(form, sample).encode("utf-8")
            got = sz.utf8_norm(sample, form)
            assert got == expected, f"Mismatch for {repr(sample)} form={form}: {got!r} != {expected!r}"


@pytest.mark.parametrize(
    "input_str, form, expected",
    [
        ("ﬁ", "NFKD", b"fi"),  # fi ligature compatibility decomposition
        ("ﬁ", "NFKC", b"fi"),  # fi ligature compatibility composition
        ("ﬀ", "NFKD", b"ff"),  # ff ligature
        ("ﬂ", "NFKD", b"fl"),  # fl ligature
        ("café", "NFD", "café".encode("utf-8")),  # precomposed → decomposed
        ("café", "NFC", "café".encode("utf-8")),  # decomposed → precomposed
        ("Å", "NFD", "Å".encode("utf-8")),  # Å → A + ring above
    ],
)
def test_utf8_norm_expansions(input_str, form, expected):
    """Test normalization with specific known transformations."""
    assert sz.utf8_norm(input_str, form) == expected


def test_unit_utf8_norm_violation():
    """Test normalization violation detection."""
    # Pure ASCII is always normalized in every form
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        assert sz.utf8_norm_violation("hello", form) is None
        assert sz.utf8_norm_violation("", form) is None

    # A precomposed character (U+00E9 é) is valid NFC but is a violation for NFD
    precomposed = "café"  # NFC: already composed
    assert sz.utf8_norm_violation(precomposed, "NFC") is None
    nfd_violation = sz.utf8_norm_violation(precomposed, "NFD")
    assert nfd_violation is not None
    assert isinstance(nfd_violation, int)
    assert nfd_violation >= 0

    # A decomposed sequence (e + combining accent) is valid NFD but violates NFC
    decomposed = "café"  # NFD: e + combining acute
    assert sz.utf8_norm_violation(decomposed, "NFD") is None
    nfc_violation = sz.utf8_norm_violation(decomposed, "NFC")
    assert nfc_violation is not None
    assert isinstance(nfc_violation, int)

    # Method form on Str
    assert sz.Str("hello").utf8_norm_violation("NFC") is None

    # Bytes input
    assert sz.utf8_norm_violation(b"hello", "NFD") is None

    # fi ligature (U+FB01) is not NFKD-normalized
    assert sz.utf8_norm_violation("ﬁ", "NFKD") is not None
    # But "fi" as two plain ASCII chars IS NFKD-normalized
    assert sz.utf8_norm_violation("fi", "NFKD") is None

    # ValueError on unknown form
    import pytest as _pytest

    with _pytest.raises(ValueError, match="unknown form"):
        sz.utf8_norm_violation("hello", "XYZ")


def test_utf8_uncased_fold_all_codepoints():
    """Compare StringZilla case folding with Unicode 17.0 CaseFolding.txt rules.

    This test downloads the official Unicode 17.0 case folding data file to validate
    StringZilla's implementation, independent of Python's Unicode version.
    The file is cached in the system temp directory for subsequent runs.
    """
    # Load Unicode 17.0 case folding rules (downloads and caches automatically)
    try:
        unicode_folds = get_uncased_folding_rules("17.0.0")
    except UnicodeDataDownloadError as e:
        pytest.skip(f"Skipping due to network issue: {e}")

    mismatches = []
    missing_folds = []
    extra_folds = []

    for codepoint in range(0x110000):
        # Skip surrogates (not valid in UTF-8)
        if 0xD800 <= codepoint <= 0xDFFF:
            continue

        try:
            char = chr(codepoint)
            char_bytes = char.encode("utf-8")
            sz_folded = sz.utf8_uncased_fold(char)

            # Get expected folding from Unicode 17.0 rules
            # If not in the table, character maps to itself
            expected = unicode_folds.get(codepoint, char_bytes)

            if sz_folded != expected:
                entry = (f"U+{codepoint:04X}", repr(char), expected.hex(), sz_folded.hex())
                if codepoint in unicode_folds and sz_folded == char_bytes:
                    missing_folds.append(entry)  # StringZilla didn't fold but should have
                elif codepoint not in unicode_folds and sz_folded != char_bytes:
                    extra_folds.append(entry)  # StringZilla folded but shouldn't have
                else:
                    mismatches.append(entry)  # Both fold but to different targets
        except (ValueError, UnicodeEncodeError):
            continue

    total_errors = len(mismatches) + len(missing_folds) + len(extra_folds)
    assert total_errors == 0, (
        f"Case folding errors vs Unicode 17.0 ({len(unicode_folds)} rules): "
        f"{len(mismatches)} wrong targets, {len(missing_folds)} missing, {len(extra_folds)} extra. "
        f"First 10: {(mismatches + missing_folds + extra_folds)[:10]}"
    )


def _reference_uncased_find(haystack_bytes: bytes, needle_folded: bytes, unicode_folds: dict) -> int:
    """Reference uncased find from UCD rules: byte offset of the first
    haystack position whose codepoint folds concatenate to exactly `needle_folded`."""
    if not needle_folded:
        return 0
    text = haystack_bytes.decode("utf-8")
    codepoint_byte_offsets = []
    codepoint_folds = []
    byte_offset = 0
    for char in text:
        char_bytes = char.encode("utf-8")
        codepoint_byte_offsets.append(byte_offset)
        codepoint_folds.append(unicode_folds.get(ord(char), char_bytes))
        byte_offset += len(char_bytes)
    for start in range(len(text)):
        accumulated = b""
        for index in range(start, len(text)):
            accumulated += codepoint_folds[index]
            if len(accumulated) >= len(needle_folded):
                break
        if accumulated == needle_folded:
            return codepoint_byte_offsets[start]
    return -1


def test_utf8_uncased_find_preimages():
    """Adversarial probes from Unicode 17.0 CaseFolding.txt: for every codepoint whose
    fold differs from itself, the fold OUTPUT is what a user types as the needle — and
    the PREIMAGE is what hides in the haystack, placed at chunk-boundary offsets.
    """
    # Load Unicode 17.0 case folding rules (downloads and caches automatically)
    try:
        unicode_folds = get_uncased_folding_rules("17.0.0")
    except UnicodeDataDownloadError as e:
        pytest.skip(f"Skipping due to network issue: {e}")

    mismatches = []
    for codepoint in sorted(unicode_folds):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        preimage_bytes = chr(codepoint).encode("utf-8")
        folded_bytes = unicode_folds[codepoint]
        if preimage_bytes == folded_bytes:
            continue  # Identity folds carry no signal here
        for offset in (0, 30, 62, 63, 64):
            haystack = b"y" * offset + preimage_bytes + b"tail"
            expected = _reference_uncased_find(haystack, folded_bytes, unicode_folds)
            actual = sz.utf8_uncased_find(haystack, folded_bytes)
            if actual != expected:
                mismatches.append((f"U+{codepoint:04X}", offset, expected, actual))

    assert not mismatches, (
        f"{len(mismatches)} uncased find mismatches vs Unicode 17.0 reference. "
        f"First 10 (codepoint, offset, expected, actual): {mismatches[:10]}"
    )


def _reference_uncased_find_subset(haystack_bytes: bytes, needle_folded: bytes, unicode_folds: dict) -> int:
    """Independent fold-subset reference matching `sz.utf8_uncased_find`'s documented semantics:
    fold every haystack codepoint, concatenate into one folded blob, and find the earliest byte
    offset at which `needle_folded` appears as a substring. The match neither needs to start nor end
    on a codepoint boundary -- it may begin or end mid-expansion. The returned value is the byte
    offset (into the original haystack) of the codepoint that CONTAINS the first matched byte, or -1
    when the folded needle is absent."""
    if not needle_folded:
        return 0
    text = haystack_bytes.decode("utf-8")
    folded_blob = b""
    source_byte_offset_for_folded_byte = []  # Maps each folded byte to its source codepoint's byte offset
    byte_offset = 0
    for char in text:
        char_bytes = char.encode("utf-8")
        folded_run = unicode_folds.get(ord(char), char_bytes)
        folded_blob += folded_run
        source_byte_offset_for_folded_byte.extend([byte_offset] * len(folded_run))
        byte_offset += len(char_bytes)
    match_position = folded_blob.find(needle_folded)
    if match_position == -1:
        return -1
    return source_byte_offset_for_folded_byte[match_position]


def test_utf8_uncased_find_crossing_expansions():
    """Crossing-expansion probes: matches whose folded runes span the boundary between two
    adjacent expanding codepoints. For example needle `ſß` folds to "sss" and must be found
    inside haystack `ßß` (folding to "ssss") at offset 0; the match is not contained within a
    single haystack codepoint's expansion but crosses from one expansion into the next.
    Each case is swept across non-folding prefix paddings to exercise chunk-boundary handling.
    """
    # Load Unicode 17.0 case folding rules (downloads and caches automatically)
    try:
        unicode_folds = get_uncased_folding_rules("17.0.0")
    except UnicodeDataDownloadError as e:
        pytest.skip(f"Skipping due to network issue: {e}")

    def fold_needle(needle_str: str) -> bytes:
        """Fold a needle string into its concatenated codepoint folds, producing the
        `needle_folded` bytes that the fold-subset reference compares against."""
        folded = b""
        for char in needle_str:
            folded += unicode_folds.get(ord(char), char.encode("utf-8"))
        return folded

    # (haystack, needle) pairs whose folded forms overlap across expansion boundaries.
    crossing_cases = [
        ("ßß", "sss"),
        ("ßß", "ſß"),
        ("ẞß", "ssss"),
        ("ẞß", "sss"),
        ("ﬃ", "fi"),
        ("ﬃ", "ffi"),
        ("ﬀﬁ", "ffi"),
    ]
    filler_char = "z"  # Folds to itself, so it never participates in a match
    prefix_paddings = [0, 30, 62, 63, 64, 65]

    mismatches = []
    for haystack_str, needle_str in crossing_cases:
        needle_folded = fold_needle(needle_str)
        for padding in prefix_paddings:
            haystack_str_padded = filler_char * padding + haystack_str
            haystack_bytes = haystack_str_padded.encode("utf-8")
            needle_bytes = needle_str.encode("utf-8")
            expected = _reference_uncased_find_subset(haystack_bytes, needle_folded, unicode_folds)
            actual = sz.utf8_uncased_find(haystack_bytes, needle_bytes)
            if actual != expected:
                mismatches.append((haystack_str, needle_str, padding, expected, actual))

    assert not mismatches, (
        f"{len(mismatches)} crossing-expansion find mismatches vs Unicode 17.0 reference. "
        f"(haystack, needle, padding, expected, actual): {mismatches[:10]}"
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_uncased_fold_random_strings(seed_value: int):
    """Test case folding on random multi-codepoint strings."""
    seed(seed_value)

    # Test with ASCII uppercase
    for _ in range(50):
        length = randint(1, 100)
        test_str = "".join(chr(randint(0x41, 0x5A)) for _ in range(length))  # A-Z
        python_folded = test_str.casefold().encode("utf-8")
        sz_folded = sz.utf8_uncased_fold(test_str)
        assert python_folded == sz_folded, f"Mismatch for: {test_str!r}"

    # Test with Latin Extended characters
    for _ in range(50):
        length = randint(1, 50)
        # Mix of ASCII uppercase and Latin Extended (includes ß, etc.)
        codepoints = [randint(0x41, 0x5A) for _ in range(length)]
        codepoints += [randint(0xC0, 0xFF) for _ in range(length // 2)]
        test_str = "".join(chr(cp) for cp in codepoints)
        python_folded = test_str.casefold().encode("utf-8")
        sz_folded = sz.utf8_uncased_fold(test_str)
        assert python_folded == sz_folded, f"Mismatch for: {test_str!r}"


@pytest.mark.parametrize(
    "haystack, needle, expected",
    [
        # ASCII basic cases
        ("Hello World", "WORLD", 6),
        ("Hello World", "hello", 0),
        ("Hello World", "xyz", -1),
        ("HELLO", "hello", 0),
        ("hello", "HELLO", 0),
        ("HeLLo WoRLd", "world", 6),
        ("abcdef", "CD", 2),
        # Latin1 accented characters (C3 lead byte range)
        ("Über allen Gipfeln", "ÜBER", 0),
        ("ÜBER", "über", 0),
        ("Das schöne Mädchen", "SCHÖNE", 4),
        ("Café au lait", "CAFÉ", 0),
        ("naïve approach", "NAÏVE", 0),
        ("El niño juega", "NIÑO", 3),
        # German Eszett: ß ↔ ss (bidirectional)
        ("Straße", "STRASSE", 0),
        ("STRASSE", "straße", 0),
        ("die Straße", "STRASSE", 4),
        ("groß", "GROSS", 0),
        ("GROSS", "groß", 0),
        ("Fußball", "FUSSBALL", 0),
        # Cyrillic
        ("ПРИВЕТ", "привет", 0),
        ("привет", "ПРИВЕТ", 0),
        ("Москва столица", "МОСКВА", 0),
        ("добрый день", "ДОБРЫЙ", 0),
        # Greek
        ("ΑΒΓΔ", "αβγδ", 0),
        ("αβγδ", "ΑΒΓΔ", 0),
        ("Ελλάδα", "ΕΛΛΆΔΑ", 0),
        # Mixed scripts
        ("Hello Мир World", "МИР", 6),
        ("Café МОСКВА", "москва", 5),
        # Empty and edge cases
        ("hello", "", 0),
        ("", "x", -1),
        ("", "", 0),
        ("a", "A", 0),
        ("A", "a", 0),
    ],
)
def test_utf8_uncased_find(haystack, needle, expected):
    """Test uncased UTF-8 substring search with various scripts."""
    assert sz.utf8_uncased_find(haystack, needle) == expected


def test_utf8_uncased_find_method():
    """Test uncased find as a method on Str objects."""
    s = sz.Str("Hello World")
    assert s.utf8_uncased_find("WORLD") == 6
    assert s.utf8_uncased_find("hello") == 0
    assert s.utf8_uncased_find("xyz") == -1


def test_utf8_uncased_find_offsets():
    """Test that str returns codepoint offsets and bytes returns byte offsets."""
    # str: codepoint offsets
    # 'Hëllo' = 5 codepoints, 'Wörld' starts at codepoint 6
    assert sz.utf8_uncased_find("Hëllo Wörld", "WÖRLD") == 6
    assert sz.utf8_uncased_find("Café", "FÉ") == 2  # C, a = 2 codepoints

    # bytes: byte offsets
    # 'Hëllo' = H(1) + ë(2) + l(1) + l(1) + o(1) + space(1) = 7 bytes
    assert sz.utf8_uncased_find("Hëllo Wörld".encode(), "WÖRLD".encode()) == 7
    assert sz.utf8_uncased_find("Café".encode(), "FÉ".encode()) == 2


def test_utf8_uncased_find_start_end():
    """Test start/end parameters with codepoint and byte offsets."""
    # str: codepoint offsets
    text = "Hëllo Hëllo"  # 11 codepoints, 13 bytes
    assert sz.utf8_uncased_find(text, "HËLLO", 0) == 0
    assert sz.utf8_uncased_find(text, "HËLLO", 1) == 6  # Skip first
    assert sz.utf8_uncased_find(text, "HËLLO", 0, 5) == 0  # Within range
    assert sz.utf8_uncased_find(text, "HËLLO", 0, 4) == -1  # Too short

    # bytes: byte offsets
    text_bytes = text.encode()
    assert sz.utf8_uncased_find(text_bytes, "HËLLO".encode(), 0) == 0
    assert sz.utf8_uncased_find(text_bytes, "HËLLO".encode(), 1) == 7
    assert sz.utf8_uncased_find(text_bytes, "HËLLO".encode(), 0, 7) == 0
    assert sz.utf8_uncased_find(text_bytes, "HËLLO".encode(), 0, 5) == -1


def test_utf8_uncased_find_bytes():
    """Test uncased find with bytes input."""
    assert sz.utf8_uncased_find(b"Hello World", b"WORLD") == 6
    assert sz.utf8_uncased_find(b"Strasse", b"STRASSE") == 0
    assert sz.utf8_uncased_find("Straße".encode(), b"STRASSE") == 0
    assert sz.utf8_uncased_find("XXStraße".encode(), b"STRASSE") == 2


def test_utf8_uncased_find_slicing():
    """Verify returned index works correctly for slicing."""
    text = "Café au lait"
    idx = sz.utf8_uncased_find(text, "AU")
    assert text[idx : idx + 2].lower() == "au"

    text_bytes = text.encode()
    idx = sz.utf8_uncased_find(text_bytes, b"AU")
    assert text_bytes[idx : idx + 2].lower() == b"au"


@pytest.mark.parametrize(
    "haystack, needle, expected_matches",
    [
        # ASCII - multiple matches
        ("Hello HELLO hello HeLLo", "hello", ["Hello", "HELLO", "hello", "HeLLo"]),
        ("Hello World", "world", ["World"]),
        ("Hello World", "xyz", []),
        # German ß/ss expansion
        ("Straße STRASSE strasse", "strasse", ["Straße", "STRASSE", "strasse"]),
        ("groß GROSS", "gross", ["groß", "GROSS"]),
        # Cyrillic
        ("ПРИВЕТ привет Привет", "привет", ["ПРИВЕТ", "привет", "Привет"]),
        # Greek
        ("ΑΒΓΔ αβγδ", "αβγδ", ["ΑΒΓΔ", "αβγδ"]),
        # Edge cases
        ("", "hello", []),
        ("hello", "xyz", []),
    ],
)
def test_utf8_uncased_matches(haystack, needle, expected_matches):
    """Parametrized test for uncased find iterator."""
    matches = [str(m) for m in sz.utf8_uncased_matches(haystack, needle)]
    assert matches == expected_matches
    # Also test method form
    matches_method = [str(m) for m in sz.Str(haystack).utf8_uncased_matches(needle)]
    assert matches_method == expected_matches


def test_utf8_uncased_matches_overlapping():
    """Test overlapping vs non-overlapping modes and bytes input."""
    # Non-overlapping (default)
    assert len(list(sz.utf8_uncased_matches("aaaa", "aa"))) == 2
    # Overlapping
    assert len(list(sz.utf8_uncased_matches("aaaa", "aa", include_overlapping=True))) == 3
    # Bytes input
    assert len(list(sz.utf8_uncased_matches(b"Hello HELLO", b"hello"))) == 2


def test_utf8_uncased_order():
    """Test uncased UTF-8 comparison."""
    # Equal strings
    assert sz.utf8_uncased_order("hello", "HELLO") == 0
    assert sz.utf8_uncased_order("HELLO", "hello") == 0
    assert sz.utf8_uncased_order("HeLLo", "hElLO") == 0

    # German sharp S equivalence
    assert sz.utf8_uncased_order("Straße", "STRASSE") == 0
    assert sz.utf8_uncased_order("strasse", "STRAẞE") == 0

    # Less than
    assert sz.utf8_uncased_order("apple", "BANANA") < 0
    assert sz.utf8_uncased_order("APPLE", "banana") < 0
    assert sz.utf8_uncased_order("a", "B") < 0

    # Greater than
    assert sz.utf8_uncased_order("ZEBRA", "apple") > 0
    assert sz.utf8_uncased_order("zebra", "APPLE") > 0
    assert sz.utf8_uncased_order("z", "A") > 0

    # Empty strings
    assert sz.utf8_uncased_order("", "") == 0
    assert sz.utf8_uncased_order("", "a") < 0
    assert sz.utf8_uncased_order("a", "") > 0

    # Prefix ordering
    assert sz.utf8_uncased_order("hello", "HELLO WORLD") < 0
    assert sz.utf8_uncased_order("HELLO WORLD", "hello") > 0

    # Greek letters
    assert sz.utf8_uncased_order("ΑΒΓΔ", "αβγδ") == 0
    assert sz.utf8_uncased_order("αβγ", "ΑΒΓ") == 0

    # Cyrillic
    assert sz.utf8_uncased_order("ПРИВЕТ", "привет") == 0

    # Method form on Str
    s = sz.Str("hello")
    assert s.utf8_uncased_order("HELLO") == 0

    # bytes input
    assert sz.utf8_uncased_order(b"hello", b"HELLO") == 0
    assert sz.utf8_uncased_order(b"HELLO", b"hello") == 0
    assert sz.utf8_uncased_order(b"apple", b"BANANA") < 0
    assert sz.utf8_uncased_order(b"ZEBRA", b"apple") > 0

    # German sharp S with bytes
    assert sz.utf8_uncased_order("Straße".encode(), b"STRASSE") == 0
    assert sz.utf8_uncased_order(b"strasse", "Straße".encode()) == 0

    # Greek with bytes
    assert sz.utf8_uncased_order("ΑΒΓΔ".encode(), "αβγδ".encode()) == 0

    # Cyrillic with bytes
    assert sz.utf8_uncased_order("ПРИВЕТ".encode(), "привет".encode()) == 0


def test_unit_utf8_count():
    """Test UTF-8 character counting (codepoints, not bytes)."""
    # ASCII strings: len == utf8_count
    assert sz.utf8_count("hello") == 5
    assert sz.utf8_count("") == 0
    assert sz.utf8_count("a") == 1

    # Multi-byte UTF-8: character count != byte count
    assert sz.utf8_count("café") == 4  # é is 2 bytes but 1 char
    assert sz.utf8_count("日本語") == 3  # Each CJK char is 3 bytes
    assert sz.utf8_count("🎉") == 1  # Emoji is 4 bytes but 1 char
    assert sz.utf8_count("👨‍👩‍👧") == 5  # Family emoji: 3 people + 2 ZWJ

    # Mixed ASCII and multi-byte
    assert sz.utf8_count("hello世界") == 7  # 5 ASCII + 2 CJK
    assert sz.utf8_count("naïve") == 5  # ï is 2 bytes

    # Method form on Str
    assert sz.Str("café").utf8_count() == 4

    # Bytes input
    assert sz.utf8_count(b"caf\xc3\xa9") == 4  # café in UTF-8 bytes

    # Various multi-byte sequences
    assert sz.utf8_count("αβγδ") == 4  # Greek letters (2 bytes each)
    assert sz.utf8_count("АБВГ") == 4  # Cyrillic letters (2 bytes each)
    assert sz.utf8_count("𐍈") == 1  # Gothic letter (4 bytes)


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


def test_utf8_words_basic():
    """Test basic word iteration."""
    # Simple two words
    result = [str(w) for w in sz.utf8_words("hello world")]
    assert result == ["hello", " ", "world"]

    # Empty string
    result = [str(w) for w in sz.utf8_words("")]
    assert result == []

    # Single character
    result = [str(w) for w in sz.utf8_words("a")]
    assert result == ["a"]


def test_utf8_words_skip_empty():
    """Test skip_empty parameter."""
    result = [str(w) for w in sz.utf8_words("hello  world", skip_empty=True)]
    # Should skip empty segments at boundaries
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)


def test_utf8_words_contractions():
    """Test English contractions per TR29 rules."""
    # Per TR29, apostrophe between letters should not break
    result = [str(w) for w in sz.utf8_words("don't")]
    # TR29: "don't" should be one word (WB6-7: MidLetter rules)
    assert "don't" in result or result == ["don't"]


def test_utf8_words_unicode():
    """Test UTF-8 multi-byte characters."""
    # German with eszett
    result = [str(w) for w in sz.utf8_words("Größe")]
    assert "Größe" in result

    # Russian text
    result = [str(w) for w in sz.utf8_words("привет мир")]
    assert len(result) >= 2

    # CJK (each character is its own word in TR29)
    result = [str(w) for w in sz.utf8_words("你好")]
    # CJK characters are typically "Other" category - each is a boundary
    assert len(result) >= 1


def test_utf8_words_emoji():
    """Test emoji handling."""
    # Simple emoji
    result = [str(w) for w in sz.utf8_words("hello 👋 world")]
    assert "hello" in result
    assert "world" in result


def test_utf8_words_numbers():
    """Test numeric handling per TR29."""
    result = [str(w) for w in sz.utf8_words("test123")]
    # ALetter followed by Numeric should not break (WB9)
    assert "test123" in result or result == ["test123"]

    result = [str(w) for w in sz.utf8_words("3.14")]
    # Numeric with MidNum should stay together (WB11-12)
    assert "3.14" in result or len(result) <= 3


def test_utf8_words_str_method():
    """Test the Str.utf8_words() method."""
    s = Str("hello world")
    # Method requires text argument, even when called on Str object
    result = [str(w) for w in s.utf8_words(s)]
    assert result == ["hello", " ", "world"]

    # Also test via module function
    result = [str(w) for w in sz.utf8_words(s)]
    assert result == ["hello", " ", "world"]


@pytest.mark.parametrize("seed_value", [42, 0, 1, 314159, 271828])
def test_utf8_word_boundary_fuzz(seed_value: int):
    """Fuzz test: compare the StringZilla word kernel against the INDEPENDENT ``uniseg`` UAX-29 implementation.

    The previous oracle (an in-repo pure-Python TR29 reimplementation, ``baseline_word_boundaries``) is itself
    ~89% wrong vs ``uniseg`` on this charset, so it could never validate a correct kernel. ``uniseg`` is a
    maintained third-party reference and agrees with the official ``WordBreakTest.txt``.
    """
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    seed(seed_value)

    # Generate random test strings. The charset deliberately spans every WB category — including the ones that a
    # naive kernel gets wrong: combining marks / Format / ZWJ (WB4, WB3c), emoji & symbol pictographs (WB3c),
    # Regional_Indicators (WB15/16), horizontal spaces incl. ideographic (WB3d), Hebrew + quotes (WB7a), Katakana,
    # CJK, CR/LF. An ASCII-only charset is what let the boundary bugs hide.
    test_chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " \t\n\r"
        ".,;:!?'\"_-"
        "äöüßαβγδабвг"
        "אב"  # Hebrew (WB7a)
        "カタ"  # Katakana (WB13)
        "中文"  # CJK
        "̀́̈"  # combining marks (Extend, WB4)
        "­⁠‍"  # Soft hyphen (Format), Word joiner (Format), ZWJ (WB3c)
        "　  "  # ideographic / thin / math spaces (WSegSpace, WB3d)
        "\U0001f600\U0001f6d1Ⓜ©❤"  # Extended_Pictographic (WB3c)
        "\U0001f1e6\U0001f1fa"  # Regional_Indicators (WB15/16)
    )

    for _ in range(200):
        length = randint(1, 100)
        text = "".join(choice(test_chars) for _ in range(length))

        # Independent UAX-29 oracle (uniseg): cumulative byte lengths of its segments.
        expected_boundaries = [0]
        for seg in uniseg_wordbreak.words(text):
            expected_boundaries.append(expected_boundaries[-1] + len(seg.encode("utf-8")))

        # StringZilla boundaries = cumulative byte lengths of the emitted words.
        sz_boundaries = [0]
        for word in sz.utf8_words(text):
            sz_boundaries.append(sz_boundaries[-1] + len(str(word).encode("utf-8")))

        # Exact match required — no `len >= 2` escape hatch.
        assert (
            sz_boundaries == expected_boundaries
        ), "Word boundary mismatch (seed {}):\n  text cps: {}\n  expected: {}\n  got:      {}".format(
            seed_value, " ".join(f"{ord(c):04X}" for c in text), expected_boundaries, sz_boundaries
        )


def test_utf8_word_boundary_official_conformance():
    """Full UAX-29 conformance: EXACT boundary match against every case in the official Unicode
    WordBreakTest.txt. This is the authoritative gate — it must match on all cases, not "first 50"
    with a `len >= 2` placeholder (which let a ~50%-conformant kernel ship undetected)."""
    try:
        test_cases = get_word_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for test_str, expected_boundaries in test_cases:
        if not test_str:
            continue

        # The StringZilla iterator yields words back-to-back; their cumulative byte lengths are the boundaries.
        sz_boundaries = [0]
        for word in sz.utf8_words(test_str):
            sz_boundaries.append(sz_boundaries[-1] + len(str(word).encode("utf-8")))

        # `get_word_break_test_cases` already returns BYTE boundaries (including 0 and len), matching `sz_boundaries`.
        if sz_boundaries != expected_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_boundaries}, got {sz_boundaries}")

    assert not failures, "UAX-29 WordBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )


@pytest.mark.parametrize("seed_value", [1, 7, 42, 1000, 271828])
def test_utf8_word_boundary_differential_uniseg(seed_value: int):
    """Differential fuzz against an INDEPENDENT UAX-29 library (``uniseg``).

    The official WordBreakTest.txt is finite (~1944 fixed cases) and does not cover the combinatorial
    explosion of interdependent rules (WB4 ignorables crossing WB3c/WB3d/WB6-7 bridges, etc.). Random
    generation over a boundary-relevant palette, checked against a second conformant implementation,
    surfaces corner cases the fixed suite misses — which is precisely how a 50%-conformant kernel once
    passed CI. Any divergence is a real bug in one of the two implementations."""
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    seed(seed_value)

    # Every WB category, including the historically-broken ones.
    palette = list(
        "abZ59 \t\n\r.,;:'\"_-"
        "éαаאבカタ中국"
        "̀́̈"  # combining Extend (WB4)
        "­⁠‍"  # Format, Word-joiner, ZWJ (WB3c / WB4)
        "　  "  # WSegSpace (WB3d)
        "\U0001f600\U0001f6d1Ⓜ©❤"  # Extended_Pictographic (WB3c)
        "\U0001f1e6\U0001f1fa\U0001f3fb"  # Regional_Indicator + skin tone (WB15/16)
    )

    def boundaries(segments):
        out = [0]
        for seg in segments:
            out.append(out[-1] + len(seg.encode("utf-8")))
        return out

    # Fixed multi-window regression strings: RI-parity and Mid-bridge carry once miscounted across the 64-byte
    # window seam. Each exceeds one window — the case the random generator below now also covers (it previously
    # used <=24 codepoints, which fits a single window and so could never exercise a seam).
    seam_regressions = [
        bytes.fromhex(
            "E382AB2D0AF09F87A662C2ADF09F87A6CC800A5FF09F87A6F09F87A6C2AD0ACC88F09F87BAF09F8FBBCC88C2ADE281A0CC88F0"
            "9F87A6F09F87A6E4B8ADE281A05F"
        ).decode("utf-8"),
        bytes.fromhex(
            "5F27F09F87BAE4B8ADCC81F09F87A6F09F87A65FCC885FD7902DE29382CC8062F09F87A6E281A0F09F87BACC80CC88F09F8FBB"
            "F09F87BACC81F09F87BA0AC2ADF09F87BACC88E382AB5F272ECC880AE4B8AD"
        ).decode("utf-8"),
        bytes.fromhex(
            "5FE281A05F352CF09F988027F09F98805FCC88CC81E4B8ADE281A05F62CC8061E4B8ADE382AB35E281A0CC88CC88F09F8FBBCC"
            "88E281A0CC80CC812735CC80E4B8ADCC81E2808DC3A9E380803527"
        ).decode("utf-8"),
    ]

    failures = []
    # Up to ~100 codepoints so most cases cross the 64-byte window boundary (multi-window seam coverage). The
    # palette keeps ignorable runs short, so every run stays inside one window — the domain where the kernel must
    # match exactly. (Pathological >64-byte ignorable runs exceed any fixed window; serial-only, out of scope here.)
    samples = seam_regressions + ["".join(choice(palette) for _ in range(randint(1, 100))) for _ in range(2000)]
    for text in samples:
        sz_boundaries = boundaries(str(w) for w in sz.utf8_words(text))
        ref_boundaries = boundaries(uniseg_wordbreak.words(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    uniseg={ref_boundaries}")

    assert not failures, "StringZilla vs uniseg word-boundary divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )


# Shared corner-case palette for grapheme / sentence / line fuzzing. Spans combining marks, ZWJ
# emoji sequences, Regional_Indicators, CJK, astral pictographs, terminators and line breaks.
_SEGMENTATION_PALETTE = list(
    "abZ59 \t\n\r.,;:!?'\"_-()"
    "éαаאבカタ中국"
    "̀́̈"  # combining Extend
    "­⁠‍"  # Format, Word-joiner, ZWJ
    "　  "  # ideographic / thin / math spaces
    "\U0001f600\U0001f6d1Ⓜ©❤"  # Extended_Pictographic
    "\U0001f1e6\U0001f1fa\U0001f3fb"  # Regional_Indicator + skin tone
    "  "  # Line Separator / Paragraph Separator (mandatory line breaks)
)


def _byte_boundaries(segments):
    """Cumulative byte-offset boundaries of an iterable of string segments."""
    out = [0]
    for seg in segments:
        out.append(out[-1] + len(str(seg).encode("utf-8")))
    return out


def _char_boundaries_to_bytes(text: str, char_boundaries) -> list:
    """Convert TR29-style character-index boundaries (incl. 0 and len) to byte offsets."""
    char_boundaries = set(char_boundaries)
    out = []
    byte_pos = 0
    for i, char in enumerate(text):
        if i in char_boundaries:
            out.append(byte_pos)
        byte_pos += len(char.encode("utf-8"))
    if len(text) in char_boundaries:
        out.append(byte_pos)
    return out


#  region Grapheme iterator


def test_utf8_graphemes_basic():
    """Test basic grapheme cluster iteration."""
    # Plain ASCII: one cluster per character.
    result = [str(g) for g in sz.utf8_graphemes("hello")]
    assert result == ["h", "e", "l", "l", "o"]

    # Empty string.
    assert [str(g) for g in sz.utf8_graphemes("")] == []

    # Base letter + combining acute accent stays a single cluster.
    result = [str(g) for g in sz.utf8_graphemes("é")]
    assert result == ["é"]

    # Emoji ZWJ family (man + ZWJ + woman + ZWJ + girl) is one cluster.
    family = "\U0001f468‍\U0001f469‍\U0001f467"
    result = [str(g) for g in sz.utf8_graphemes(family)]
    assert result == [family]


def test_utf8_graphemes_skip_empty():
    """Test skip_empty parameter for grapheme iteration."""
    result = [str(g) for g in sz.utf8_graphemes("héllo", skip_empty=True)]
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)


def test_utf8_graphemes_unicode():
    """Test UTF-8 multi-byte / combining / emoji grapheme coverage."""
    # Precomposed vs decomposed: decomposed should still be a single cluster.
    assert [str(g) for g in sz.utf8_graphemes("ä")] == ["ä"]

    # Regional indicator pair forms a single flag cluster.
    flag = "\U0001f1e6\U0001f1fa"
    assert [str(g) for g in sz.utf8_graphemes(flag)] == [flag]

    # Emoji with skin-tone modifier is one cluster.
    waver = "\U0001f44b\U0001f3fb"
    assert [str(g) for g in sz.utf8_graphemes(waver)] == [waver]


def test_utf8_graphemes_str_method():
    """The Str.utf8_graphemes() method must agree with the module function."""
    s = Str("héllo")
    method_result = [str(g) for g in s.utf8_graphemes(s)]
    module_result = [str(g) for g in sz.utf8_graphemes(s)]
    assert method_result == module_result
    assert method_result == [str(g) for g in sz.utf8_graphemes("héllo")]


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_boundary_fuzz(seed_value: int):
    """Fuzz: compare the C grapheme iterator vs the pure-Python UAX-29 baseline."""
    seed(seed_value)
    try:
        props = get_grapheme_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode data files")

    for _ in range(200):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 100)))
        try:
            expected = baseline_grapheme_boundaries(text, props)
        except Exception:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(text))
        assert (
            sz_boundaries == expected
        ), "Grapheme boundary mismatch (seed {}):\n  text cps: {}\n  expected: {}\n  got:      {}".format(
            seed_value, " ".join(f"{ord(c):04X}" for c in text), expected, sz_boundaries
        )


def test_utf8_grapheme_boundary_official_conformance():
    """Full UAX-29 conformance against every case in the official GraphemeBreakTest.txt."""
    try:
        test_cases = get_grapheme_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for test_str, expected_byte_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(test_str))
        if sz_boundaries != expected_byte_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_byte_boundaries}, got {sz_boundaries}")

    assert not failures, "UAX-29 GraphemeBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_differential_grapheme(seed_value: int):
    """Differential fuzz against an independent grapheme library (``grapheme`` / ``uniseg``)."""
    seed(seed_value)
    try:
        graphemes_fn = pytest.importorskip("grapheme", reason="grapheme not installed").graphemes
    except Exception:
        uniseg_gc = pytest.importorskip("uniseg.graphemecluster", reason="uniseg not installed")
        graphemes_fn = uniseg_gc.grapheme_clusters

    failures = []
    for _ in range(2000):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(text))
        ref_boundaries = _byte_boundaries(graphemes_fn(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    ref={ref_boundaries}")

    assert not failures, "StringZilla vs reference grapheme-boundary divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )


#  endregion Grapheme iterator

#  region Sentence iterator


def test_utf8_sentences_basic():
    """Test basic sentence iteration."""
    result = [str(s) for s in sz.utf8_sentences("Hello. World!")]
    assert len(result) == 2
    assert "".join(result) == "Hello. World!"

    # Empty string.
    assert [str(s) for s in sz.utf8_sentences("")] == []

    # No terminator: the whole text is one sentence.
    assert [str(s) for s in sz.utf8_sentences("no terminator here")] == ["no terminator here"]


def test_utf8_sentences_skip_empty():
    """Test skip_empty parameter for sentence iteration."""
    result = [str(s) for s in sz.utf8_sentences("A. B. C.", skip_empty=True)]
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)


def test_utf8_sentences_unicode():
    """Test UTF-8 multi-byte sentence coverage."""
    # Cyrillic sentences separated by a period and space.
    result = [str(s) for s in sz.utf8_sentences("Привет. Мир.")]
    assert "".join(result) == "Привет. Мир."
    assert len(result) >= 2

    # Ideographic full stop terminates a sentence.
    result = [str(s) for s in sz.utf8_sentences("你好。世界。")]
    assert "".join(result) == "你好。世界。"


def test_utf8_sentences_str_method():
    """The Str.utf8_sentences() method must agree with the module function."""
    s = Str("Hello. World!")
    method_result = [str(x) for x in s.utf8_sentences(s)]
    module_result = [str(x) for x in sz.utf8_sentences(s)]
    assert method_result == module_result
    assert method_result == [str(x) for x in sz.utf8_sentences("Hello. World!")]


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_boundary_fuzz(seed_value: int):
    """Fuzz: compare the C sentence iterator vs the pure-Python UAX-29 baseline."""
    seed(seed_value)
    try:
        props = get_sentence_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode data files")

    for _ in range(200):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 100)))
        try:
            expected = baseline_sentence_boundaries(text, props)
        except Exception:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(text))
        assert (
            sz_boundaries == expected
        ), "Sentence boundary mismatch (seed {}):\n  text cps: {}\n  expected: {}\n  got:      {}".format(
            seed_value, " ".join(f"{ord(c):04X}" for c in text), expected, sz_boundaries
        )


def test_utf8_sentence_boundary_official_conformance():
    """Full UAX-29 conformance against every case in the official SentenceBreakTest.txt."""
    try:
        test_cases = get_sentence_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for test_str, expected_byte_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(test_str))
        if sz_boundaries != expected_byte_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_byte_boundaries}, got {sz_boundaries}")

    assert not failures, "UAX-29 SentenceBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_differential_icu(seed_value: int):
    """Differential fuzz against an independent sentence segmenter (``icu``, else ``pysbd``)."""
    seed(seed_value)

    def icu_boundaries():
        icu = pytest.importorskip("icu", reason="PyICU not installed")
        bi = icu.BreakIterator.createSentenceInstance(icu.Locale.getRoot())

        def segments(text):
            uni = icu.UnicodeString(text)
            bi.setText(uni)
            parts = []
            start = bi.first()
            end = bi.nextBoundary()
            while end != icu.BreakIterator.DONE:
                parts.append(str(uni[start:end]))
                start = end
                end = bi.nextBoundary()
            return parts

        return segments

    try:
        ref_segments = icu_boundaries()
    except Exception:
        pysbd = pytest.importorskip("pysbd", reason="neither PyICU nor pysbd installed")
        segmenter = pysbd.Segmenter(language="en", clean=False)
        ref_segments = lambda text: segmenter.segment(text)  # noqa: E731

    # Restrict the palette to characters ICU/pysbd and StringZilla agree on the meaning of: terminators,
    # letters, spaces and newlines. Emoji/RI sentence behavior is not portable across segmenters.
    palette = list("abZ59 \t\n.,;:!?'\"()éαаカ中" "  ")

    failures = []
    for _ in range(500):
        text = "".join(choice(palette) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(text))
        ref_boundaries = _byte_boundaries(ref_segments(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    ref={ref_boundaries}")

    # Sentence segmentation differs in defensible ways across libraries; require broad agreement, not
    # bit-exactness, so a portable-but-imperfect reference does not produce spurious CI failures.
    agreement = 1.0 - len(failures) / 500.0
    assert agreement >= 0.80, "StringZilla vs reference sentence agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )


#  endregion Sentence iterator

#  region Line iterator


def test_utf8_linewraps_basic():
    """Test basic line iteration yielding line-break-opportunity Str segments."""
    result = [str(seg) for seg in sz.utf8_linewraps("first\nsecond")]
    assert "".join(result) == "first\nsecond"  # Segments tile the input.

    # Empty string yields nothing.
    assert [str(seg) for seg in sz.utf8_linewraps("")] == []

    # A single short line still tiles back to the input.
    result = [str(seg) for seg in sz.utf8_linewraps("just one line")]
    assert "".join(result) == "just one line"


def test_utf8_linewraps_tiling():
    """Linewrap is forward-only; its segments tile the input contiguously."""
    text = "alpha\nbeta\ngamma"
    forward = [str(seg) for seg in sz.utf8_linewraps(text)]
    assert "".join(forward) == text


def test_utf8_linewraps_skip_empty():
    """Test skip_empty parameter for line iteration."""
    result = [str(seg) for seg in sz.utf8_linewraps("a\n\nb", skip_empty=True)]
    assert len(result) > 0
    assert all(len(seg) > 0 for seg in result)


def test_utf8_linewraps_unicode():
    """Test UTF-8 multi-byte / Unicode coverage."""
    result = [str(seg) for seg in sz.utf8_linewraps("Größe привет")]
    assert "".join(result) == "Größe привет"
    # Segments tile the input.

    # CRLF is a single break opportunity, not two.
    result = [str(seg) for seg in sz.utf8_linewraps("a\r\nb")]
    assert "".join(result) == "a\r\nb"


def test_utf8_linewraps_str_method():
    """The Str.utf8_linewraps() method must agree with the module function."""
    s = Str("first\nsecond")
    method_result = [str(seg) for seg in s.utf8_linewraps(s)]
    module_result = [str(seg) for seg in sz.utf8_linewraps(s)]
    assert method_result == module_result
    assert method_result == [str(seg) for seg in sz.utf8_linewraps("first\nsecond")]


def test_utf8_linewrap_boundary_official_conformance():
    """UAX-14 break-opportunity conformance against the official LineBreakTest.txt.

    The iterator emits a segment at every break opportunity (mandatory or soft); their cumulative
    byte lengths must match the official break positions.
    """
    try:
        test_cases = get_line_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for test_str, expected_byte_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_linewraps(test_str))
        if sz_boundaries != expected_byte_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_byte_boundaries}, got {sz_boundaries}")

    # UAX-14 has many tailorable rules; require broad agreement rather than bit-exactness so a
    # defensible tailoring difference does not break CI.
    agreement = 1.0 - len(failures) / max(1, len(test_cases))
    assert agreement >= 0.80, "UAX-14 LineBreakTest agreement {:.2%} too low ({} / {}):\n{}".format(
        agreement, len(test_cases) - len(failures), len(test_cases), "\n".join(failures[:40])
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_linewrap_differential_uniseg(seed_value: int):
    """Differential fuzz against ``uniseg.linebreak`` for line-break-opportunity segments."""
    uniseg_linebreak = pytest.importorskip("uniseg.linebreak", reason="uniseg not installed")
    seed(seed_value)

    def ref_segments(text):
        return list(uniseg_linebreak.line_break_units(text))

    failures = []
    for _ in range(2000):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_linewraps(text))
        ref_boundaries = _byte_boundaries(ref_segments(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    uniseg={ref_boundaries}")

    # uniseg implements default UAX-14 tailorings that may differ from StringZilla; require broad
    # agreement on segment positions rather than bit-exactness.
    agreement = 1.0 - len(failures) / 2000.0
    assert agreement >= 0.80, "StringZilla vs uniseg line-boundary agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )


#  endregion Line iterator


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", "-s", __file__]))
