from random import choice, randint
from string import ascii_lowercase
from typing import Optional

import pytest

import stringzilla as sz
from stringzilla import Str, Strs

# NumPy is available on most platforms and is required for most tests.
# When using PyPy on some platforms NumPy has internal issues, that will
# raise a weird error, not an `ImportError`. That's why we intentionally
# use a naked `except:`. Necessary evil!
try:
    import numpy as np

    numpy_available = True
except:
    # NumPy is not installed, most tests will be skipped
    numpy_available = False


# PyArrow is not available on most platforms.
# When using PyPy on some platforms PyArrow has internal issues, that will
# raise a weird error, not an `ImportError`. That's why we intentionally
# use a naked `except:`. Necessary evil!
try:
    import pyarrow as pa

    pyarrow_available = True
except:
    # NumPy is not installed, most tests will be skipped
    pyarrow_available = False


def test_library_properties():
    assert len(sz.__version__.split(".")) == 3, "Semantic versioning must be preserved"
    assert "serial" in sz.__capabilities__.split(","), "Serial backend must be present"


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


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_unit_buffer_protocol():
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


def test_unit_strs_sequence():
    native = "p3\np2\np1"
    big = Str(native)

    lines = big.splitlines()
    assert [2, 1, 0] == list(lines.order())
    assert "p3" in lines
    assert "p4" not in lines

    lines.sort()
    assert [0, 1, 2] == list(lines.order())
    assert ["p1", "p2", "p3"] == list(lines)

    # Reverse order
    assert [2, 1, 0] == list(lines.order(reverse=True))
    lines.sort(reverse=True)
    assert ["p3", "p2", "p1"] == list(lines)

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
    assert big_sequence[::-1] == ["6", "5", "4", "3", "2", "1"], to_str(
        big_sequence[::-1]
    )

    # Let's go harder with nested slicing
    assert big_sequence[1:][::-1] == ["6", "5", "4", "3", "2"]
    assert big_sequence[1:][::-2] == ["6", "4", "2"]
    assert big_sequence[1:][::-3] == ["6", "3"]


def test_unit_globals():
    """Validates that the previously unit-tested member methods are also visible as global functions."""

    assert sz.find("abcdef", "bcdef") == 1
    assert sz.find("abcdef", "x") == -1

    assert sz.count("abcdef", "x") == 0
    assert sz.count("aaaaa", "a") == 5
    assert sz.count("aaaaa", "aa") == 2
    assert sz.count("aaaaa", "aa", allowoverlap=True) == 4

    assert sz.hamming_distance("aaa", "aaa") == 0
    assert sz.hamming_distance("aaa", "bbb") == 3
    assert sz.hamming_distance("abababab", "aaaaaaaa") == 4
    assert sz.hamming_distance("abababab", "aaaaaaaa", 2) == 2
    assert sz.hamming_distance("abababab", "aaaaaaaa", bound=2) == 2

    assert sz.edit_distance("aaa", "aaa") == 0
    assert sz.edit_distance("aaa", "bbb") == 3
    assert sz.edit_distance("abababab", "aaaaaaaa") == 4
    assert sz.edit_distance("abababab", "aaaaaaaa", 2) == 2
    assert sz.edit_distance("abababab", "aaaaaaaa", bound=2) == 2


def test_unit_len():
    w = sz.Str("abcd")
    assert 4 == len(w)


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
        assert (
            native_slice == big_slice
        ), f"Mismatch between `{native_slice}` and `{str(big_slice)}`"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def baseline_edit_distance(s1, s2) -> int:
    """
    Compute the Levenshtein distance between two strings.
    """

    # Create a matrix of size (len(s1)+1) x (len(s2)+1)
    matrix = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)

    # Initialize the first column and first row of the matrix
    for i in range(len(s1) + 1):
        matrix[i, 0] = i
    for j in range(len(s2) + 1):
        matrix[0, j] = j

    # Compute Levenshtein distance
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i, j] = min(
                matrix[i - 1, j] + 1,  # Deletion
                matrix[i, j - 1] + 1,  # Insertion
                matrix[i - 1, j - 1] + cost,  # Substitution
            )

    # Return the Levenshtein distance
    return matrix[len(s1), len(s2)]


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
    assert native.find(needle, len_quarter, 3 * len_quarter) == big.find(
        needle, len_quarter, 3 * len_quarter
    )

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
def test_fuzzy_substrings(pattern_length: int, haystack_length: int, variability: int):
    native = get_random_string(variability=variability, length=haystack_length)
    big = Str(native)
    pattern = get_random_string(variability=variability, length=pattern_length)
    assert (pattern in native) == big.contains(
        pattern
    ), f"Failed to check if {pattern} at offset {native.find(pattern)} is present in {native}"
    assert native.find(pattern) == big.find(
        pattern
    ), f"Failed to locate {pattern} at offset {native.find(pattern)} in {native}"


@pytest.mark.repeat(100)
@pytest.mark.parametrize("max_edit_distance", [150])
def test_edit_distance_insertions(max_edit_distance: int):
    # Create a new string by slicing and concatenating
    def insert_char_at(s, char_to_insert, index):
        return s[:index] + char_to_insert + s[index:]

    a = get_random_string(length=20)
    b = a
    for i in range(max_edit_distance):
        source_offset = randint(0, len(ascii_lowercase) - 1)
        target_offset = randint(0, len(b) - 1)
        b = insert_char_at(b, ascii_lowercase[source_offset], target_offset)
        assert sz.edit_distance(a, b, bound=200) == i + 1


def test_edit_distances():

    assert sz.hamming_distance("hello", "hello") == 0
    assert sz.hamming_distance("hello", "hell") == 1
    assert sz.hamming_distance("abc", "adc") == 1, "one substitution"
    assert sz.hamming_distance("αβγδ", "αxxγδ") == 2, "replace Beta UTF8 codepoint"
    assert (
        sz.hamming_distance_unicode("abcdefgh", "_bcdefg_") == 2
    ), "replace ASCI prefix and suffix"
    assert (
        sz.hamming_distance_unicode("αβγδ", "αγγδ") == 1
    ), "replace Beta UTF8 codepoint"

    assert sz.edit_distance("hello", "hello") == 0
    assert sz.edit_distance("hello", "hell") == 1
    assert sz.edit_distance("", "") == 0
    assert sz.edit_distance("", "abc") == 3
    assert sz.edit_distance("abc", "") == 3
    assert sz.edit_distance("abc", "ac") == 1, "one deletion"
    assert sz.edit_distance("abc", "a_bc") == 1, "one insertion"
    assert sz.edit_distance("abc", "adc") == 1, "one substitution"
    assert (
        sz.edit_distance("ggbuzgjux{}l", "gbuzgjux{}l") == 1
    ), "one insertion (prepended)"
    assert sz.edit_distance("abcdefgABCDEFG", "ABCDEFGabcdefg") == 14

    assert (
        sz.edit_distance_unicode("hello", "hell") == 1
    ), "no unicode symbols, just ASCII"
    assert (
        sz.edit_distance_unicode("𠜎 𠜱 𠝹 𠱓", "𠜎𠜱𠝹𠱓") == 3
    ), "add 3 whitespaces in Chinese"
    assert sz.edit_distance_unicode("💖", "💗") == 1

    assert sz.edit_distance_unicode("αβγδ", "αγδ") == 1, "insert Beta"
    assert (
        sz.edit_distance_unicode("école", "école") == 2
    ), "etter 'é' as a single character vs 'e' + '´'"
    assert (
        sz.edit_distance_unicode("façade", "facade") == 1
    ), "'ç' with cedilla vs. plain"
    assert (
        sz.edit_distance_unicode("Schön", "Scho\u0308n") == 2
    ), "'ö' represented as 'o' + '¨'"
    assert (
        sz.edit_distance_unicode("München", "Muenchen") == 2
    ), "German with umlaut vs. transcription"
    assert sz.edit_distance_unicode("こんにちは世界", "こんばんは世界") == 2


@pytest.mark.repeat(30)
@pytest.mark.parametrize("first_length", [20, 100])
@pytest.mark.parametrize("second_length", [20, 100])
def test_edit_distance_random(first_length: int, second_length: int):
    a = get_random_string(length=first_length)
    b = get_random_string(length=second_length)
    assert sz.edit_distance(a, b) == baseline_edit_distance(a, b)


@pytest.mark.repeat(30)
@pytest.mark.parametrize("first_length", [20, 100])
@pytest.mark.parametrize("second_length", [20, 100])
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_alignment_score_random(first_length: int, second_length: int):

    a = get_random_string(length=first_length)
    b = get_random_string(length=second_length)
    character_substitutions = np.zeros((256, 256), dtype=np.int8)
    character_substitutions.fill(-1)
    np.fill_diagonal(character_substitutions, 0)

    assert sz.alignment_score(
        a,
        b,
        substitution_matrix=character_substitutions,
        gap_score=-1,
    ) == -baseline_edit_distance(a, b)


@pytest.mark.parametrize("list_length", [10, 20, 30, 40, 50])
@pytest.mark.parametrize("part_length", [5, 10])
@pytest.mark.parametrize("variability", [2, 3])
def test_fuzzy_sorting(list_length: int, part_length: int, variability: int):
    native_list = [
        get_random_string(variability=variability, length=part_length)
        for _ in range(list_length)
    ]
    native_joined = ".".join(native_list)
    big_joined = Str(native_joined)
    big_list = big_joined.split(".")

    native_ordered = sorted(native_list)
    native_order = big_list.order()
    for i in range(list_length):
        assert native_ordered[i] == native_list[native_order[i]], "Order is wrong"
        assert native_ordered[i] == str(
            big_list[int(native_order[i])]
        ), "Split is wrong?!"

    native_list.sort()
    big_list.sort()

    assert len(native_list) == len(big_list)
    for native_str, big_str in zip(native_list, big_list):
        assert native_str == str(big_str), "Order is wrong"


@pytest.mark.parametrize("list_length", [10, 20, 30, 40, 50])
@pytest.mark.parametrize("part_length", [5, 10])
@pytest.mark.parametrize("variability", [2, 3])
def test_fuzzy_sorting(list_length: int, part_length: int, variability: int):
    native_list = [
        get_random_string(variability=variability, length=part_length)
        for _ in range(list_length)
    ]
    native_joined = ".".join(native_list)
    big_joined = Str(native_joined)
    big_list = big_joined.split(".")

    native_ordered = sorted(native_list)
    native_order = big_list.order()
    for i in range(list_length):
        assert native_ordered[i] == native_list[native_order[i]], "Order is wrong"
        assert native_ordered[i] == str(
            big_list[int(native_order[i])]
        ), "Split is wrong?!"

    native_list.sort()
    big_list.sort()

    assert len(native_list) == len(big_list)
    for native_str, big_str in zip(native_list, big_list):
        assert native_str == str(big_str), "Order is wrong"


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_pyarrow_str_conversion():
    native = "hello"
    big = Str(native)
    assert isinstance(big.address, int) and big.address != 0
    assert isinstance(big.nbytes, int) and big.nbytes == len(native)

    arrow_buffer = pa.foreign_buffer(big.address, big.nbytes, big)
    assert arrow_buffer.to_pybytes() == native.encode("utf-8")
