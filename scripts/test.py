from random import choice, randint
from string import ascii_lowercase
from typing import Optional
import numpy as np

import pytest

from random import choice, randint
from string import ascii_lowercase
import stringzilla as sz
from stringzilla import Str, Strs
from typing import Optional


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
    s2 = Str("abb")
    assert s2[1:] == "bb"
    assert s2[:-1] == "ab"
    assert s2[-1:] == "b"


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

    assert sz.edit_distance("aaa", "aaa") == 0
    assert sz.edit_distance("aaa", "bbb") == 3
    assert sz.edit_distance("abababab", "aaaaaaaa") == 4
    assert sz.edit_distance("abababab", "aaaaaaaa", 2) == 2
    assert sz.edit_distance("abababab", "aaaaaaaa", bound=2) == 2


def test_unit_len():
    w = sz.Str("abcd")
    assert 4 == len(w)


def test_slice_of_split():
    def impl(native_str):
        native_split = native_str.split()
        text = sz.Str(native_str)
        split = text.split()
        for split_idx in range(len(native_split)):
            native_slice = native_split[split_idx:]
            idx = split_idx
            for word in split[split_idx:]:
                assert str(word) == native_split[idx]
                idx += 1

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
def test_alignment_score_random(first_length: int, second_length: int):
    a = get_random_string(length=first_length)
    b = get_random_string(length=second_length)
    character_substitutions = np.ones((256, 256), dtype=np.int8)
    np.fill_diagonal(character_substitutions, 0)

    assert sz.alignment_score(
        a, b, substitution_matrix=character_substitutions, gap_score=1
    ) == baseline_edit_distance(a, b)


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
