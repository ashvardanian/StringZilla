from typing import Union, Optional
from random import choice, randint
from string import ascii_lowercase

import pytest

import stringzilla as sz
from stringzilla import Str, Strs


def get_random_string(
    length: Optional[int] = None, variability: Optional[int] = None
) -> str:
    if length is None:
        length = randint(3, 300)
    if variability is None:
        variability = len(ascii_lowercase)
    return "".join(choice(ascii_lowercase[:variability]) for _ in range(length))


def is_equal_strings(native_strings, big_strings):
    for native_slice, big_slice in zip(native_strings, big_strings):
        assert native_slice == big_slice


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
    assert native.count(needle) == big.count(needle)

    native_strings = native.split(needle)
    big_strings: Strs = big.split(needle)
    assert len(native_strings) == len(big_strings)

    if check_iterators:
        for i in range(len(native_strings)):
            assert len(native_strings[i]) == len(big_strings[i])
            assert native_strings[i] == big_strings[i]
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


@pytest.mark.parametrize("iterations", range(100))
@pytest.mark.parametrize("max_edit_distance", [150])
def test_levenstein(iterations: int, max_edit_distance: int):
    # Create a new string by slicing and concatenating
    def insert_char_at(s, char_to_insert, index):
        return s[:index] + char_to_insert + s[index:]

    for _ in range(iterations):
        a = get_random_string(length=20)
        b = a
        for i in range(max_edit_distance):
            source_offset = randint(0, len(ascii_lowercase) - 1)
            target_offset = randint(0, len(b) - 1)
            b = insert_char_at(b, ascii_lowercase[source_offset], target_offset)
            assert sz.levenstein(a, b, 200) == i + 1


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
