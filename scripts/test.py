from typing import Union, Optional
from random import choice, randint
from string import ascii_lowercase

import pytest

from stringzilla import Str, File, Slices


def get_random_string(
    length: Optional[int] = None, variability: Optional[int] = None
) -> str:
    if length is None:
        length = randint(3, 300)
    if variability is None:
        variability = len(ascii_lowercase)
    return "".join(choice(ascii_lowercase[:variability]) for _ in range(length))


def check_identical(
    native: str,
    big: Union[Str, File],
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

    native_slices = native.split(needle)
    big_slices: Slices = big.split(needle)
    assert len(native_slices) == len(big_slices)

    if check_iterators:
        for i in range(len(native_slices)):
            assert len(native_slices[i]) == len(big_slices[i])
            assert native_slices[i] == str(big_slices[i])
            assert [c for c in native_slices[i]] == [c for c in big_slices[i]]

        for native_slice, big_slice in zip(native_slices, big_slices):
            assert native_slice == str(big_slice)


@pytest.mark.parametrize("repetitions", range(1, 10))
def test_basic(repetitions: int):
    native = "abcd" * repetitions
    big = Str(native)

    check_identical(native, big, "a", True)
    check_identical(native, big, "ab", True)
    check_identical(native, big, "abc", True)
    check_identical(native, big, "abcd", True)
    check_identical(native, big, "abcde", True)  # Missing pattern


@pytest.mark.parametrize("pattern_length", [1, 2, 4, 5])
@pytest.mark.parametrize("haystack_length", range(1, 65))
@pytest.mark.parametrize("variability", [2, 25])
def test_fuzzy(pattern_length: int, haystack_length: int, variability: int):
    native = get_random_string(variability=variability, length=haystack_length)
    big = Str(native)

    for _ in range(haystack_length // pattern_length):
        pattern = get_random_string(variability=variability, length=pattern_length)
        check_identical(native, big, pattern)
