from typing import Union, Optional
from random import choice, randint
from string import ascii_lowercase
import math

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


def is_equal_slices(native_slices, big_slices):
    for native_slice, big_slice in zip(native_slices, big_slices):
        assert native_slice == big_slice


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
            assert native_slices[i] == big_slices[i]
            assert [c for c in native_slices[i]] == [c for c in big_slices[i]]

    is_equal_slices(native_slices, big_slices)


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
@pytest.mark.parametrize("variability", range(1, 25))
def test_fuzzy(pattern_length: int, haystack_length: int, variability: int):
    native = get_random_string(variability=variability, length=haystack_length)
    big = Str(native)

    # Start by matching the prefix and the suffix
    check_identical(native, big, native[:pattern_length])
    check_identical(native, big, native[-pattern_length:])

    # Continue with random slices
    for _ in range(haystack_length // pattern_length):
        pattern = get_random_string(variability=variability, length=pattern_length)
        check_identical(native, big, pattern)


def test_slices():
    native = get_random_string(length=10)
    big = Str(native)

    assert native[0:5] == big.sub(0, 5) and native[0:5] == big[0:5]
    assert native[5:10] == big.sub(5, 10) and native[5:10] == big[5:10]

    assert native[5:5] == big.sub(5, 5) and native[5:5] == big[5:5]
    assert native[-5:-5] == big.sub(-5, -5) and native[-5:-5] == big[-5:-5]
    assert native[2:-2] == big.sub(2, -2) and native[2:-2] == big[2:-2]
    assert native[7:-7] == big.sub(7, -7) and native[7:-7] == big[7:-7]

    assert native[5:3] == big.sub(5, 3) and native[5:3] == big[5:3]
    assert native[5:7] == big.sub(5, 7) and native[5:7] == big[5:7]
    assert native[5:-3] == big.sub(5, -3) and native[5:-3] == big[5:-3]
    assert native[5:-7] == big.sub(5, -7) and native[5:-7] == big[5:-7]

    assert native[-5:3] == big.sub(-5, 3) and native[-5:3] == big[-5:3]
    assert native[-5:7] == big.sub(-5, 7) and native[-5:7] == big[-5:7]
    assert native[-5:-3] == big.sub(-5, -3) and native[-5:-3] == big[-5:-3]
    assert native[-5:-7] == big.sub(-5, -7) and native[-5:-7] == big[-5:-7]

    assert native[2:] == big.sub(2) and native[2:] == big[2:]
    assert native[:7] == big.sub(end=7) and native[:7] == big[:7]
    assert native[-2:] == big.sub(-2) and native[-2:] == big[-2:]
    assert native[:-7] == big.sub(end=-7) and native[:-7] == big[:-7]
    assert native[:-10] == big.sub(end=-10) and native[:-10] == big[:-10]
    assert native[:-1] == big.sub(end=-1) and native[:-1] == big[:-1]

    length = 10000
    native = get_random_string(length=length)
    big = Str(native)

    needle = native[0 : randint(2, 5)]
    native_slices = native.split(needle)
    big_slices: Slices = big.split(needle)

    length = len(native_slices)
    for i in range(length):
        start = randint(1 - length, length - 1)
        stop = randint(1 - length, length - 1)
        step = 0
        while step == 0:
            step = randint(-int(math.sqrt(length)), int(math.sqrt(length)))

        is_equal_slices(native_slices[start:stop:step], big_slices[start:stop:step])
        is_equal_slices(
            native_slices[start:stop:step],
            big_slices.sub(start, stop, step),
        )
