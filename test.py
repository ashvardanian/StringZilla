from typing import Union, Optional
from random import choice, randint
from string import ascii_lowercase

import pytest

from stringzilla.compiled import Str, File, Slices


def get_random_string(
    length: Optional[int] = None, variability: Optional[int] = None
) -> str:
    if length is None:
        length = randint(3, 300)
    if variability is None:
        variability = len(ascii_lowercase)
    return "".join(choice(ascii_lowercase[:variability]) for _ in range(length))


def check_identical(native: str, big: Union[Str, File], needle: Optional[str] = None):
    if needle is None:
        part_offset = randint(0, len(native) - 1)
        part_length = randint(1, len(native) - part_offset)
        needle = native[part_offset:part_length]

    present_in_native: bool = needle in native
    present_in_big = needle in big
    assert present_in_native == present_in_big
    assert native.find(needle) == big.find(needle)
    assert native.count(needle) == big.count(needle)


def test_basic():
    pattern = "abc"
    native = "abcabcabc"
    big = Str("abcabcabc")

    check_identical(native, big, pattern)


@pytest.mark.parametrize("pattern_length", [4, 5])
@pytest.mark.parametrize("haystack_length", range(1, 200))
def test_fuzzy(pattern_length: int, haystack_length: int):
    native = get_random_string(variability=3, length=haystack_length)
    big = Str(native)

    for _ in range(haystack_length // pattern_length):
        pattern = get_random_string(variability=3, length=pattern_length)
        check_identical(native, big, pattern)
