from typing import Union, Optional
from random import choice, randint
from string import ascii_lowercase
import math

import pytest

import stringzilla as sz
from stringzilla import Str


def test_globals():
    assert sz.find("abcdef", "bcdef") == 1
    assert sz.find("abcdef", "x") == -1

    assert sz.count("abcdef", "x") == 0
    assert sz.count("aaaaa", "a") == 5
    assert sz.count("aaaaa", "aa") == 2
    assert sz.count("aaaaa", "aa", allowoverlap=True) == 4

    assert sz.levenstein("aaa", "aaa") == 0
    assert sz.levenstein("aaa", "bbb") == 3
    assert sz.levenstein("abababab", "aaaaaaaa") == 4
    assert sz.levenstein("abababab", "aaaaaaaa", 2) == 2
    assert sz.levenstein("abababab", "aaaaaaaa", bound=2) == 2


def test_construct():
    native = "aaaaa"
    big = Str(native)
    assert len(big) == len(native)


def test_indexing():
    native = "abcdef"
    big = Str(native)
    for i in range(len(native)):
        assert big[i] == native[i]


# def test_contains():
#     big = Str("abcdef")
#     assert "a" in big
#     assert "ab" in big
#     assert "xxx" not in big


# def test_rich_comparisons():
#     assert Str("aa") == "aa"
#     assert Str("aa") < "b"
#     assert Str("abb")[1:] == "bb"


# def get_random_string(
#     length: Optional[int] = None, variability: Optional[int] = None
# ) -> str:
#     if length is None:
#         length = randint(3, 300)
#     if variability is None:
#         variability = len(ascii_lowercase)
#     return "".join(choice(ascii_lowercase[:variability]) for _ in range(length))


# def is_equal_strings(native_strings, big_strings):
#     for native_slice, big_slice in zip(native_strings, big_strings):
#         assert native_slice == big_slice


# def check_identical(
#     native: str,
#     big: Union[Str, File],
#     needle: Optional[str] = None,
#     check_iterators: bool = False,
# ):
#     if needle is None:
#         part_offset = randint(0, len(native) - 1)
#         part_length = randint(1, len(native) - part_offset)
#         needle = native[part_offset:part_length]

#     present_in_native: bool = needle in native
#     present_in_big = needle in big
#     assert present_in_native == present_in_big
#     assert native.find(needle) == big.find(needle)
#     assert native.count(needle) == big.count(needle)

#     native_strings = native.split(needle)
#     big_strings: Strs = big.split(needle)
#     assert len(native_strings) == len(big_strings)

#     if check_iterators:
#         for i in range(len(native_strings)):
#             assert len(native_strings[i]) == len(big_strings[i])
#             assert native_strings[i] == big_strings[i]
#             assert [c for c in native_strings[i]] == [c for c in big_strings[i]]

#     is_equal_strings(native_strings, big_strings)


# @pytest.mark.parametrize("haystack_length", range(1, 65))
# @pytest.mark.parametrize("variability", range(1, 25))
# def test_contains(haystack_length: int, variability: int):
#     native = get_random_string(variability=variability, length=haystack_length)
#     big = Str(native)
#     pattern = get_random_string(variability=variability, length=randint(1, 5))
#     assert (pattern in native) == big.contains(pattern)


# def test_count_overlap():
#     native = "aaaaa"
#     big = Str(native)
#     assert native.count("aa") == big.count("aa")
#     assert 4 == big.count("aa", allowoverlap=True)


# def test_splitlines():
#     native = "line1\nline2\nline3"
#     big = Str(native)
#     assert native.splitlines() == list(big.splitlines())
#     assert native.splitlines(True) == list(big.splitlines(keeplinebreaks=True))


def test_split_keepseparator():
    native = "word1_word2_word3"
    big = Str(native)

    words = sz.split(big, "_")
    assert len(words) == 3

    parts = sz.split(big, "_", keepseparator=True)
    assert len(parts) == 3

    assert str(words[0]) == "word1"
    assert str(parts[0]) == "word1_"
    assert str(words[2]) == "word3"
    assert str(parts[2]) == "word3"


# def test_strs_operations():
#     native = "line1\nline2\nline3"
#     big = Str(native)
#     lines = big.splitlines()
#     lines.sort()
#     assert ["line1", "line2", "line3"] == list(lines)

#     shuffled_copy = lines.shuffled(seed=42)
#     assert set(lines) == set(shuffled_copy)

#     lines.append("line4")
#     assert 4 == len(lines)
#     lines.extend(["line5", "line6"])
#     assert 6 == len(lines)

#     lines.append(lines[0])
#     assert 7 == len(lines)
#     assert lines[6] == "line1"

#     lines.extend(lines)
#     assert 14 == len(lines)
#     assert lines[7] == "line1"
#     assert lines[8] == "line2"
#     assert lines[12] == "line6"

#     # Test that shuffles are reproducible with the same `seed`
#     a = [str(s) for s in lines.shuffled(seed=42)]
#     b = [str(s) for s in lines.shuffled(seed=42)]
#     assert a == b


# @pytest.mark.parametrize("repetitions", range(1, 10))
# def test_basic(repetitions: int):
#     native = "abcd" * repetitions
#     big = Str(native)

#     check_identical(native, big, "a", True)
#     check_identical(native, big, "ab", True)
#     check_identical(native, big, "abc", True)
#     check_identical(native, big, "abcd", True)
#     check_identical(native, big, "abcde", True)  # Missing pattern


# @pytest.mark.parametrize("pattern_length", [1, 2, 4, 5])
# @pytest.mark.parametrize("haystack_length", range(1, 69, 3))
# @pytest.mark.parametrize("variability", range(1, 27, 3))
# def test_fuzzy(pattern_length: int, haystack_length: int, variability: int):
#     native = get_random_string(variability=variability, length=haystack_length)
#     big = Str(native)

#     # Start by matching the prefix and the suffix
#     check_identical(native, big, native[:pattern_length])
#     check_identical(native, big, native[-pattern_length:])

#     # Continue with random strs
#     for _ in range(haystack_length // pattern_length):
#         pattern = get_random_string(variability=variability, length=pattern_length)
#         check_identical(native, big, pattern)


# def test_strs():
#     native = get_random_string(length=10)
#     big = Str(native)

#     assert native[0:5] == big.sub(0, 5) and native[0:5] == big[0:5]
#     assert native[5:10] == big.sub(5, 10) and native[5:10] == big[5:10]

#     assert native[5:5] == big.sub(5, 5) and native[5:5] == big[5:5]
#     assert native[-5:-5] == big.sub(-5, -5) and native[-5:-5] == big[-5:-5]
#     assert native[2:-2] == big.sub(2, -2) and native[2:-2] == big[2:-2]
#     assert native[7:-7] == big.sub(7, -7) and native[7:-7] == big[7:-7]

#     assert native[5:3] == big.sub(5, 3) and native[5:3] == big[5:3]
#     assert native[5:7] == big.sub(5, 7) and native[5:7] == big[5:7]
#     assert native[5:-3] == big.sub(5, -3) and native[5:-3] == big[5:-3]
#     assert native[5:-7] == big.sub(5, -7) and native[5:-7] == big[5:-7]

#     assert native[-5:3] == big.sub(-5, 3) and native[-5:3] == big[-5:3]
#     assert native[-5:7] == big.sub(-5, 7) and native[-5:7] == big[-5:7]
#     assert native[-5:-3] == big.sub(-5, -3) and native[-5:-3] == big[-5:-3]
#     assert native[-5:-7] == big.sub(-5, -7) and native[-5:-7] == big[-5:-7]

#     assert native[2:] == big.sub(2) and native[2:] == big[2:]
#     assert native[:7] == big.sub(end=7) and native[:7] == big[:7]
#     assert native[-2:] == big.sub(-2) and native[-2:] == big[-2:]
#     assert native[:-7] == big.sub(end=-7) and native[:-7] == big[:-7]
#     assert native[:-10] == big.sub(end=-10) and native[:-10] == big[:-10]
#     assert native[:-1] == big.sub(end=-1) and native[:-1] == big[:-1]

#     length = 1000
#     native = get_random_string(length=length)
#     big = Str(native)

#     needle = native[0 : randint(2, 5)]
#     native_strings = native.split(needle)
#     big_strings: Strs = big.split(needle)

#     length = len(native_strings)
#     for i in range(length):
#         start = randint(1 - length, length - 1)
#         stop = randint(1 - length, length - 1)
#         step = 0
#         while step == 0:
#             step = randint(-int(math.sqrt(length)), int(math.sqrt(length)))

#         is_equal_strings(native_strings[start:stop:step], big_strings[start:stop:step])
#         is_equal_strings(
#             native_strings[start:stop:step],
#             big_strings.sub(start, stop, step),
#         )


# def test_levenstein():
#     # Create a new string by slicing and concatenating
#     def insert_char_at(s, char_to_insert, index):
#         return s[:index] + char_to_insert + s[index:]

#     for _ in range(100):
#         a = get_random_string(length=20)
#         b = a
#         for i in range(150):
#             source_offset = randint(0, len(ascii_lowercase) - 1)
#             target_offset = randint(0, len(b) - 1)
#             b = insert_char_at(b, ascii_lowercase[source_offset], target_offset)
#             assert levenstein(a, b, 200) == i + 1
