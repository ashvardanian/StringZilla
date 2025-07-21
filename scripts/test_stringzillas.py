from random import choice, randint
from string import ascii_lowercase
from typing import Optional, Sequence, Dict
import tempfile
import os

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
    # PyArrow is not installed, most tests will be skipped
    pyarrow_available = False


def test_library_properties():
    assert len(sz.__version__.split(".")) == 3, "Semantic versioning must be preserved"
    assert "serial" in sz.__capabilities__.split(","), "Serial backend must be present"


def test_unit_globals():
    """Validates that the previously unit-tested member methods are also visible as global functions."""

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
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
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


def baseline_translate(body: str, lut: Sequence) -> str:
    return "".join([chr(lut[ord(c)]) for c in body])


def translation_table_to_dict(lut: Sequence) -> Dict[str, str]:
    return {chr(i): chr(lut[i]) for i in range(256)}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("length", range(1, 300))
def test_translations(length: int):

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
    assert sz.translate(body_bytes, view_identity) == body_bytes.translate(
        view_identity
    )
    assert sz.translate(body_bytes, view_invert) == body_bytes.translate(view_invert)
    assert sz.translate(body_bytes, view_threshold) == body_bytes.translate(
        view_threshold
    )

    # Check in-place translations - all of them return nothing
    after_identity = memoryview(body_bytes)
    assert sz.translate(after_identity, view_identity, inplace=True) == None
    assert sz.equal(after_identity, body.translate(dict_identity))
    after_invert = memoryview(body_bytes)
    assert sz.translate(after_invert, view_invert, inplace=True) == None
    assert sz.equal(after_invert, body.translate(dict_invert))
    after_threshold = memoryview(body_bytes)
    assert sz.translate(after_threshold, view_threshold, inplace=True) == None
    assert sz.equal(after_threshold, body.translate(dict_threshold))


@pytest.mark.repeat(3)
@pytest.mark.parametrize("length", list(range(0, 300)) + [1024, 4096, 100000])
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_translations_random(length: int):
    body = get_random_string(length=length)
    lut = np.random.randint(0, 256, size=256, dtype=np.uint8)
    assert sz.translate(body, memoryview(lut)) == baseline_translate(body, lut)


@pytest.mark.repeat(3)
@pytest.mark.parametrize("length", list(range(0, 300)) + [1024, 4096, 100000])
def test_bytesums_random(length: int):
    def sum_bytes(body: str) -> int:
        return sum([ord(c) for c in body])

    body = get_random_string(length=length)
    assert sum_bytes(body) == sz.bytesum(body)


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
    native_order = big_list.argsort()
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
    native_order = big_list.argsort()
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-x", "-s", __file__]))
