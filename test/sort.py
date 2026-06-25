#!/usr/bin/env python3
"""Sequence sorting tests: Strs.sorted / argsort (top-K, uncased, out-buffer) and fuzzy sort equivalence.

Mirrors the C++ test/sort.cpp translation unit.
"""

import pytest

from stringzilla import Str

from test.helpers import SEED_VALUES, seed_random_generators, get_random_string, numpy_available

# NumPy is optional; the defensive naked `except` mirrors the old monolith (PyPy raises non-ImportError).
try:
    import numpy as np
except:  # noqa: E722
    pass


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
