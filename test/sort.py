#!/usr/bin/env python3
"""Sequence sorting: Strs.sorted / argsort, top-K and reverse selection, uncased ordering, and the
argsort(out=...) buffer protocol.

Mirrors the C++ test/sort.cpp translation unit.

Covers: Strs.sorted() and argsort() over Strs collections, repr/str truncation, top-K and reverse
ordering, uncased case-folded sorting, the argsort(out=...) buffer protocol and its rejections,
Strs.intersect() position pairing and duplicate collapsing, sampling, and a batch-size and
corpus-shape sweep across every capability_sweep() backend.
Compares against: CPython sorted() in byte mode, and cross-backend permutation agreement across
every capability_sweep() backend.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/sort.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/sort.py -q
"""

from random import randint

import pytest

from stringzilla import Str, Strs

from test.sz_helpers import (
    SEED_VALUES,
    seed_random_generators,
    get_random_string,
    numpy_available,
    run_across_backends,
    assert_backends_agree,
)

# NumPy is optional; the naked `except` also catches PyPy, which fails this import with an error
# other than `ImportError`.
try:
    import numpy as np
except:  # noqa: E722
    pass


# region Unit


def test_unit_strs_sequence():
    """`Strs` sequence smoke test: splitlines, repr/str truncation on large arrays, sorted()/argsort()
    with reverse and top-K, uncased stable ordering that keeps fold-equal strings in input order, the
    keyword-only argument surface, and sample()."""
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


def test_unit_strs_intersect():
    """`Strs.intersect` returns parallel position tuples mapping equal strings across two collections,
    matching each distinct shared value exactly once even when either side holds duplicates."""
    first = Str("banana\napple\ncherry").splitlines()
    second = Str("cherry\norange\npineapple\nbanana").splitlines()

    ours, theirs = first.intersect(second)
    assert len(ours) == len(theirs) == 2
    assert {str(first[i]) for i in ours} == {"banana", "cherry"}
    for i, j in zip(ours, theirs):
        assert str(first[i]) == str(second[j])

    # Seeds only reshuffle the hash table; the matched set is identical.
    seeded_ours, seeded_theirs = first.intersect(second, seed=42)
    assert sorted(seeded_ours) == sorted(ours)
    assert sorted(seeded_theirs) == sorted(theirs)

    # No overlap and empty inputs produce empty tuples.
    assert first.intersect(Str("kiwi\nmango").splitlines()) == ((), ())
    assert first.intersect(Str("").splitlines()) == ((), ())

    # Duplicates on either side collapse to a single match per distinct value.
    dup_ours, dup_theirs = Str("apple\napple\nbanana").splitlines().intersect(Str("apple\napple").splitlines())
    assert len(dup_ours) == len(dup_theirs) == 1

    # Only Strs-to-Strs intersections are supported.
    try:
        first.intersect(["banana"])
        assert False, "intersect() should reject non-Strs arguments"
    except TypeError:
        pass


def test_unit_strs_argsort_out():
    """`argsort(out=...)` writes the permutation into a caller pointer-width buffer and returns it."""
    import array
    import struct

    strs = Str("banana\napple\ncherry\nApple\nBANANA").splitlines()
    n = len(strs)

    # `out` must match `sz_sorted_idx_t` exactly: one pointer-width unsigned integer per index.
    index_code = next(code for code in "LQ" if array.array(code).itemsize == struct.calcsize("P"))
    buf = array.array(index_code, [0] * n)
    assert strs.argsort(uncased=True, out=buf) is buf
    assert tuple(buf) == strs.argsort(uncased=True)

    # Top-K writes exactly `top` indices and leaves the rest of a wider buffer untouched.
    wide = array.array(index_code, [12345] * n)
    strs.argsort(top=2, out=wide)
    assert tuple(wide[:2]) == strs.argsort(top=2)
    assert all(x == 12345 for x in wide[2:]), "argsort(out=...) clobbered past `top`"

    # Rejections: wrong itemsize (2-byte `'H'` on every platform), undersized, read-only, and
    # `sorted()` has no `out=`.
    with pytest.raises(TypeError):
        strs.argsort(out=array.array("H", [0] * n))
    with pytest.raises(ValueError):
        strs.argsort(out=array.array(index_code, [0] * (n - 1)))
    with pytest.raises((TypeError, BufferError)):
        strs.argsort(out=bytes(8 * n))
    with pytest.raises(TypeError):
        strs.sorted(out=buf)


# endregion Unit


# region Oracles


def randomly_cased(text: str) -> str:
    """Flip each character's case with 50% probability, so `uncased=True` sorting has something to
    fold while the byte-case oracle stays well-defined via `str.casefold`."""
    return "".join(char.upper() if randint(0, 1) else char for char in text)


def assert_sort_family_matches_oracles(native_list: list, *, top=None, reverse: bool = False, uncased: bool = False):
    """Sweep every backend for `.sorted()`/`.argsort()` on one batch, asserting backend parity, the
    permutation/order relationship, and (for both byte and uncased modes) a CPython oracle."""
    strs = Strs(native_list)

    sorted_results = run_across_backends(lambda: list(map(str, strs.sorted(top=top, reverse=reverse, uncased=uncased))))
    argsort_results = run_across_backends(lambda: strs.argsort(top=top, reverse=reverse, uncased=uncased))

    assert_backends_agree(sorted_results)
    assert_backends_agree(argsort_results)

    sorted_order = next(iter(sorted_results.values()))
    permutation = next(iter(argsort_results.values()))
    # The permutation must reorder the original list to exactly the reported sorted order.
    assert [native_list[index] for index in permutation] == sorted_order

    key = (lambda item: item.casefold()) if uncased else None
    expected_full = sorted(native_list, key=key, reverse=reverse)
    expected_count = len(native_list) if top is None else min(top, len(native_list))
    assert sorted_order == expected_full[:expected_count]


@pytest.mark.parametrize("list_length", [10, 20, 30, 40, 50])
@pytest.mark.parametrize("part_length", [5, 10])
@pytest.mark.parametrize("variability", [2, 3])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_fuzzy_sorting(list_length: int, part_length: int, variability: int, seed_value: int):
    """Fuzzed `Strs.argsort()`/`.sorted()` agree with Python's native `sorted()` on randomized batches:
    the pairwise `<`/`>` comparator, the permutation, and the re-split substrings all match, and the
    `argsort(out=...)` NumPy buffer path returns the same permutation."""
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


# endregion Oracles


# region Backend differential


# Batch sizes bracketing: a single element, a pair, just past a typical SIMD/threading tile (33, 64),
# and a size large enough to engage any parallel/chunked sort path (1000).
BATCH_SIZES = [1, 2, 33, 64, 1000]


def random_batch(size: int) -> list:
    """A batch of random-length, randomly-cased small-alphabet strings (frequent ties and prefixes)."""
    return [randomly_cased(get_random_string(variability=6, length=randint(0, 16))) for _ in range(size)]


def all_equal_batch(size: int) -> list:
    """A batch where every element is byte-for-byte identical, the sort must stay stable / no-op."""
    return ["repeated"] * size


def single_char_batch(size: int) -> list:
    """A batch of single-character (possibly differently-cased) strings."""
    return [randomly_cased(get_random_string(variability=20, length=1)) for _ in range(size)]


def empty_mixed_batch(size: int) -> list:
    """A batch interleaving empty strings with non-empty ones, every third element empty."""
    return [
        "" if index % 3 == 0 else randomly_cased(get_random_string(variability=6, length=randint(1, 10)))
        for index in range(size)
    ]


BATCH_BUILDERS = {
    "random": random_batch,
    "all_equal": all_equal_batch,
    "single_char": single_char_batch,
    "empty_mixed": empty_mixed_batch,
}


@pytest.mark.parametrize("uncased", [False, True])
@pytest.mark.parametrize("batch_name", sorted(BATCH_BUILDERS))
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_sorted_argsort(seed_value, batch_size, batch_name, uncased):
    """`Strs.sorted()`/`.argsort()` agree across every `capability_sweep()` backend and with CPython
    `sorted()`, byte mode or case-folded when `uncased=True`, across batch sizes bracketing
    SIMD/threading thresholds and the all-equal, single-char, length-1, and empty-mixed corner shapes;
    a divergence is a kernel bug, not a binding bug."""
    seed_random_generators(seed_value)
    native_list = BATCH_BUILDERS[batch_name](batch_size)
    assert_sort_family_matches_oracles(native_list, uncased=uncased)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("top", [None, 1, 5, 50, 10_000])
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_sorted_argsort_top_reverse(seed_value, batch_size, top, reverse):
    """`top=K` partial sort and `reverse=True` descending sort agree across every `capability_sweep()`
    backend and with CPython `sorted()[:K]`, for `top` values from below the batch size up to 10_000,
    far beyond it; a divergence is a kernel bug, not a binding bug."""
    seed_random_generators(seed_value)
    native_list = random_batch(batch_size)
    assert_sort_family_matches_oracles(native_list, top=top, reverse=reverse)


def test_unit_backend_differential_sorted_argsort_empty():
    """An empty `Strs` collection sorts to empty on every `capability_sweep()` backend for every
    `uncased`/`reverse` combination; a divergence is a kernel bug, not a binding bug."""
    for uncased in (False, True):
        for reverse in (False, True):
            assert_sort_family_matches_oracles([], uncased=uncased, reverse=reverse)


# endregion Backend differential
