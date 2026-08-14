#!/usr/bin/env python3
"""Substrings: multi-pattern Aho-Corasick search via `szs.Substrings` construction and its four verbs.

Mirrors the C++ test/substrings.cuh translation unit.

Covers: engine construction from every `Strs` layout, hand-pinned counts and match records for the
classic `he`/`she`/`his`/`hers` automaton, the three overlap policies, full Unicode case folding,
BM25 against hand-computed values, rewriting including the self-replacement identity, the arithmetic
`replace_bound` ceiling, and haystacks large enough to cross the per-core slicing path.
Compares against: hand-written expectations, `sz.utf8_uncased_matches` as the single-pattern oracle the
folded walk must agree with, every other device scope as a backend differential, and `pyahocorasick` as
an independent automaton when it imports.

Run:
    uv pip install numpy pytest
    uv pip install --group differential      # optional, unlocks the third-party differential
    SZ_TARGET=stringzillas-cpus uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/substrings.py -q
    SZ_TARGET=stringzillas-cuda uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/substrings.py -q
"""

import random
import sys

import pytest
import numpy as np

import stringzilla as sz
import stringzillas as szs
from stringzilla import Strs

from test.sz_helpers import SEED_VALUES, malformed_utf8_corpus, scale_iterations, seed_random_generators
from test.szs_helpers import DEVICE_NAMES, device_scope_and_capabilities

CLASSIC_NEEDLES = ["he", "she", "his", "hers"]
CLASSIC_HAYSTACKS = ["ushers", "nothing", "hishers"]

#: Every walk the engine offers, so a sweep names them rather than repeating three literals.
POLICIES = ["overlapping", "leftmost-longest", "leftmost-first"]

#: The two covers, which are the only policies a rewrite accepts.
COVER_POLICIES = ["leftmost-longest", "leftmost-first"]

#: Every scope the differential region sweeps, each paired with the capabilities its engine is built
#: for. The first entry is the reference every other must reach.
DIFFERENTIAL_DEVICE_SCOPES = [(name, *device_scope_and_capabilities(name)) for name in DEVICE_NAMES]


def matches_as_tuples(engine, haystacks, **kwargs):
    """Zips the four columns `find` returns into records, so expectations read as literals."""
    haystack_ix, needle_ix, offsets, lengths = engine.find(haystacks, **kwargs)
    return sorted((int(h), int(n), int(o), int(l)) for h, n, o, l in zip(haystack_ix, needle_ix, offsets, lengths))


def random_ascii_corpus(rng: random.Random, documents: int, length: int, alphabet: str = "abcdefg ") -> list:
    """A corpus drawn from a tiny alphabet, so needles of two or three letters hit often enough to
    exercise the walk rather than measuring an empty one."""
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, length))) for _ in range(documents)]


# region Unit


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_counts_and_matches(device_name: str):
    """The textbook automaton, pinned by hand: `ushers` holds `she`, `he` and `hers`, `hishers`
    additionally holds `his`, and `nothing` holds none of them."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES), device=device, capabilities=capabilities)

    counts, total = engine.count(Strs(CLASSIC_HAYSTACKS), device=device)
    assert list(int(c) for c in counts) == [3, 0, 4]
    assert total == 7
    assert counts.dtype == np.uint64

    found = matches_as_tuples(engine, Strs(CLASSIC_HAYSTACKS), device=device)
    assert found == [
        (0, 0, 2, 2),  # ushers -> "he"
        (0, 1, 1, 3),  # ushers -> "she"
        (0, 3, 2, 4),  # ushers -> "hers"
        (2, 0, 3, 2),  # hishers -> "he"
        (2, 1, 2, 3),  # hishers -> "she"
        (2, 2, 0, 3),  # hishers -> "his"
        (2, 3, 3, 4),  # hishers -> "hers"
    ]


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_leftmost_policies(device_name: str):
    """A cover keeps no two matches sharing a byte. Longest prefers the widest match at the earliest
    start, first prefers the lowest needle index there, and both are subsets of the overlapping walk."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES), device=device, capabilities=capabilities)
    haystacks = Strs(CLASSIC_HAYSTACKS)

    overlapping = set(matches_as_tuples(engine, haystacks, device=device, policy="overlapping"))
    longest = matches_as_tuples(engine, haystacks, device=device, policy="leftmost-longest")
    first = matches_as_tuples(engine, haystacks, device=device, policy="leftmost-first")

    assert set(longest) <= overlapping
    assert set(first) <= overlapping
    # Longest *at the leftmost start*, not the longest overall: `ushers` commits `she` at 1, which rules
    # out the wider `hers` at 2 because the two would share bytes.
    assert longest == [(0, 1, 1, 3), (2, 2, 0, 3), (2, 3, 3, 4)]
    # First takes the same starts but the lowest needle index there, so `hishers` commits `his` at 0 and
    # then `he` at 3 rather than the wider `hers`.
    assert first == [(0, 1, 1, 3), (2, 0, 3, 2), (2, 2, 0, 3)]
    # Each cover is disjoint within itself; the two covers are different answers, so they are not
    # compared against each other.
    for cover in (longest, first):
        by_haystack = {}
        for haystack_index, _, offset, length in cover:
            spans = by_haystack.setdefault(haystack_index, [])
            for other_offset, other_length in spans:
                assert offset + length <= other_offset or other_offset + other_length <= offset
            spans.append((offset, length))


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_uncased_folds_both_sides(device_name: str):
    """Folding applies to needle and haystack alike, so a capitalized corpus matches a lowercase
    dictionary, and the byte-exact engine over the same inputs does not."""
    device, capabilities = device_scope_and_capabilities(device_name)
    haystacks = Strs(["The Hershey Company"])

    cased = szs.Substrings(Strs(["hershey"]), device=device, capabilities=capabilities)
    uncased = szs.Substrings(Strs(["hershey"]), case_sensitivity="uncased", device=device, capabilities=capabilities)
    assert cased.count(haystacks, device=device)[1] == 0
    assert uncased.count(haystacks, device=device)[1] == 1


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_uncased_conformance_table(device_name: str):
    """Full `CaseFolding.txt` conformance, ported from the C++ table. Folding is not length-preserving,
    so a needle matches spans of a byte length its own does not predict, and the equivalence classes are
    exactly the ones status codes `C` and `F` draw."""
    device, capabilities = device_scope_and_capabilities(device_name)

    # Corruption guard: the two load-bearing fixtures re-spelled as bytes, so a tool that silently
    # renormalizes the escapes below is caught here rather than in a confusing match count.
    assert "ß".encode() == b"\xc3\x9f", "Sharp S literal corrupted"
    assert "K".encode() == b"\xe2\x84\xaa", "Kelvin sign literal corrupted"

    # | needle | must match                                              | must not match |
    table = [
        ("ss", ["ss", "SS", "sS", "Ss", "ß", "ẞ"], ["s"]),
        ("ß", ["ß", "ẞ", "ss", "SS", "sS", "Ss"], ["s"]),
        ("K", ["K", "k", "tempKvalue"], []),
        ("Å", ["Å", "å", "tempÅvalue"], ["Å"]),
        ("İ", ["i̇"], ["i", "I"]),
        ("é", ["é", "É"], ["é"]),
        # A length-changing fold in the MIDDLE of a needle, so the byte-delta state keying that
        # reconverges variable-length preimages is exercised mid-walk rather than only at acceptance.
        ("weißrd", ["weissrd", "weiSSrd", "weiẞrd"], ["weisrd", "weird"]),
    ]
    for needle, matching, missing in table:
        engine = szs.Substrings(Strs([needle]), case_sensitivity="uncased", device=device, capabilities=capabilities)
        for haystack in matching:
            assert engine.count(Strs([haystack]), device=device)[1] >= 1, f"{needle!r} must match {haystack!r}"
        for haystack in missing:
            assert engine.count(Strs([haystack]), device=device)[1] == 0, f"{needle!r} must miss {haystack!r}"


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_uncased_matches_single_pattern_engine(device_name: str):
    """Agreement with `sz.utf8_uncased_matches` is the entire semantic claim of the uncased mode: a
    one-needle dictionary must report exactly the spans the shipped single-pattern engine reports."""
    device, capabilities = device_scope_and_capabilities(device_name)
    needles = ["ss", "ß", "K", "Å", "İ", "é", "weißrd", "the"]
    haystacks = [
        "the Straße was STRASSE",
        "tempKvalue and K and k",
        "Ångström Å å",
        "weißrd weissrd weiSSrd",
        "İstanbul i̇ I i",
        "café CAFÉ café",
    ]

    for needle in needles:
        engine = szs.Substrings(Strs([needle]), case_sensitivity="uncased", device=device, capabilities=capabilities)
        for haystack in haystacks:
            view = sz.Str(haystack)
            oracle = sorted(
                (match.offset_within(view), match.nbytes)
                for match in sz.utf8_uncased_matches(view, needle, include_overlapping=True)
            )
            found = [
                (offset, length) for _, _, offset, length in matches_as_tuples(engine, Strs([haystack]), device=device)
            ]
            assert found == oracle, f"needle {needle!r} over {haystack!r}"


# endregion Unit


# region Scoring


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_score_bm25_hand_computed(device_name: str):
    """The exact values the C++ suite pins, so a constant both implementations share cannot hide."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(["cat", "dog"]), device=device, capabilities=capabilities)
    haystacks = Strs(["catcat", "dog", "nothing"])
    weights = np.array([1.0, 2.0], dtype=np.float32)

    scores = engine.score_bm25(haystacks, weights, 6.0, device=device)
    # "catcat" is 6 bytes against a 6-byte mean, so the length term is exactly one and "cat" twice scores
    #   1.0 * 2 * (1.2 + 1) / (2 + 1.2 * 1) = 4.4 / 3.2
    assert abs(float(scores[0]) - 4.4 / 3.2) < 1e-5, "Two occurrences must saturate, not double"
    # "dog" is 3 bytes against the same mean, so the length term is 1 - 0.75 + 0.75 * 3 / 6 = 0.625, and
    # the needle's own weight of two scales the whole term.
    assert abs(float(scores[1]) - 2.0 * 2.2 / (1.0 + 1.2 * 0.625)) < 1e-5, "A weight scales its whole term"
    assert float(scores[2]) == 0.0, "A haystack no needle hits scores exactly zero"

    # Bit-stability: the same call on the same engine must reproduce every score exactly.
    for _ in range(scale_iterations(4)):
        repeated = engine.score_bm25(haystacks, weights, 6.0, device=device)
        assert np.array_equal(repeated, scores), "Scores must be bit-identical across runs of one backend"

    # Omitted `document_lengths` means byte lengths, which the caller can also state outright.
    byte_lengths = np.array([6.0, 3.0, 7.0], dtype=np.float32)
    stated = engine.score_bm25(haystacks, weights, 6.0, device=device, document_lengths=byte_lengths)
    assert np.array_equal(stated, scores), "Byte lengths stated outright must score identically"

    # A zero weight removes its needle from the ranking without removing it from the automaton.
    muted = engine.score_bm25(haystacks, np.zeros(2, dtype=np.float32), 6.0, device=device)
    assert not muted.any(), "Zero weights must score zero"


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_score_bm25_orders_documents(device_name: str):
    """A document holding none of the query terms scores exactly zero, one holding more of them scores
    higher, and the score rises with a term's weight."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES), device=device, capabilities=capabilities)
    haystacks = Strs(CLASSIC_HAYSTACKS)

    uniform = np.ones(len(CLASSIC_NEEDLES), dtype=np.float32)
    scores = engine.score_bm25(haystacks, uniform, 6.0, device=device)
    assert scores.dtype == np.float32 and len(scores) == 3
    assert scores[1] == 0.0  # "nothing" holds no needle
    assert scores[2] > scores[0] > 0.0  # "hishers" holds four, "ushers" three

    louder = engine.score_bm25(haystacks, uniform * 2.0, 6.0, device=device)
    assert louder[0] > scores[0] and louder[1] == 0.0


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_score_bm25_checks_the_weight_count(device_name: str):
    """One weight per needle, no more and no fewer, since the engine reads the dictionary's width."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES), device=device, capabilities=capabilities)

    with pytest.raises(Exception):
        engine.score_bm25(Strs(CLASSIC_HAYSTACKS), np.ones(2, dtype=np.float32), 6.0, device=device)


# endregion Scoring


# region Rewriting


@pytest.mark.parametrize("policy", COVER_POLICIES)
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_replace(device_name: str, policy: str):
    """Each match is substituted by its needle's replacement under a cover, and the offsets delimit one
    rewritten haystack each. Both covers commit `she` in `ushers`, so both rewrite it the same way."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES), device=device, capabilities=capabilities)
    haystacks = Strs(CLASSIC_HAYSTACKS)
    replacements = Strs(["[HE]", "[SHE]", "[HIS]", "[HERS]"])

    data, offsets = engine.replace(haystacks, replacements, device=device, policy=policy)
    assert len(offsets) == len(CLASSIC_HAYSTACKS) + 1
    pieces = [data[offsets[i] : offsets[i + 1]].decode() for i in range(len(CLASSIC_HAYSTACKS))]
    assert pieces[0] == "u[SHE]rs"
    assert pieces[1] == "nothing"
    # `hishers` commits `his` at 0 either way, then the widest match at 3 under longest and the lowest
    # needle index there under first.
    assert pieces[2] == ("[HIS][HERS]" if policy == "leftmost-longest" else "[HIS][HE]rs")


@pytest.mark.parametrize("policy", COVER_POLICIES)
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_replace_identity(device_name: str, policy: str):
    """Replacing every needle with itself must reproduce the input byte for byte - an oracle that needs
    no second implementation to be trustworthy, and one that holds under either cover."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES), device=device, capabilities=capabilities)
    haystacks = Strs(CLASSIC_HAYSTACKS)

    data, offsets = engine.replace(haystacks, Strs(CLASSIC_NEEDLES), device=device, policy=policy)
    pieces = [data[offsets[i] : offsets[i + 1]].decode() for i in range(len(CLASSIC_HAYSTACKS))]
    assert pieces == CLASSIC_HAYSTACKS


def test_substrings_replace_deletes_on_empty():
    """An empty replacement deletes its match rather than being refused."""
    engine = szs.Substrings(Strs(["cat"]))
    data, offsets = engine.replace(Strs(["a cat here"]), Strs([""]))
    assert data[offsets[0] : offsets[1]].decode() == "a  here"


def test_substrings_replace_inserts_verbatim_under_folding():
    """No case adaptation: every fold preimage takes the replacement's exact bytes, so a rewrite over a
    3-byte Kelvin sign shrinks."""
    engine = szs.Substrings(Strs(["k"]), case_sensitivity="uncased")
    data, offsets = engine.replace(Strs(["KKk"]), Strs(["x"]))
    assert data[offsets[0] : offsets[1]] == b"xxx"


def test_substrings_replace_bound_covers_the_widest_expansion():
    """The bound is arithmetic over the needle set, so sizing a tape to it makes the rewrite a single
    call that cannot be refused."""
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES))
    haystacks = Strs(CLASSIC_HAYSTACKS)
    replacements = Strs(["[HE]", "[SHE]", "[HIS]", "[HERS]"])

    input_bytes = sum(len(h) for h in CLASSIC_HAYSTACKS)
    bound = engine.replace_bound(replacements, input_bytes)
    data, _ = engine.replace(haystacks, replacements)
    assert bound >= len(data) and bound >= input_bytes

    # "a" expands 1 byte into 4 and "bb" expands 2 into 4, so the widest ratio is 4 and a fully tiled
    # 8-byte haystack cannot exceed 32; a dictionary that only shrinks still bounds at the input length.
    assert szs.Substrings(Strs(["a", "bb"])).replace_bound(Strs(["wxyz", "wxyz"]), 8) == 32
    assert szs.Substrings(Strs(["aaaa"])).replace_bound(Strs(["z"]), 8) == 8


# endregion Rewriting


# region Corner cases


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_empty_and_missing(device_name: str):
    """A corpus with no match scores zero everywhere, and an empty needle is refused outright."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES), device=device, capabilities=capabilities)

    counts, total = engine.count(Strs(["xyz", "abc"]), device=device)
    assert total == 0 and list(int(c) for c in counts) == [0, 0]
    assert len(engine.find(Strs(["xyz"]), device=device)[0]) == 0

    with pytest.raises(Exception):
        szs.Substrings(Strs(["ok", ""]), device=device, capabilities=capabilities)


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_empty_batch(device_name: str):
    """A batch of no haystacks is a legal batch: every verb returns an empty result rather than failing."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES), device=device, capabilities=capabilities)
    empty = Strs([])

    counts, total = engine.count(empty, device=device)
    assert len(counts) == 0 and total == 0
    assert all(len(column) == 0 for column in engine.find(empty, device=device))
    assert len(engine.score_bm25(empty, np.ones(4, dtype=np.float32), 6.0, device=device)) == 0

    data, offsets = engine.replace(empty, Strs(CLASSIC_NEEDLES), device=device)
    assert len(data) == 0 and list(offsets) == [0]


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_degenerate_shapes(device_name: str):
    """One-byte haystacks, an all-same run that tiles, and a needle set of nested prefixes - the shapes
    that put a match at every position or none."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(["a", "aa", "aaa"]), device=device, capabilities=capabilities)

    counts, _ = engine.count(Strs(["", "a", "aaaa"]), device=device, policy="overlapping")
    # "aaaa" holds "a" four times, "aa" three times and "aaa" twice.
    assert list(int(c) for c in counts) == [0, 1, 9]

    # A cover tiles the same haystack with the widest needle it can commit at each start, then resumes
    # past it: "aaa" at 0, and the single "a" left over at 3.
    cover = matches_as_tuples(engine, Strs(["aaaa"]), device=device, policy="leftmost-longest")
    assert cover == [(0, 0, 3, 1), (0, 2, 0, 3)]


@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_substrings_large_haystacks(device_name: str):
    """A haystack far past any per-core slice, so matches that straddle a slice boundary are exercised.
    Every backend must reach the count a single-threaded scan of the same text reaches."""
    device, capabilities = device_scope_and_capabilities(device_name)
    engine = szs.Substrings(Strs(["cat", "dog", "concatenate"]), device=device, capabilities=capabilities)

    unit = "cat dog concatenate at the "
    text = unit * (256 * 1024 // len(unit))
    # "concatenate" contains "cat", and each unit holds one of each plus the "cat" inside "concatenate".
    per_unit = 4
    repeats = len(text) // len(unit)

    counts, total = engine.count(Strs([text, text]), device=device, policy="overlapping")
    assert list(int(c) for c in counts) == [per_unit * repeats, per_unit * repeats]
    assert total == 2 * per_unit * repeats

    # Self-replacement must survive every slice boundary byte for byte.
    data, offsets = engine.replace(Strs([text]), Strs(["cat", "dog", "concatenate"]), device=device)
    assert data[offsets[0] : offsets[1]].decode() == text


# endregion Corner cases


# region Backend differential


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_substrings_backend_differential(seed_value: int, policy: str):
    """Every scope must reach the answer the default scope reaches, over a randomized corpus. The scopes
    differ in how a batch is split across cores and devices, never in what a walk reports."""
    seed_random_generators(seed_value)
    rng = random.Random(seed_value)
    needles = sorted(
        {"".join(rng.choice("abcdefg") for _ in range(rng.randint(1, 4))) for _ in range(scale_iterations(12))}
    )
    corpus = random_ascii_corpus(rng, documents=24, length=200)
    haystacks = Strs(corpus)

    reference = None
    for name, device, capabilities in DIFFERENTIAL_DEVICE_SCOPES:
        engine = szs.Substrings(Strs(needles), device=device, capabilities=capabilities)
        counts, total = engine.count(haystacks, device=device, policy=policy)
        answer = (
            list(int(c) for c in counts),
            total,
            matches_as_tuples(engine, haystacks, device=device, policy=policy),
        )
        if reference is None:
            reference, reference_name = answer, name
            assert answer[1] == sum(answer[0]), "The total must be the sum of the per-haystack counts"
            assert len(answer[2]) == total, "Every counted match must be locatable"
        else:
            assert answer == reference, f"{name} disagrees with {reference_name}"


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_substrings_covers_are_subsets_of_the_overlapping_walk(seed_value: int):
    """Over a randomized corpus, each cover is a disjoint subset of the overlapping walk - the property
    the policies are defined by, checked where hand-pinned fixtures cannot reach."""
    seed_random_generators(seed_value)
    rng = random.Random(seed_value)
    needles = sorted(
        {"".join(rng.choice("abc") for _ in range(rng.randint(1, 4))) for _ in range(scale_iterations(10))}
    )
    haystacks = Strs(random_ascii_corpus(rng, documents=16, length=120, alphabet="abc "))
    engine = szs.Substrings(Strs(needles))

    overlapping = set(matches_as_tuples(engine, haystacks, policy="overlapping"))
    for policy in COVER_POLICIES:
        cover = matches_as_tuples(engine, haystacks, policy=policy)
        assert set(cover) <= overlapping, f"{policy} reported a match the overlapping walk did not"
        spans = {}
        for haystack_index, _, offset, length in cover:
            taken = spans.setdefault(haystack_index, [])
            for other_offset, other_length in taken:
                assert offset + length <= other_offset or other_offset + other_length <= offset
            taken.append((offset, length))


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_substrings_replace_identity_over_random_corpora(seed_value: int):
    """Self-replacement is a byte-exact identity for any needle set and any corpus, so a randomized sweep
    of it catches every off-by-one a fixed fixture would not."""
    seed_random_generators(seed_value)
    rng = random.Random(seed_value)
    needles = sorted(
        {"".join(rng.choice("abcd") for _ in range(rng.randint(1, 3))) for _ in range(scale_iterations(8))}
    )
    corpus = random_ascii_corpus(rng, documents=12, length=150, alphabet="abcd ")
    engine = szs.Substrings(Strs(needles))

    for policy in COVER_POLICIES:
        data, offsets = engine.replace(Strs(corpus), Strs(needles), policy=policy)
        pieces = [data[offsets[i] : offsets[i + 1]].decode() for i in range(len(corpus))]
        assert pieces == corpus, f"self-replacement changed the corpus under {policy}"


def test_substrings_malformed_utf8():
    """Cased matching is over bytes and accepts anything; uncased matching is over codepoints and refuses
    a needle that is not well-formed UTF-8. Neither may crash on a malformed haystack."""
    malformed = malformed_utf8_corpus()

    cased = szs.Substrings(Strs([b"\xff\xfe", b"ab"]))
    counts, total = cased.count(Strs(malformed))
    assert total == sum(int(c) for c in counts)

    for needle in malformed:
        try:
            needle.decode("utf-8")
        except UnicodeDecodeError:
            with pytest.raises(Exception):
                szs.Substrings(Strs([needle]), case_sensitivity="uncased")
        else:
            # An embedded NUL is a legal codepoint, so a well-formed needle is accepted whatever it holds.
            szs.Substrings(Strs([needle]), case_sensitivity="uncased")

    # A well-formed uncased dictionary over malformed haystacks must still answer without crashing.
    uncased = szs.Substrings(Strs(["ab"]), case_sensitivity="uncased")
    uncased.count(Strs(malformed))


# endregion Backend differential


# region Third-party differential


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_stress_matches_pyahocorasick(seed_value: int):
    """An independent C automaton must report the same overlapping match set over a randomized corpus.

    ASCII only, since `pyahocorasick` indexes a `str` by codepoint while this engine reports bytes. Only
    `Automaton.iter` is used as an oracle: its `iter_long` drops a match it has already banked when the
    input ends mid-way through a longer candidate - with needles `a` and `dace` over `"da"` it reports
    nothing where `iter` reports `a` - so it does not witness the leftmost-longest cover. The Rust suite
    differentials all three covers against the `aho-corasick` crate instead.
    """
    ahocorasick = pytest.importorskip("ahocorasick")
    seed_random_generators(seed_value)
    rng = random.Random(seed_value)
    needles = sorted(
        {"".join(rng.choice("abcde") for _ in range(rng.randint(1, 4))) for _ in range(scale_iterations(12))}
    )
    corpus = random_ascii_corpus(rng, documents=20, length=180, alphabet="abcde ")
    engine = szs.Substrings(Strs(needles))

    automaton = ahocorasick.Automaton()
    for index, needle in enumerate(needles):
        automaton.add_word(needle, (index, len(needle)))
    automaton.make_automaton()

    oracle = sorted(
        (haystack_index, needle_index, end_index - needle_length + 1, needle_length)
        for haystack_index, haystack in enumerate(corpus)
        for end_index, (needle_index, needle_length) in automaton.iter(haystack)
    )
    assert matches_as_tuples(engine, Strs(corpus), policy="overlapping") == oracle


# endregion Third-party differential


# region Interop


def test_substrings_needle_layouts_agree():
    """Needles reach the engine as a callback-addressed sequence whatever layout they arrived in, so a
    tape-backed `Strs` and a sliced one must compile to the same automaton."""
    tape = Strs(CLASSIC_NEEDLES)
    fragmented = Strs(CLASSIC_NEEDLES + ["unused"])[:4]
    haystacks = Strs(CLASSIC_HAYSTACKS)

    from_tape = szs.Substrings(tape)
    from_fragmented = szs.Substrings(fragmented)
    assert matches_as_tuples(from_tape, haystacks) == matches_as_tuples(from_fragmented, haystacks)
    assert from_tape.count(haystacks)[1] == from_fragmented.count(haystacks)[1]


def test_substrings_haystack_layouts_agree():
    """Haystacks reach three different C entry points - a 32-bit tape, a 64-bit one, and the callback
    addressed sequence a reordered `Strs` becomes - and all three must report one answer."""
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES))
    tape = Strs(CLASSIC_HAYSTACKS)
    # A contiguous slice stays a tape view, so only a reordering step makes the fragmented layout; two
    # reversals restore the order while leaving the layout behind.
    fragmented = Strs(CLASSIC_HAYSTACKS)[::-1][::-1]

    assert matches_as_tuples(engine, tape) == matches_as_tuples(engine, fragmented)
    assert engine.count(tape)[1] == engine.count(fragmented)[1]

    # Rewriting is tape in, tape out, so the reordered layout is refused rather than silently reordered.
    with pytest.raises(TypeError):
        engine.replace(fragmented, Strs(CLASSIC_NEEDLES))


# endregion Interop


# region Argument surface


def test_substrings_rejects_wrong_types():
    """Every entry point names the conversion a caller needs rather than failing obscurely."""
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES))
    with pytest.raises(TypeError):
        engine.count(["not", "a", "Strs"])
    with pytest.raises(ValueError):
        engine.count(Strs(CLASSIC_HAYSTACKS), policy="sideways")
    with pytest.raises(ValueError):
        szs.Substrings(Strs(CLASSIC_NEEDLES), case_sensitivity="sideways")
    with pytest.raises(TypeError):
        engine.count(Strs(CLASSIC_HAYSTACKS), device="not a scope")
    with pytest.raises(ValueError):
        engine.replace(Strs(CLASSIC_HAYSTACKS), Strs(CLASSIC_NEEDLES), policy="overlapping")


def test_substrings_repr_and_capabilities():
    """`repr` names the dictionary's shape, and the engine reports what it was built for."""
    engine = szs.Substrings(Strs(CLASSIC_NEEDLES))
    assert "4 needles" in repr(engine) and "cased" in repr(engine)
    assert isinstance(engine.__capabilities__, tuple)


# endregion Argument surface


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", "-s", __file__]))
