#!/usr/bin/env python3
"""UAX-29 grapheme-cluster tests: utf8_graphemes behavior, official GraphemeBreakTest conformance, and a
bit-exact differential against ICU (createCharacterInstance) plus uniseg/grapheme. Adds malformed/seam sweeps.

Mirrors the C++ scripts/test_utf8_graphemes.cpp translation unit.
"""

from random import Random, choice, randint, seed

import pytest

import stringzilla as sz
from stringzilla import Str

from test_helpers import (
    SEED_VALUES,
    UnicodeDataDownloadError,
    get_grapheme_break_properties,
    get_grapheme_break_test_cases,
    baseline_grapheme_boundaries,
)
from test_utf8_helpers import (
    SEGMENTATION_PALETTE,
    adversarial_utf8_inputs,
    assert_segments_tile,
    byte_boundaries,
    corpus_of_byte_length,
    icu_segmenter,
    window_seam_lengths,
)

_byte_boundaries = byte_boundaries
_SEGMENTATION_PALETTE = SEGMENTATION_PALETTE


#  region Grapheme iterator


def test_utf8_graphemes_basic():
    """Test basic grapheme cluster iteration."""
    # Plain ASCII: one cluster per character.
    result = [str(g) for g in sz.utf8_graphemes("hello")]
    assert result == ["h", "e", "l", "l", "o"]

    # Empty string.
    assert [str(g) for g in sz.utf8_graphemes("")] == []

    # Base letter + combining acute accent stays a single cluster.
    result = [str(g) for g in sz.utf8_graphemes("é")]
    assert result == ["é"]

    # Emoji ZWJ family (man + ZWJ + woman + ZWJ + girl) is one cluster.
    family = "\U0001f468‍\U0001f469‍\U0001f467"
    result = [str(g) for g in sz.utf8_graphemes(family)]
    assert result == [family]


def test_utf8_graphemes_skip_empty():
    """Test skip_empty parameter for grapheme iteration."""
    result = [str(g) for g in sz.utf8_graphemes("héllo", skip_empty=True)]
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)


def test_utf8_graphemes_unicode():
    """Test UTF-8 multi-byte / combining / emoji grapheme coverage."""
    # Precomposed vs decomposed: decomposed should still be a single cluster.
    assert [str(g) for g in sz.utf8_graphemes("ä")] == ["ä"]

    # Regional indicator pair forms a single flag cluster.
    flag = "\U0001f1e6\U0001f1fa"
    assert [str(g) for g in sz.utf8_graphemes(flag)] == [flag]

    # Emoji with skin-tone modifier is one cluster.
    waver = "\U0001f44b\U0001f3fb"
    assert [str(g) for g in sz.utf8_graphemes(waver)] == [waver]


def test_utf8_graphemes_str_method():
    """The Str.utf8_graphemes() method must agree with the module function."""
    s = Str("héllo")
    method_result = [str(g) for g in s.utf8_graphemes(s)]
    module_result = [str(g) for g in sz.utf8_graphemes(s)]
    assert method_result == module_result
    assert method_result == [str(g) for g in sz.utf8_graphemes("héllo")]


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_boundary_fuzz(seed_value: int):
    """Fuzz: compare the C grapheme iterator vs the pure-Python UAX-29 baseline."""
    seed(seed_value)
    try:
        props = get_grapheme_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode data files")

    for _ in range(200):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 100)))
        try:
            expected = baseline_grapheme_boundaries(text, props)
        except Exception:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(text))
        assert (
            sz_boundaries == expected
        ), "Grapheme boundary mismatch (seed {}):\n  text cps: {}\n  expected: {}\n  got:      {}".format(
            seed_value, " ".join(f"{ord(c):04X}" for c in text), expected, sz_boundaries
        )


def test_utf8_grapheme_boundary_official_conformance():
    """Full UAX-29 conformance against every case in the official GraphemeBreakTest.txt."""
    try:
        test_cases = get_grapheme_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for test_str, expected_byte_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(test_str))
        if sz_boundaries != expected_byte_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_byte_boundaries}, got {sz_boundaries}")

    assert not failures, "UAX-29 GraphemeBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_differential_grapheme(seed_value: int):
    """Differential fuzz against an independent grapheme library (``grapheme`` / ``uniseg``)."""
    seed(seed_value)
    try:
        graphemes_fn = pytest.importorskip("grapheme", reason="grapheme not installed").graphemes
    except Exception:
        uniseg_gc = pytest.importorskip("uniseg.graphemecluster", reason="uniseg not installed")
        graphemes_fn = uniseg_gc.grapheme_clusters

    failures = []
    for _ in range(2000):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(text))
        ref_boundaries = _byte_boundaries(graphemes_fn(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    ref={ref_boundaries}")

    assert not failures, "StringZilla vs reference grapheme-boundary divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )


#  endregion Grapheme iterator

#  region Synthetic corner cases (safety / seam / ICU)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_safety(seed_value: int):
    """Adversarial-byte safety: the grapheme iterator must survive the malformed battery and still tile its input.

    Feeds named malformed shapes, the astral fixtures, all 256 single bytes, every UTF-8 lead-class byte pair,
    and random garbage; each output must reconstruct the exact input bytes (no dropped, duplicated, or
    out-of-bounds span). Replaces the monolith's repr-only smoke test with the real tiling invariant.
    """
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        assert_segments_tile(sz.utf8_graphemes(raw), raw)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_seam(seed_value: int):
    """Window-seam sweep: inputs sized to land on 63/64/65/127/128/129/… byte boundaries relative to the
    kernel's 64-byte SIMD window must tile and stay bit-exact against ICU, catching off-by-one seam bugs."""
    rng = Random(seed_value)
    icu_graphemes = icu_segmenter("grapheme")
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_graphemes(raw), raw)
        text = raw.decode("utf-8")
        assert byte_boundaries(sz.utf8_graphemes(text)) == byte_boundaries(icu_graphemes(text))


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_differential_icu(seed_value: int):
    """Bit-exact differential against ICU's `createCharacterInstance` (extended grapheme clusters == UAX-29
    GraphemeBreakTest). Measured 0/4000 divergence on this boundary-rich palette, so an exact match is required."""
    icu_graphemes = icu_segmenter("grapheme")
    rng = Random(seed_value)
    failures = []
    for _ in range(2000):
        text = "".join(rng.choice(SEGMENTATION_PALETTE) for _ in range(rng.randint(1, 32)))
        sz_boundaries = byte_boundaries(sz.utf8_graphemes(text))
        icu_boundaries = byte_boundaries(icu_graphemes(text))
        if sz_boundaries != icu_boundaries:
            codepoints = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {codepoints}\n    sz={sz_boundaries}\n    icu={icu_boundaries}")

    assert not failures, "StringZilla vs ICU grapheme-boundary divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )


#  endregion Synthetic corner cases (safety / seam / ICU)
