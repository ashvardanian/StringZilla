#!/usr/bin/env python3
"""UAX-29 grapheme cluster segmentation: sz.utf8_graphemes and Str.utf8_graphemes().

Mirrors the C++ test/utf8_graphemes.cpp translation unit.

Covers: ASCII, combining-mark, ZWJ-emoji-family, regional-indicator-flag, and skin-tone-modifier
cluster iteration and the Str method surface, malformed-byte and SIMD window-seam safety sweeps,
class-adjacency pairs and triples, fuzz against the pure-Python UAX-29 baseline and multi-script
prose paragraphs, and cross-backend agreement on regular and malformed inputs.
Compares against: the official UCD GraphemeBreakTest, ICU's createCharacterInstance segmenter,
the grapheme and uniseg libraries, and cross-backend self-consistency.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat PyICU grapheme uniseg
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/utf8_graphemes.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/utf8_graphemes.py -q
"""

from random import Random, choice, randint

import pytest

import stringzilla as sz
from stringzilla import Str

from test.sz_helpers import (
    vector_width_bracketing_strings,
    SEED_VALUES,
    scale_iterations,
    UnicodeDataDownloadError,
    seed_random_generators,
    run_across_backends,
    assert_backends_agree,
    get_grapheme_break_properties,
    get_grapheme_break_test_cases,
    get_indic_conjunct_break_properties,
    get_extended_pictographic,
    get_random_string,
    malformed_utf8_corpus,
    representatives_by_class,
    baseline_grapheme_boundaries,
)
from test.utf8_helpers import (
    SEGMENTATION_PALETTE,
    adversarial_utf8_inputs,
    assert_segments_tile,
    byte_boundaries,
    class_adjacency_strings,
    corpus_of_byte_length,
    icu_segmenter,
    icu_unicode_at_least,
    window_seam_lengths,
)

_byte_boundaries = byte_boundaries
_SEGMENTATION_PALETTE = SEGMENTATION_PALETTE


# region Unit


def test_utf8_graphemes_basic():
    """Plain ASCII splits one cluster per character, the empty string yields no clusters, a base letter
    plus a combining accent stays one cluster, and a ZWJ emoji family collapses to one cluster."""
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
    """Grapheme iteration over a multi-byte string yields only non-empty clusters."""
    result = [str(g) for g in sz.utf8_graphemes("héllo")]
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)


def test_utf8_graphemes_unicode():
    """A decomposed accented letter, a regional-indicator flag pair, and an emoji with a skin-tone
    modifier each form a single grapheme cluster."""
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


# endregion Unit

# region Corner cases


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_safety(seed_value: int):
    """Adversarial-byte safety: the grapheme iterator must survive the malformed battery and still tile its input.

    Feeds named malformed shapes, the astral fixtures, all 256 single bytes, every UTF-8 lead-class byte pair,
    and random garbage; each output must reconstruct the exact input bytes with no dropped, duplicated, or
    out-of-bounds span.
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
    icu_current = icu_unicode_at_least("17")  # ICU-exact only when its Unicode matches our 17.0 tables
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_graphemes(raw), raw)
        text = raw.decode("utf-8")
        if icu_current:
            assert byte_boundaries(sz.utf8_graphemes(text)) == byte_boundaries(icu_graphemes(text))
    # Phase sweep: shift a fixed corpus by every byte offset 0..63 so its content lands at every alignment
    # relative to the 64-byte window, an exhaustive deterministic complement to the length sweep.
    body = corpus_of_byte_length(96, rng).decode("utf-8")
    for phase in range(64):
        text = "a" * phase + body
        assert_segments_tile(sz.utf8_graphemes(text), text)
        if icu_current:
            assert byte_boundaries(sz.utf8_graphemes(text)) == byte_boundaries(icu_graphemes(text))


# endregion Corner cases

# region Conformance


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


def test_baseline_grapheme_matches_official():
    """Validate the oracle itself: baseline_grapheme_boundaries must reproduce every official
    GraphemeBreakTest.txt case bit-exactly. A baseline that cannot pass the official suite cannot be trusted to
    gate the random fuzz above; this catches a regressed or approximate baseline, for example an
    Extended_Pictographic or Indic-conjunct gap."""
    try:
        test_cases = get_grapheme_break_test_cases()
        properties = get_grapheme_break_properties()
        indic_conjunct_breaks = get_indic_conjunct_break_properties()
        extended_pictographic = get_extended_pictographic()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for text, expected_byte_boundaries in test_cases:
        if not text:
            continue
        got = baseline_grapheme_boundaries(text, properties, indic_conjunct_breaks, extended_pictographic)
        if got != expected_byte_boundaries:
            codepoints = " ".join(f"{ord(character):04X}" for character in text)
            failures.append(f"  {codepoints}: expected {expected_byte_boundaries}, got {got}")

    assert not failures, "baseline_grapheme_boundaries vs official GraphemeBreakTest ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )


def test_utf8_grapheme_class_adjacency(grapheme_break_props):
    """Every Grapheme_Cluster_Break class-adjacency pair and triple, built from one representative codepoint
    per class, must segment bit-exactly the same as ICU. Derives the corpus from the property table rather
    than a hand-picked palette, so rare class interactions are exercised exhaustively."""
    icu = pytest.importorskip("icu", reason="PyICU not installed")
    icu_graphemes = icu_segmenter("grapheme")
    representatives = representatives_by_class(grapheme_break_props, count=1)
    cases = class_adjacency_strings(representatives, arity=2)
    cases += class_adjacency_strings(representatives, arity=3, max_cases=8000)

    failures = []
    for text in cases:
        if not all(icu.Char.isdefined(ord(character)) for character in text):
            continue  # skip representatives unassigned in the local ICU's Unicode version
        sz_boundaries = byte_boundaries(sz.utf8_graphemes(text))
        icu_boundaries = byte_boundaries(icu_graphemes(text))
        if sz_boundaries != icu_boundaries:
            codepoints = " ".join(f"{ord(character):04X}" for character in text)
            failures.append(f"  {codepoints}\n    sz={sz_boundaries}\n    icu={icu_boundaries}")

    assert not failures, "StringZilla vs ICU grapheme class-adjacency divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )


# endregion Conformance

# region Oracles


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_boundary_fuzz(seed_value: int, grapheme_break_props):
    """Fuzz: compare the C grapheme iterator vs the pure-Python UAX-29 baseline."""
    seed_random_generators(seed_value)
    for _ in range(scale_iterations(200)):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 100)))
        try:
            expected = baseline_grapheme_boundaries(text, grapheme_break_props)
        except Exception:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(text))
        assert (
            sz_boundaries == expected
        ), "Grapheme boundary mismatch (seed {}):\n  text cps: {}\n  expected: {}\n  got:      {}".format(
            seed_value, " ".join(f"{ord(c):04X}" for c in text), expected, sz_boundaries
        )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_differential_grapheme(seed_value: int):
    """Differential fuzz against an independent grapheme library (``grapheme`` / ``uniseg``)."""
    seed_random_generators(seed_value)
    try:
        graphemes_fn = pytest.importorskip("grapheme", reason="grapheme not installed").graphemes
    except Exception:
        uniseg_gc = pytest.importorskip("uniseg.graphemecluster", reason="uniseg not installed")
        graphemes_fn = uniseg_gc.grapheme_clusters

    failures = []
    for _ in range(scale_iterations(2000)):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_graphemes(text))
        ref_boundaries = _byte_boundaries(graphemes_fn(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    ref={ref_boundaries}")

    assert not failures, "StringZilla vs reference grapheme-boundary divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_grapheme_differential_icu(seed_value: int):
    """Differential against ICU's `createCharacterInstance` (extended grapheme clusters == UAX-29 GraphemeBreakTest).

    Bit-exact when the local PyICU is at our Unicode version (≥17), since the palette measured 0/4000 divergence;
    when PyICU is older its GCB tables differ from our 17.0 tables on changed codepoints, so the gate loosens to
    broad agreement (≥99%) to avoid spurious version-induced failures. The strict oracle is the official
    GraphemeBreakTest (`test_utf8_grapheme_boundary_official_conformance`), version-pinned to UNICODE_VERSION."""
    icu_graphemes = icu_segmenter("grapheme")
    icu_current = icu_unicode_at_least("17")
    rng = Random(seed_value)
    iterations = scale_iterations(2000)
    failures = []
    for _ in range(iterations):
        text = "".join(rng.choice(SEGMENTATION_PALETTE) for _ in range(rng.randint(1, 32)))
        sz_boundaries = byte_boundaries(sz.utf8_graphemes(text))
        icu_boundaries = byte_boundaries(icu_graphemes(text))
        if sz_boundaries != icu_boundaries:
            codepoints = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {codepoints}\n    sz={sz_boundaries}\n    icu={icu_boundaries}")

    if icu_current:
        assert not failures, "StringZilla vs ICU grapheme-boundary divergences ({}):\n{}".format(
            len(failures), "\n".join(failures[:20])
        )
    else:
        agreement = 1.0 - len(failures) / iterations
        assert agreement >= 0.99, "StringZilla vs stale-ICU grapheme agreement {:.2%} too low:\n{}".format(
            agreement, "\n".join(failures[:20])
        )


def test_utf8_graphemes_prose():
    """Realistic multi-script paragraphs: grapheme count matches the ICU root oracle; clusters != codepoints."""
    from test.utf8_helpers import (
        PROSE_PRIDE_CAPTION,
        PROSE_DEVANAGARI_TIP,
        PROSE_CONCERT_POST,
        PROSE_RTL_SCRIPTS,
        PROSE_MICRO_PREPEND,
        assert_segments_tile,
        icu_segmenter,
    )

    cases = [
        (PROSE_PRIDE_CAPTION, 206),
        (PROSE_DEVANAGARI_TIP, 252),
        (PROSE_CONCERT_POST, 134),
        (PROSE_RTL_SCRIPTS, 256),
        (PROSE_MICRO_PREPEND, 3),
    ]
    for text, expected in cases:
        clusters = list(sz.utf8_graphemes(text))
        assert_segments_tile(clusters, text)
        assert len(clusters) == expected
    oracle = icu_segmenter("grapheme")
    for text, _ in cases:
        assert len(list(sz.utf8_graphemes(text))) == len(oracle(text))

    # Codepoints are not clusters: the emoji paragraph has more runes than grapheme clusters.
    assert sz.utf8_count(PROSE_PRIDE_CAPTION) == 222
    assert sz.utf8_count(PROSE_PRIDE_CAPTION) > sum(1 for _ in sz.utf8_graphemes(PROSE_PRIDE_CAPTION))


# endregion Oracles

# region Backend differential


def _grapheme_segment_bytes(text_or_bytes) -> list:
    """Per-cluster raw bytes of a grapheme iteration, safe for malformed UTF-8 where `str(segment)`
    (as used by `byte_boundaries`) would raise, unlike `bytes(segment)`."""
    return [bytes(segment) for segment in sz.utf8_graphemes(text_or_bytes)]


GRAPHEME_BACKEND_DIFFERENTIAL_TEXTS = [
    "hello",
    "héllo",
    "\U0001f468‍\U0001f469‍\U0001f467",  # man-ZWJ-woman-ZWJ-girl family
    "\U0001f1e6\U0001f1fa",  # regional-indicator flag pair
    "\U0001f44b\U0001f3fb",  # emoji + skin-tone modifier
    "é",  # e + combining acute accent (decomposed e)
    "\r\n",
    "a\r\nb\r\n",
    "café\r\ncafé",  # precomposed accents around CRLF
] + vector_width_bracketing_strings()


@pytest.mark.parametrize("text", GRAPHEME_BACKEND_DIFFERENTIAL_TEXTS)
def test_utf8_graphemes_backend_differential(text):
    """utf8_graphemes must segment identically across every SIMD backend, covering emoji ZWJ sequences,
    combining marks, CRLF, and inputs straddling the 16/32/64-byte SIMD lanes."""
    results = run_across_backends(lambda: _grapheme_segment_bytes(text))
    assert_backends_agree(results, format_inputs=lambda: repr(text))


@pytest.mark.parametrize("raw", malformed_utf8_corpus())
def test_utf8_graphemes_backend_differential_malformed(raw):
    """Malformed UTF-8 must segment identically and never crash across every SIMD backend, and every
    backend's output must still tile the raw input bytes exactly."""

    def segment_and_check():
        segments = _grapheme_segment_bytes(raw)
        assert_segments_tile(segments, raw)
        return segments

    results = run_across_backends(segment_and_check)
    assert_backends_agree(results, format_inputs=lambda: raw.hex())


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_graphemes_backend_differential_random(seed_value):
    """Random ASCII corpora, one cluster per byte, must segment identically across every SIMD backend."""
    seed_random_generators(seed_value)
    text = get_random_string()
    results = run_across_backends(lambda: _grapheme_segment_bytes(text))
    assert_backends_agree(results, format_inputs=lambda: repr(text))


# endregion Backend differential
