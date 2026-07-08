#!/usr/bin/env python3
"""UAX-29 word-boundary tests: utf8_wordbreaks behavior, official WordBreakTest conformance, and differential
fuzz against uniseg. Words are NOT validated against ICU (its word iterator is editor-tailored, ~17-20%
off raw UAX-29); WordBreakTest.txt + uniseg are the bit-exact oracles. Adds malformed/seam safety sweeps.

Mirrors the C++ test/utf8_wordbreaks.cpp translation unit.
"""

from random import Random, choice, randint, seed

import pytest

import stringzilla as sz
from stringzilla import Str

from test.helpers import (
    SEED_VALUES,
    UnicodeDataDownloadError,
    get_word_break_properties,
    get_word_break_test_cases,
    representatives_by_class,
)
from test.utf8_helpers import (
    adversarial_utf8_inputs,
    assert_segments_tile,
    byte_boundaries,
    class_adjacency_strings,
    corpus_of_byte_length,
    window_seam_lengths,
)


def test_utf8_wordbreaks_basic():
    """Test basic word iteration."""
    # Simple two words
    result = [str(w) for w in sz.utf8_wordbreaks("hello world")]
    assert result == ["hello", " ", "world"]

    # Empty string
    result = [str(w) for w in sz.utf8_wordbreaks("")]
    assert result == []

    # Single character
    result = [str(w) for w in sz.utf8_wordbreaks("a")]
    assert result == ["a"]


def test_utf8_wordbreaks_skip_empty():
    """Test skip_empty parameter."""
    result = [str(w) for w in sz.utf8_wordbreaks("hello  world")]
    # Should skip empty segments at boundaries
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)


def test_utf8_wordbreaks_contractions():
    """Test English contractions per TR29 rules."""
    # Per TR29, apostrophe between letters should not break
    result = [str(w) for w in sz.utf8_wordbreaks("don't")]
    # TR29: "don't" should be one word (WB6-7: MidLetter rules)
    assert "don't" in result or result == ["don't"]


def test_utf8_wordbreaks_unicode():
    """Test UTF-8 multi-byte characters."""
    # German with eszett
    result = [str(w) for w in sz.utf8_wordbreaks("Größe")]
    assert "Größe" in result

    # Russian text
    result = [str(w) for w in sz.utf8_wordbreaks("привет мир")]
    assert len(result) >= 2

    # CJK (each character is its own word in TR29)
    result = [str(w) for w in sz.utf8_wordbreaks("你好")]
    # CJK characters are typically "Other" category - each is a boundary
    assert len(result) >= 1


def test_utf8_wordbreaks_emoji():
    """Test emoji handling."""
    # Simple emoji
    result = [str(w) for w in sz.utf8_wordbreaks("hello 👋 world")]
    assert "hello" in result
    assert "world" in result


def test_utf8_wordbreaks_numbers():
    """Test numeric handling per TR29."""
    result = [str(w) for w in sz.utf8_wordbreaks("test123")]
    # ALetter followed by Numeric should not break (WB9)
    assert "test123" in result or result == ["test123"]

    result = [str(w) for w in sz.utf8_wordbreaks("3.14")]
    # Numeric with MidNum should stay together (WB11-12)
    assert "3.14" in result or len(result) <= 3


def test_utf8_wordbreaks_str_method():
    """Test the Str.utf8_wordbreaks() method."""
    s = Str("hello world")
    # Method requires text argument, even when called on Str object
    result = [str(w) for w in s.utf8_wordbreaks(s)]
    assert result == ["hello", " ", "world"]

    # Also test via module function
    result = [str(w) for w in sz.utf8_wordbreaks(s)]
    assert result == ["hello", " ", "world"]


@pytest.mark.parametrize("seed_value", [42, 0, 1, 314159, 271828])
def test_utf8_word_boundary_fuzz(seed_value: int):
    """Fuzz test: compare the StringZilla word kernel against the INDEPENDENT ``uniseg`` UAX-29 implementation.

    The previous oracle (an in-repo pure-Python TR29 reimplementation, ``baseline_word_boundaries``) is itself
    ~89% wrong vs ``uniseg`` on this charset, so it could never validate a correct kernel. ``uniseg`` is a
    maintained third-party reference and agrees with the official ``WordBreakTest.txt``.
    """
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    seed(seed_value)

    # Generate random test strings. The charset deliberately spans every WB category — including the ones that a
    # naive kernel gets wrong: combining marks / Format / ZWJ (WB4, WB3c), emoji & symbol pictographs (WB3c),
    # Regional_Indicators (WB15/16), horizontal spaces incl. ideographic (WB3d), Hebrew + quotes (WB7a), Katakana,
    # CJK, CR/LF. An ASCII-only charset is what let the boundary bugs hide.
    test_chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " \t\n\r"
        ".,;:!?'\"_-"
        "äöüßαβγδабвг"
        "אב"  # Hebrew (WB7a)
        "カタ"  # Katakana (WB13)
        "中文"  # CJK
        "̀́̈"  # combining marks (Extend, WB4)
        "­⁠‍"  # Soft hyphen (Format), Word joiner (Format), ZWJ (WB3c)
        "　  "  # ideographic / thin / math spaces (WSegSpace, WB3d)
        "\U0001f600\U0001f6d1Ⓜ©❤"  # Extended_Pictographic (WB3c)
        "\U0001f1e6\U0001f1fa"  # Regional_Indicators (WB15/16)
    )

    for _ in range(200):
        length = randint(1, 100)
        text = "".join(choice(test_chars) for _ in range(length))

        # Independent UAX-29 oracle (uniseg): cumulative byte lengths of its segments.
        expected_boundaries = [0]
        for seg in uniseg_wordbreak.words(text):
            expected_boundaries.append(expected_boundaries[-1] + len(seg.encode("utf-8")))

        # StringZilla boundaries = cumulative byte lengths of the emitted words.
        sz_boundaries = [0]
        for word in sz.utf8_wordbreaks(text):
            sz_boundaries.append(sz_boundaries[-1] + len(str(word).encode("utf-8")))

        # Exact match required — no `len >= 2` escape hatch.
        assert (
            sz_boundaries == expected_boundaries
        ), "Word boundary mismatch (seed {}):\n  text cps: {}\n  expected: {}\n  got:      {}".format(
            seed_value, " ".join(f"{ord(c):04X}" for c in text), expected_boundaries, sz_boundaries
        )


def test_utf8_word_boundary_official_conformance():
    """Full UAX-29 conformance: EXACT boundary match against every case in the official Unicode
    WordBreakTest.txt. This is the authoritative gate — it must match on all cases, not "first 50"
    with a `len >= 2` placeholder (which let a ~50%-conformant kernel ship undetected)."""
    try:
        test_cases = get_word_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for test_str, expected_boundaries in test_cases:
        if not test_str:
            continue

        # The StringZilla iterator yields words back-to-back; their cumulative byte lengths are the boundaries.
        sz_boundaries = [0]
        for word in sz.utf8_wordbreaks(test_str):
            sz_boundaries.append(sz_boundaries[-1] + len(str(word).encode("utf-8")))

        # `get_word_break_test_cases` already returns BYTE boundaries (including 0 and len), matching `sz_boundaries`.
        if sz_boundaries != expected_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_boundaries}, got {sz_boundaries}")

    assert not failures, "UAX-29 WordBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )


@pytest.mark.parametrize("seed_value", [1, 7, 42, 1000, 271828])
def test_utf8_word_boundary_differential_uniseg(seed_value: int):
    """Differential fuzz against an INDEPENDENT UAX-29 library (``uniseg``).

    The official WordBreakTest.txt is finite (~1944 fixed cases) and does not cover the combinatorial
    explosion of interdependent rules (WB4 ignorables crossing WB3c/WB3d/WB6-7 bridges, etc.). Random
    generation over a boundary-relevant palette, checked against a second conformant implementation,
    surfaces corner cases the fixed suite misses — which is precisely how a 50%-conformant kernel once
    passed CI. Any divergence is a real bug in one of the two implementations."""
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    seed(seed_value)

    # Every WB category, including the historically-broken ones.
    palette = list(
        "abZ59 \t\n\r.,;:'\"_-"
        "éαаאבカタ中국"
        "̀́̈"  # combining Extend (WB4)
        "­⁠‍"  # Format, Word-joiner, ZWJ (WB3c / WB4)
        "　  "  # WSegSpace (WB3d)
        "\U0001f600\U0001f6d1Ⓜ©❤"  # Extended_Pictographic (WB3c)
        "\U0001f1e6\U0001f1fa\U0001f3fb"  # Regional_Indicator + skin tone (WB15/16)
    )

    def boundaries(segments):
        out = [0]
        for seg in segments:
            out.append(out[-1] + len(seg.encode("utf-8")))
        return out

    # Fixed multi-window regression strings: RI-parity and Mid-bridge carry once miscounted across the 64-byte
    # window seam. Each exceeds one window — the case the random generator below now also covers (it previously
    # used <=24 codepoints, which fits a single window and so could never exercise a seam).
    seam_regressions = [
        bytes.fromhex(
            "E382AB2D0AF09F87A662C2ADF09F87A6CC800A5FF09F87A6F09F87A6C2AD0ACC88F09F87BAF09F8FBBCC88C2ADE281A0CC88F0"
            "9F87A6F09F87A6E4B8ADE281A05F"
        ).decode("utf-8"),
        bytes.fromhex(
            "5F27F09F87BAE4B8ADCC81F09F87A6F09F87A65FCC885FD7902DE29382CC8062F09F87A6E281A0F09F87BACC80CC88F09F8FBB"
            "F09F87BACC81F09F87BA0AC2ADF09F87BACC88E382AB5F272ECC880AE4B8AD"
        ).decode("utf-8"),
        bytes.fromhex(
            "5FE281A05F352CF09F988027F09F98805FCC88CC81E4B8ADE281A05F62CC8061E4B8ADE382AB35E281A0CC88CC88F09F8FBBCC"
            "88E281A0CC80CC812735CC80E4B8ADCC81E2808DC3A9E380803527"
        ).decode("utf-8"),
    ]

    failures = []
    # Up to ~100 codepoints so most cases cross the 64-byte window boundary (multi-window seam coverage). The
    # palette keeps ignorable runs short, so every run stays inside one window — the domain where the kernel must
    # match exactly. (Pathological >64-byte ignorable runs exceed any fixed window; serial-only, out of scope here.)
    samples = seam_regressions + ["".join(choice(palette) for _ in range(randint(1, 100))) for _ in range(2000)]
    for text in samples:
        sz_boundaries = boundaries(str(w) for w in sz.utf8_wordbreaks(text))
        ref_boundaries = boundaries(uniseg_wordbreak.words(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    uniseg={ref_boundaries}")

    assert not failures, "StringZilla vs uniseg word-boundary divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )


#  region Synthetic corner cases (safety / seam)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_word_safety(seed_value: int):
    """Adversarial-byte safety: the word iterator must survive the malformed battery and still tile its input
    (named malformed shapes, astral fixtures, all single bytes, every lead-class byte pair, random garbage)."""
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        assert_segments_tile(sz.utf8_wordbreaks(raw), raw)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_word_seam(seed_value: int):
    """Window-seam sweep: inputs sized to land on 63/64/65/127/128/129/… byte boundaries relative to the
    kernel's 64-byte SIMD window must still tile, catching segments dropped or duplicated at the seam.
    (Boundary correctness is gated bit-exact by the WordBreakTest + uniseg differentials above.)"""
    rng = Random(seed_value)
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_wordbreaks(raw), raw)
    # Phase sweep: shift a fixed corpus by every byte offset 0..63 so its content lands at every alignment
    # relative to the 64-byte window — exhaustive deterministic complement to the length sweep.
    body = corpus_of_byte_length(96, rng)
    for phase in range(64):
        raw = b"a" * phase + body
        assert_segments_tile(sz.utf8_wordbreaks(raw), raw)


#  endregion Synthetic corner cases (safety / seam)


#  region Rule-derived coverage (class adjacency)


def test_utf8_word_class_adjacency():
    """Every Word_Break class-adjacency pair and triple — built from one representative codepoint per class —
    must segment bit-exactly the same as the independent UAX-29 reference (uniseg). Derived from the property
    table, this exercises rare class interactions (Mid* bridges, RI parity, ExtendNumLet) the palette misses."""
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    try:
        properties = get_word_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode data files")

    representatives = representatives_by_class(properties, count=1)
    cases = class_adjacency_strings(representatives, arity=2)
    cases += class_adjacency_strings(representatives, arity=3, max_cases=8000)

    failures = []
    for text in cases:
        sz_boundaries = byte_boundaries(sz.utf8_wordbreaks(text))
        reference_boundaries = byte_boundaries(list(uniseg_wordbreak.words(text)))
        if sz_boundaries != reference_boundaries:
            codepoints = " ".join(f"{ord(character):04X}" for character in text)
            failures.append(f"  {codepoints}\n    sz={sz_boundaries}\n    uniseg={reference_boundaries}")

    assert not failures, "StringZilla vs uniseg word class-adjacency divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )


#  endregion Rule-derived coverage (class adjacency)


def test_utf8_wordbreaks_prose():
    """Realistic multi-script paragraphs: wordbreak count matches the uniseg UAX-29 oracle; segments tile."""
    from test.utf8_helpers import (
        PROSE_HOTEL_REVIEW,
        PROSE_NEWS_LEDE,
        PROSE_CONCERT_POST,
        PROSE_RTL_SCRIPTS,
        PROSE_MICRO_APOSTROPHE,
        assert_segments_tile,
    )

    cases = [
        (PROSE_HOTEL_REVIEW, 100),
        (PROSE_NEWS_LEDE, 83),
        (PROSE_CONCERT_POST, 69),
        (PROSE_RTL_SCRIPTS, 98),
        (PROSE_MICRO_APOSTROPHE, 5),
    ]
    for text, expected in cases:
        segments = list(sz.utf8_wordbreaks(text))
        assert_segments_tile(segments, text)
        assert len(segments) == expected
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak")
    for text, _ in cases:
        assert len(list(sz.utf8_wordbreaks(text))) == len(list(uniseg_wordbreak.words(text)))
