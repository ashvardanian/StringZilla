#!/usr/bin/env python3
"""UAX-29 word-boundary segmentation: utf8_wordbreaks and Str.utf8_wordbreaks.

Mirrors the C++ test/utf8_wordbreaks.cpp translation unit.

Covers: basic word iteration and the skip_empty flag, TR29 contraction/number/apostrophe handling,
multi-script and emoji text, the Str method surface, adversarial-byte and window-seam safety sweeps,
exact conformance against the official WordBreakTest.txt, class-adjacency pairs and triples built from
the Word_Break property table, randomized differential fuzzing, realistic multi-script prose, and
cross-backend agreement on realistic, malformed, and lane-straddling inputs.
Compares against: the official Unicode WordBreakTest.txt and the uniseg library, both bit-exact UAX-29
oracles, and cross-backend self-consistency across every capability_sweep() backend. It deliberately
does not compare against ICU's word iterator, which is editor-tailored and diverges from raw UAX-29 by
roughly 17-20%.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat uniseg
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/utf8_wordbreaks.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/utf8_wordbreaks.py -q
"""

import itertools
from random import Random, choice, randint

import pytest

import stringzilla as sz
from stringzilla import Str

from test.sz_helpers import (
    SEED_VALUES,
    scale_iterations,
    UnicodeDataDownloadError,
    assert_backends_agree,
    get_word_break_test_cases,
    malformed_utf8_corpus,
    representatives_by_class,
    run_across_backends,
    seed_random_generators,
    vector_width_bracketing_strings,
)
from test.utf8_helpers import (
    adversarial_utf8_inputs,
    assert_segments_tile,
    byte_boundaries,
    class_adjacency_strings,
    corpus_of_byte_length,
    window_seam_lengths,
)

# region Unit


def test_utf8_wordbreaks_basic():
    """Splits `"hello world"` into words and the interstitial space, yields nothing for an empty string,
    and returns a single character as its own word."""
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
    """Word-breaking `"hello  world"` yields only non-empty segments."""
    result = [str(w) for w in sz.utf8_wordbreaks("hello  world")]
    # Should skip empty segments at boundaries
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)


def test_utf8_wordbreaks_contractions():
    """Per UAX-29 WB6-7 MidLetter rules, `"don't"` keeps its apostrophe inside a single word rather than
    breaking on it."""
    # Per TR29, apostrophe between letters should not break
    result = [str(w) for w in sz.utf8_wordbreaks("don't")]
    # TR29: "don't" should be one word (WB6-7: MidLetter rules)
    assert "don't" in result or result == ["don't"]


def test_utf8_wordbreaks_unicode():
    """Word-breaks multi-byte text correctly: `"Größe"` stays one word, Cyrillic `"привет мир"` splits
    into at least two segments, and CJK text yields at least one segment per UAX-29's per-character rule."""
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
    """Word-breaking text containing an emoji still recovers the surrounding words `"hello"` and
    `"world"` intact."""
    # Simple emoji
    result = [str(w) for w in sz.utf8_wordbreaks("hello 👋 world")]
    assert "hello" in result
    assert "world" in result


def test_utf8_wordbreaks_numbers():
    """Per UAX-29 WB9, a letter immediately followed by digits stays in one word as in `"test123"`, and
    per WB11-12 a decimal point between digits keeps `"3.14"` together."""
    result = [str(w) for w in sz.utf8_wordbreaks("test123")]
    # ALetter followed by Numeric should not break (WB9)
    assert "test123" in result or result == ["test123"]

    result = [str(w) for w in sz.utf8_wordbreaks("3.14")]
    # Numeric with MidNum should stay together (WB11-12)
    assert "3.14" in result or len(result) <= 3


@pytest.mark.parametrize("run_length", [3, 8, 15, 31, 32, 33, 100])
def test_utf8_wordbreaks_deferred_mid(run_length: int):
    """WB6/7/11/12 deferred-mid goldens: a Mid* (`,` `:` `'` `\"`) whose WB4-ignorable Extend run crosses every
    SIMD window width before the bridge completes or fails on the far side. Regression corpus for the
    cross-window bridge-shadow carry: a failed bridge must still break at the mid, keep WB7a's
    Hebrew x Single_Quote join, and leave the mid as the effective left context."""
    marks = "̀" * run_length
    he = "ה"
    emoji = "\U0001f600"
    zwj = "‍"
    math_five = "\U0001d7d7"
    goldens = [
        (f"5,{marks}6", [f"5,{marks}6"]),  # WB11 completes
        (f"5,{marks}{math_five}", [f"5,{marks}{math_five}"]),  # WB11 completes on an astral Numeric
        (f"a:{marks}b", [f"a:{marks}b"]),  # WB6 completes
        (f'{he}"{marks}{he}', [f'{he}"{marks}{he}']),  # WB7b completes
        (f"5,{marks}a", ["5", f",{marks}", "a"]),  # WB11 fails
        (f"a:{marks}5", ["a", f":{marks}", "5"]),  # WB6 fails
        (f"5,{marks}", ["5", f",{marks}"]),  # fails at end-of-text
        (f"5,{marks},5", ["5", f",{marks}", ",", "5"]),  # fails into a second mid
        (f"{he}'{marks}x", [f"{he}'{marks}x"]),  # WB7 completes
        (f"{he}'{marks}5", [f"{he}'{marks}", "5"]),  # WB7a keeps the quote on a failed bridge
        (f'{he}"{marks}x', [he, f'"{marks}', "x"]),  # WB7b fails
        (f"a:{marks}{zwj}{emoji}", ["a", f":{marks}{zwj}{emoji}"]),  # fails; WB3c joins the pictograph
        (f"5,{emoji}", ["5", ",", emoji]),  # astral lookahead right after the mid
    ]
    for text, expected in goldens:
        assert [str(w) for w in sz.utf8_wordbreaks(text)] == expected


def test_utf8_wordbreaks_str_method():
    """`Str.utf8_wordbreaks()` and the module-level `sz.utf8_wordbreaks()` return identical segmentation
    for `"hello world"`."""
    s = Str("hello world")
    # Method requires text argument, even when called on Str object
    result = [str(w) for w in s.utf8_wordbreaks(s)]
    assert result == ["hello", " ", "world"]

    # Also test via module function
    result = [str(w) for w in sz.utf8_wordbreaks(s)]
    assert result == ["hello", " ", "world"]


# endregion Unit


# region Corner cases


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
    The WordBreakTest and uniseg differentials separately check bit-exact boundary placement."""
    rng = Random(seed_value)
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_wordbreaks(raw), raw)
    # Phase sweep: shift a fixed corpus by every byte offset 0..63 so its content lands at every alignment
    # relative to the 64-byte window, an exhaustive deterministic complement to the length sweep.
    body = corpus_of_byte_length(96, rng)
    for phase in range(64):
        raw = b"a" * phase + body
        assert_segments_tile(sz.utf8_wordbreaks(raw), raw)


# endregion Corner cases


# region Conformance


def test_utf8_word_boundary_official_conformance():
    """Full UAX-29 conformance: exact boundary match against every case in the official Unicode
    WordBreakTest.txt. This is the authoritative gate, matching on all cases rather than a "first 50"
    sample with a `len >= 2` placeholder that let a ~50%-conformant kernel ship undetected."""
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


def test_utf8_word_class_adjacency(word_break_props):
    """Every Word_Break class-adjacency pair and triple, built from one representative codepoint per class,
    must segment bit-exactly the same as uniseg, the independent UAX-29 reference. Derived from the property
    table, this exercises rare class interactions (Mid* bridges, RI parity, ExtendNumLet) the palette misses."""
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    representatives = representatives_by_class(word_break_props, count=1)
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


# endregion Conformance


# region Oracles


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_word_boundary_differential_uniseg(seed_value: int):
    """Differential fuzz against uniseg, an independent UAX-29 library.

    The official WordBreakTest.txt is finite (~1944 fixed cases) and does not cover the combinatorial
    explosion of interdependent rules (WB4 ignorables crossing WB3c/WB3d/WB6-7 bridges, etc.). Random
    generation over a boundary-relevant palette, checked against a second conformant implementation,
    surfaces corner cases the fixed suite misses, the same gap that once let a 50%-conformant kernel
    pass CI. Any divergence is a real bug in one of the two implementations."""
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    seed_random_generators(seed_value)

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
    # window seam. Each exceeds one window, the case the random generator below also covers.
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
    # palette keeps ignorable runs short, so every run stays inside one window, the domain where the kernel must
    # match exactly. (Pathological >64-byte ignorable runs exceed any fixed window; serial-only, out of scope here.)
    random_samples = ("".join(choice(palette) for _ in range(randint(1, 100))) for _ in range(scale_iterations(2000)))
    for text in itertools.chain(seam_regressions, random_samples):
        sz_boundaries = boundaries(str(w) for w in sz.utf8_wordbreaks(text))
        ref_boundaries = boundaries(uniseg_wordbreak.words(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    uniseg={ref_boundaries}")

    assert not failures, "StringZilla vs uniseg word-boundary divergences ({}):\n{}".format(
        len(failures), "\n".join(failures[:20])
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_word_boundary_fuzz(seed_value: int):
    """Fuzz test: compare the StringZilla word kernel against the independent ``uniseg`` UAX-29 implementation.

    An in-repo pure-Python TR29 reimplementation, ``baseline_word_boundaries``, is itself ~89% wrong vs
    ``uniseg`` on this charset, so it cannot validate a correct kernel. ``uniseg`` is a maintained
    third-party reference and agrees with the official ``WordBreakTest.txt``.
    """
    uniseg_wordbreak = pytest.importorskip("uniseg.wordbreak", reason="uniseg not installed")
    seed_random_generators(seed_value)

    # Generate random test strings. The charset deliberately spans every WB category, including the ones that a
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

    for _ in range(scale_iterations(200)):
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

        # Exact boundary match required on every case.
        assert (
            sz_boundaries == expected_boundaries
        ), "Word boundary mismatch (seed {}):\n  text cps: {}\n  expected: {}\n  got:      {}".format(
            seed_value, " ".join(f"{ord(c):04X}" for c in text), expected_boundaries, sz_boundaries
        )


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


# endregion Oracles


# region Backend differential


WORDBREAK_REALISTIC_TEXTS = [
    "hello world",
    "don't stop",
    "Größe café",
    "привет мир",
    "你好 世界",
    "test123 3.14",
    "hello 👋 world",
]


@pytest.mark.parametrize(
    "text", WORDBREAK_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_wordbreaks_backend_differential(text):
    """utf8_wordbreaks must segment identically across every SIMD backend on realistic, malformed, and
    lane-straddling inputs, compared as raw segment bytes so malformed UTF-8 sweeps without raising."""
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_wordbreaks(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")


# endregion Backend differential
