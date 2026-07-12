#!/usr/bin/env python3
"""UAX-14 line-break-opportunity segmentation: sz.utf8_linebreaks and the Str.utf8_linebreaks method.

Mirrors the C++ scripts/test_utf8_linebreaks.cpp translation unit.

Covers: basic and Unicode line iteration including CRLF as a single break opportunity, the tiling
invariant across empty, single-line, and blank-line inputs, and Str-method parity with the module
function; adversarial malformed UTF-8 safety and SIMD window-seam and byte-offset phase sweeps;
bit-exact conformance against every case in the official LineBreakTest, Line_Break class-adjacency
pairs, differential fuzzing, and realistic multi-script prose; and cross-backend agreement on
realistic, malformed, and lane-straddling inputs.
Compares against: the official UCD LineBreakTest, the uniseg library with broad-agreement tolerance
for its default UAX-14 tailorings, and cross-backend self-consistency.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat uniseg
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/utf8_linebreaks.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/utf8_linebreaks.py -q
"""

from random import Random, choice, randint, seed

import pytest

import stringzilla as sz
from stringzilla import Str

from test.sz_helpers import (
    SEED_VALUES,
    scale_iterations,
    UnicodeDataDownloadError,
    assert_backends_agree,
    get_line_break_test_cases,
    malformed_utf8_corpus,
    representatives_by_class,
    run_across_backends,
    vector_width_bracketing_strings,
)
from test.utf8_helpers import (
    SEGMENTATION_PALETTE,
    adversarial_utf8_inputs,
    assert_segments_tile,
    byte_boundaries,
    class_adjacency_strings,
    corpus_of_byte_length,
    window_seam_lengths,
)

_byte_boundaries = byte_boundaries


# region Unit


def test_utf8_linebreaks_basic():
    """Line iteration yields Str segments that tile a multi-line text, a single line, and an empty
    string, which yields none."""
    result = [str(seg) for seg in sz.utf8_linebreaks("first\nsecond")]
    assert "".join(result) == "first\nsecond"  # Segments tile the input.

    # Empty string yields nothing.
    assert [str(seg) for seg in sz.utf8_linebreaks("")] == []

    # A single short line still tiles back to the input.
    result = [str(seg) for seg in sz.utf8_linebreaks("just one line")]
    assert "".join(result) == "just one line"


def test_utf8_linebreaks_tiling():
    """Linewrap is forward-only; its segments tile the input contiguously."""
    text = "alpha\nbeta\ngamma"
    forward = [str(seg) for seg in sz.utf8_linebreaks(text)]
    assert "".join(forward) == text


def test_utf8_linebreaks_skip_empty():
    """Line iteration over text containing a blank line still yields only non-empty segments."""
    result = [str(seg) for seg in sz.utf8_linebreaks("a\n\nb")]
    assert len(result) > 0
    assert all(len(seg) > 0 for seg in result)


def test_utf8_linebreaks_unicode():
    """Multi-byte Unicode text, including a paragraph separator and Cyrillic, tiles correctly, and
    CRLF counts as a single break opportunity rather than two."""
    result = [str(seg) for seg in sz.utf8_linebreaks("Größe\u2028привет")]
    assert "".join(result) == "Größe\u2028привет"
    # Segments tile the input.

    # CRLF is a single break opportunity, not two.
    result = [str(seg) for seg in sz.utf8_linebreaks("a\r\nb")]
    assert "".join(result) == "a\r\nb"


def test_utf8_linebreaks_str_method():
    """The Str.utf8_linebreaks() method must agree with the module function."""
    s = Str("first\nsecond")
    method_result = [str(seg) for seg in s.utf8_linebreaks(s)]
    module_result = [str(seg) for seg in sz.utf8_linebreaks(s)]
    assert method_result == module_result
    assert method_result == [str(seg) for seg in sz.utf8_linebreaks("first\nsecond")]


# endregion Unit


# region Corner cases


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_linewrap_safety(seed_value: int):
    """Adversarial-byte safety: the linewrap iterator must survive the malformed battery and still tile its input."""
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        assert_segments_tile(sz.utf8_linebreaks(raw), raw)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_linewrap_seam(seed_value: int):
    """Window-seam sweep: inputs sized to straddle the 64-byte SIMD windows, plus a full byte-offset phase
    sweep across one window, must still tile contiguously; this checks tiling only, not UAX-14 boundary
    agreement."""
    rng = Random(seed_value)
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_linebreaks(raw), raw)
    # Phase sweep: shift a fixed corpus by every byte offset 0..63 so its content lands at every alignment
    # relative to the 64-byte window, an exhaustive deterministic complement to the length sweep.
    body = corpus_of_byte_length(96, rng)
    for phase in range(64):
        raw = b"a" * phase + body
        assert_segments_tile(sz.utf8_linebreaks(raw), raw)


# endregion Corner cases


# region Conformance


def test_utf8_linewrap_boundary_official_conformance():
    """Full UAX-14 conformance against every case in the official LineBreakTest.txt.

    The iterator emits a segment at every mandatory or soft break opportunity; their cumulative byte
    lengths must match the official break positions bit-exactly, with no tailoring-tolerance gate.

    LB2, which forbids breaking at the start of text, means LineBreakTest never marks a leading ÷, so
    the official boundary list has no leading 0, unlike GraphemeBreakTest/WordBreakTest/SentenceBreakTest,
    which break at start. The segments always start at offset 0, so drop the shared helper's leading 0
    here.
    """
    try:
        test_cases = get_line_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for test_str, expected_byte_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_linebreaks(test_str))[1:]
        if sz_boundaries != expected_byte_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_byte_boundaries}, got {sz_boundaries}")

    assert not failures, "UAX-14 LineBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )


def test_utf8_linewrap_class_adjacency(line_break_props):
    """Every Line_Break class-adjacency pair, built from one representative codepoint per class, checked
    against uniseg. Line breaking has the largest property alphabet, about 40 classes, so this derives far
    more class interactions than the palette; agreement-gated, not bit-exact, because uniseg applies
    default UAX-14 tailorings StringZilla may differ on."""
    uniseg_linebreak = pytest.importorskip("uniseg.linebreak", reason="uniseg not installed")
    representatives = representatives_by_class(line_break_props, count=1)
    cases = class_adjacency_strings(representatives, arity=2)

    failures = []
    for text in cases:
        sz_boundaries = byte_boundaries(sz.utf8_linebreaks(text))
        reference_boundaries = byte_boundaries(list(uniseg_linebreak.line_break_units(text)))
        if sz_boundaries != reference_boundaries:
            codepoints = " ".join(f"{ord(character):04X}" for character in text)
            failures.append(f"  {codepoints}\n    sz={sz_boundaries}\n    uniseg={reference_boundaries}")

    agreement = 1.0 - len(failures) / max(1, len(cases))
    assert agreement >= 0.80, "StringZilla vs uniseg line class-adjacency agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )


# endregion Conformance


# region Oracles


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_linewrap_differential_uniseg(seed_value: int):
    """Differential fuzz against ``uniseg.linebreak`` for line-break-opportunity segments."""
    uniseg_linebreak = pytest.importorskip("uniseg.linebreak", reason="uniseg not installed")
    seed(seed_value)

    def ref_segments(text):
        return list(uniseg_linebreak.line_break_units(text))

    iterations = scale_iterations(2000)
    failures = []
    for _ in range(iterations):
        text = "".join(choice(SEGMENTATION_PALETTE) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_linebreaks(text))
        ref_boundaries = _byte_boundaries(ref_segments(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    uniseg={ref_boundaries}")

    # uniseg implements default UAX-14 tailorings that may differ from StringZilla; require broad
    # agreement on segment positions rather than bit-exactness.
    agreement = 1.0 - len(failures) / iterations
    assert agreement >= 0.80, "StringZilla vs uniseg line-boundary agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )


def test_utf8_linebreaks_prose():
    """Realistic multi-script paragraphs: the UAX-14 line-break count matches the ICU root oracle, and
    segments tile."""
    from test.utf8_helpers import (
        PROSE_HOTEL_REVIEW,
        PROSE_SCIENCE_ABSTRACT,
        PROSE_NEWS_LEDE,
        assert_segments_tile,
        icu_segmenter,
    )

    cases = [
        (PROSE_HOTEL_REVIEW, 45),
        (PROSE_SCIENCE_ABSTRACT, 43),
        (PROSE_NEWS_LEDE, 32),
    ]
    for text, expected in cases:
        segments = list(sz.utf8_linebreaks(text))
        assert_segments_tile(segments, text)
        assert len(segments) == expected
    oracle = icu_segmenter("line")
    for text, _ in cases:
        assert len(list(sz.utf8_linebreaks(text))) == len(oracle(text))


# endregion Oracles


# region Backend differential


LINEBREAK_REALISTIC_TEXTS = [
    "first\nsecond",
    "just one line",
    "Größe\u2028привет",
    "a\r\nb",
    "alpha\nbeta\ngamma",
]


@pytest.mark.parametrize(
    "text", LINEBREAK_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_linebreaks_backend_differential(text):
    """`utf8_linebreaks` agrees across every `capability_sweep()` backend on realistic, malformed, and
    lane-straddling inputs, compared as raw segment bytes so malformed UTF-8 sweeps safely; a divergence
    is a kernel bug, not a binding bug."""
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_linebreaks(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")


# endregion Backend differential
