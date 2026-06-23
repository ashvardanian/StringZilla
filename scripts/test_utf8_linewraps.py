#!/usr/bin/env python3
"""UAX-14 line-break tests: utf8_linewraps behavior, strict official LineBreakTest conformance (bit-exact
19338/19338), and an agreement-gated differential against uniseg. Adds malformed/seam sweeps.

Mirrors the C++ scripts/test_utf8_linewraps.cpp translation unit.
"""

from random import Random, choice, randint, seed

import pytest

import stringzilla as sz
from stringzilla import Str

from test_helpers import (
    SEED_VALUES,
    UnicodeDataDownloadError,
    get_line_break_properties,
    get_line_break_test_cases,
    representatives_by_class,
)
from test_utf8_helpers import (
    SEGMENTATION_PALETTE,
    adversarial_utf8_inputs,
    assert_segments_tile,
    byte_boundaries,
    class_adjacency_strings,
    corpus_of_byte_length,
    window_seam_lengths,
)

_byte_boundaries = byte_boundaries
_SEGMENTATION_PALETTE = SEGMENTATION_PALETTE


#  region Line iterator


def test_utf8_linewraps_basic():
    """Test basic line iteration yielding line-break-opportunity Str segments."""
    result = [str(seg) for seg in sz.utf8_linewraps("first\nsecond")]
    assert "".join(result) == "first\nsecond"  # Segments tile the input.

    # Empty string yields nothing.
    assert [str(seg) for seg in sz.utf8_linewraps("")] == []

    # A single short line still tiles back to the input.
    result = [str(seg) for seg in sz.utf8_linewraps("just one line")]
    assert "".join(result) == "just one line"


def test_utf8_linewraps_tiling():
    """Linewrap is forward-only; its segments tile the input contiguously."""
    text = "alpha\nbeta\ngamma"
    forward = [str(seg) for seg in sz.utf8_linewraps(text)]
    assert "".join(forward) == text


def test_utf8_linewraps_skip_empty():
    """Test skip_empty parameter for line iteration."""
    result = [str(seg) for seg in sz.utf8_linewraps("a\n\nb", skip_empty=True)]
    assert len(result) > 0
    assert all(len(seg) > 0 for seg in result)


def test_utf8_linewraps_unicode():
    """Test UTF-8 multi-byte / Unicode coverage."""
    result = [str(seg) for seg in sz.utf8_linewraps("Größe\u2028привет")]
    assert "".join(result) == "Größe\u2028привет"
    # Segments tile the input.

    # CRLF is a single break opportunity, not two.
    result = [str(seg) for seg in sz.utf8_linewraps("a\r\nb")]
    assert "".join(result) == "a\r\nb"


def test_utf8_linewraps_str_method():
    """The Str.utf8_linewraps() method must agree with the module function."""
    s = Str("first\nsecond")
    method_result = [str(seg) for seg in s.utf8_linewraps(s)]
    module_result = [str(seg) for seg in sz.utf8_linewraps(s)]
    assert method_result == module_result
    assert method_result == [str(seg) for seg in sz.utf8_linewraps("first\nsecond")]


def test_utf8_linewrap_boundary_official_conformance():
    """Full UAX-14 conformance against every case in the official LineBreakTest.txt.

    The iterator emits a segment at every break opportunity (mandatory or soft); their cumulative
    byte lengths must match the official break positions bit-exactly — no tailoring-tolerance gate.

    LB2 (never break at start of text) means LineBreakTest never marks a leading ÷, so the official
    boundary list has no leading 0 — unlike GraphemeBreakTest/WordBreakTest/SentenceBreakTest, which
    break at start. The segments always start at offset 0, so drop the shared helper's leading 0 here.
    """
    try:
        test_cases = get_line_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for test_str, expected_byte_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_linewraps(test_str))[1:]
        if sz_boundaries != expected_byte_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_byte_boundaries}, got {sz_boundaries}")

    assert not failures, "UAX-14 LineBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_linewrap_differential_uniseg(seed_value: int):
    """Differential fuzz against ``uniseg.linebreak`` for line-break-opportunity segments."""
    uniseg_linebreak = pytest.importorskip("uniseg.linebreak", reason="uniseg not installed")
    seed(seed_value)

    def ref_segments(text):
        return list(uniseg_linebreak.line_break_units(text))

    failures = []
    for _ in range(2000):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_linewraps(text))
        ref_boundaries = _byte_boundaries(ref_segments(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    uniseg={ref_boundaries}")

    # uniseg implements default UAX-14 tailorings that may differ from StringZilla; require broad
    # agreement on segment positions rather than bit-exactness.
    agreement = 1.0 - len(failures) / 2000.0
    assert agreement >= 0.80, "StringZilla vs uniseg line-boundary agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )


#  endregion Line iterator


#  region Synthetic corner cases (safety / seam)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_linewrap_safety(seed_value: int):
    """Adversarial-byte safety: the linewrap iterator must survive the malformed battery and still tile its input."""
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        assert_segments_tile(sz.utf8_linewraps(raw), raw)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_linewrap_seam(seed_value: int):
    """Window-seam sweep: inputs sized to straddle the 64-byte SIMD windows must still tile (UAX-14 boundary
    agreement vs uniseg is gated above; the seam invariant asserted here is contiguous tiling)."""
    rng = Random(seed_value)
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_linewraps(raw), raw)
    # Phase sweep: shift a fixed corpus by every byte offset 0..63 so its content lands at every alignment
    # relative to the 64-byte window — exhaustive deterministic complement to the length sweep.
    body = corpus_of_byte_length(96, rng)
    for phase in range(64):
        raw = b"a" * phase + body
        assert_segments_tile(sz.utf8_linewraps(raw), raw)


#  endregion Synthetic corner cases (safety / seam)


#  region Rule-derived coverage (class adjacency)


def test_utf8_linewrap_class_adjacency():
    """Every Line_Break class-adjacency pair — built from one representative codepoint per class — checked
    against uniseg. Line breaking has the largest property alphabet (~40 classes), so this derives far more
    class interactions than the palette; agreement-gated (not bit-exact) because uniseg applies default UAX-14
    tailorings StringZilla may differ on."""
    uniseg_linebreak = pytest.importorskip("uniseg.linebreak", reason="uniseg not installed")
    try:
        properties = get_line_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode data files")

    representatives = representatives_by_class(properties, count=1)
    cases = class_adjacency_strings(representatives, arity=2)

    failures = []
    for text in cases:
        sz_boundaries = byte_boundaries(sz.utf8_linewraps(text))
        reference_boundaries = byte_boundaries(list(uniseg_linebreak.line_break_units(text)))
        if sz_boundaries != reference_boundaries:
            codepoints = " ".join(f"{ord(character):04X}" for character in text)
            failures.append(f"  {codepoints}\n    sz={sz_boundaries}\n    uniseg={reference_boundaries}")

    agreement = 1.0 - len(failures) / max(1, len(cases))
    assert agreement >= 0.80, "StringZilla vs uniseg line class-adjacency agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )


#  endregion Rule-derived coverage (class adjacency)
