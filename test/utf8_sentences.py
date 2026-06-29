#!/usr/bin/env python3
"""UAX-29 sentence-boundary tests: utf8_sentences behavior, official SentenceBreakTest conformance, and an
agreement-gated differential against ICU (else pysbd). Adds malformed/seam safety sweeps.

Mirrors the C++ scripts/test_utf8_sentences.cpp translation unit.
"""

from random import Random, choice, randint, seed

import pytest

import stringzilla as sz
from stringzilla import Str

from test.helpers import (
    SEED_VALUES,
    UnicodeDataDownloadError,
    get_sentence_break_properties,
    get_sentence_break_test_cases,
    baseline_sentence_boundaries,
    representatives_by_class,
)
from test.utf8_helpers import (
    SEGMENTATION_PALETTE,
    adversarial_utf8_inputs,
    assert_segments_tile,
    byte_boundaries,
    class_adjacency_strings,
    corpus_of_byte_length,
    icu_segmenter,
    window_seam_lengths,
)

_byte_boundaries = byte_boundaries
_SEGMENTATION_PALETTE = SEGMENTATION_PALETTE


#  region Sentence iterator


def test_utf8_sentences_basic():
    """Test basic sentence iteration."""
    result = [str(s) for s in sz.utf8_sentences("Hello. World!")]
    assert len(result) == 2
    assert "".join(result) == "Hello. World!"

    # Empty string.
    assert [str(s) for s in sz.utf8_sentences("")] == []

    # No terminator: the whole text is one sentence.
    assert [str(s) for s in sz.utf8_sentences("no terminator here")] == ["no terminator here"]


def test_utf8_sentences_skip_empty():
    """Test skip_empty parameter for sentence iteration."""
    result = [str(s) for s in sz.utf8_sentences("A. B. C.")]
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)


def test_utf8_sentences_unicode():
    """Test UTF-8 multi-byte sentence coverage."""
    # Cyrillic sentences separated by a period and space.
    result = [str(s) for s in sz.utf8_sentences("Привет. Мир.")]
    assert "".join(result) == "Привет. Мир."
    assert len(result) >= 2

    # Ideographic full stop terminates a sentence.
    result = [str(s) for s in sz.utf8_sentences("你好。世界。")]
    assert "".join(result) == "你好。世界。"


def test_utf8_sentences_str_method():
    """The Str.utf8_sentences() method must agree with the module function."""
    s = Str("Hello. World!")
    method_result = [str(x) for x in s.utf8_sentences(s)]
    module_result = [str(x) for x in sz.utf8_sentences(s)]
    assert method_result == module_result
    assert method_result == [str(x) for x in sz.utf8_sentences("Hello. World!")]


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_boundary_fuzz(seed_value: int):
    """Fuzz: compare the C sentence iterator vs the pure-Python UAX-29 baseline."""
    seed(seed_value)
    try:
        props = get_sentence_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode data files")

    # The pure-Python `baseline_sentence_boundaries` is an imperfect SB reference (e.g. it misses the SB11 break
    # after `ATerm Sp OP`). The authoritative oracles are the official UCD vector (tested strictly in
    # `test_utf8_sentence_boundary_official_conformance`, 512/512) and ICU. Use ICU to arbitrate a kernel-vs-baseline
    # mismatch: exonerate the kernel when it matches ICU — so a real kernel bug, disagreeing with BOTH, still fails.
    icu_sentences = None
    try:
        icu_sentences = icu_segmenter("sentence")
    except Exception:
        icu_sentences = None

    total = 0
    disagreements = []
    for _ in range(200):
        text = "".join(choice(_SEGMENTATION_PALETTE) for _ in range(randint(1, 100)))
        try:
            expected = baseline_sentence_boundaries(text, props)
        except Exception:
            continue
        total += 1
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(text))
        if sz_boundaries == expected:
            continue
        if icu_sentences is not None and sz_boundaries == _byte_boundaries(icu_sentences(text)):
            continue  # baseline bug; kernel agrees with ICU (and the strict UCD gate)
        # Neither imperfect oracle agrees on this text — the baseline misses some SB11 interactions and ICU is a
        # stale Unicode version, so they disagree on different boundaries. The official UCD vector (strict, elsewhere)
        # is the real gate; require broad agreement here to catch only gross regressions, not oracle imperfection.
        disagreements.append((" ".join(f"{ord(c):04X}" for c in text), expected, sz_boundaries))

    agreement = 1.0 - len(disagreements) / max(total, 1)
    assert agreement >= 0.98, "Sentence boundary agreement {:.2%} too low (seed {}):\n{}".format(
        agreement, seed_value, "\n".join(f"  {c}\n    baseline={e}\n    got={g}" for c, e, g in disagreements[:10])
    )


def test_utf8_sentence_boundary_official_conformance():
    """Full UAX-29 conformance against every case in the official SentenceBreakTest.txt."""
    try:
        test_cases = get_sentence_break_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for test_str, expected_byte_boundaries in test_cases:
        if not test_str:
            continue
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(test_str))
        if sz_boundaries != expected_byte_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in test_str)
            failures.append(f"  {cps}: expected {expected_byte_boundaries}, got {sz_boundaries}")

    assert not failures, "UAX-29 SentenceBreakTest conformance failures ({} / {}):\n{}".format(
        len(failures), len(test_cases), "\n".join(failures[:40])
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_differential_icu(seed_value: int):
    """Differential fuzz against an independent sentence segmenter (``icu``, else ``pysbd``)."""
    seed(seed_value)

    def icu_boundaries():
        icu = pytest.importorskip("icu", reason="PyICU not installed")
        bi = icu.BreakIterator.createSentenceInstance(icu.Locale.getRoot())

        def segments(text):
            uni = icu.UnicodeString(text)
            bi.setText(uni)
            parts = []
            start = bi.first()
            end = bi.nextBoundary()
            while end != icu.BreakIterator.DONE:
                parts.append(str(uni[start:end]))
                start = end
                end = bi.nextBoundary()
            return parts

        return segments

    try:
        ref_segments = icu_boundaries()
    except Exception:
        pysbd = pytest.importorskip("pysbd", reason="neither PyICU nor pysbd installed")
        segmenter = pysbd.Segmenter(language="en", clean=False)
        ref_segments = lambda text: segmenter.segment(text)  # noqa: E731

    # Restrict the palette to characters ICU/pysbd and StringZilla agree on the meaning of: terminators,
    # letters, spaces and newlines. Emoji/RI sentence behavior is not portable across segmenters.
    palette = list("abZ59 \t\n.,;:!?'\"()éαаカ中" "\u2028\u2029")

    failures = []
    for _ in range(500):
        text = "".join(choice(palette) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(text))
        ref_boundaries = _byte_boundaries(ref_segments(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    ref={ref_boundaries}")

    # Sentence segmentation differs in defensible ways across libraries; require broad agreement, not
    # bit-exactness, so a portable-but-imperfect reference does not produce spurious CI failures.
    agreement = 1.0 - len(failures) / 500.0
    assert agreement >= 0.80, "StringZilla vs reference sentence agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )


#  endregion Sentence iterator


#  region Synthetic corner cases (safety / seam)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_safety(seed_value: int):
    """Adversarial-byte safety: the sentence iterator must survive the malformed battery and still tile its input."""
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        assert_segments_tile(sz.utf8_sentences(raw), raw)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_seam(seed_value: int):
    """Window-seam sweep: inputs sized to straddle the 64-byte SIMD windows must still tile (segmentation
    against ICU is agreement-gated above, so the seam invariant asserted here is contiguous tiling). The phase
    sweep additionally shifts a fixed corpus by every byte offset 0..63 so the content lands at every alignment
    relative to the 64-byte window — the exhaustive deterministic complement to the length sweep."""
    rng = Random(seed_value)
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_sentences(raw), raw)
    body = corpus_of_byte_length(96, rng)
    for phase in range(64):
        raw = b"a" * phase + body
        assert_segments_tile(sz.utf8_sentences(raw), raw)


def test_utf8_sentence_class_adjacency():
    """Every Sentence_Break class-adjacency pair — one representative codepoint per class — checked against ICU.
    Derives far more class interactions than the random palette; agreement-gated (ICU is a stale-Unicode, locale-
    tailored reference), with the official SentenceBreakTest the strict oracle."""
    icu_sentences = icu_segmenter("sentence")
    try:
        properties = get_sentence_break_properties()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode data files")

    representatives = representatives_by_class(properties, count=1)
    cases = class_adjacency_strings(representatives, arity=2)

    failures = []
    for text in cases:
        if byte_boundaries(sz.utf8_sentences(text)) != byte_boundaries(icu_sentences(text)):
            failures.append(" ".join(f"{ord(character):04X}" for character in text))

    agreement = 1.0 - len(failures) / max(1, len(cases))
    assert agreement >= 0.80, "StringZilla vs ICU sentence class-adjacency agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )


#  endregion Synthetic corner cases (safety / seam)
