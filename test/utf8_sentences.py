#!/usr/bin/env python3
"""UAX-29 sentence-boundary segmentation: sz.utf8_sentences and Str.utf8_sentences.

Mirrors the C++ scripts/test_utf8_sentences.cpp translation unit.

Covers: sentence iteration and the Str method mirror across ASCII, Cyrillic, and ideographic
terminators, malformed-byte and window-seam tiling safety, full UAX-29 SentenceBreakTest.txt
conformance and Sentence_Break class-adjacency pairs, differential fuzzing against ICU or pysbd
and against a pure-Python UAX-29 baseline arbitrated by ICU, realistic multi-script prose, and
cross-backend agreement on realistic, malformed, and lane-straddling inputs.
Compares against: the official UCD SentenceBreakTest.txt, ICU BreakIterator sentence segmentation
with pysbd as a fallback oracle, a pure-Python UAX-29 baseline, and cross-backend self-consistency
via capability_sweep.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat PyICU pysbd
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/utf8_sentences.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/utf8_sentences.py -q
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
    get_sentence_break_test_cases,
    baseline_sentence_boundaries,
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
    icu_segmenter,
    window_seam_lengths,
)

_byte_boundaries = byte_boundaries


# region Unit


def test_utf8_sentences_basic():
    """Splits `"Hello. World!"` into two sentences that concatenate back to the input, yields nothing
    for an empty string, and treats unterminated text as a single sentence."""
    result = [str(s) for s in sz.utf8_sentences("Hello. World!")]
    assert len(result) == 2
    assert "".join(result) == "Hello. World!"

    # Empty string.
    assert [str(s) for s in sz.utf8_sentences("")] == []

    # No terminator: the whole text is one sentence.
    assert [str(s) for s in sz.utf8_sentences("no terminator here")] == ["no terminator here"]


def test_utf8_sentences_skip_empty():
    """Splitting `"A. B. C."` into sentences yields only non-empty segments."""
    result = [str(s) for s in sz.utf8_sentences("A. B. C.")]
    assert len(result) > 0
    assert all(len(s) > 0 for s in result)


def test_utf8_sentences_unicode():
    """Sentence splitting on Cyrillic text yields at least two sentences that concatenate back to the
    input, and the CJK ideographic full stop terminates sentences the same way."""
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


# endregion Unit


# region Corner cases


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_safety(seed_value: int):
    """Adversarial-byte safety: the sentence iterator must survive the malformed battery and still tile its input."""
    rng = Random(seed_value)
    for raw in adversarial_utf8_inputs(rng):
        assert_segments_tile(sz.utf8_sentences(raw), raw)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_seam(seed_value: int):
    """Window-seam sweep: inputs sized to straddle the 64-byte SIMD windows must still tile, since
    segmentation against ICU is agreement-gated above and the invariant asserted here is contiguous
    tiling. The phase sweep additionally shifts a fixed corpus by every byte offset 0..63 so the
    content lands at every alignment relative to the 64-byte window, the exhaustive deterministic
    complement to the length sweep."""
    rng = Random(seed_value)
    for length in window_seam_lengths():
        raw = corpus_of_byte_length(length, rng)
        assert_segments_tile(sz.utf8_sentences(raw), raw)
    body = corpus_of_byte_length(96, rng)
    for phase in range(64):
        raw = b"a" * phase + body
        assert_segments_tile(sz.utf8_sentences(raw), raw)


# endregion Corner cases


# region Conformance


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


def test_utf8_sentence_class_adjacency(sentence_break_props):
    """Every Sentence_Break class-adjacency pair, one representative codepoint per class, checked against ICU.
    Derives far more class interactions than the random palette; agreement-gated since ICU is a stale-Unicode,
    locale-tailored reference, with the official SentenceBreakTest the strict oracle."""
    icu_sentences = icu_segmenter("sentence")
    representatives = representatives_by_class(sentence_break_props, count=1)
    cases = class_adjacency_strings(representatives, arity=2)

    failures = []
    for text in cases:
        if byte_boundaries(sz.utf8_sentences(text)) != byte_boundaries(icu_sentences(text)):
            failures.append(" ".join(f"{ord(character):04X}" for character in text))

    agreement = 1.0 - len(failures) / max(1, len(cases))
    assert agreement >= 0.80, "StringZilla vs ICU sentence class-adjacency agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )


# endregion Conformance


# region Oracles


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

    iterations = scale_iterations(500)
    failures = []
    for _ in range(iterations):
        text = "".join(choice(palette) for _ in range(randint(1, 24)))
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(text))
        ref_boundaries = _byte_boundaries(ref_segments(text))
        if sz_boundaries != ref_boundaries:
            cps = " ".join(f"{ord(c):04X}" for c in text)
            failures.append(f"  {cps}\n    sz={sz_boundaries}\n    ref={ref_boundaries}")

    # Sentence segmentation differs in defensible ways across libraries; require broad agreement, not
    # bit-exactness, so a portable-but-imperfect reference does not produce spurious CI failures.
    agreement = 1.0 - len(failures) / iterations
    assert agreement >= 0.80, "StringZilla vs reference sentence agreement {:.2%} too low:\n{}".format(
        agreement, "\n".join(failures[:20])
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_sentence_boundary_fuzz(seed_value: int, sentence_break_props):
    """Fuzz: compare the C sentence iterator vs the pure-Python UAX-29 baseline."""
    seed(seed_value)

    # The pure-Python `baseline_sentence_boundaries` is an imperfect SB reference, missing the SB11 break
    # after `ATerm Sp OP`. The authoritative oracles are the official UCD vector, tested strictly in
    # `test_utf8_sentence_boundary_official_conformance` at 512/512, and ICU. Use ICU to arbitrate a
    # kernel-vs-baseline mismatch, exonerating the kernel when it matches ICU, so a real kernel bug,
    # disagreeing with both, still fails.
    icu_sentences = None
    try:
        icu_sentences = icu_segmenter("sentence")
    except Exception:
        icu_sentences = None

    total = 0
    disagreements = []
    for _ in range(scale_iterations(200)):
        text = "".join(choice(SEGMENTATION_PALETTE) for _ in range(randint(1, 100)))
        try:
            expected = baseline_sentence_boundaries(text, sentence_break_props)
        except Exception:
            continue
        total += 1
        sz_boundaries = _byte_boundaries(sz.utf8_sentences(text))
        if sz_boundaries == expected:
            continue
        if icu_sentences is not None and sz_boundaries == _byte_boundaries(icu_sentences(text)):
            continue  # baseline bug, kernel agrees with ICU and the strict UCD gate
        # Neither imperfect oracle agrees on this text. The baseline misses some SB11 interactions and ICU
        # is a stale Unicode version, so they disagree on different boundaries. The official UCD vector is
        # the real gate, strictly enforced elsewhere; require broad agreement here to catch only gross
        # regressions, not oracle imperfection.
        disagreements.append((" ".join(f"{ord(c):04X}" for c in text), expected, sz_boundaries))

    agreement = 1.0 - len(disagreements) / max(total, 1)
    assert agreement >= 0.98, "Sentence boundary agreement {:.2%} too low (seed {}):\n{}".format(
        agreement, seed_value, "\n".join(f"  {c}\n    baseline={e}\n    got={g}" for c, e, g in disagreements[:10])
    )


def test_utf8_sentences_prose():
    """Realistic multi-script paragraphs: sentence count matches the ICU root oracle; segments tile."""
    from test.utf8_helpers import (
        PROSE_HOTEL_REVIEW,
        PROSE_CONCERT_POST,
        PROSE_NEWS_LEDE,
        PROSE_MICRO_HARDBREAKS,
        assert_segments_tile,
        icu_segmenter,
    )

    cases = [
        (PROSE_HOTEL_REVIEW, 5),
        (PROSE_CONCERT_POST, 4),
        (PROSE_NEWS_LEDE, 6),
        (PROSE_MICRO_HARDBREAKS, 3),
    ]
    for text, expected in cases:
        segments = list(sz.utf8_sentences(text))
        assert_segments_tile(segments, text)
        assert len(segments) == expected
    oracle = icu_segmenter("sentence")
    for text, _ in cases:
        assert len(list(sz.utf8_sentences(text))) == len(oracle(text))


# endregion Oracles


# region Backend differential


SENTENCE_REALISTIC_TEXTS = [
    "Hello. World!",
    "no terminator here",
    "Привет. Мир.",
    "你好。世界。",
    "Dr. Smith went home. He slept.",
]


@pytest.mark.parametrize(
    "text", SENTENCE_REALISTIC_TEXTS + list(malformed_utf8_corpus()) + vector_width_bracketing_strings()
)
def test_utf8_sentences_backend_differential(text):
    """utf8_sentences must segment identically across every SIMD backend on realistic, malformed, and
    lane-straddling inputs, compared as raw segment bytes so malformed UTF-8 sweeps without raising."""
    results = run_across_backends(lambda: [bytes(segment) for segment in sz.utf8_sentences(text)])
    assert_backends_agree(results, format_inputs=lambda: f"text={text!r}")


# endregion Backend differential
