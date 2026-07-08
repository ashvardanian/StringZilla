#!/usr/bin/env python3
"""UTF-8 normalization tests: NFC/NFD/NFKC/NFKD, expansions, violation offsets, and an exhaustive
all-codepoints cross-check against ICU's Normalizer2 (gated on the local ICU's Unicode version).

Mirrors the C++ test/utf8_norm.cpp translation unit.
"""

import unicodedata
from random import Random

import pytest

import stringzilla as sz

from test.helpers import (
    SEED_VALUES,
    UnicodeDataDownloadError,
    get_combining_classes,
    get_normalization_test_cases,
)
from test.utf8_helpers import combining_scrambles, icu_normalizer


def test_unit_utf8_norm():
    """Test basic Unicode normalization functionality."""

    # ASCII is unchanged by all forms
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        assert sz.utf8_norm("hello", form) == b"hello"
        assert sz.utf8_norm("", form) == b""

    # NFC vs NFD differ for precomposed characters
    # "café" with precomposed é (U+00E9)
    cafe_nfc = "café"
    cafe_nfd = "café"  # e + combining acute accent
    # NFC of precomposed → precomposed bytes
    assert sz.utf8_norm(cafe_nfc, "NFC") == cafe_nfc.encode("utf-8")
    # NFD of precomposed → decomposed bytes
    assert sz.utf8_norm(cafe_nfc, "NFD") == cafe_nfd.encode("utf-8")
    # NFC of decomposed → precomposed bytes (composition)
    assert sz.utf8_norm(cafe_nfd, "NFC") == cafe_nfc.encode("utf-8")

    # NFKD of fi ligature (U+FB01) decomposes to "fi"
    assert sz.utf8_norm("ﬁ", "NFKD") == b"fi"
    assert sz.utf8_norm("ﬁ", "NFKC") == b"fi"

    # Idempotence: normalizing an already-normalized string gives the same result
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        once = sz.utf8_norm("café", form)
        twice = sz.utf8_norm(once, form)
        assert once == twice, f"Idempotence failed for form {form}"

    # Method form on Str
    assert sz.Str("café").utf8_norm("NFD") == "café".encode("utf-8")

    # Bytes input
    assert sz.utf8_norm("café".encode("utf-8"), "NFC") == "café".encode("utf-8")

    # Cross-check with Python unicodedata.normalize
    for sample in ("café", "Ångström", "ﬁle", "ẛ̣"):
        for form in ("NFC", "NFD", "NFKC", "NFKD"):
            expected = unicodedata.normalize(form, sample).encode("utf-8")
            got = sz.utf8_norm(sample, form)
            assert got == expected, f"Mismatch for {repr(sample)} form={form}: {got!r} != {expected!r}"


@pytest.mark.parametrize(
    "input_str, form, expected",
    [
        ("ﬁ", "NFKD", b"fi"),  # fi ligature compatibility decomposition
        ("ﬁ", "NFKC", b"fi"),  # fi ligature compatibility composition
        ("ﬀ", "NFKD", b"ff"),  # ff ligature
        ("ﬂ", "NFKD", b"fl"),  # fl ligature
        ("café", "NFD", "café".encode("utf-8")),  # precomposed → decomposed
        ("café", "NFC", "café".encode("utf-8")),  # decomposed → precomposed
        ("Å", "NFD", "Å".encode("utf-8")),  # Å → A + ring above
    ],
)
def test_utf8_norm_expansions(input_str, form, expected):
    """Test normalization with specific known transformations."""
    assert sz.utf8_norm(input_str, form) == expected


def test_unit_utf8_find_denormalized():
    """Test normalization violation detection."""
    # Pure ASCII is always normalized in every form
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        assert sz.utf8_find_denormalized("hello", form) is None
        assert sz.utf8_find_denormalized("", form) is None

    # A precomposed character (U+00E9 é) is valid NFC but is a violation for NFD
    precomposed = "café"  # NFC: already composed
    assert sz.utf8_find_denormalized(precomposed, "NFC") is None
    nfd_violation = sz.utf8_find_denormalized(precomposed, "NFD")
    assert nfd_violation is not None
    assert isinstance(nfd_violation, int)
    assert nfd_violation >= 0

    # A decomposed sequence (e + combining accent) is valid NFD but violates NFC
    decomposed = "café"  # NFD: e + combining acute
    assert sz.utf8_find_denormalized(decomposed, "NFD") is None
    nfc_violation = sz.utf8_find_denormalized(decomposed, "NFC")
    assert nfc_violation is not None
    assert isinstance(nfc_violation, int)

    # Method form on Str
    assert sz.Str("hello").utf8_find_denormalized("NFC") is None

    # Bytes input
    assert sz.utf8_find_denormalized(b"hello", "NFD") is None

    # fi ligature (U+FB01) is not NFKD-normalized
    assert sz.utf8_find_denormalized("ﬁ", "NFKD") is not None
    # But "fi" as two plain ASCII chars IS NFKD-normalized
    assert sz.utf8_find_denormalized("fi", "NFKD") is None

    # ValueError on unknown form
    import pytest as _pytest

    with _pytest.raises(ValueError, match="unknown form"):
        sz.utf8_find_denormalized("hello", "XYZ")


#  region Exhaustive ICU cross-check


@pytest.mark.parametrize("form", ["NFC", "NFD", "NFKC", "NFKD"])
def test_utf8_norm_all_codepoints_icu(form: str):
    """Exhaustive single-codepoint normalization vs ICU's Normalizer2, for every codepoint assigned in the
    locally-installed ICU's Unicode version.

    StringZilla targets a newer Unicode than many ICU builds ship; codepoints unassigned in the local ICU are
    skipped (via `Char.isdefined`) rather than flagged as spurious mismatches. This is the version-pinned twin
    of the case-folding all-codepoints sweep, independent of the host Python's `unicodedata` version.
    """
    icu = pytest.importorskip("icu", reason="PyICU not installed")
    normalize = icu_normalizer(form)

    mismatches = []
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:  # surrogates are not valid in UTF-8
            continue
        if not icu.Char.isdefined(codepoint):  # skip codepoints unassigned in the local ICU's Unicode version
            continue
        char = chr(codepoint)
        expected = normalize(char).encode("utf-8")
        got = sz.utf8_norm(char, form)
        if got != expected:
            mismatches.append((f"U+{codepoint:04X}", expected.hex(), got.hex()))

    assert not mismatches, "{} codepoints differ from ICU {} (icu unicode {}). First 10: {}".format(
        len(mismatches), form, icu.UNICODE_VERSION, mismatches[:10]
    )


@pytest.mark.parametrize("form", ["NFC", "NFD", "NFKC", "NFKD"])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_norm_random_sequences_icu(form: str, seed_value: int):
    """Multi-codepoint normalization vs ICU: random base-letter + combining-mark sequences exercise composition
    and canonical ordering across codepoint boundaries (the exhaustive single-codepoint sweep cannot)."""
    icu = pytest.importorskip("icu", reason="PyICU not installed")  # noqa: F841
    normalize = icu_normalizer(form)
    rng = Random(seed_value)

    # Bases + combining marks all predate Unicode 15.1, so StringZilla and any modern ICU agree on them.
    bases = "aeiounAEIOUNcsz"
    combining = [chr(cp) for cp in (0x0300, 0x0301, 0x0302, 0x0303, 0x0308, 0x0327, 0x0323)]
    alphabet = list(bases) + combining

    mismatches = []
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 16)))
        expected = normalize(text).encode("utf-8")
        got = sz.utf8_norm(text, form)
        if got != expected:
            mismatches.append((" ".join(f"{ord(c):04X}" for c in text), expected.hex(), got.hex()))

    assert not mismatches, "{} sequences differ from ICU {}. First 10: {}".format(
        len(mismatches), form, mismatches[:10]
    )


#  endregion Exhaustive ICU cross-check


#  region Official conformance + canonical ordering


def test_utf8_norm_official_conformance():
    """Authoritative NFC/NFD/NFKC/NFKD conformance against the official NormalizationTest.txt.

    Version-pinned to the project's target Unicode (17.0) and independent of ICU's or the host Python's
    Unicode version, this is the normalization analogue of the *BreakTest.txt segmentation suites: for every
    listed source string, each form must reproduce the file's expected column bit-exactly.
    """
    try:
        cases = get_normalization_test_cases()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode test data")

    failures = []
    for source, nfc, nfd, nfkc, nfkd in cases:
        expected_by_form = {"NFC": nfc, "NFD": nfd, "NFKC": nfkc, "NFKD": nfkd}
        for form, expected_text in expected_by_form.items():
            expected = expected_text.encode("utf-8")
            got = sz.utf8_norm(source, form)
            if got != expected:
                codepoints = " ".join(f"{ord(character):04X}" for character in source)
                failures.append((form, codepoints, expected.hex(), got.hex()))

    assert not failures, "NormalizationTest.txt conformance failures ({} / {} cases). First 10: {}".format(
        len(failures), len(cases), failures[:10]
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_norm_canonical_ordering(seed_value: int):
    """Independent canonical-ordering invariant: within each combining run, NFD output must order marks by
    non-decreasing Canonical_Combining_Class. Checked over random non-canonical combining scrambles, using the
    CCC table directly (no external normalizer needed)."""
    try:
        combining_classes = get_combining_classes()
    except UnicodeDataDownloadError:
        pytest.skip("Could not download Unicode data files")

    rng = Random(seed_value)
    for text in combining_scrambles(combining_classes, rng, 300):
        normalized = sz.utf8_norm(text, "NFD").decode("utf-8")
        previous_class = 0
        for character in normalized:
            combining_class = combining_classes.get(ord(character), 0)
            if combining_class == 0:
                previous_class = 0
                continue
            assert combining_class >= previous_class, "Non-canonical CCC order in NFD of {}: {}".format(
                " ".join(f"{ord(c):04X}" for c in text), " ".join(f"{ord(c):04X}" for c in normalized)
            )
            previous_class = combining_class


#  endregion Official conformance + canonical ordering


def test_utf8_norm_prose():
    """Realistic paragraphs normalize bit-identically to ICU across all four forms; NFD content is detected."""
    from test.utf8_helpers import (
        PROSE_HOTEL_REVIEW,
        PROSE_SCIENCE_ABSTRACT,
        PROSE_DEVANAGARI_TIP,
        PROSE_RTL_SCRIPTS,
        icu_normalizer,
    )

    # P1 carries an NFD 'cafe' (e + U+0301), so it is not already NFC.
    assert sz.utf8_find_denormalized(PROSE_HOTEL_REVIEW, "NFC") is not None
    for text in (PROSE_HOTEL_REVIEW, PROSE_SCIENCE_ABSTRACT, PROSE_DEVANAGARI_TIP, PROSE_RTL_SCRIPTS):
        for form in ("NFC", "NFD", "NFKC", "NFKD"):
            assert sz.utf8_norm(text, form) == icu_normalizer(form)(text).encode("utf-8")
