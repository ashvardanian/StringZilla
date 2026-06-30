#!/usr/bin/env python3
"""Uncased UTF-8 tests: case folding (incl. exhaustive UCD 17.0 sweep), uncased find/matches/order.

Mirrors the C++ test/uncased.cpp translation unit.
"""

from random import randint, seed

import pytest

import stringzilla as sz

from test.helpers import SEED_VALUES, UnicodeDataDownloadError, get_uncased_folding_rules


def test_unit_utf8_uncased_fold():
    """Test basic case folding functionality."""
    # ASCII
    assert sz.utf8_uncased_fold("HELLO") == b"hello"
    assert sz.utf8_uncased_fold("Hello World") == b"hello world"
    assert sz.utf8_uncased_fold("") == b""
    assert sz.utf8_uncased_fold("already lowercase") == b"already lowercase"

    # German sharp S expansion (ß → ss)
    assert sz.utf8_uncased_fold("Straße") == b"strasse"
    assert sz.utf8_uncased_fold("GROSSE") == b"grosse"

    # Method form on Str
    assert sz.Str("HELLO").utf8_uncased_fold() == b"hello"

    # Bytes input
    assert sz.utf8_uncased_fold(b"HELLO") == b"hello"


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("A", b"a"),
        ("Z", b"z"),
        ("ß", b"ss"),  # German sharp S (U+00DF)
        ("ẞ", b"ss"),  # Capital sharp S (U+1E9E)
        ("ﬁ", b"fi"),  # fi ligature (U+FB01)
        ("ﬀ", b"ff"),  # ff ligature (U+FB00)
        ("ﬂ", b"fl"),  # fl ligature (U+FB02)
        ("ﬃ", b"ffi"),  # ffi ligature (U+FB03)
        ("ﬄ", b"ffl"),  # ffl ligature (U+FB04)
        ("Σ", "σ".encode("utf-8")),  # Greek Sigma
        ("Ω", "ω".encode("utf-8")),  # Greek Omega
        ("Ä", "ä".encode("utf-8")),  # German umlaut
        ("É", "é".encode("utf-8")),  # French accent
        ("Ñ", "ñ".encode("utf-8")),  # Spanish tilde
    ],
)
def test_utf8_uncased_fold_expansions(input_str, expected):
    """Test case folding with specific known transformations including expansions."""
    assert sz.utf8_uncased_fold(input_str) == expected


def test_utf8_uncased_fold_all_codepoints():
    """Compare StringZilla case folding with Unicode 17.0 CaseFolding.txt rules.

    This test downloads the official Unicode 17.0 case folding data file to validate
    StringZilla's implementation, independent of Python's Unicode version.
    The file is cached in the system temp directory for subsequent runs.
    """
    # Load Unicode 17.0 case folding rules (downloads and caches automatically)
    try:
        unicode_folds = get_uncased_folding_rules("17.0.0")
    except UnicodeDataDownloadError as e:
        pytest.skip(f"Skipping due to network issue: {e}")

    mismatches = []
    missing_folds = []
    extra_folds = []

    for codepoint in range(0x110000):
        # Skip surrogates (not valid in UTF-8)
        if 0xD800 <= codepoint <= 0xDFFF:
            continue

        try:
            char = chr(codepoint)
            char_bytes = char.encode("utf-8")
            sz_folded = sz.utf8_uncased_fold(char)

            # Get expected folding from Unicode 17.0 rules
            # If not in the table, character maps to itself
            expected = unicode_folds.get(codepoint, char_bytes)

            if sz_folded != expected:
                entry = (f"U+{codepoint:04X}", repr(char), expected.hex(), sz_folded.hex())
                if codepoint in unicode_folds and sz_folded == char_bytes:
                    missing_folds.append(entry)  # StringZilla didn't fold but should have
                elif codepoint not in unicode_folds and sz_folded != char_bytes:
                    extra_folds.append(entry)  # StringZilla folded but shouldn't have
                else:
                    mismatches.append(entry)  # Both fold but to different targets
        except (ValueError, UnicodeEncodeError):
            continue

    total_errors = len(mismatches) + len(missing_folds) + len(extra_folds)
    assert total_errors == 0, (
        f"Case folding errors vs Unicode 17.0 ({len(unicode_folds)} rules): "
        f"{len(mismatches)} wrong targets, {len(missing_folds)} missing, {len(extra_folds)} extra. "
        f"First 10: {(mismatches + missing_folds + extra_folds)[:10]}"
    )


def _reference_uncased_find(haystack_bytes: bytes, needle_folded: bytes, unicode_folds: dict) -> int:
    """Reference uncased find from UCD rules: byte offset of the first
    haystack position whose codepoint folds concatenate to exactly `needle_folded`."""
    if not needle_folded:
        return 0
    text = haystack_bytes.decode("utf-8")
    codepoint_byte_offsets = []
    codepoint_folds = []
    byte_offset = 0
    for char in text:
        char_bytes = char.encode("utf-8")
        codepoint_byte_offsets.append(byte_offset)
        codepoint_folds.append(unicode_folds.get(ord(char), char_bytes))
        byte_offset += len(char_bytes)
    for start in range(len(text)):
        accumulated = b""
        for index in range(start, len(text)):
            accumulated += codepoint_folds[index]
            if len(accumulated) >= len(needle_folded):
                break
        if accumulated == needle_folded:
            return codepoint_byte_offsets[start]
    return -1


def test_utf8_uncased_search_preimages():
    """Adversarial probes from Unicode 17.0 CaseFolding.txt: for every codepoint whose
    fold differs from itself, the fold OUTPUT is what a user types as the needle — and
    the PREIMAGE is what hides in the haystack, placed at chunk-boundary offsets.
    """
    # Load Unicode 17.0 case folding rules (downloads and caches automatically)
    try:
        unicode_folds = get_uncased_folding_rules("17.0.0")
    except UnicodeDataDownloadError as e:
        pytest.skip(f"Skipping due to network issue: {e}")

    mismatches = []
    for codepoint in sorted(unicode_folds):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        preimage_bytes = chr(codepoint).encode("utf-8")
        folded_bytes = unicode_folds[codepoint]
        if preimage_bytes == folded_bytes:
            continue  # Identity folds carry no signal here
        for offset in (0, 30, 62, 63, 64):
            haystack = b"y" * offset + preimage_bytes + b"tail"
            expected = _reference_uncased_find(haystack, folded_bytes, unicode_folds)
            actual = sz.utf8_uncased_search(haystack, folded_bytes)
            if actual != expected:
                mismatches.append((f"U+{codepoint:04X}", offset, expected, actual))

    assert not mismatches, (
        f"{len(mismatches)} uncased find mismatches vs Unicode 17.0 reference. "
        f"First 10 (codepoint, offset, expected, actual): {mismatches[:10]}"
    )


def _reference_uncased_find_subset(haystack_bytes: bytes, needle_folded: bytes, unicode_folds: dict) -> int:
    """Independent fold-subset reference matching `sz.utf8_uncased_search`'s documented semantics:
    fold every haystack codepoint, concatenate into one folded blob, and find the earliest byte
    offset at which `needle_folded` appears as a substring. The match neither needs to start nor end
    on a codepoint boundary -- it may begin or end mid-expansion. The returned value is the byte
    offset (into the original haystack) of the codepoint that CONTAINS the first matched byte, or -1
    when the folded needle is absent."""
    if not needle_folded:
        return 0
    text = haystack_bytes.decode("utf-8")
    folded_blob = b""
    source_byte_offset_for_folded_byte = []  # Maps each folded byte to its source codepoint's byte offset
    byte_offset = 0
    for char in text:
        char_bytes = char.encode("utf-8")
        folded_run = unicode_folds.get(ord(char), char_bytes)
        folded_blob += folded_run
        source_byte_offset_for_folded_byte.extend([byte_offset] * len(folded_run))
        byte_offset += len(char_bytes)
    match_position = folded_blob.find(needle_folded)
    if match_position == -1:
        return -1
    return source_byte_offset_for_folded_byte[match_position]


def test_utf8_uncased_search_crossing_expansions():
    """Crossing-expansion probes: matches whose folded runes span the boundary between two
    adjacent expanding codepoints. For example needle `ſß` folds to "sss" and must be found
    inside haystack `ßß` (folding to "ssss") at offset 0; the match is not contained within a
    single haystack codepoint's expansion but crosses from one expansion into the next.
    Each case is swept across non-folding prefix paddings to exercise chunk-boundary handling.
    """
    # Load Unicode 17.0 case folding rules (downloads and caches automatically)
    try:
        unicode_folds = get_uncased_folding_rules("17.0.0")
    except UnicodeDataDownloadError as e:
        pytest.skip(f"Skipping due to network issue: {e}")

    def fold_needle(needle_str: str) -> bytes:
        """Fold a needle string into its concatenated codepoint folds, producing the
        `needle_folded` bytes that the fold-subset reference compares against."""
        folded = b""
        for char in needle_str:
            folded += unicode_folds.get(ord(char), char.encode("utf-8"))
        return folded

    # (haystack, needle) pairs whose folded forms overlap across expansion boundaries.
    crossing_cases = [
        ("ßß", "sss"),
        ("ßß", "ſß"),
        ("ẞß", "ssss"),
        ("ẞß", "sss"),
        ("ﬃ", "fi"),
        ("ﬃ", "ffi"),
        ("ﬀﬁ", "ffi"),
    ]
    filler_char = "z"  # Folds to itself, so it never participates in a match
    prefix_paddings = [0, 30, 62, 63, 64, 65]

    mismatches = []
    for haystack_str, needle_str in crossing_cases:
        needle_folded = fold_needle(needle_str)
        for padding in prefix_paddings:
            haystack_str_padded = filler_char * padding + haystack_str
            haystack_bytes = haystack_str_padded.encode("utf-8")
            needle_bytes = needle_str.encode("utf-8")
            expected = _reference_uncased_find_subset(haystack_bytes, needle_folded, unicode_folds)
            actual = sz.utf8_uncased_search(haystack_bytes, needle_bytes)
            if actual != expected:
                mismatches.append((haystack_str, needle_str, padding, expected, actual))

    assert not mismatches, (
        f"{len(mismatches)} crossing-expansion find mismatches vs Unicode 17.0 reference. "
        f"(haystack, needle, padding, expected, actual): {mismatches[:10]}"
    )


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_uncased_fold_random_strings(seed_value: int):
    """Test case folding on random multi-codepoint strings."""
    seed(seed_value)

    # Test with ASCII uppercase
    for _ in range(50):
        length = randint(1, 100)
        test_str = "".join(chr(randint(0x41, 0x5A)) for _ in range(length))  # A-Z
        python_folded = test_str.casefold().encode("utf-8")
        sz_folded = sz.utf8_uncased_fold(test_str)
        assert python_folded == sz_folded, f"Mismatch for: {test_str!r}"

    # Test with Latin Extended characters
    for _ in range(50):
        length = randint(1, 50)
        # Mix of ASCII uppercase and Latin Extended (includes ß, etc.)
        codepoints = [randint(0x41, 0x5A) for _ in range(length)]
        codepoints += [randint(0xC0, 0xFF) for _ in range(length // 2)]
        test_str = "".join(chr(cp) for cp in codepoints)
        python_folded = test_str.casefold().encode("utf-8")
        sz_folded = sz.utf8_uncased_fold(test_str)
        assert python_folded == sz_folded, f"Mismatch for: {test_str!r}"


@pytest.mark.parametrize(
    "haystack, needle, expected",
    [
        # ASCII basic cases
        ("Hello World", "WORLD", 6),
        ("Hello World", "hello", 0),
        ("Hello World", "xyz", -1),
        ("HELLO", "hello", 0),
        ("hello", "HELLO", 0),
        ("HeLLo WoRLd", "world", 6),
        ("abcdef", "CD", 2),
        # Latin1 accented characters (C3 lead byte range)
        ("Über allen Gipfeln", "ÜBER", 0),
        ("ÜBER", "über", 0),
        ("Das schöne Mädchen", "SCHÖNE", 4),
        ("Café au lait", "CAFÉ", 0),
        ("naïve approach", "NAÏVE", 0),
        ("El niño juega", "NIÑO", 3),
        # German Eszett: ß ↔ ss (bidirectional)
        ("Straße", "STRASSE", 0),
        ("STRASSE", "straße", 0),
        ("die Straße", "STRASSE", 4),
        ("groß", "GROSS", 0),
        ("GROSS", "groß", 0),
        ("Fußball", "FUSSBALL", 0),
        # Cyrillic
        ("ПРИВЕТ", "привет", 0),
        ("привет", "ПРИВЕТ", 0),
        ("Москва столица", "МОСКВА", 0),
        ("добрый день", "ДОБРЫЙ", 0),
        # Greek
        ("ΑΒΓΔ", "αβγδ", 0),
        ("αβγδ", "ΑΒΓΔ", 0),
        ("Ελλάδα", "ΕΛΛΆΔΑ", 0),
        # Mixed scripts
        ("Hello Мир World", "МИР", 6),
        ("Café МОСКВА", "москва", 5),
        # Empty and edge cases
        ("hello", "", 0),
        ("", "x", -1),
        ("", "", 0),
        ("a", "A", 0),
        ("A", "a", 0),
    ],
)
def test_utf8_uncased_search(haystack, needle, expected):
    """Test uncased UTF-8 substring search with various scripts."""
    assert sz.utf8_uncased_search(haystack, needle) == expected


def test_utf8_uncased_search_method():
    """Test uncased find as a method on Str objects."""
    s = sz.Str("Hello World")
    assert s.utf8_uncased_search("WORLD") == 6
    assert s.utf8_uncased_search("hello") == 0
    assert s.utf8_uncased_search("xyz") == -1


def test_utf8_uncased_search_offsets():
    """Test that str returns codepoint offsets and bytes returns byte offsets."""
    # str: codepoint offsets
    # 'Hëllo' = 5 codepoints, 'Wörld' starts at codepoint 6
    assert sz.utf8_uncased_search("Hëllo Wörld", "WÖRLD") == 6
    assert sz.utf8_uncased_search("Café", "FÉ") == 2  # C, a = 2 codepoints

    # bytes: byte offsets
    # 'Hëllo' = H(1) + ë(2) + l(1) + l(1) + o(1) + space(1) = 7 bytes
    assert sz.utf8_uncased_search("Hëllo Wörld".encode(), "WÖRLD".encode()) == 7
    assert sz.utf8_uncased_search("Café".encode(), "FÉ".encode()) == 2


def test_utf8_uncased_search_start_end():
    """Test start/end parameters with codepoint and byte offsets."""
    # str: codepoint offsets
    text = "Hëllo Hëllo"  # 11 codepoints, 13 bytes
    assert sz.utf8_uncased_search(text, "HËLLO", 0) == 0
    assert sz.utf8_uncased_search(text, "HËLLO", 1) == 6  # Skip first
    assert sz.utf8_uncased_search(text, "HËLLO", 0, 5) == 0  # Within range
    assert sz.utf8_uncased_search(text, "HËLLO", 0, 4) == -1  # Too short

    # bytes: byte offsets
    text_bytes = text.encode()
    assert sz.utf8_uncased_search(text_bytes, "HËLLO".encode(), 0) == 0
    assert sz.utf8_uncased_search(text_bytes, "HËLLO".encode(), 1) == 7
    assert sz.utf8_uncased_search(text_bytes, "HËLLO".encode(), 0, 7) == 0
    assert sz.utf8_uncased_search(text_bytes, "HËLLO".encode(), 0, 5) == -1


def test_utf8_uncased_search_bytes():
    """Test uncased find with bytes input."""
    assert sz.utf8_uncased_search(b"Hello World", b"WORLD") == 6
    assert sz.utf8_uncased_search(b"Strasse", b"STRASSE") == 0
    assert sz.utf8_uncased_search("Straße".encode(), b"STRASSE") == 0
    assert sz.utf8_uncased_search("XXStraße".encode(), b"STRASSE") == 2


def test_utf8_uncased_search_slicing():
    """Verify returned index works correctly for slicing."""
    text = "Café au lait"
    idx = sz.utf8_uncased_search(text, "AU")
    assert text[idx : idx + 2].lower() == "au"

    text_bytes = text.encode()
    idx = sz.utf8_uncased_search(text_bytes, b"AU")
    assert text_bytes[idx : idx + 2].lower() == b"au"


@pytest.mark.parametrize(
    "haystack, needle, expected_matches",
    [
        # ASCII - multiple matches
        ("Hello HELLO hello HeLLo", "hello", ["Hello", "HELLO", "hello", "HeLLo"]),
        ("Hello World", "world", ["World"]),
        ("Hello World", "xyz", []),
        # German ß/ss expansion
        ("Straße STRASSE strasse", "strasse", ["Straße", "STRASSE", "strasse"]),
        ("groß GROSS", "gross", ["groß", "GROSS"]),
        # Cyrillic
        ("ПРИВЕТ привет Привет", "привет", ["ПРИВЕТ", "привет", "Привет"]),
        # Greek
        ("ΑΒΓΔ αβγδ", "αβγδ", ["ΑΒΓΔ", "αβγδ"]),
        # Edge cases
        ("", "hello", []),
        ("hello", "xyz", []),
    ],
)
def test_utf8_uncased_matches(haystack, needle, expected_matches):
    """Parametrized test for uncased find iterator."""
    matches = [str(m) for m in sz.utf8_uncased_matches(haystack, needle)]
    assert matches == expected_matches
    # Also test method form
    matches_method = [str(m) for m in sz.Str(haystack).utf8_uncased_matches(needle)]
    assert matches_method == expected_matches


def test_utf8_uncased_matches_overlapping():
    """Test overlapping vs non-overlapping modes and bytes input."""
    # Non-overlapping (default)
    assert len(list(sz.utf8_uncased_matches("aaaa", "aa"))) == 2
    # Overlapping
    assert len(list(sz.utf8_uncased_matches("aaaa", "aa", include_overlapping=True))) == 3
    # Bytes input
    assert len(list(sz.utf8_uncased_matches(b"Hello HELLO", b"hello"))) == 2


def test_utf8_uncased_order():
    """Test uncased UTF-8 comparison."""
    # Equal strings
    assert sz.utf8_uncased_order("hello", "HELLO") == 0
    assert sz.utf8_uncased_order("HELLO", "hello") == 0
    assert sz.utf8_uncased_order("HeLLo", "hElLO") == 0

    # German sharp S equivalence
    assert sz.utf8_uncased_order("Straße", "STRASSE") == 0
    assert sz.utf8_uncased_order("strasse", "STRAẞE") == 0

    # Less than
    assert sz.utf8_uncased_order("apple", "BANANA") < 0
    assert sz.utf8_uncased_order("APPLE", "banana") < 0
    assert sz.utf8_uncased_order("a", "B") < 0

    # Greater than
    assert sz.utf8_uncased_order("ZEBRA", "apple") > 0
    assert sz.utf8_uncased_order("zebra", "APPLE") > 0
    assert sz.utf8_uncased_order("z", "A") > 0

    # Empty strings
    assert sz.utf8_uncased_order("", "") == 0
    assert sz.utf8_uncased_order("", "a") < 0
    assert sz.utf8_uncased_order("a", "") > 0

    # Prefix ordering
    assert sz.utf8_uncased_order("hello", "HELLO WORLD") < 0
    assert sz.utf8_uncased_order("HELLO WORLD", "hello") > 0

    # Greek letters
    assert sz.utf8_uncased_order("ΑΒΓΔ", "αβγδ") == 0
    assert sz.utf8_uncased_order("αβγ", "ΑΒΓ") == 0

    # Cyrillic
    assert sz.utf8_uncased_order("ПРИВЕТ", "привет") == 0

    # Method form on Str
    s = sz.Str("hello")
    assert s.utf8_uncased_order("HELLO") == 0

    # bytes input
    assert sz.utf8_uncased_order(b"hello", b"HELLO") == 0
    assert sz.utf8_uncased_order(b"HELLO", b"hello") == 0
    assert sz.utf8_uncased_order(b"apple", b"BANANA") < 0
    assert sz.utf8_uncased_order(b"ZEBRA", b"apple") > 0

    # German sharp S with bytes
    assert sz.utf8_uncased_order("Straße".encode(), b"STRASSE") == 0
    assert sz.utf8_uncased_order(b"strasse", "Straße".encode()) == 0

    # Greek with bytes
    assert sz.utf8_uncased_order("ΑΒΓΔ".encode(), "αβγδ".encode()) == 0

    # Cyrillic with bytes
    assert sz.utf8_uncased_order("ПРИВЕТ".encode(), "привет".encode()) == 0


def test_utf8_uncased_prose():
    """Case-folded search finds realistic terms that a plain byte search misses."""
    from test.utf8_helpers import PROSE_HOTEL_REVIEW, PROSE_SCIENCE_ABSTRACT, PROSE_LANGUAGE_LESSON

    cases = [
        (PROSE_HOTEL_REVIEW, "strasse"),
        (PROSE_SCIENCE_ABSTRACT, "film"),
        (PROSE_LANGUAGE_LESSON, "strasse"),
    ]
    for haystack, needle in cases:
        assert haystack.find(needle) == -1  # plain byte search misses
        result = sz.utf8_uncased_search(haystack, needle)
        assert result is not None and result >= 0
