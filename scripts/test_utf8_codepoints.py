#!/usr/bin/env python3
"""UTF-8 codepoint tests: utf8_count (codepoints, not bytes), plus round-trip and malformed-byte safety.

Mirrors the C++ scripts/test_utf8_codepoints.cpp translation unit.
"""

from random import Random

import pytest

import stringzilla as sz

from test_helpers import SEED_VALUES
from test_utf8_helpers import adversarial_utf8_inputs, random_segmentation_corpus


def test_unit_utf8_count():
    """Test UTF-8 character counting (codepoints, not bytes)."""
    # ASCII strings: len == utf8_count
    assert sz.utf8_count("hello") == 5
    assert sz.utf8_count("") == 0
    assert sz.utf8_count("a") == 1

    # Multi-byte UTF-8: character count != byte count
    assert sz.utf8_count("café") == 4  # é is 2 bytes but 1 char
    assert sz.utf8_count("日本語") == 3  # Each CJK char is 3 bytes
    assert sz.utf8_count("🎉") == 1  # Emoji is 4 bytes but 1 char
    assert sz.utf8_count("👨‍👩‍👧") == 5  # Family emoji: 3 people + 2 ZWJ

    # Mixed ASCII and multi-byte
    assert sz.utf8_count("hello世界") == 7  # 5 ASCII + 2 CJK
    assert sz.utf8_count("naïve") == 5  # ï is 2 bytes

    # Method form on Str
    assert sz.Str("café").utf8_count() == 4

    # Bytes input
    assert sz.utf8_count(b"caf\xc3\xa9") == 4  # café in UTF-8 bytes

    # Various multi-byte sequences
    assert sz.utf8_count("αβγδ") == 4  # Greek letters (2 bytes each)
    assert sz.utf8_count("АБВГ") == 4  # Cyrillic letters (2 bytes each)
    assert sz.utf8_count("𐍈") == 1  # Gothic letter (4 bytes)


#  region Synthetic corner cases (round-trip / malformed)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_utf8_count_roundtrip(seed_value: int):
    """On valid UTF-8 corpora, utf8_count equals Python's codepoint count for both str and bytes inputs."""
    rng = Random(seed_value)
    for _ in range(200):
        raw = random_segmentation_corpus(rng.randint(1, 400), "valid", None, rng)
        text = raw.decode("utf-8")
        assert sz.utf8_count(text) == len(text)
        assert sz.utf8_count(raw) == len(text)


def test_utf8_count_malformed():
    """Malformed-byte safety: utf8_count must survive the adversarial battery and stay within [0, byte length]."""
    rng = Random(0)
    for raw in adversarial_utf8_inputs(rng, random_input_count=500):
        count = sz.utf8_count(raw)
        assert 0 <= count <= len(raw)


#  endregion Synthetic corner cases (round-trip / malformed)
