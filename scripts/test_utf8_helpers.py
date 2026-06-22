"""
Shared UTF-8 segmentation test driver for the per-family test modules.

The Python analog of the C++ `scripts/test_utf8.hpp`: a single place that owns the boundary-relevant
palettes, the SMP/astral fixtures, the malformed-UTF-8 corpus generators, the window-seam length sweep,
the adversarial-byte battery, and the metamorphic tiling invariant — so every family TU
(test_utf8_words.py, test_utf8_graphemes.py, …) shares one driver instead of copying corpora.

Differential oracles live here too: `icu_segmenter` / `icu_normalizer` wrap PyICU (skipped when absent),
generalizing the sentence-only ICU idiom that previously lived inline in the monolith.

Palette members are built from explicit codepoints (the source stays pure ASCII) so an editor or a
delegated agent cannot silently NFC-normalize a raw multi-byte literal.
"""

import itertools
from typing import Callable, Iterable, List, Optional, Sequence, Union

import pytest

#  region Palettes & fixtures

# Boundary-relevant codepoints spanning every category the segmentation kernels must get right: combining
# marks, Format / ZWJ, Regional-Indicators, CJK / Hangul / Kana, astral pictographs, and the mandatory line
# separators. Built by codepoint so the file holds no raw multi-byte literals.
_PALETTE_CODEPOINTS = [
    0x00E9,  # é  Latin-1 letter with combining-equivalent (1 codepoint)
    0x03B1,  # α  Greek
    0x0430,  # а  Cyrillic
    0x05D0,  # א  Hebrew (WB7a)
    0x05D1,  # ב  Hebrew
    0x30AB,  # カ Katakana (WB13, ICU dictionary script)
    0x30BF,  # タ Katakana
    0x4E2D,  # 中 CJK (ICU dictionary script)
    0xAD6D,  # 국 Hangul syllable
    0x0300,  # combining grave (Extend, WB4 / GB9)
    0x0301,  # combining acute
    0x0308,  # combining diaeresis
    0x00AD,  # soft hyphen (Format)
    0x2060,  # word joiner (Format)
    0x200D,  # ZWJ (WB3c / GB11)
    0x3000,  # ideographic space (WSegSpace)
    0x0020,  # extra ASCII spaces, weighting whitespace as in the original segmentation palette
    0x0020,
    0x1F600,  # 😀 Extended_Pictographic (astral)
    0x1F6D1,  # 🛑 Extended_Pictographic (astral)
    0x24C2,  # Ⓜ  Enclosed M (pictographic)
    0x00A9,  # ©  copyright
    0x2764,  # ❤  heart
    0x1F1E6,  # Regional_Indicator A (GB12/13 / WB15/16)
    0x1F1FA,  # Regional_Indicator U
    0x1F3FB,  # emoji skin-tone modifier
    0x2028,  # Line Separator (mandatory break)
    0x2029,  # Paragraph Separator (mandatory break)
]

# Shared corner-case palette for grapheme / sentence / line fuzzing (ASCII control + the codepoints above).
# Words are NOT validated against ICU: its word BreakIterator is tailored for editor word-selection and
# diverges ~17-20% from raw UAX-29 (separators / spaces / word-joiner), so the word family keeps its
# bit-exact gate on the official WordBreakTest.txt + uniseg instead. Grapheme vs ICU is bit-exact (0/4000).
SEGMENTATION_PALETTE: List[str] = list("abZ59 \t\n\r.,;:!?'\"_-()") + [chr(cp) for cp in _PALETTE_CODEPOINTS]

# SMP / astral fixtures the pure-BMP random corpora miss (Regional-Indicator pairs, ZWJ sequences, lone
# astral codepoints). Stored as raw UTF-8 bytes — reused by every family's safety + differential sweeps.
ASTRAL_FIXTURES: List[bytes] = [
    b"\xf0\x9f\x87\xba\xf0\x9f\x87\xb8",  # RI(U) RI(S) flag pair
    b"\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f\x87\xab\xf0\x9f\x87\xb7",  # two flags
    b"\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa7",  # family ZWJ sequence
    b"\xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd",  # emoji + skin-tone modifier
    b"\xf0\x9f\x98\x80\xf0\x9f\x98\x81",  # two astral emoji
    b"a\xf0\x9f\x98\x80b",  # ASCII around an astral codepoint
    b"\xf0\x90\x80\x80\xf0\x90\x80\x81",  # U+10000 U+10001 plain astral pair
    b"\xf0\x9f\x87\xbaa\xf0\x9f\x87\xb8",  # RI ASCII RI - RI parity must reset
]

# One malformed-UTF-8 instance per class: overlong encodings, surrogates, lone continuations, invalid leads,
# truncated tails, out-of-range leads, and noncharacters.
MALFORMED_CLASSES: List[bytes] = [
    b"\xc0\x80",  # overlong 2-byte encoding of NUL
    b"\xc1\xbf",  # overlong 2-byte encoding of U+007F
    b"\xe0\x80\x80",  # overlong 3-byte encoding of NUL
    b"\xe0\x9f\xbf",  # overlong 3-byte encoding of U+07FF
    b"\xf0\x80\x80\x80",  # overlong 4-byte encoding of NUL
    b"\xf0\x8f\xbf\xbf",  # overlong 4-byte encoding of U+FFFF
    b"\xed\xa0\x80",  # surrogate U+D800 in 3-byte form
    b"\xed\xbf\xbf",  # surrogate U+DFFF in 3-byte form
    b"\x80",  # lone continuation (low)
    b"\xbf",  # lone continuation (high)
    b"\xfe",  # invalid lead 0xFE
    b"\xff",  # invalid lead 0xFF
    b"\xc2",  # truncated 2-byte tail
    b"\xe2\x80",  # truncated 3-byte tail
    b"\xf0\x9f\x98",  # truncated 4-byte tail
    b"\xf5\x80\x80\x80",  # out-of-range lead >U+10FFFF (0xF5)
    b"\xef\xb7\x90",  # noncharacter U+FDD0
    b"\xef\xb7\xaf",  # noncharacter U+FDEF
    b"\xef\xbf\xbe",  # plane-ender noncharacter U+FFFE
    b"\xef\xbf\xbf",  # plane-ender noncharacter U+FFFF
]

# Named adversarial shapes fed first by the safety sweep.
_NAMED_ADVERSARIAL: List[bytes] = [
    b"\x80",  # lone continuation byte
    b"\xc0\x80",  # overlong encoding of NUL
    b"\xed\xa0\x80",  # surrogate-encoded codepoint (U+D800)
    b"hello\xf0\x9f\x98",  # truncated 4-byte sequence at the very end
]

# UTF-8 lead bytes that drive the byte-pair sweep (one per byte-length class + the invalid/edge leads), each
# paired against all 256 trailing bytes so lead/continuation interactions are exhaustively probed.
_LEAD_BYTES = [0x00, 0x41, 0x7F, 0x80, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xED, 0xEF, 0xF0, 0xF4, 0xF5, 0xFF]

# Byte-length boundary codepoints + BOM / ZWJ / VS16 sprinkled into the random corpus.
_BOUNDARY_CODEPOINTS = [0x007F, 0x0080, 0x07FF, 0x0800, 0xFFFF, 0x10000, 0x10FFFF, 0xFEFF, 0x200D, 0xFE0F]

# Valid UTF-8 snippets forming the bulk of the random corpus (one codepoint per byte length, contractions,
# line/soft-wrap context, separators). Stored as bytes so the file holds no raw multi-byte literals.
_SNIPPETS: List[bytes] = [
    b"a",
    b"Hello, world! ",
    b"don't ",
    b"3,14 ",
    b"\xe4\xbd\xa0\xe5\xa5\xbd",  # 你好
    b"\r\n",
    b"one. two. ",
    b"A! B? ",
    b"e\xcc\x81",  # e + combining acute
    b"\xea\xb0\x80",  # 가
    b"\xe0\xa4\x95\xe0\xa5\x8d\xe0\xa4\xb7",  # Devanagari k-virama-ssa
    b"line\nbreak ",
    b"soft wrap here ",
    b"\xe2\x80\xa8",  # Line Separator
    b"\xe2\x80\xa9",  # Paragraph Separator
]

# Max input bytes used by the safety sweep's random garbage, mirroring `utf8_unit_capacity_k`.
_UNIT_CAPACITY = 70

#  endregion Palettes & fixtures

#  region Boundary helpers


def byte_boundaries(segments: Iterable) -> List[int]:
    """Cumulative byte-offset boundaries of an iterable of string segments (always starting at 0)."""
    out = [0]
    for segment in segments:
        out.append(out[-1] + len(str(segment).encode("utf-8")))
    return out


def char_boundaries_to_bytes(text: str, char_boundaries: Iterable[int]) -> List[int]:
    """Convert TR29-style character-index boundaries (incl. 0 and len) to byte offsets."""
    char_boundaries = set(char_boundaries)
    out = []
    byte_position = 0
    for index, char in enumerate(text):
        if index in char_boundaries:
            out.append(byte_position)
        byte_position += len(char.encode("utf-8"))
    if len(text) in char_boundaries:
        out.append(byte_position)
    return out


def segment_bytes(segments: Iterable) -> bytes:
    """Concatenate the raw bytes of an iterable of StringZilla segments."""
    return b"".join(bytes(segment) for segment in segments)


def assert_segments_tile(segments: Iterable, original: Union[str, bytes]) -> None:
    """Metamorphic invariant: a boundary segmenter's output must reconstruct the input exactly.

    Applies to words / graphemes / sentences / linewraps (which tile `[0, length)` contiguously), not to the
    delimiter splitters `utf8_tokens` / `utf8_lines`, which drop their separators.
    """
    original_bytes = original.encode("utf-8") if isinstance(original, str) else bytes(original)
    rebuilt = segment_bytes(segments)
    assert rebuilt == original_bytes, "segments do not tile the input: in={} out={}".format(
        original_bytes.hex(), rebuilt.hex()
    )


#  endregion Boundary helpers

#  region Synthetic corpora


def append_malformed_class(rng) -> bytes:
    """Return one randomly chosen malformed-UTF-8 class drawn from `rng`."""
    return rng.choice(MALFORMED_CLASSES)


def apply_mutation_passes(data: bytes, rng) -> bytes:
    """Structurally corrupt `data`: NUL injection (10%), byte-swap (10%), truncate-last, stray continuation.

    Mirrors the C++ `apply_mutation_passes_`; breaks codepoint alignment so the malformed-flavor invariants
    (no crash, in-bounds, still-tiling) are exercised on otherwise-valid text.
    """
    buffer = bytearray(data)
    if not buffer:
        return bytes(buffer)
    to_corrupt = len(buffer) // 10
    for _ in range(to_corrupt):
        buffer[rng.randrange(len(buffer))] = 0
    for _ in range(to_corrupt):
        first, second = rng.randrange(len(buffer)), rng.randrange(len(buffer))
        buffer[first], buffer[second] = buffer[second], buffer[first]
    # Truncate the last 1-3 bytes so a trailing multi-byte sequence loses its tail.
    truncate_by = min(rng.randint(1, 3), len(buffer))
    del buffer[len(buffer) - truncate_by :]
    # Insert a stray continuation byte, breaking codepoint alignment.
    if buffer:
        buffer.insert(rng.randint(0, len(buffer)), rng.randint(0x80, 0xBF))
    return bytes(buffer)


def random_segmentation_corpus(
    min_length: int, flavor: str, motifs: Optional[Sequence[bytes]], rng
) -> bytes:
    """Build a weighted random UTF-8 corpus until at least `min_length` bytes.

    `flavor` is ``"valid"`` (well-formed only) or ``"malformed"`` (malformed classes mixed in). `motifs` are
    the family's corner-case byte inputs, sprinkled in. Weights mirror the C++ `utf8_random_segmentation_corpus_`.
    """
    motifs = list(motifs or [])
    out = bytearray()
    while len(out) < min_length:
        roll = rng.randint(0, 99)
        if flavor == "malformed" and roll < 10:
            out += append_malformed_class(rng)
        elif roll < 30:
            out += chr(rng.choice(_BOUNDARY_CODEPOINTS)).encode("utf-8")
        elif roll < 45:
            out += rng.choice(ASTRAL_FIXTURES)
        elif roll < 65 and motifs:
            out += rng.choice(motifs)
        else:
            out += rng.choice(_SNIPPETS)
    return bytes(out)


def corpus_of_byte_length(target_length: int, rng, palette: Optional[Sequence[str]] = None) -> bytes:
    """Build valid UTF-8 of exactly `target_length` bytes from `palette`, padding with ASCII to land the seam.

    Used by the window-seam sweep so the input ends precisely on a 63/64/65/… byte size relative to the
    kernel's 64-byte SIMD window.
    """
    palette = palette or SEGMENTATION_PALETTE
    out = bytearray()
    while len(out) < target_length:
        encoded = rng.choice(palette).encode("utf-8")
        if len(out) + len(encoded) > target_length:
            break
        out += encoded
    out += b"x" * (target_length - len(out))
    return bytes(out)


def window_seam_lengths() -> List[int]:
    """Byte lengths that straddle the kernel's 64-byte SIMD windows (generalizes the words-only seam regressions)."""
    return [1, 3, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256]


def adversarial_utf8_inputs(rng, random_input_count: int = 4000) -> Iterable[bytes]:
    """Yield the adversarial-byte battery: named shapes, astral fixtures, all 256 single bytes, every
    lead-class byte-pair, and random garbage. The Python analog of `for_each_adversarial_utf8_input_`.
    """
    yield from _NAMED_ADVERSARIAL
    yield from ASTRAL_FIXTURES
    for byte in range(256):
        yield bytes([byte])
    for lead in _LEAD_BYTES:
        for trailing in range(256):
            yield bytes([lead, trailing])
    for _ in range(random_input_count):
        length = rng.randint(1, _UNIT_CAPACITY)
        yield bytes(rng.randrange(256) for _ in range(length))


#  endregion Synthetic corpora

#  region ICU oracles


def icu_segmenter(kind: str) -> Callable[[str], List[str]]:
    """Return an ICU `BreakIterator`-backed segmenter `text -> list[str]` for the given boundary `kind`.

    `kind` is one of ``"word"`` / ``"grapheme"`` / ``"sentence"`` / ``"line"``. Skips the test when PyICU is
    absent. Generalizes the sentence-only idiom that previously lived inline in the monolith.
    """
    icu = pytest.importorskip("icu", reason="PyICU not installed")
    factories = {
        "word": icu.BreakIterator.createWordInstance,
        "grapheme": icu.BreakIterator.createCharacterInstance,
        "sentence": icu.BreakIterator.createSentenceInstance,
        "line": icu.BreakIterator.createLineInstance,
    }
    iterator = factories[kind](icu.Locale.getRoot())

    def segments(text: str) -> List[str]:
        unicode_text = icu.UnicodeString(text)
        iterator.setText(unicode_text)
        parts = []
        start = iterator.first()
        end = iterator.nextBoundary()
        while end != icu.BreakIterator.DONE:
            parts.append(str(unicode_text[start:end]))
            start = end
            end = iterator.nextBoundary()
        return parts

    return segments


def icu_normalizer(form: str) -> Callable[[str], str]:
    """Return an ICU `Normalizer2`-backed normalizer `text -> str` for NFC/NFD/NFKC/NFKD. Skips without PyICU."""
    icu = pytest.importorskip("icu", reason="PyICU not installed")
    getters = {
        "NFC": icu.Normalizer2.getNFCInstance,
        "NFD": icu.Normalizer2.getNFDInstance,
        "NFKC": icu.Normalizer2.getNFKCInstance,
        "NFKD": icu.Normalizer2.getNFKDInstance,
    }
    normalizer = getters[form]()
    return lambda text: normalizer.normalize(text)


#  endregion ICU oracles

#  region Rule-derived generators

# Generators that turn the UCD break-property / combining-class / decomposition tables (extracted in
# test_helpers.py) into hard synthetic corner cases, rather than relying on a hand-picked palette.


def class_adjacency_strings(
    representatives: dict, arity: int = 2, representatives_per_class: int = 1, max_cases: Optional[int] = None
) -> List[str]:
    """Build strings over the cartesian product of break classes — one representative codepoint per cell.

    With one representative per class this enumerates every class-adjacency pair (arity 2) or triple (arity 3)
    the segmentation rules can encounter, surfacing rule interactions the random palette rarely hits. `max_cases`
    bounds the explosion (e.g. Line_Break's ~40 classes make full triples huge).
    """
    class_names = sorted(representatives)
    chosen = {name: representatives[name][:representatives_per_class] for name in class_names}
    results: List[str] = []
    for class_combination in itertools.product(class_names, repeat=arity):
        for codepoint_combination in itertools.product(*[chosen[name] for name in class_combination]):
            results.append("".join(chr(codepoint) for codepoint in codepoint_combination))
            if max_cases is not None and len(results) >= max_cases:
                return results
    return results


def combining_scrambles(combining_classes: dict, rng, count: int, max_marks: int = 4) -> List[str]:
    """Build `count` strings of a base letter followed by several non-zero-CCC combining marks in random order.

    The non-canonical mark order stresses canonical reordering: NFD must reorder by combining class.
    """
    marks = [codepoint for codepoint, combining_class in combining_classes.items() if codepoint <= 0xFFFF]
    base_letters = "aeoun"
    results: List[str] = []
    for _ in range(count):
        base_letter = rng.choice(base_letters)
        mark_count = rng.randint(2, max_marks)
        scrambled_marks = [rng.choice(marks) for _ in range(mark_count)]
        results.append(base_letter + "".join(chr(codepoint) for codepoint in scrambled_marks))
    return results


def decomposition_pairs(decomposition_mappings: dict, canonical_only: bool = True, limit: Optional[int] = None) -> list:
    """Build (precomposed, decomposed) canonical-equivalent string pairs from the decomposition mappings."""
    pairs = []
    for codepoint in sorted(decomposition_mappings):
        kind, targets = decomposition_mappings[codepoint]
        if canonical_only and kind != "canonical":
            continue
        pairs.append((chr(codepoint), "".join(chr(target) for target in targets)))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


#  endregion Rule-derived generators
