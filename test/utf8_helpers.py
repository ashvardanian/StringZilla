"""
Shared UTF-8 segmentation test driver for the per-family test modules.

The Python analog of the C++ `scripts/test_utf8.hpp`: a single place that owns the boundary-relevant
palettes, the SMP/astral fixtures, the malformed-UTF-8 corpus generators, the window-seam length sweep,
the adversarial-byte battery, and the metamorphic tiling invariant, so every family TU
(test_utf8_wordbreaks.py, test_utf8_graphemes.py, …) shares one driver instead of copying corpora.

Differential oracles live here too: `icu_segmenter` / `icu_normalizer` wrap PyICU (skipped when absent),
generalizing the sentence-only ICU idiom that previously lived inline in the monolith.

Palette members are built from explicit codepoints (the source stays pure ASCII) so an editor or a
delegated agent cannot silently NFC-normalize a raw multi-byte literal.
"""

import itertools
from typing import Callable, Iterable, List, Optional, Sequence, Union

import pytest

from test.sz_helpers import scale_iterations

# region Palettes & fixtures

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
# Words are not validated against ICU: its word BreakIterator is tailored for editor word-selection and
# diverges ~17-20% from raw UAX-29 (separators / spaces / word-joiner), so the word family keeps its
# bit-exact gate on the official WordBreakTest.txt + uniseg instead. Grapheme vs ICU is bit-exact (0/4000).
SEGMENTATION_PALETTE: List[str] = list("abZ59 \t\n\r.,;:!?'\"_-()") + [chr(cp) for cp in _PALETTE_CODEPOINTS]

# SMP / astral fixtures the pure-BMP random corpora miss (Regional-Indicator pairs, ZWJ sequences, lone
# astral codepoints). Stored as raw UTF-8 bytes, reused by every family's safety and differential sweeps.
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

# endregion Palettes & fixtures

# region Boundary helpers


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

    Applies to words / graphemes / sentences / linebreaks (which tile `[0, length)` contiguously), not to the
    delimiter splitters `utf8_tokens` / `utf8_lines`, which drop their separators.
    """
    original_bytes = original.encode("utf-8") if isinstance(original, str) else bytes(original)
    rebuilt = segment_bytes(segments)
    assert rebuilt == original_bytes, "segments do not tile the input: in={} out={}".format(
        original_bytes.hex(), rebuilt.hex()
    )


# endregion Boundary helpers

# region Synthetic corpora


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


def random_segmentation_corpus(min_length: int, flavor: str, motifs: Optional[Sequence[bytes]], rng) -> bytes:
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


def adversarial_utf8_inputs(rng, random_input_count: Optional[int] = None) -> Iterable[bytes]:
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
    count = scale_iterations(4000) if random_input_count is None else random_input_count
    for _ in range(count):
        length = rng.randint(1, _UNIT_CAPACITY)
        yield bytes(rng.randrange(256) for _ in range(length))


# endregion Synthetic corpora

# region ICU oracles


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


def icu_unicode_at_least(version: str) -> bool:
    """True when the local PyICU's bundled Unicode version is >= ``version`` (compared on major.minor, e.g. "17").

    A stale PyICU (older Unicode than our 17.0 tables) disagrees with the kernel on codepoints whose break
    properties changed between releases, so a *bit-exact* ICU differential would log spurious version-induced
    failures. Callers gate the strict assertion behind this and fall back to an agreement gate when ICU is behind.
    """
    icu = pytest.importorskip("icu", reason="PyICU not installed")

    def major_minor(value: str) -> tuple:
        return tuple(int(piece) for piece in str(value).split(".")[:2] if piece.isdigit())

    return major_minor(icu.UNICODE_VERSION) >= major_minor(version)


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


# endregion ICU oracles

# region Rule-derived generators

# Generators that turn the UCD break-property / combining-class / decomposition tables (extracted in
# test_helpers.py) into hard synthetic corner cases, rather than relying on a hand-picked palette.


def class_adjacency_strings(
    representatives: dict, arity: int = 2, representatives_per_class: int = 1, max_cases: Optional[int] = None
) -> List[str]:
    """Build strings over the cartesian product of break classes, one representative codepoint per cell.

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


# endregion Rule-derived generators


# region Prose fixtures

# Realistic multi-script paragraphs (ASCII-source \uXXXX escapes; rendered prose in each comment).
# Segment counts are asserted live against the ICU / uniseg oracles in the per-family modules.

# Hotel review (German + Japanese): NFD cafe, NBSP-glued units, a sentence-ending abbreviation, a CJK run.
#   Last spring we strolled down Münchner Straße; the café cortado cost 3,50 € and was unreal. Dr. Vogel, our
#   guide, swore it's the city's finest. Worth the detour?! Absolutely — and 東京タワー the next week, all 333 m of
#   it, was breathtaking at dusk…
PROSE_HOTEL_REVIEW = (
    "Last spring we strolled down M\u00fcnchner Stra\u00dfe; the cafe\u0301 cortado cost 3,50"
    "\u00a0\u20ac and was unreal. Dr. Vogel, our guide, swore it's the city's finest. Worth the d"
    "etour?! Absolutely \u2014 and \u6771\u4eac\u30bf\u30ef\u30fc the next week, all 333\u00a0m o"
    "f it, was breathtaking at dusk\u2026"
)

# Pride caption: a ZWJ family and VS16 rainbow flag, a skin-tone modifier, a keycap, an odd regional-indicator run.
#   Best Pride yet 🏳️‍🌈 — the whole crew showed up. Even my parents 👨‍👩‍👧‍👦 and grandma 👍🏽 came through! We met
#   at booth 5️⃣, then waved every flag we packed 🇺🇸🇯🇵🇫. Texting ☎︎ over calling ✈️ all day; 10/10, would march
#   again.
PROSE_PRIDE_CAPTION = (
    "Best Pride yet \U0001f3f3\ufe0f\u200d\U0001f308 \u2014 the whole crew showed up. Even my par"
    "ents \U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466 and grandma \U0001f44d"
    "\U0001f3fd came through! We met at booth 5\ufe0f\u20e3, then waved every flag we packed "
    "\U0001f1fa\U0001f1f8\U0001f1ef\U0001f1f5\U0001f1eb. Texting \u260e\ufe0e over calling \u2708"
    "\ufe0f all day; 10/10, would march again."
)

# Concert post (Korean + Japanese): conjoining L+V+T jamo, a Katakana run, an ideographic stop, a 'p.m.' no-break.
#   오늘 콘서트, 진짜 미쳤다!! 한국 팬들이 다 모였고, the staff bowed and said 안녕히 가세요. Setlist was pure ハードコア; 今日は最高だった。 We
#   screamed 사랑해 till 11 p.m. sharp.
PROSE_CONCERT_POST = (
    "\uc624\ub298 \ucf58\uc11c\ud2b8, \uc9c4\uc9dc \ubbf8\ucce4\ub2e4!! \u1112\u1161\u11ab\uad6d "
    "\ud32c\ub4e4\uc774 \ub2e4 \ubaa8\uc600\uace0, the staff bowed and said \uc548\ub155\ud788 "
    "\uac00\uc138\uc694. Setlist was pure \u30cf\u30fc\u30c9\u30b3\u30a2; \u4eca\u65e5\u306f"
    "\u6700\u9ad8\u3060\u3063\u305f\u3002 We screamed \uc0ac\ub791\ud574 till 11 p.m. sharp."
)

# Devanagari note: a virama conjunct, ZWJ/ZWNJ half-forms, a spacing vowel sign, and an NFKC vulgar fraction.
#   Quick Devanagari tip: क्ष is one cluster (क + ् + ष), not three. Force the half-form with ZWJ — क्‍ष — or
#   split it with ZWNJ — क्‌ष. The same logic hits क्षत्रिय and spacing vowel signs like की. Renderers disagree,
#   so test (½ the bugs are font bugs) before you ship!
PROSE_DEVANAGARI_TIP = (
    "Quick Devanagari tip: \u0915\u094d\u0937 is one cluster (\u0915 + \u094d + \u0937), not thre"
    "e. Force the half-form with ZWJ \u2014 \u0915\u094d\u200d\u0937 \u2014 or split it with ZWNJ"
    " \u2014 \u0915\u094d\u200c\u0937. The same logic hits \u0915\u094d\u0937\u0924\u094d\u0930"
    "\u093f\u092f and spacing vowel signs like \u0915\u0940. Renderers disagree, so test (\u00bd "
    "the bugs are font bugs) before you ship!"
)

# Science abstract: NFKC ligatures/superscripts/Roman/full-width, Kelvin and Angstrom singletons, NBSP, WJ + ZWSP.
#   The ﬁlm grew at 300 K on a 5 Å buffer (≈ 2² monolayers). Section Ⅻ covers the Ａ-phase; see Fig. 2 for the
#   Σ-band dispersion. Resistivity scaled as T², vanishing at the 4.2 K transition. Full dataset:
#   doi:10.1000⁠/​xyz (mirror in Box ②).
PROSE_SCIENCE_ABSTRACT = (
    "The \ufb01lm grew at 300\u00a0\u212a on a 5\u00a0\u212b buffer (\u2248 2\u00b2 monolayers). "
    "Section \u216b covers the \uff21-phase; see Fig. 2 for the \u03a3-band dispersion. Resistivi"
    "ty scaled as T\u00b2, vanishing at the 4.2\u00a0\u212a transition. Full dataset: doi:10.1000"
    "\u2060/\u200bxyz (mirror in Box \u2461)."
)

# News lede: 'U.S.A.' before a lowercase word (no break), curly quotes, thousands, currency, a date range.
#   The U.S.A. wasn't ready, analysts said. “We lost 1,000 jobs,” the mayor warned. “Recovery starts now.”
#   Filings spiked 2024/06–2024/09, topping $1,000 per claim. Will it hold?! No one knows for sure.
PROSE_NEWS_LEDE = (
    "The U.S.A. wasn't ready, analysts said. \u201cWe lost 1,000 jobs,\u201d the mayor warned. "
    "\u201cRecovery starts now.\u201d Filings spiked 2024/06\u20132024/09, topping $1,000 per cla"
    "im. Will it hold?! No one knows for sure."
)

# Language lesson: a Greek final sigma, Cyrillic case pairs, a Croatian titlecase digraph, and a fold-only match.
#   Greek lesson: ΟΔΟΣ becomes οδός when lowercased, ending in a final ς. Russian's easy too — МОСКВА ↔ москва,
#   no drama. Croatian has the digraph Ǆ: titlecase ǅ, lowercase ǆ. Quiz — does “straße” match STRASSE? Yes,
#   once you fold.
PROSE_LANGUAGE_LESSON = (
    "Greek lesson: \u039f\u0394\u039f\u03a3 becomes \u03bf\u03b4\u03cc\u03c2 when lowercased, end"
    "ing in a final \u03c2. Russian's easy too \u2014 \u041c\u041e\u0421\u041a\u0412\u0410 \u2194"
    " \u043c\u043e\u0441\u043a\u0432\u0430, no drama. Croatian has the digraph \u01c4: titlecase "
    "\u01c5, lowercase \u01c6. Quiz \u2014 does \u201cstra\u00dfe\u201d match STRASSE? Yes, once "
    "you fold."
)

# RTL scripts: Hebrew gershayim, Arabic, a number-sign Prepend, an NFC niqqud reorder, a Malayalam dot-reph.
#   Hebrew acronyms take gershayim: צה״ל and ארה״ב aren't typos. Arabic flows right-to-left too — مرحبا بالعالم
#   — and finance text can carry the number sign ؀٤. Niqqud stacks marks: שָׁלוֹם must reorder under NFC.
#   Malayalam even has a true prepend, the dot-reph ൎക.
PROSE_RTL_SCRIPTS = (
    "Hebrew acronyms take gershayim: \u05e6\u05d4\u05f4\u05dc and \u05d0\u05e8\u05d4\u05f4\u05d1 "
    "aren't typos. Arabic flows right-to-left too \u2014 \u0645\u0631\u062d\u0628\u0627 \u0628"
    "\u0627\u0644\u0639\u0627\u0644\u0645 \u2014 and finance text can carry the number sign "
    "\u0600\u0664. Niqqud stacks marks: \u05e9\u05c1\u05b8\u05dc\u05d5\u05b9\u05dd must reorder u"
    "nder NFC. Malayalam even has a true prepend, the dot-reph \u0d4e\u0d15."
)

# A U+2019 contraction tiles as a single word, like the ASCII apostrophe.
#   it’s worth it
PROSE_MICRO_APOSTROPHE = "it\u2019s worth it"

# Two Prepend characters (Arabic number sign, Malayalam dot-reph): clusters fewer than codepoints.
#   ؀٤ ൎക
PROSE_MICRO_PREPEND = "\u0600\u0664 \u0d4e\u0d15"

# A CR-LF pair and a U+2028 line separator: both Sep (force sentence and line breaks); CR-LF is one grapheme.
#   A.\r\nB.\u2028C.
PROSE_MICRO_HARDBREAKS = "A.\u000d\u000aB.\u2028C."

# endregion
