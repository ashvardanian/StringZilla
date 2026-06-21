#!/usr/bin/env python3
"""Generate authoritative UTF-8 segmentation golden vectors for `scripts/test_utf8.cpp`.

The C unit tests keep small, hand-auditable known-answer vectors so they stay self-contained (no vendored
`*BreakTest.txt` at runtime). Rather than hand-guessing the expected boundaries for tricky UAX motifs (emoji-ZWJ,
Regional-Indicator parity, Indic conjuncts, Hangul jamo, line-break adjacency), this script computes them once,
offline, from an independent reference implementation (`uniseg`), and prints C initializer lists that a maintainer
pastes into `test_utf8.cpp`'s unified `case_t` arrays.

The emitted strings follow the repository convention: printable ASCII stays literal, every other byte is a `\\xHH`
escape, and a string is split into adjacent literals whenever an escape would greedily merge with a following hex
digit (e.g. `"a\\xE2\\x80\\xA8" "b"`).

Reference: `uniseg` (https://pypi.org/project/uniseg/) — grapheme clusters, sentences, and line-break units.

Usage:
    python3 scripts/generate_utf8_golden.py            # all three families
    python3 scripts/generate_utf8_golden.py grapheme   # one family
"""

import sys

import uniseg.graphemecluster as graphemecluster
import uniseg.linebreak as linebreak
import uniseg.sentencebreak as sentencebreak

# Codepoints whose LineBreak class makes the break that follows them *mandatory* (UAX-14 LB4/LB5).
_MANDATORY_LINE_BREAK_CODEPOINTS = {
    0x000A,  # LF  LINE FEED
    0x000B,  # VT  LINE TABULATION
    0x000C,  # FF  FORM FEED
    0x000D,  # CR  CARRIAGE RETURN
    0x0085,  # NEL NEXT LINE
    0x2028,  # LS  LINE SEPARATOR
    0x2029,  # PS  PARAGRAPH SEPARATOR
}


def c_escape(text: str) -> str:
    """Render a Python string as one or more concatenated C string literals, per the repo's escaping style."""
    raw = text.encode("utf-8")
    pieces: list[str] = []
    previous_was_hex_escape = False
    for byte in raw:
        character = chr(byte)
        is_plain_ascii = 0x20 <= byte < 0x7F and character not in ('"', "\\")
        if is_plain_ascii:
            # Close and reopen the literal so a hex digit cannot extend the preceding `\xHH` escape.
            if previous_was_hex_escape and character in "0123456789abcdefABCDEF":
                pieces.append('" "')
            pieces.append(character)
            previous_was_hex_escape = False
        else:
            pieces.append(f"\\x{byte:02X}")
            previous_was_hex_escape = True
    return '"' + "".join(pieces) + '"'


def grapheme_segments(text: str) -> list[tuple[str, bool]]:
    return [(cluster, False) for cluster in graphemecluster.grapheme_clusters(text)]


def sentence_segments(text: str) -> list[tuple[str, bool]]:
    return [(sentence, False) for sentence in sentencebreak.sentences(text)]


def line_segments(text: str) -> list[tuple[str, bool]]:
    """Line-break units plus whether each unit ends at a UAX-14 mandatory break."""
    segments: list[tuple[str, bool]] = []
    for unit in linebreak.line_break_units(text):
        mandatory = len(unit) > 0 and ord(unit[-1]) in _MANDATORY_LINE_BREAK_CODEPOINTS
        segments.append((unit, mandatory))
    return segments


# Curated, deterministic corpora. Keep these small and hand-auditable; they are the "examples known ahead of time"
# layer that complements the random differential. Comments name the UAX rule each motif exercises.
_CORPORA: dict[str, list[str]] = {
    "grapheme": [
        "abc",
        "a\r\nb",  # GB3: CR x LF stays one cluster
        "é",  # GB9: e + combining acute -> one cluster
        "\U0001F1FA\U0001F1F8",  # GB12/13: RI pair -> one flag
        "\U0001F1FA\U0001F1F8\U0001F1EB\U0001F1F7",  # two flags (even RI run)
        "\U0001F1FAa\U0001F1F8",  # RI ASCII RI: parity reset
        "\U0001F1FA\U0001F1F8\U0001F1EB",  # odd RI run (3)
        "\U0001F469‍\U0001F469‍\U0001F467",  # GB11: ZWJ family -> one cluster
        "\U0001F44D\U0001F3FD",  # emoji + skin-tone modifier
        "☺️",  # emoji + VS16
        "가",  # Hangul LV syllable
        "각",  # GB6/7/8: Hangul L+V+T jamo -> one cluster
        "क्ष",  # Devanagari conjunct (GB9c InCB)
        "ত্ত",  # Bengali conjunct
        "க்க",  # Tamil conjunct
    ],
    "sentence": [
        "Hello.",
        "Hello. World.",  # terminator + space ends a sentence
        "Mr. Smith went.",  # UAX-29 has no abbreviation list: '. ' before uppercase breaks
        "etc. and so on",  # SB8: '. ' before a lowercase word does not break
        "One.\nTwo.",  # ParaSep after terminator
        "A! B? C.",  # multiple terminators
        'He said "Go." She left.',  # SB6/7: Close after terminator
        "Wait... really?",  # repeated ATerm
        "U.S.A. is large.",  # interior periods
    ],
    "line": [
        "hello world",  # soft wrap after the space
        "a\nb",  # LF mandatory
        "a\r\nb",  # CRLF mandatory, one unit
        "a b",  # U+2028 LINE SEPARATOR mandatory
        "a b",  # U+2029 PARAGRAPH SEPARATOR mandatory
        "ab",  # U+0085 NEL mandatory
        "a b",  # NBSP (GL): no break
        "3,000 and 4",  # numeric: no break between digits
        "foo-bar baz",  # hyphen break opportunity
    ],
}

_SEGMENTERS = {
    "grapheme": grapheme_segments,
    "sentence": sentence_segments,
    "line": line_segments,
}


def emit_family(family: str) -> None:
    segmenter = _SEGMENTERS[family]
    print(f"// {family} golden vectors (generated by scripts/generate_utf8_golden.py via uniseg)")
    print(f"std::vector<utf8_segment_unit_case_t> const {family}_golden_cases = {{")
    for text in _CORPORA[family]:
        segments = segmenter(text)
        rendered = ", ".join(f"{{{c_escape(bytes_)}, {'true' if mandatory else 'false'}}}"
                             for bytes_, mandatory in segments)
        print(f"    {{{c_escape(text)}, {{{rendered}}}}},")
    print("};")
    print()


def main() -> None:
    families = sys.argv[1:] or ["grapheme", "sentence", "line"]
    for family in families:
        if family not in _SEGMENTERS:
            raise SystemExit(f"unknown family {family!r}; choose from grapheme/sentence/line")
        emit_family(family)


if __name__ == "__main__":
    main()
