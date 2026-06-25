/**
 *  @brief  UAX-29 word-boundary (Word_Break) tests: known-answer goldens, malformed-input safety, and the
 *          serial-vs-ISA differential over hardened corpora.
 *  @file   scripts/test_utf8_words.cpp
 *  @author Ash Vardanian
 */
#undef NDEBUG // ! Enable all assertions for testing

#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#define SZ_USE_MISALIGNED_LOADS 0
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // ! Enforce aggressive logging in this translation unit

#include <cassert> // `assert`
#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::printf`

#include <random> // `std::mt19937`, `std::uniform_int_distribution`
#include <string> // `std::string`
#include <vector> // `std::vector`

#include "utf8.hpp" // shared segmentation harness (pulls in StringZilla + `test_stringzilla.hpp`)

#pragma region Unit

/** @brief Hand-checked UAX-29 word-break golden vectors: each source text and its expected word segments. */
static utf8_unit_case_t const utf8_words_unit_cases[] = {
    {""_sv, {}},
    {"a"_sv, {"a"_sv}},
    {"don't"_sv, {"don't"_sv}},                                              // WB6/7: apostrophe bridges letter runs
    {"3,14"_sv, {"3,14"_sv}},                                                // WB11/12: comma between digits
    {"3,"_sv, {"3"_sv, ","_sv}},                                             // digit then bare comma breaks
    {"can't_stop"_sv, {"can't_stop"_sv}},                                    // WB13a/b: ExtendNumLet underscore
    {"a\r\nb"_sv, {"a"_sv, "\r\n"_sv, "b"_sv}},                              // WB3: CR x LF stay together
    {"\xE4\xBD\xA0\xE5\xA5\xBD"_sv, {"\xE4\xBD\xA0"_sv, "\xE5\xA5\xBD"_sv}}, // CJK: each ideograph is its own word
    {"Hello, world!"_sv, {"Hello"_sv, ","_sv, " "_sv, "world"_sv, "!"_sv}},  // letter/punct/space boundaries
};
static constexpr std::size_t utf8_words_unit_cases_count = sizeof(utf8_words_unit_cases) /
                                                           sizeof(utf8_words_unit_cases[0]);

/** @brief Known-answer property table for `sz_rune_is_word_char` (UAX-29 word-character classification). */
static void test_utf8_words_classification_() {
    // ASCII letters, digits, underscore, and the mid-word apostrophe are word characters.
    assert(sz_rune_is_word_char('A') == sz_true_k);
    assert(sz_rune_is_word_char('z') == sz_true_k);
    assert(sz_rune_is_word_char('0') == sz_true_k);
    assert(sz_rune_is_word_char('9') == sz_true_k);
    assert(sz_rune_is_word_char('_') == sz_true_k);
    assert(sz_rune_is_word_char('\'') == sz_true_k);

    // ASCII whitespace and punctuation are boundaries.
    assert(sz_rune_is_word_char(' ') == sz_false_k);
    assert(sz_rune_is_word_char('\n') == sz_false_k);
    assert(sz_rune_is_word_char('\t') == sz_false_k);
    assert(sz_rune_is_word_char('!') == sz_false_k);
    assert(sz_rune_is_word_char('-') == sz_false_k);

    // Latin Extended, Greek, Cyrillic, Hebrew, Arabic letters and Hangul syllables are word characters.
    assert(sz_rune_is_word_char(0x00DF) == sz_true_k); // ß
    assert(sz_rune_is_word_char(0x0100) == sz_true_k); // Latin Extended-A start
    assert(sz_rune_is_word_char(0x03B1) == sz_true_k); // Greek alpha
    assert(sz_rune_is_word_char(0x0430) == sz_true_k); // Cyrillic a
    assert(sz_rune_is_word_char(0x05D0) == sz_true_k); // Hebrew alef
    assert(sz_rune_is_word_char(0x0627) == sz_true_k); // Arabic alef
    assert(sz_rune_is_word_char(0xAC00) == sz_true_k); // Hangul first
    assert(sz_rune_is_word_char(0xD7A3) == sz_true_k); // Hangul last

    // CJK ideographs, spaces, dashes, and emoji are boundaries (not word characters under TR29).
    assert(sz_rune_is_word_char(0x4E00) == sz_false_k);  // CJK first
    assert(sz_rune_is_word_char(0x9FFF) == sz_false_k);  // CJK last
    assert(sz_rune_is_word_char(0x3000) == sz_false_k);  // Ideographic space
    assert(sz_rune_is_word_char(0x2014) == sz_false_k);  // Em dash
    assert(sz_rune_is_word_char(0x1F600) == sz_false_k); // emoji

    // Edge cases.
    assert(sz_rune_is_word_char(0x0000) == sz_false_k); // NUL
    assert(sz_rune_is_word_char(0x007F) == sz_false_k); // DEL
    assert(sz_rune_is_word_char(0xFFFF) == sz_false_k); // BMP max
}

/** @brief Known-answer word-break vectors through dispatched, serial, and each ISA backend + the C++ range. */
void test_utf8_words_unit() {
    std::printf("  - testing UTF-8 word-break known-answer vectors...\n");

    test_utf8_words_classification_();

    check_utf8_segment_unit_("word", sz_utf8_words, utf8_words_unit_cases, utf8_words_unit_cases_count); // Dispatched
    check_utf8_segment_unit_("word", sz_utf8_words_serial, utf8_words_unit_cases, utf8_words_unit_cases_count);
#if SZ_USE_ICELAKE
    check_utf8_segment_unit_("word", sz_utf8_words_icelake, utf8_words_unit_cases, utf8_words_unit_cases_count);
#endif

    // C++ range wrapper known-answer: the view must faithfully expose the kernel's segments.
    std::vector<std::string> const hello =
        sz::string_view("Hello, world!").utf8_words().template to<std::vector<std::string>>();
    assert(hello.size() == 5 && hello[0] == "Hello" && hello[3] == "world" && "C++ utf8_words range");
}

#pragma endregion // Unit

#pragma region Equivalence

/** @brief UAX-29 word-break corner motifs (sprinkled into the random corpus): Mid-bridges, MidNum, RI parity, etc. */
static sz::string_view const utf8_words_motifs[] = {
    "don't"_sv,                                // WB6/7: apostrophe bridges two letter runs
    "l'avion"_sv,                              // WB7a/b: leading apostrophe shape
    "can't_stop"_sv,                           // WB13a/b: ExtendNumLet underscore keeps the word whole
    "3,14"_sv,                                 // WB11/12: comma between digits stays one word
    "1,2,3"_sv,                                // chained numeric MidNum grouping
    "3,"_sv,                                   // digit then bare comma: break after the number
    "Hello, world!"_sv,                        // punctuation/space boundaries between letter runs
    "\xE4\xBD\xA0\xE5\xA5\xBD"_sv,             // CJK: each ideograph is its own word
    "\xEC\x95\x88\xEB\x85\x95"_sv,             // Hangul syllables are word chars
    "\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D"_sv,     // Hebrew letter run
    "\xD7\x90\x27\xD7\x91"_sv,                 // Hebrew aleph + apostrophe + bet (WB7a single-quote)
    "\xE3\x82\xAB\xE3\x82\xBF\xE3\x82\xAB"_sv, // Katakana run (WB13 Katakana x Katakana)
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv,     // Regional-Indicator pair (WB15/16 parity)
    "\xF0\x9F\x87\xBA\x61\xF0\x9F\x87\xB8"_sv, // RI ASCII RI: parity reset between flags
    "word\xC2\xB7word"_sv,                     // U+00B7 MIDDLE DOT as MidLetter between letters
};
static constexpr std::size_t utf8_words_motifs_count = sizeof(utf8_words_motifs) / sizeof(utf8_words_motifs[0]);

/**
 *  @brief Multi-window seam regressions (each > 64 bytes): WB15/16 Regional_Indicator parity and WB6/7/11/12
 *         Mid-bridge carry once miscounted across the 64-byte window boundary. Stored as raw bytes so the
 *         differential driver feeds them to serial-vs-ISA directly (no inline agreement asserts).
 */
static sz::string_view const utf8_words_seam_regressions[] = {
    // ri_after_newline (65 bytes)
    "\xE3\x82\xAB\x2D\x0A\xF0\x9F\x87\xA6\x62\xC2\xAD\xF0\x9F\x87\xA6\xCC\x80\x0A\x5F\xF0\x9F\x87\xA6" //
    "\xF0\x9F\x87\xA6\xC2\xAD\x0A\xCC\x88\xF0\x9F\x87\xBA\xF0\x9F\x8F\xBB\xCC\x88\xC2\xAD\xE2\x81\xA0" //
    "\xCC\x88\xF0\x9F\x87\xA6\xF0\x9F\x87\xA6\xE4\xB8\xAD\xE2\x81\xA0\x5F"_sv,
    // ri_word_joiner (82 bytes)
    "\x5F\x27\xF0\x9F\x87\xBA\xE4\xB8\xAD\xCC\x81\xF0\x9F\x87\xA6\xF0\x9F\x87\xA6\x5F\xCC\x88\x5F\xD7" //
    "\x90\x2D\xE2\x93\x82\xCC\x80\x62\xF0\x9F\x87\xA6\xE2\x81\xA0\xF0\x9F\x87\xBA\xCC\x80\xCC\x88\xF0" //
    "\x9F\x8F\xBB\xF0\x9F\x87\xBA\xCC\x81\xF0\x9F\x87\xBA\x0A\xC2\xAD\xF0\x9F\x87\xBA\xCC\x88\xE3\x82" //
    "\xAB\x5F\x27\x2E\xCC\x88\x0A\xE4\xB8\xAD"_sv,
    // wb12_bridge (78 bytes)
    "\x5F\xE2\x81\xA0\x5F\x35\x2C\xF0\x9F\x98\x80\x27\xF0\x9F\x98\x80\x5F\xCC\x88\xCC\x81\xE4\xB8\xAD" //
    "\xE2\x81\xA0\x5F\x62\xCC\x80\x61\xE4\xB8\xAD\xE3\x82\xAB\x35\xE2\x81\xA0\xCC\x88\xCC\x88\xF0\x9F" //
    "\x8F\xBB\xCC\x88\xE2\x81\xA0\xCC\x80\xCC\x81\x27\x35\xCC\x80\xE4\xB8\xAD\xCC\x81\xE2\x80\x8D\xC3" //
    "\xA9\xE3\x80\x80\x35\x27"_sv,
};
static constexpr std::size_t utf8_words_seam_regressions_count = sizeof(utf8_words_seam_regressions) /
                                                                 sizeof(utf8_words_seam_regressions[0]);

/** @brief Katakana run @p link_count codepoints long (WB13 Katakana x Katakana), into @p out (cleared first). */
static void utf8_words_dense_katakana_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) append_codepoint_(out, 0x30AB); // カ
}

/** @brief Numeric run with MidNum and Extend marks, @p link_count groups (WB11/12 + Extend), into @p out. */
static void utf8_words_dense_numeric_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        out.append("12");
        append_codepoint_(out, 0x0301); // Extend combining mark inside a number
        out.append(",34 ");             // MidNum comma
    }
}

/** @brief MidLetter (`'` and U+00B7) between letters, dense across @p link_count groups (WB6/7), into @p out. */
static void utf8_words_dense_midletter_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(out, 0x0061);                         // 'a'
        append_codepoint_(out, (index & 1u) ? 0x00B7 : 0x0027); // MIDDLE DOT or apostrophe
        append_codepoint_(out, 0x0062);                         // 'b'
    }
}

/** @brief Hebrew letters bridged by single-quote, @p link_count groups (WB7a Hebrew_Letter MidLetter), into @p out. */
static void utf8_words_dense_hebrew_quote_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(out, 0x05D0); // א
        append_codepoint_(out, 0x0027); // single quote
        append_codepoint_(out, 0x05D1); // ב
    }
}

/** @brief Stream the word family's high-density homogeneous runs (each spans several 64-byte windows) to @p sink. */
static void utf8_words_dense_runs_(std::mt19937 &rng, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    std::size_t const wide_count = std::uniform_int_distribution<std::size_t>(60, 220)(rng);
    utf8_dense_regional_indicators_(scratch, rng, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_words_dense_katakana_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_words_dense_numeric_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_words_dense_midletter_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_words_dense_hebrew_quote_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
}

/** @brief Stream the word family's long-range straddling constructions for a given @p gap to @p sink. */
static void utf8_words_straddles_(std::mt19937 &rng, std::size_t gap, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    utf8_dense_regional_indicators_(scratch, rng, gap);
    scratch.append("a"); // ASCII tail forces the WB15/16 parity decision after the long run
    sink(context, scratch.data(), scratch.size());
    utf8_words_dense_midletter_(scratch, gap), sink(context, scratch.data(), scratch.size());
}

/** @brief Word-biased random-corpus snippets: apostrophe/underscore bridges, numeric groups, CJK/Hangul/Hebrew/Katakana. */
static char const *const utf8_words_snippets[] = {
    "don't ",
    "can't_stop ",
    "3,14 ",
    "1,2,3 ",
    "Hello, world! ",
    "\xE4\xBD\xA0\xE5\xA5\xBD",         // CJK ideographs
    "\xEC\x95\x88\xEB\x85\x95",         // Hangul syllables
    "\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D", // Hebrew run
    "\xE3\x82\xAB\xE3\x82\xBF ",        // Katakana
    "word\xC2\xB7word ",                // MIDDLE DOT MidLetter
    "a b ",
    "_a ",
};

/** @brief Word family alphabet: weights bias toward the family snippets and motifs (WB6/7/11/12/13/15/16). */
static utf8_corpus_alphabet_t const utf8_words_alphabet = {
    utf8_words_snippets,
    sizeof(utf8_words_snippets) / sizeof(utf8_words_snippets[0]),
    utf8_default_boundary_codepoints,
    sizeof(utf8_default_boundary_codepoints) / sizeof(utf8_default_boundary_codepoints[0]),
    {40, 15, 10, 30, 5}, // snippet, boundary, astral, motif, malformed
};

/** @brief Assemble the word family's differential corpora (motifs + dense + straddle + seam regressions + alphabet). */
static utf8_segment_corpora_t utf8_words_corpora_() {
    utf8_segment_corpora_t corpora = {"word",
                                      utf8_words_motifs,
                                      utf8_words_motifs_count,
                                      &utf8_words_dense_runs_,
                                      &utf8_words_straddles_,
                                      utf8_words_seam_regressions,
                                      utf8_words_seam_regressions_count,
                                      &utf8_words_alphabet};
    return corpora;
}

#pragma endregion // Equivalence

#pragma region Rule coverage

/** @brief Rule-coverage gate: every WB rule motif agrees serial-vs-ISA (at window phases), no rule left unexercised. */
void test_utf8_words_rules() {
    std::printf("  - testing UTF-8 word rule-coverage matrix...\n");

    // One motif per UAX-29 Word_Break rule, tagged with the direction it demonstrates; rules with both senses also
    // carry an opposite-direction motif (the gate compares serial-vs-ISA on every motif).
    utf8_rule_case_t const rule_cases[] = {
        {"WB3", utf8_rule_joins_k, "\r\n"_sv},                          // CR x LF (no break)
        {"WB3a", utf8_rule_breaks_k, "\rb"_sv},                         // (Newline|CR|LF) / break after CR
        {"WB3b", utf8_rule_breaks_k, "a\n"_sv},                         // / (Newline|CR|LF) break before LF
        {"WB3c", utf8_rule_joins_k, "\xE2\x80\x8D\xF0\x9F\x98\x80"_sv}, // ZWJ x Extended_Pictographic (no break)
        {"WB3d", utf8_rule_joins_k, "  "_sv},                           // WSegSpace x WSegSpace (no break)
        {"WB4", utf8_rule_joins_k, "a\xCC\x81"_sv},                     // X (Extend|Format|ZWJ)* absorbed
        {"WB5", utf8_rule_joins_k, "ab"_sv},                            // AHLetter x AHLetter
        {"WB6", utf8_rule_joins_k, "a'b"_sv},                           // AHLetter x (MidLetter) AHLetter
        {"WB7", utf8_rule_joins_k, "a'b"_sv},                           // AHLetter (MidLetter) x AHLetter
        {"WB7a", utf8_rule_joins_k, "\xD7\x90'"_sv},                    // Hebrew_Letter x Single_Quote
        {"WB7b", utf8_rule_joins_k, "\xD7\x90\"\xD7\x90"_sv},           // Hebrew_Letter x Double_Quote Hebrew_Letter
        {"WB7c", utf8_rule_joins_k, "\xD7\x90\"\xD7\x90"_sv},           // Hebrew_Letter Double_Quote x Hebrew_Letter
        {"WB8", utf8_rule_joins_k, "12"_sv},                            // Numeric x Numeric
        {"WB9", utf8_rule_joins_k, "a1"_sv},                            // AHLetter x Numeric
        {"WB10", utf8_rule_joins_k, "1a"_sv},                           // Numeric x AHLetter
        {"WB11", utf8_rule_joins_k, "1,2"_sv},                          // Numeric (MidNum) x Numeric
        {"WB12", utf8_rule_joins_k, "1,2"_sv},                          // Numeric x (MidNum) Numeric
        {"WB13", utf8_rule_joins_k, "\xE3\x82\xA2\xE3\x82\xA2"_sv},     // Katakana x Katakana
        {"WB13a", utf8_rule_joins_k, "a_"_sv},                          // (AHLetter|Numeric|Katakana) x ExtendNumLet
        {"WB13b", utf8_rule_joins_k, "_a"_sv},                          // ExtendNumLet x (AHLetter|Numeric|Katakana)
        {"WB15", utf8_rule_joins_k, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv},     // sot (RI RI)* RI x RI (even parity)
        {"WB16", utf8_rule_joins_k, "\xF0\x9F\x87\xBA\x61\xF0\x9F\x87\xB8"_sv}, // [^RI] (RI RI)* RI x RI (parity reset)
        {"WB999", utf8_rule_breaks_k, "a b"_sv}, // Any / Any (default break at the space)
        // Opposite-direction motifs (E1): the same rule firing the other way.
        {"WB6", utf8_rule_breaks_k, "a''b"_sv},           // two MidLetters: the bridge fails -> break
        {"WB11", utf8_rule_breaks_k, "1,,2"_sv},          // two MidNum: numeric bridge fails -> break
        {"WB13", utf8_rule_breaks_k, "\xE3\x82\xA2z"_sv}, // Katakana then Latin: script change -> break
        {"WB15", utf8_rule_breaks_k,
         "\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA"_sv}, // 3 RI: break after the pair
    };
    // Every Word_Break rule id the gate requires a motif for (spec-derived checklist).
    char const *const required_rules[] = {
        "WB3", "WB3a", "WB3b", "WB3c", "WB3d", "WB4",  "WB5",   "WB6",   "WB7",  "WB7a", "WB7b",  "WB7c",
        "WB8", "WB9",  "WB10", "WB11", "WB12", "WB13", "WB13a", "WB13b", "WB15", "WB16", "WB999",
    };
#if SZ_USE_ICELAKE
    check_utf8_rule_coverage_("word", sz_utf8_words_serial, sz_utf8_words_icelake, rule_cases,
                              sizeof(rule_cases) / sizeof(rule_cases[0]), required_rules,
                              sizeof(required_rules) / sizeof(required_rules[0]));
#endif
}

#pragma endregion // Rule coverage

#pragma region Safety

/** @brief Malformed-input safety of the UTF-8 word kernels (serial / dispatched / icelake). */
void test_utf8_words_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 word kernels...\n");
    check_utf8_segment_safety_("word (serial)", sz_utf8_words_serial);
    check_utf8_segment_safety_("word (dispatched)", sz_utf8_words);
#if SZ_USE_ICELAKE
    check_utf8_segment_safety_("word (icelake)", sz_utf8_words_icelake);
#endif
    std::printf("    word safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/** @brief Serial-vs-ISA word differential over the hardened corpora (high-density + long-range + seam regressions). */
void test_utf8_words_all() {
#if SZ_USE_ICELAKE
    utf8_segment_corpora_t const corpora = utf8_words_corpora_();
    test_utf8_segment_equivalence_(sz_utf8_words_serial, sz_utf8_words_icelake, corpora);
#endif
}

#pragma endregion // Drivers
