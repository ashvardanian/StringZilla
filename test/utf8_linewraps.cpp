/**
 *  @brief  UAX-14 line-break (linewrap) tests: known-answer goldens, malformed-input safety, and the
 *          serial-vs-ISA differential over hardened corpora.
 *  @file   scripts/test_utf8_linewraps.cpp
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

/** @brief Hand-checked UAX-14 line-break golden vectors: each source text and its expected line segments. */
static utf8_unit_case_t const utf8_linewraps_unit_cases[] = {
    {""_sv, {}},
    {"hello world"_sv, {"hello "_sv, "world"_sv}},          // soft wrap after the space
    {"a\nb"_sv, {"a\n"_sv, "b"_sv}},                        // LF hard break
    {"a\r\nb"_sv, {"a\r\n"_sv, "b"_sv}},                    // CRLF hard break, one segment
    {"a\xE2\x80\xA8" "b"_sv, {"a\xE2\x80\xA8"_sv, "b"_sv}}, // U+2028 LINE SEPARATOR
    {"a\xE2\x80\xA9" "b"_sv, {"a\xE2\x80\xA9"_sv, "b"_sv}}, // U+2029 PARAGRAPH SEPARATOR
};
static constexpr std::size_t utf8_linewraps_unit_cases_count = sizeof(utf8_linewraps_unit_cases) /
                                                               sizeof(utf8_linewraps_unit_cases[0]);

/** @brief Known-answer line-break vectors through dispatched, serial, and each ISA backend + the C++ range. */
void test_utf8_linewraps_unit() {
    std::printf("  - testing UTF-8 line-break known-answer vectors...\n");

    check_utf8_segment_unit_("linewrap", sz_utf8_linewraps, utf8_linewraps_unit_cases,
                             utf8_linewraps_unit_cases_count); // Dispatched
    check_utf8_segment_unit_("linewrap", sz_utf8_linewraps_serial, utf8_linewraps_unit_cases,
                             utf8_linewraps_unit_cases_count);
#if SZ_USE_HASWELL
    check_utf8_segment_unit_("linewrap", sz_utf8_linewraps_haswell, utf8_linewraps_unit_cases,
                             utf8_linewraps_unit_cases_count);
#endif
#if SZ_USE_ICELAKE
    check_utf8_segment_unit_("linewrap", sz_utf8_linewraps_icelake, utf8_linewraps_unit_cases,
                             utf8_linewraps_unit_cases_count);
#endif
#if SZ_USE_NEON
    check_utf8_segment_unit_("linewrap", sz_utf8_linewraps_neon, utf8_linewraps_unit_cases,
                             utf8_linewraps_unit_cases_count);
#endif

    // C++ range wrapper known-answer: the view must faithfully expose the kernel's segments.
    std::vector<std::string> const wrapped =
        sz::string_view("a\nb").utf8_linewraps().template to<std::vector<std::string>>();
    assert(wrapped.size() == 2 && wrapped[0] == "a\n" && wrapped[1] == "b" && "C++ utf8_linewraps range");
}

#pragma endregion // Unit

#pragma region Equivalence

/** @brief UAX-14 line-break corner motifs (sprinkled into the random corpus): mandatory breaks, OP/CL, HY, GL, NU. */
static sz::string_view const utf8_linewraps_motifs[] = {
    "a\nb"_sv,              // LF mandatory break
    "a\rb"_sv,              // CR mandatory break
    "a\r\nb"_sv,            // CRLF mandatory break (one segment)
    "a\xE2\x80\xA8" "b"_sv, // U+2028 LINE SEPARATOR mandatory
    "a\xE2\x80\xA9" "b"_sv, // U+2029 PARAGRAPH SEPARATOR mandatory
    "a\xC2\x85" "b"_sv,     // U+0085 NEL mandatory
    "a\xE2\x80\x8D" "b"_sv, // ZWJ (U+200D) adjacency
    "(word"_sv,             // OP open-punctuation before
    "word)"_sv,             // CL close-punctuation after
    "co-op"_sv,             // HY hyphen
    "a\xC2\xA0" "b"_sv,     // GL glue NBSP (U+00A0)
    "a b"_sv,               // BA break-after space
    "word!"_sv,             // EX exclamation
    "\"word\""_sv,          // QU quotation
    "a:b"_sv,               // IS infix separator
    "12345"_sv,             // NU numbers
    "$50"_sv,               // PR prefix
    "3,000"_sv,             // numeric grouping comma
    // Regression motifs for fuzzer-found icelake-vs-serial divergences (sprinkled at window-edge offsets so the
    // cross-window carry is exercised):
    "\xD7\x90-a"_sv,               // LB21a: HL (U+05D0) HY x AL -- no break before the AL, incl. when HL carries
    "\xD7\x90\xE2\x80\x90" "a"_sv, // LB21a with HH (U+2010) instead of HY
    "\xE3\x80\xAF\xE2\x80\x98" "a"_sv, // LB10/LB19: lone CM (U+302F, East-Asian) then Pi quote (U+2018) -- side bits cleared
    "\xCC\x88\xE2\x80\x9C"_sv,         // lone combining diaeresis (U+0308) then QU (U+201C)
};
static constexpr std::size_t utf8_linewraps_motifs_count = sizeof(utf8_linewraps_motifs) /
                                                           sizeof(utf8_linewraps_motifs[0]);

/** @brief Mandatory-break-dense line run cycling CRLF/U+2028/U+2029/U+000B, @p link_count cycles (LB4/5), into @p out. */
static void utf8_linewraps_dense_mandatory_breaks_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(out, 0x0061); // 'a'
        switch (index & 0x3u) {
        case 0: out.append("\r\n"); break;              // CRLF
        case 1: append_codepoint_(out, 0x2028); break;  // LINE SEPARATOR
        case 2: append_codepoint_(out, 0x2029); break;  // PARAGRAPH SEPARATOR
        default: append_codepoint_(out, 0x000B); break; // vertical tab (BK)
        }
    }
}

/** @brief OP/CL/QU/HY/BA/GL nesting cycled @p link_count times (LB13/14/15/18 adjacency), into @p out. */
static void utf8_linewraps_dense_nesting_(std::string &out, std::size_t link_count) {
    out.clear();
    static char const *cycle[] = {"(", "word", ")", "\"", "-", " ", "\xC2\xA0"}; // OP CL QU HY BA SP GL(NBSP)
    for (std::size_t index = 0; index != link_count; ++index) out.append(cycle[index % 7u]);
}

/** @brief Numeric `NU` runs interleaved with IS (`.`) and SY (`/`), @p link_count groups (LB25 numbers), into @p out. */
static void utf8_linewraps_dense_numeric_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) out.append("1.234/56 ");
}

/** @brief Stream the linewrap family's high-density homogeneous runs (each spans several 64-byte windows) to @p sink. */
static void utf8_linewraps_dense_runs_(std::mt19937 &rng, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    std::size_t const wide_count = std::uniform_int_distribution<std::size_t>(60, 220)(rng);
    utf8_linewraps_dense_mandatory_breaks_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_linewraps_dense_nesting_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_linewraps_dense_numeric_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
}

/** @brief Stream the linewrap family's long-range straddling constructions for a given @p gap to @p sink. */
static void utf8_linewraps_straddles_(std::mt19937 & /*rng*/, std::size_t gap, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    utf8_linewraps_dense_mandatory_breaks_(scratch, gap), sink(context, scratch.data(), scratch.size());
    utf8_linewraps_dense_nesting_(scratch, gap), sink(context, scratch.data(), scratch.size());
}

/** @brief Linewrap-biased random-corpus snippets: mandatory breaks, separators, NBSP/GL, nesting, hyphen, numeric. */
static char const *const utf8_linewraps_snippets[] = {
    "a\r\nb",        // mandatory break (CRLF)
    "a\x0C" "b",     // mandatory break (form feed, BK)
    "\xE2\x80\xA8",  // U+2028 LINE SEPARATOR
    "\xE2\x80\xA9",  // U+2029 PARAGRAPH SEPARATOR
    "a\xC2\xA0" "b", // GL glue NBSP (U+00A0)
    "(a)",           // OP/CL nesting
    "\"x\"",         // QU quotes
    "a-b",           // HY hyphen
    "1,234.5 ",      // numeric grouping
    "\xD7\x90-a",    // Hebrew letter + hyphen (LB21a)
};

/** @brief Linewrap family alphabet: weights bias toward family snippets and motifs (LB mandatory/GL/HY/NU/nesting). */
static utf8_corpus_alphabet_t const utf8_linewraps_alphabet = {
    utf8_linewraps_snippets,
    sizeof(utf8_linewraps_snippets) / sizeof(utf8_linewraps_snippets[0]),
    utf8_default_boundary_codepoints,
    sizeof(utf8_default_boundary_codepoints) / sizeof(utf8_default_boundary_codepoints[0]),
    {45, 15, 5, 30, 5}, // snippet, boundary, astral, motif, malformed
};

/** @brief Assemble the linewrap family's differential corpora (motifs + dense + straddle + alphabet). */
static utf8_segment_corpora_t utf8_linewraps_corpora_() {
    utf8_segment_corpora_t corpora = {"linewrap",
                                      utf8_linewraps_motifs,
                                      utf8_linewraps_motifs_count,
                                      &utf8_linewraps_dense_runs_,
                                      &utf8_linewraps_straddles_,
                                      nullptr,
                                      0,
                                      &utf8_linewraps_alphabet};
    return corpora;
}

#pragma endregion // Equivalence

#pragma region Rule coverage

/** @brief Rule-coverage gate: every LB rule motif agrees serial-vs-ISA (at window phases), no rule left unexercised. */
void test_utf8_linewraps_rules() {
    std::printf("  - testing UTF-8 line-break rule-coverage matrix...\n");

    // One motif per UAX-14 Line_Break rule, tagged with the direction it demonstrates; rules with both senses also
    // carry an opposite-direction motif (the gate compares serial-vs-ISA on every motif).
    utf8_rule_case_t const rule_cases[] = {
        {"LB4", utf8_rule_breaks_k, "a\x0C" "b"_sv},                             // BK ! (mandatory break after FF)
        {"LB5", utf8_rule_breaks_k, "a\r\nb"_sv},                                // CR x LF, then LF ! (CRLF, break)
        {"LB6", utf8_rule_joins_k, "a\x0C"_sv},                                  // x (BK|CR|LF|NL): no break before HB
        {"LB7", utf8_rule_joins_k, "a b"_sv},                                    // x SP / x ZW: no break before space
        {"LB8", utf8_rule_breaks_k, "a\xE2\x80\x8B" "b"_sv},                     // ZW SP* / break after ZWSP
        {"LB8a", utf8_rule_joins_k, "a\xE2\x80\x8D\xF0\x9F\x98\x80"_sv},         // ZWJ x (no break)
        {"LB9", utf8_rule_joins_k, "a\xCC\x81"_sv},                              // treat X CM* as X
        {"LB10", utf8_rule_joins_k, "\xCC\x81" "a"_sv},                          // lone CM -> AL
        {"LB11", utf8_rule_joins_k, "a\xE2\x81\xA0" "b"_sv},                     // x WJ / WJ x (Word Joiner, no break)
        {"LB12", utf8_rule_joins_k, "\xC2\xA0" "a"_sv},                          // GL x (no break after NBSP)
        {"LB12a", utf8_rule_joins_k, "a\xC2\xA0"_sv},                            // [^SP BA HY] x GL
        {"LB13", utf8_rule_joins_k, "a)"_sv},                                    // x CL/CP/EX/SY (no break before `)`)
        {"LB14", utf8_rule_joins_k, "(a"_sv},                                    // OP SP* x (no break after OP)
        {"LB15a", utf8_rule_joins_k, "\xE2\x80\x9C" "a"_sv},                     // (sot|...) [QU & Pi] SP* x (no break)
        {"LB15b", utf8_rule_joins_k, "a\xE2\x80\x9D"_sv},                        // x [QU & Pf] (no break before close)
        {"LB16", utf8_rule_joins_k, ")\xE3\x82\xA1"_sv},                         // (CL|CP) SP* x NS (no break)
        {"LB17", utf8_rule_joins_k, "\xE2\x80\x94\xE2\x80\x94"_sv},              // B2 SP* B2 (no break between dashes)
        {"LB18", utf8_rule_breaks_k, "a b"_sv},                                  // SP / break after space
        {"LB19", utf8_rule_joins_k, "a\"b"_sv},                                  // x QU / QU x (no break around quote)
        {"LB20", utf8_rule_breaks_k, "a\xEF\xBF\xBC" "b"_sv},                    // / CB ; CB / (break around U+FFFC)
        {"LB20a", utf8_rule_joins_k, "-a"_sv},                                   // (sot|...) (HY|HH) x AL (no break)
        {"LB21", utf8_rule_joins_k, "a-b"_sv},                                   // x BA/HY/HH/NS, BB x (no break)
        {"LB21a", utf8_rule_joins_k, "\xD7\x90-a"_sv},                           // HL (HY|HH) x [^HL] (no break)
        {"LB21b", utf8_rule_joins_k, "/\xD7\x90"_sv},                            // SY x HL (no break)
        {"LB22", utf8_rule_joins_k, "a\xE2\x80\xA6"_sv},                         // x IN (no break before ellipsis)
        {"LB23", utf8_rule_joins_k, "a1"_sv},                                    // (AL|HL) x NU / NU x (AL|HL)
        {"LB23a", utf8_rule_joins_k, "$\xE4\xB8\xAD"_sv},                        // PR x (ID|EB|EM)
        {"LB24", utf8_rule_joins_k, "$a"_sv},                                    // (PR|PO) x (AL|HL) / (AL|HL) x ...
        {"LB25", utf8_rule_joins_k, "1,5"_sv},                                   // numeric clusters (no break inside)
        {"LB26", utf8_rule_joins_k, "\xE1\x84\x80\xE1\x85\xA1"_sv},              // Hangul L x V (no break)
        {"LB27", utf8_rule_joins_k, "\xEA\xB0\x80\xE1\x86\xA8"_sv},              // Hangul (H2 x T) continuation
        {"LB28", utf8_rule_joins_k, "ab"_sv},                                    // AL x AL (no break)
        {"LB28a", utf8_rule_joins_k, "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\x95"_sv}, // Brahmic aksara (C virama C)
        {"LB29", utf8_rule_joins_k, ".a"_sv},                                    // IS x (AL|HL) (no break)
        {"LB30", utf8_rule_joins_k, "a("_sv},                                    // (AL|HL|NU) x OP[^EAW] (no break)
        {"LB30a", utf8_rule_joins_k, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv},     // RI RI (even-parity pair, no break)
        {"LB30b", utf8_rule_joins_k, "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBB"_sv},     // EB x EM (base + modifier, no break)
        {"LB31", utf8_rule_breaks_k, "\xE4\xB8\xAD\xE4\xB8\xAD"_sv},             // break everywhere else (ideographs)
        // Opposite-direction motifs (E1): the same rule firing the other way.
        {"LB28", utf8_rule_breaks_k, "a b"_sv}, // AL SP AL -> break at the space
        {"LB25", utf8_rule_breaks_k, "1 2"_sv}, // numerals split by space -> break
        {"LB30a", utf8_rule_breaks_k, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA"_sv}, // 3 RI -> break
    };
    // Every Line_Break rule id the gate requires a motif for (spec-derived checklist).
    char const *const required_rules[] = {
        "LB4",   "LB5",   "LB6",  "LB7",  "LB8",  "LB8a", "LB9",   "LB10",  "LB11", "LB12",  "LB12a", "LB13", "LB14",
        "LB15a", "LB15b", "LB16", "LB17", "LB18", "LB19", "LB20",  "LB20a", "LB21", "LB21a", "LB21b", "LB22", "LB23",
        "LB23a", "LB24",  "LB25", "LB26", "LB27", "LB28", "LB28a", "LB29",  "LB30", "LB30a", "LB30b", "LB31",
    };
#if SZ_USE_HASWELL
    check_utf8_rule_coverage_("linewrap", sz_utf8_linewraps_serial, sz_utf8_linewraps_haswell, rule_cases,
                              sizeof(rule_cases) / sizeof(rule_cases[0]), required_rules,
                              sizeof(required_rules) / sizeof(required_rules[0]));
#endif
#if SZ_USE_ICELAKE
    check_utf8_rule_coverage_("linewrap", sz_utf8_linewraps_serial, sz_utf8_linewraps_icelake, rule_cases,
                              sizeof(rule_cases) / sizeof(rule_cases[0]), required_rules,
                              sizeof(required_rules) / sizeof(required_rules[0]));
#endif
#if SZ_USE_NEON
    check_utf8_rule_coverage_("linewrap", sz_utf8_linewraps_serial, sz_utf8_linewraps_neon, rule_cases,
                              sizeof(rule_cases) / sizeof(rule_cases[0]), required_rules,
                              sizeof(required_rules) / sizeof(required_rules[0]));
#endif
}

#pragma endregion // Rule coverage

#pragma region Safety

/** @brief Malformed-input safety of the UTF-8 line kernels (serial / dispatched / icelake). */
void test_utf8_linewraps_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 line kernels...\n");
    check_utf8_segment_safety_("linewrap (serial)", sz_utf8_linewraps_serial);
    check_utf8_segment_safety_("linewrap (dispatched)", sz_utf8_linewraps);
#if SZ_USE_HASWELL
    check_utf8_segment_safety_("linewrap (haswell)", sz_utf8_linewraps_haswell);
#endif
#if SZ_USE_ICELAKE
    check_utf8_segment_safety_("linewrap (icelake)", sz_utf8_linewraps_icelake);
#endif
#if SZ_USE_NEON
    check_utf8_segment_safety_("linewrap (neon)", sz_utf8_linewraps_neon);
#endif
    std::printf("    linewrap safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/** @brief Serial-vs-ISA line differential over the hardened corpora (high-density + long-range). */
void test_utf8_linewraps_all() {
    utf8_segment_corpora_t const corpora = utf8_linewraps_corpora_();
    sz_unused_(corpora);
#if SZ_USE_HASWELL
    test_utf8_segment_equivalence_(sz_utf8_linewraps_serial, sz_utf8_linewraps_haswell, corpora);
#endif
#if SZ_USE_ICELAKE
    test_utf8_segment_equivalence_(sz_utf8_linewraps_serial, sz_utf8_linewraps_icelake, corpora);
#endif
#if SZ_USE_NEON
    test_utf8_segment_equivalence_(sz_utf8_linewraps_serial, sz_utf8_linewraps_neon, corpora);
#endif
}

#pragma endregion // Drivers
