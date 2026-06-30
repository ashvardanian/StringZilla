/**
 *  @brief  UAX-29 sentence-boundary (Sentence_Break) tests: known-answer goldens, malformed-input safety, and the
 *          serial-vs-ISA differential over hardened corpora.
 *  @file   scripts/test_utf8_sentences.cpp
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

/** @brief Hand-checked UAX-29 sentence-break golden vectors: each source text and its expected sentence segments. */
static utf8_unit_case_t const utf8_sentences_unit_cases[] = {
    {""_sv, {}},
    {"Hello."_sv, {"Hello."_sv}},
    {"Hello. World."_sv, {"Hello. "_sv, "World."_sv}}, // terminator + space ends the sentence
    {"Mr. Smith went."_sv,
     {"Mr. "_sv, "Smith went."_sv}},                // UAX-29 has no abbreviation list: '. ' before uppercase breaks
    {"etc. and so on"_sv, {"etc. and so on"_sv}},   // SB8: '. ' before a lowercase word does NOT break
    {"One.\nTwo."_sv, {"One.\n"_sv, "Two."_sv}},    // ParaSep after terminator
    {"A! B? C."_sv, {"A! "_sv, "B? "_sv, "C."_sv}}, // multiple terminators
};
static constexpr std::size_t utf8_sentences_unit_cases_count = sizeof(utf8_sentences_unit_cases) /
                                                               sizeof(utf8_sentences_unit_cases[0]);

/**
 *  @brief The UTF-8 sentence-break segmenters compiled on this target. The always-present `dispatched` entry keeps
 *         the table non-empty on a baseline build; the unit / rule-coverage / safety / equivalence drivers all
 *         iterate this one ladder so their ISA coverage stays in lockstep.
 */
static utf8_segment_backend_t const utf8_sentences_backends[] = {
    {"dispatched", sz_utf8_sentences},
#if SZ_USE_HASWELL
    {"haswell", sz_utf8_sentences_haswell},
#endif
#if SZ_USE_ICELAKE
    {"icelake", sz_utf8_sentences_icelake},
#endif
#if SZ_USE_NEON
    {"neon", sz_utf8_sentences_neon},
#endif
};

/** @brief Known-answer sentence-break vectors through dispatched, serial, and each ISA backend + the C++ range. */
void test_utf8_sentences_unit() {
    std::printf("  - testing UTF-8 sentence-break known-answer vectors...\n");

    check_utf8_segment_unit_("sentence", sz_utf8_sentences_serial, utf8_sentences_unit_cases,
                             utf8_sentences_unit_cases_count);
    for (utf8_segment_backend_t const &backend : utf8_sentences_backends)
        check_utf8_segment_unit_("sentence", backend.finder, utf8_sentences_unit_cases,
                                 utf8_sentences_unit_cases_count);

    // C++ range wrapper known-answer: the view must faithfully expose the kernel's sentence segments.
    std::vector<std::string> const sentences =
        sz::string_view("Hi. Yo.").utf8_sentences().template to<std::vector<std::string>>();
    assert(sentences.size() == 2 && "C++ utf8_sentences range");
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief UAX-29 sentence-break corner motifs (sprinkled into the random corpus): ATerm vs STerm, terminator +
 *         Close + Space + case, paragraph separators after a terminator, SContinue, abbreviation-like and numeric.
 */
static sz::string_view const utf8_sentences_motifs[] = {
    "End. Next"_sv,               // ATerm + space + uppercase (SB break)
    "End! Next"_sv,               // STerm + space + uppercase (SB break)
    "End? Next"_sv,               // STerm '?' + space + uppercase
    "end. next"_sv,               // SB8: '. ' before lowercase does not break
    "(End.) Next"_sv,             // terminator + Close ')' + space + uppercase
    "\"End.\" Next"_sv,           // terminator + Close '"' + space + uppercase
    "End.\xE2\x80\xA8" "Next"_sv, // ParaSep U+2028 after terminator
    "End.\xE2\x80\xA9" "Next"_sv, // ParaSep U+2029 after terminator
    "End.\xC2\x85" "Next"_sv,     // ParaSep U+0085 NEL after terminator
    "Item, more"_sv,              // SContinue ',' continuation
    "Item: more"_sv,              // SContinue ':' continuation
    "Mr. Smith"_sv,               // abbreviation-like
    "Dr. House"_sv,               // abbreviation-like
    "U.S.A. ok"_sv,               // dotted acronym
    "3.14 value"_sv,              // numeric ATerm (no break)
    // Dense-decode edge motifs (SB5 Extend/Format transparency + cross-window SB8), sprinkled at window seams:
    "End.\xCC\x88 next"_sv,      // ATerm + combining mark (Extend, SB5-transparent) before the space
    "End\xCC\x88. Next"_sv,      // combining mark on the terminator's preceding letter
    "End.\xE2\x80\x8B Next"_sv,  // ATerm + Format (U+200B ZWSP is Format here) transparency
    "A.\xCC\x80\xCC\x80 b"_sv,   // ATerm + stacked Extend marks then lowercase (SB8 across transparents)
    "End. \xE2\x80\x8C Next"_sv, // ATerm Sp then Format then Upper (SB8 neutral chain)
};
static constexpr std::size_t utf8_sentences_motifs_count = sizeof(utf8_sentences_motifs) /
                                                           sizeof(utf8_sentences_motifs[0]);

/** @brief ATerm + Close* + Sp* run @p link_count wide, then a Lower (SB8 continuation, no break), into @p out. */
static void utf8_sentences_dense_aterm_sp_lower_(std::string &out, std::size_t link_count) {
    out.clear();
    append_codepoint_(out, 0x0055);                                                           // 'U' Upper
    append_codepoint_(out, 0x002E);                                                           // '.' ATerm
    for (std::size_t index = 0; index != link_count; ++index) append_codepoint_(out, 0x0020); // Sp run
    append_codepoint_(out, 0x0061);                                                           // 'a' Lower
}

/** @brief Terminator-dense `A. A. A. ...` repeated @p link_count times (one break per terminator), into @p out. */
static void utf8_sentences_dense_terminators_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) out.append("A. ");
}

/** @brief Ideographic full stop (U+3002) + Sp run + CJK, repeated @p link_count times (multibyte STerm), into @p out. */
static void utf8_sentences_dense_cjk_term_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(out, 0x4E2D); // 中
        append_codepoint_(out, 0x3002); // 。 ideographic full stop (STerm)
        append_codepoint_(out, 0x0020); // Sp
    }
}

/** @brief Stream the sentence family's high-density homogeneous runs (each spans several 64-byte windows) to @p sink. */
static void utf8_sentences_dense_runs_(std::mt19937 &rng, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    std::size_t const wide_count = std::uniform_int_distribution<std::size_t>(60, 220)(rng);
    utf8_sentences_dense_aterm_sp_lower_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_sentences_dense_terminators_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_sentences_dense_cjk_term_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
}

/** @brief SB8: `Upper ATerm Sp{gap} Lower` — long Sp run before a lowercase keeps one sentence, into @p out. */
static void utf8_straddle_sb8_lower_(std::string &out, std::size_t gap) {
    out.clear();
    append_codepoint_(out, 0x0055);                                                    // 'U'
    append_codepoint_(out, 0x002E);                                                    // '.'
    for (std::size_t index = 0; index != gap; ++index) append_codepoint_(out, 0x0020); // Sp run across windows
    append_codepoint_(out, 0x0061);                                                    // 'a' Lower
}

/** @brief SB11: `Upper ATerm Sp{gap} Upper` — same run before an uppercase must break, into @p out. */
static void utf8_straddle_sb11_upper_(std::string &out, std::size_t gap) {
    out.clear();
    append_codepoint_(out, 0x0055);                                                    // 'U'
    append_codepoint_(out, 0x002E);                                                    // '.'
    for (std::size_t index = 0; index != gap; ++index) append_codepoint_(out, 0x0020); // Sp run across windows
    append_codepoint_(out, 0x0042);                                                    // 'B' Upper
}

/** @brief Stream the sentence family's long-range straddling constructions for a given @p gap to @p sink. */
static void utf8_sentences_straddles_(std::mt19937 & /*rng*/, std::size_t gap, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    utf8_straddle_sb8_lower_(scratch, gap), sink(context, scratch.data(), scratch.size());
    utf8_straddle_sb11_upper_(scratch, gap), sink(context, scratch.data(), scratch.size());
}

/** @brief Sentence-biased random-corpus snippets: ATerm/STerm + space + case, abbreviations, numeric, CJK stop, ParaSep. */
static char const *const utf8_sentences_snippets[] = {
    "End. Next ",               // ATerm + space + Upper (SB break)
    "end. next ",               // SB8: '. ' before Lower does not break
    "(End.) ",                  // terminator + Close ')' + space
    "Mr. ",                     // abbreviation-like
    "U.S.A. ",                  // dotted acronym
    "3.14 ",                    // numeric ATerm (no break)
    "Item, ",                   // SContinue ',' continuation
    "\xE4\xB8\xAD\xE3\x80\x82", // CJK ideograph + ideographic full stop (STerm)
    "\xE2\x80\xA8",             // ParaSep U+2028
};

/** @brief Sentence family alphabet: weights bias toward the family snippets and motifs (SB6/7/8/8a/9/10/11). */
static utf8_corpus_alphabet_t const utf8_sentences_alphabet = {
    utf8_sentences_snippets,
    sizeof(utf8_sentences_snippets) / sizeof(utf8_sentences_snippets[0]),
    utf8_default_boundary_codepoints,
    sizeof(utf8_default_boundary_codepoints) / sizeof(utf8_default_boundary_codepoints[0]),
    {45, 15, 5, 30, 5}, // snippet, boundary, astral, motif, malformed
};

/** @brief Assemble the sentence family's differential corpora (motifs + dense + straddle + alphabet; no seam regressions). */
static utf8_segment_corpora_t utf8_sentences_corpora_() {
    utf8_segment_corpora_t corpora = {"sentence",
                                      utf8_sentences_motifs,
                                      utf8_sentences_motifs_count,
                                      &utf8_sentences_dense_runs_,
                                      &utf8_sentences_straddles_,
                                      nullptr,
                                      0,
                                      &utf8_sentences_alphabet};
    return corpora;
}

#pragma endregion // Equivalence

#pragma region Rule coverage

/** @brief Rule-coverage gate: every SB rule motif agrees serial-vs-ISA (at window phases), no rule left unexercised. */
void test_utf8_sentences_rules() {
    std::printf("  - testing UTF-8 sentence rule-coverage matrix...\n");

    // One motif per UAX-29 Sentence_Break rule, tagged with the direction it demonstrates (break or no-break).
    utf8_rule_case_t const rule_cases[] = {
        {"SB3", utf8_rule_joins_k, "\r\n"_sv},      // CR x LF (no break)
        {"SB4", utf8_rule_breaks_k, "a\nb"_sv},     // (Sep|CR|LF) / break after LF
        {"SB5", utf8_rule_joins_k, "a\xCC\x81"_sv}, // X (Extend|Format)* absorbed (no break)
        {"SB6", utf8_rule_joins_k, "3.4"_sv},       // ATerm x Numeric (no break)
        {"SB7", utf8_rule_joins_k, "A.B"_sv},       // (Upper|Lower) ATerm x Upper (no break)
        {"SB8", utf8_rule_joins_k, "A. a"_sv},      // ATerm Close* Sp* x (not stop)* Lower (no break)
        {"SB8a", utf8_rule_joins_k, "a.,"_sv},      // (STerm|ATerm) Close* Sp* x (SContinue|STerm|ATerm) (no break)
        {"SB9", utf8_rule_joins_k, "a.)"_sv},       // (STerm|ATerm) Close* x (Close|Sp|Sep|CR|LF) (no break)
        {"SB10", utf8_rule_joins_k, "a. "_sv},      // (STerm|ATerm) Close* Sp* x (Sp|Sep|CR|LF) (no break)
        {"SB11", utf8_rule_breaks_k, "A. B"_sv},    // (STerm|ATerm) Close* Sp* / break before the next sentence
        {"SB998", utf8_rule_joins_k, "ab"_sv},      // Any x Any (no break by default)
        // Opposite-direction motifs (E1): the same rule firing the other way (the gate compares serial-vs-ISA on each).
        {"SB998", utf8_rule_breaks_k,
         "ab cd"_sv}, // default no-break interior, but the terminator-less run still varies
        {"SB5", utf8_rule_joins_k, "a\xE2\x80\x8B"_sv}, // Format (U+200B ZWSP) transparency absorbed (no break)
    };
    // Every Sentence_Break rule id the gate requires a motif for (spec-derived checklist).
    char const *const required_rules[] = {
        "SB3", "SB4", "SB5", "SB6", "SB7", "SB8", "SB8a", "SB9", "SB10", "SB11", "SB998",
    };
    for (utf8_segment_backend_t const &backend : utf8_sentences_backends)
        check_utf8_rule_coverage_("sentence", sz_utf8_sentences_serial, backend.finder, rule_cases,
                                  sizeof(rule_cases) / sizeof(rule_cases[0]), required_rules,
                                  sizeof(required_rules) / sizeof(required_rules[0]));
}

#pragma endregion // Rule coverage

#pragma region Safety

/** @brief Malformed-input safety of the UTF-8 sentence kernels (serial / dispatched / icelake). */
void test_utf8_sentences_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 sentence kernels...\n");
    check_utf8_segment_safety_("sentence (serial)", sz_utf8_sentences_serial);
    for (utf8_segment_backend_t const &backend : utf8_sentences_backends) {
        std::string const label = std::string("sentence (") + backend.name + ")";
        check_utf8_segment_safety_(label.c_str(), backend.finder);
    }
    std::printf("    sentence safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/** @brief Serial-vs-ISA sentence differential over the hardened corpora (high-density + long-range). */
void test_utf8_sentences_all() {
    utf8_segment_corpora_t const corpora = utf8_sentences_corpora_();
    for (utf8_segment_backend_t const &backend : utf8_sentences_backends)
        test_utf8_segment_equivalence_(sz_utf8_sentences_serial, backend.finder, corpora);
}

#pragma endregion // Drivers
