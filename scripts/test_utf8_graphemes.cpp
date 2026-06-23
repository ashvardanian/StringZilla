/**
 *  @brief  UAX-29 grapheme-cluster (Grapheme_Cluster_Break) tests: known-answer goldens, malformed-input safety,
 *          and the serial-vs-ISA differential over hardened corpora.
 *  @file   scripts/test_utf8_graphemes.cpp
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

#include "test_utf8.hpp" // shared segmentation harness (pulls in StringZilla + `test_stringzilla.hpp`)

#pragma region Unit

/** @brief Hand-checked UAX-29 grapheme-cluster golden vectors: each source text and its expected cluster segments. */
static utf8_unit_case_t const utf8_graphemes_unit_cases[] = {
    {""_sv, {}},
    {"a"_sv, {"a"_sv}},
    {"abc"_sv, {"a"_sv, "b"_sv, "c"_sv}},
    {"a\r\nb"_sv, {"a"_sv, "\r\n"_sv, "b"_sv}}, // GB3: CR x LF stay one cluster
    {"e\xCC\x81"_sv, {"e\xCC\x81"_sv}},         // e + U+0301 combining acute -> one cluster (GB9)
    {"\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv,
     {"\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv}}, // RI pair -> one cluster (GB12/13)
    {"\xF0\x9F\x87\xBA\x61\xF0\x9F\x87\xB8"_sv,
     {"\xF0\x9F\x87\xBA"_sv, "a"_sv, "\xF0\x9F\x87\xB8"_sv}}, // RI ASCII RI: parity reset
    {"\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9"_sv,       // ZWJ joins -> one cluster (GB11)
     {"\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9"_sv}},
    {"\xEA\xB0\x80"_sv, {"\xEA\xB0\x80"_sv}}, // Hangul LV syllable -> one cluster
    {"\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8"_sv,
     {"\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8"_sv}}, // Hangul L+V+T jamo -> one cluster (GB6/7/8)
    {"\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7"_sv,
     {"\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7"_sv}}, // Indic consonant + virama + consonant -> one cluster (GB9c InCB)
};
static constexpr std::size_t utf8_graphemes_unit_cases_count = sizeof(utf8_graphemes_unit_cases) /
                                                               sizeof(utf8_graphemes_unit_cases[0]);

/** @brief Known-answer grapheme-cluster vectors through dispatched, serial, and each ISA backend + the C++ range. */
void test_utf8_graphemes_unit() {
    std::printf("  - testing UTF-8 grapheme-cluster known-answer vectors...\n");

    check_utf8_segment_unit_("grapheme", sz_utf8_graphemes, utf8_graphemes_unit_cases,
                             utf8_graphemes_unit_cases_count); // Dispatched
    check_utf8_segment_unit_("grapheme", sz_utf8_graphemes_serial, utf8_graphemes_unit_cases,
                             utf8_graphemes_unit_cases_count);
#if SZ_USE_ICELAKE
    check_utf8_segment_unit_("grapheme", sz_utf8_graphemes_icelake, utf8_graphemes_unit_cases,
                             utf8_graphemes_unit_cases_count);
#endif

    // C++ range wrapper known-answer: the view must faithfully expose the kernel's clusters.
    std::vector<std::string> const clusters =
        sz::string_view("ab").utf8_graphemes().template to<std::vector<std::string>>();
    assert(clusters.size() == 2 && clusters[0] == "a" && clusters[1] == "b" && "C++ utf8_graphemes range");
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief UAX-29 grapheme-cluster corner motifs (sprinkled into the random corpus): emoji-ZWJ chains, VS16,
 *         skin-tone modifiers, regional-indicator runs of varying parity, Indic virama clusters, Hangul jamo
 *         combinations, and bare CR/LF shapes. All non-ASCII bytes are `\xHH` escapes.
 */
static sz::string_view const utf8_graphemes_motifs[] = {
    "\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D" // three-link ZWJ chain
    "\xF0\x9F\x91\xA7"_sv,
    "\xF0\x9F\x91\xA9\xE2\x80\x8D"_sv,     // dangling trailing ZWJ
    "\xE2\x9C\x8B\xEF\xB8\x8F"_sv,         // emoji + VS16 (U+FE0F)
    "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD"_sv, // emoji + skin-tone modifier (U+1F3FD)
    "\xF0\x9F\x87\xBA"_sv,                 // single regional indicator (run length 1)
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv, // regional-indicator run length 2
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"     // regional-indicator run length 3 (odd parity)
    "\xF0\x9F\x87\xAB"_sv,
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8" // regional-indicator run length 5 (odd parity)
    "\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7\xF0\x9F\x87\xA9"_sv,
    "\xE0\xA6\x95\xE0\xA7\x8D\xE0\xA6\x95"_sv, // Bengali consonant + virama + consonant (U+09CD)
    "\xE0\xB4\x95\xE0\xB5\x8D\xE0\xB4\x95"_sv, // Malayalam consonant + virama + consonant (U+0D4D)
    "\xE0\xAE\x95\xE0\xAF\x8D\xE0\xAE\x95"_sv, // Tamil consonant + virama + consonant (U+0BCD)
    "\xEA\xB0\x80"_sv,                         // Hangul LV syllable
    "\xEC\x95\x8A"_sv,                         // Hangul LVT syllable
    "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8"_sv, // Hangul L+V+T jamo combination
    "\r"_sv,                                   // CR alone
    "\n"_sv,                                   // LF alone
    "\r\r\n"_sv,                               // CR CR LF (parity of CR runs)
    "a\xDF\xBF\xE0\xA0\x80z"_sv,               // ASCII + U+07FF (2-byte) + U+0800 (3-byte cold) + ASCII: every BMP gate
    "x\xF0\x9F\x98\x80y"_sv,           // ASCII + astral emoji (U+1F600) + ASCII: ASCII and astral gates together
    "\xD0\x90\xF0\x9F\x98\x80"_sv,     // Cyrillic (2-byte) + astral emoji: page LUT and astral, no cold cascade
    "\xE0\xA0\x80\xF0\x9F\x98\x80"_sv, // U+0800 (cold 3-byte) + astral emoji: cold cascade and astral together
    "\xDF\xBF\xE0\xA0\x80"_sv,         // page-LUT edge: last 2-byte U+07FF abutting first 3-byte U+0800
};
static constexpr std::size_t utf8_graphemes_motifs_count = sizeof(utf8_graphemes_motifs) /
                                                           sizeof(utf8_graphemes_motifs[0]);

/** @brief A pictograph ZWJ chain @p link_count deep, every other link followed by VS16 (GB11). */
static std::string utf8_graphemes_dense_zwj_pictograph_chain_(std::size_t link_count) {
    std::string text;
    static sz_rune_t const pictographs[] = {0x1F468, 0x1F469, 0x1F467, 0x1F466}; // man, woman, girl, boy
    append_codepoint_(text, pictographs[0]);
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(text, 0x200D);                          // ZWJ
        append_codepoint_(text, pictographs[(index + 1) & 0x3u]); // next pictograph
        if (index & 1u) append_codepoint_(text, 0xFE0F);          // VS16 on alternating links
    }
    return text;
}

/** @brief One base letter followed by @p link_count combining marks (GB9 Extend run). */
static std::string utf8_graphemes_dense_combining_marks_(std::size_t link_count) {
    std::string text;
    static sz_rune_t const marks[] = {0x0301, 0x0300, 0x0308, 0x0327, 0x0323}; // acute, grave, diaeresis, cedilla, dot
    append_codepoint_(text, 0x0061);                                           // base 'a'
    for (std::size_t index = 0; index != link_count; ++index) append_codepoint_(text, marks[index % 5u]);
    return text;
}

/** @brief @p link_count emoji each immediately followed by a skin-tone modifier drawn from @p rng (GB9 Extend). */
static std::string utf8_graphemes_dense_skin_tone_run_(std::mt19937 &rng, std::size_t link_count) {
    std::string text;
    std::uniform_int_distribution<sz_rune_t> modifier(0x1F3FB, 0x1F3FF); // U+1F3FB..U+1F3FF
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(text, 0x1F44D); // thumbs up
        append_codepoint_(text, modifier(rng));
    }
    return text;
}

/** @brief @p link_count Indic consonant+virama conjunct links (GB9c InCB Consonant Linker chains). */
static std::string utf8_graphemes_dense_indic_conjunct_(std::size_t link_count) {
    std::string text;
    append_codepoint_(text, 0x0915); // Devanagari KA
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(text, 0x094D); // virama (Linker)
        append_codepoint_(text, 0x0915); // KA
    }
    return text;
}

/** @brief @p link_count Hangul L/V/T jamo triples (GB6/7/8 jamo composition runs). */
static std::string utf8_graphemes_dense_hangul_jamo_(std::size_t link_count) {
    std::string text;
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(text, 0x1100); // L (Choseong Kiyeok)
        append_codepoint_(text, 0x1161); // V (Jungseong A)
        append_codepoint_(text, 0x11A8); // T (Jongseong Kiyeok)
    }
    return text;
}

/** @brief The grapheme family's high-density homogeneous runs, sized from @p rng to span several 64-byte windows. */
static std::vector<std::string> utf8_graphemes_dense_runs_(std::mt19937 &rng) {
    std::uniform_int_distribution<std::size_t> wide(60, 220);
    std::uniform_int_distribution<std::size_t> chain(20, 80);
    std::size_t const wide_count = wide(rng);
    std::size_t const chain_count = chain(rng);
    return {utf8_dense_regional_indicators_(rng, wide_count),  utf8_graphemes_dense_zwj_pictograph_chain_(chain_count),
            utf8_graphemes_dense_combining_marks_(wide_count), utf8_graphemes_dense_skin_tone_run_(rng, chain_count),
            utf8_graphemes_dense_indic_conjunct_(wide_count),  utf8_graphemes_dense_hangul_jamo_(wide_count)};
}

/** @brief The grapheme family's long-range straddling constructions for a given @p gap. */
static std::vector<std::string> utf8_graphemes_straddles_(std::mt19937 &rng, std::size_t gap) {
    std::string regional_parity = utf8_dense_regional_indicators_(rng, gap);
    regional_parity.append("a"); // ASCII tail forces the GB12/13 parity decision after the long run
    std::string zwj_chain = utf8_graphemes_dense_zwj_pictograph_chain_(gap);
    append_codepoint_(zwj_chain, 0x0061); // ASCII break after the chain
    std::string indic_conjunct = utf8_graphemes_dense_indic_conjunct_(gap);
    append_codepoint_(indic_conjunct, 0x0061); // ASCII break after the conjunct
    // A long RI run, then a ZWJ before a final RI: RI...RI ZWJ RI. The ZWJ does NOT bridge two RIs (GB11 bridges only
    // Extended_Pictographic), so this must break after the ZWJ - and the ZWJ resets the GB12/13 RI parity. Straddles
    // the 64-byte window so the parity carry and the post-ZWJ break are exercised across the edge.
    std::string regional_zwj = utf8_dense_regional_indicators_(rng, gap);
    append_codepoint_(regional_zwj, 0x200D);  // ZWJ
    append_codepoint_(regional_zwj, 0x1F1E6); // Regional_Indicator after the ZWJ
    append_codepoint_(regional_zwj, 0x0061);  // ASCII tail
    return {regional_parity, zwj_chain, indic_conjunct, regional_zwj};
}

/** @brief Assemble the grapheme family's differential corpora (motifs + dense + straddle; no seam regressions). */
static utf8_segment_corpora_t utf8_graphemes_corpora_() {
    utf8_segment_corpora_t corpora = {
        "grapheme",
        utf8_graphemes_motifs,
        utf8_graphemes_motifs_count,
        &utf8_graphemes_dense_runs_,
        &utf8_graphemes_straddles_,
        nullptr, // no fixed seam regressions for graphemes
        0,
    };
    return corpora;
}

#pragma endregion // Equivalence

#pragma region Rule coverage

/** @brief Rule-coverage gate: every GB rule motif agrees serial-vs-ISA (at window phases), no rule left unexercised. */
void test_utf8_graphemes_rules() {
    std::printf("  - testing UTF-8 grapheme rule-coverage matrix...\n");

    // One motif per UAX-29 Grapheme_Cluster_Break rule (break or no-break direction).
    utf8_rule_case_t const rule_cases[] = {
        {"GB3", "\r\n"_sv},                                          // CR x LF (no break)
        {"GB4", "\na"_sv},                                           // (Control|CR|LF) / break after LF
        {"GB5", "a\n"_sv},                                           // / (Control|CR|LF) break before LF
        {"GB6", "\xE1\x84\x80\xE1\x85\xA1"_sv},                      // L x (L|V|LV|LVT): Hangul L + V
        {"GB7", "\xEA\xB0\x80\xE1\x85\xA1"_sv},                      // (LV|V) x (V|T): Hangul LV + V
        {"GB8", "\xEA\xB0\x81\xE1\x86\xA8"_sv},                      // (LVT|T) x T: Hangul LVT + T
        {"GB9", "a\xCC\x81"_sv},                                     // x (Extend|ZWJ)
        {"GB9a", "\xE0\xA4\x95\xE0\xA4\xBE"_sv},                     // x SpacingMark: consonant + dependent vowel
        {"GB9b", "\xD8\x80" "a"_sv},                                 // Prepend x: U+0600 then letter
        {"GB9c", "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\x95"_sv},         // Indic conjunct: consonant virama consonant
        {"GB11", "\xF0\x9F\x98\x80\xE2\x80\x8D\xF0\x9F\x98\x81"_sv}, // ExtPict Extend* ZWJ x ExtPict
        {"GB12", "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv},             // sot (RI RI)* RI x RI
        {"GB13", "a\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv},            // [^RI] (RI RI)* RI x RI
        {"GB999", "ab"_sv},                                          // Any / Any (default break)
    };
    // Every Grapheme_Cluster_Break rule id the gate requires a motif for (spec-derived checklist).
    char const *const required_rules[] = {
        "GB3", "GB4", "GB5", "GB6", "GB7", "GB8", "GB9", "GB9a", "GB9b", "GB9c", "GB11", "GB12", "GB13", "GB999",
    };
#if SZ_USE_ICELAKE
    check_utf8_rule_coverage_("grapheme", sz_utf8_graphemes_serial, sz_utf8_graphemes_icelake, rule_cases,
                              sizeof(rule_cases) / sizeof(rule_cases[0]), required_rules,
                              sizeof(required_rules) / sizeof(required_rules[0]));
#endif
}

#pragma endregion // Rule coverage

#pragma region Safety

/** @brief Malformed-input safety of the UTF-8 grapheme kernels (serial / dispatched / icelake). */
void test_utf8_graphemes_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 grapheme kernels...\n");
    check_utf8_segment_safety_("grapheme (serial)", sz_utf8_graphemes_serial);
    check_utf8_segment_safety_("grapheme (dispatched)", sz_utf8_graphemes);
#if SZ_USE_ICELAKE
    check_utf8_segment_safety_("grapheme (icelake)", sz_utf8_graphemes_icelake);
#endif
    std::printf("    grapheme safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/** @brief Serial-vs-ISA grapheme differential over the hardened corpora (high-density + long-range). */
void test_utf8_graphemes_all() {
#if SZ_USE_ICELAKE
    utf8_segment_corpora_t const corpora = utf8_graphemes_corpora_();
    test_utf8_segment_equivalence_(sz_utf8_graphemes_serial, sz_utf8_graphemes_icelake, corpora);
#endif
}

#pragma endregion // Drivers
