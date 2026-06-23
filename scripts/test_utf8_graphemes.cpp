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

/** @brief A pictograph ZWJ chain @p link_count deep, every other link followed by VS16 (GB11), into @p out (cleared). */
static void utf8_graphemes_dense_zwj_pictograph_chain_(std::string &out, std::size_t link_count) {
    out.clear();
    static sz_rune_t const pictographs[] = {0x1F468, 0x1F469, 0x1F467, 0x1F466}; // man, woman, girl, boy
    append_codepoint_(out, pictographs[0]);
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(out, 0x200D);                          // ZWJ
        append_codepoint_(out, pictographs[(index + 1) & 0x3u]); // next pictograph
        if (index & 1u) append_codepoint_(out, 0xFE0F);          // VS16 on alternating links
    }
}

/** @brief One base letter followed by @p link_count combining marks (GB9 Extend run), into @p out (cleared first). */
static void utf8_graphemes_dense_combining_marks_(std::string &out, std::size_t link_count) {
    out.clear();
    static sz_rune_t const marks[] = {0x0301, 0x0300, 0x0308, 0x0327, 0x0323}; // acute, grave, diaeresis, cedilla, dot
    append_codepoint_(out, 0x0061);                                            // base 'a'
    for (std::size_t index = 0; index != link_count; ++index) append_codepoint_(out, marks[index % 5u]);
}

/** @brief @p link_count emoji each followed by a skin-tone modifier drawn from @p rng (GB9 Extend), into @p out. */
static void utf8_graphemes_dense_skin_tone_run_(std::string &out, std::mt19937 &rng, std::size_t link_count) {
    out.clear();
    std::uniform_int_distribution<sz_rune_t> modifier(0x1F3FB, 0x1F3FF); // U+1F3FB..U+1F3FF
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(out, 0x1F44D); // thumbs up
        append_codepoint_(out, modifier(rng));
    }
}

/** @brief @p link_count Indic consonant+virama conjunct links (GB9c InCB Consonant Linker chains), into @p out. */
static void utf8_graphemes_dense_indic_conjunct_(std::string &out, std::size_t link_count) {
    out.clear();
    append_codepoint_(out, 0x0915); // Devanagari KA
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(out, 0x094D); // virama (Linker)
        append_codepoint_(out, 0x0915); // KA
    }
}

/** @brief @p link_count Hangul L/V/T jamo triples (GB6/7/8 jamo composition runs), into @p out (cleared first). */
static void utf8_graphemes_dense_hangul_jamo_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(out, 0x1100); // L (Choseong Kiyeok)
        append_codepoint_(out, 0x1161); // V (Jungseong A)
        append_codepoint_(out, 0x11A8); // T (Jongseong Kiyeok)
    }
}

/** @brief Stream the grapheme family's high-density homogeneous runs (each spans several 64-byte windows) to @p sink. */
static void utf8_graphemes_dense_runs_(std::mt19937 &rng, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    std::uniform_int_distribution<std::size_t> wide(60, 220);
    std::uniform_int_distribution<std::size_t> chain(20, 80);
    std::size_t const wide_count = wide(rng);
    std::size_t const chain_count = chain(rng);
    utf8_dense_regional_indicators_(scratch, rng, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_zwj_pictograph_chain_(scratch, chain_count), sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_combining_marks_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_skin_tone_run_(scratch, rng, chain_count), sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_indic_conjunct_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_hangul_jamo_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
}

/** @brief Stream the grapheme family's long-range straddling constructions for a given @p gap to @p sink. */
static void utf8_graphemes_straddles_(std::mt19937 &rng, std::size_t gap, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    utf8_dense_regional_indicators_(scratch, rng, gap);
    scratch.append("a"); // ASCII tail forces the GB12/13 parity decision after the long run
    sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_zwj_pictograph_chain_(scratch, gap);
    append_codepoint_(scratch, 0x0061); // ASCII break after the chain
    sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_indic_conjunct_(scratch, gap);
    append_codepoint_(scratch, 0x0061); // ASCII break after the conjunct
    sink(context, scratch.data(), scratch.size());
    // A long RI run, then a ZWJ before a final RI: RI...RI ZWJ RI. The ZWJ does NOT bridge two RIs (GB11 bridges only
    // Extended_Pictographic), so this must break after the ZWJ - and the ZWJ resets the GB12/13 RI parity. Straddles
    // the 64-byte window so the parity carry and the post-ZWJ break are exercised across the edge.
    utf8_dense_regional_indicators_(scratch, rng, gap);
    append_codepoint_(scratch, 0x200D);  // ZWJ
    append_codepoint_(scratch, 0x1F1E6); // Regional_Indicator after the ZWJ
    append_codepoint_(scratch, 0x0061);  // ASCII tail
    sink(context, scratch.data(), scratch.size());
}

/** @brief Grapheme-biased random-corpus snippets: emoji ZWJ sequences, combining marks, skin-tone, jamo, conjuncts. */
static char const *const utf8_graphemes_snippets[] = {
    "\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9", // woman ZWJ woman (GB11 emoji-ZWJ sequence)
    "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD",             // thumbs-up + skin-tone modifier (GB9 Extend)
    "e\xCC\x81",                                    // base 'e' + combining acute (GB9)
    "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8",         // Hangul L+V+T jamo (GB6/7/8)
    "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7",         // Devanagari consonant + virama + consonant (GB9c)
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8",             // regional-indicator pair (GB12/13)
    "\r\n",                                         // CRLF (GB3)
    "abc",                                          // plain ASCII (GB999)
};

/** @brief Grapheme family alphabet: weights bias toward astral/motif clusters (GB9/9c/11/12/13). */
static utf8_corpus_alphabet_t const utf8_graphemes_alphabet = {
    utf8_graphemes_snippets,
    sizeof(utf8_graphemes_snippets) / sizeof(utf8_graphemes_snippets[0]),
    utf8_default_boundary_codepoints,
    sizeof(utf8_default_boundary_codepoints) / sizeof(utf8_default_boundary_codepoints[0]),
    {35, 15, 25, 20, 5}, // snippet, boundary, astral, motif, malformed
};

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
        &utf8_graphemes_alphabet,
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
        {"GB3", utf8_rule_joins_k, "\r\n"_sv},                                  // CR x LF (no break)
        {"GB4", utf8_rule_breaks_k, "\na"_sv},                                  // (Control|CR|LF) / break after LF
        {"GB5", utf8_rule_breaks_k, "a\n"_sv},                                  // / (Control|CR|LF) break before LF
        {"GB6", utf8_rule_joins_k, "\xE1\x84\x80\xE1\x85\xA1"_sv},              // L x (L|V|LV|LVT): Hangul L + V
        {"GB7", utf8_rule_joins_k, "\xEA\xB0\x80\xE1\x85\xA1"_sv},              // (LV|V) x (V|T): Hangul LV + V
        {"GB8", utf8_rule_joins_k, "\xEA\xB0\x81\xE1\x86\xA8"_sv},              // (LVT|T) x T: Hangul LVT + T
        {"GB9", utf8_rule_joins_k, "a\xCC\x81"_sv},                             // x (Extend|ZWJ)
        {"GB9a", utf8_rule_joins_k, "\xE0\xA4\x95\xE0\xA4\xBE"_sv},             // x SpacingMark: consonant + vowel
        {"GB9b", utf8_rule_joins_k, "\xD8\x80" "a"_sv},                         // Prepend x: U+0600 then letter
        {"GB9c", utf8_rule_joins_k, "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\x95"_sv}, // Indic conjunct: cons virama cons
        {"GB11", utf8_rule_joins_k, "\xF0\x9F\x98\x80\xE2\x80\x8D\xF0\x9F\x98\x81"_sv}, // ExtPict Extend* ZWJ x ExtPict
        {"GB12", utf8_rule_joins_k, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv},             // sot (RI RI)* RI x RI
        {"GB13", utf8_rule_joins_k, "a\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv},            // [^RI] (RI RI)* RI x RI
        {"GB999", utf8_rule_breaks_k, "ab"_sv},                                         // Any / Any (default break)
        // Opposite-direction motifs (E1): the same rule firing the other way.
        {"GB11", utf8_rule_breaks_k, "\xF0\x9F\x98\x80z"_sv}, // ExtPict then ASCII: no ZWJ bridge -> break
        {"GB12", utf8_rule_breaks_k,
         "\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA"_sv}, // 3 RI: break after the pair
        {"GB6", utf8_rule_breaks_k, "\xE1\x84\x80z"_sv},         // Hangul L then ASCII: not L|V|LV|LVT -> break
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
