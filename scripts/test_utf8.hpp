/**
 *  @brief  Shared harness for the UTF-8 segmentation family tests (words / graphemes / sentences / linewraps).
 *  @file   scripts/test_utf8.hpp
 *  @author Ash Vardanian
 *
 *  Each segmentation family lives in its own translation unit (`test_utf8_<family>.cpp`) and pulls its substrate
 *  from here: the known-answer driver (`check_utf8_segment_unit_`), the malformed-input safety sweep
 *  (`check_utf8_segment_safety_`), and the serial-vs-ISA differential driver (`test_utf8_segment_equivalence_`).
 *  The driver is parameterized over a @ref utf8_segment_corpora_t the family fills with its own motifs and its
 *  high-density / long-range generators, so there is no central family enum — each TU owns its corner cases.
 *
 *  Everything here is `inline` (the header is included into four TUs) and C++11-clean (uses `sz::string_view`,
 *  never `std::string_view`/`std::span`).
 */
#ifndef STRINGZILLA_TEST_UTF8_HPP_
#define STRINGZILLA_TEST_UTF8_HPP_

#include <cassert> // `assert`
#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::fprintf`
#include <cstring> // `std::memcpy`

#include <initializer_list> // `std::initializer_list`
#include <random>           // `std::mt19937`, `std::uniform_int_distribution`
#include <string>           // `std::string`
#include <vector>           // `std::vector`

#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // `sz::string_view`

#include "test_stringzilla.hpp" // `global_random_generator`, `scale_iterations`, `for_each_cacheline_offset_`

namespace sz = ashvardanian::stringzilla;
using sz::scripts::for_each_cacheline_offset_; // alignment sweep used by the safety + differential drivers
using sz::scripts::global_random_generator;    // shared seeded RNG (honors `SZ_TESTS_SEED`)
using sz::scripts::scale_iterations;           // scales fuzz counts by `SZ_TESTS_MULTIPLIER`
using sz::literals::operator""_sv;

#pragma region Shared types

/** @brief Upper bound on the inputs the unit/safety segmentation layers feed (<=70 bytes, so <=70 segments). */
static constexpr sz_size_t utf8_unit_capacity_k = 70;

/** @brief Whether a generated corpus must stay well-formed UTF-8 or may have malformed classes mixed in. */
enum class utf8_corpus_flavor_t { valid_k, malformed_k };

/** @brief One hand-checked segmentation golden vector: the source text and its expected segment list (borrowed). */
struct utf8_unit_case_t {
    sz::string_view text;
    std::initializer_list<sz::string_view> expected;
};

/**
 *  @brief A family's corpora, passed to the shared differential driver (replaces a central family enum). Each
 *         family TU fills this with its own corner-case motifs and its high-density / long-range generators.
 *
 *  @param family_name   Human label printed by the driver (e.g. "word").
 *  @param motifs         Pointer to the family's corner-case motif inputs (sprinkled into the random corpus).
 *  @param motif_count    Number of @p motifs.
 *  @param dense_runs     Builds the family's high-density homogeneous runs (each spans several 64-byte windows).
 *  @param straddles      Builds the family's long-range constructions for a given straddling @p gap.
 *  @param regressions    Optional fixed hand-found regression inputs (each checked serial-vs-ISA directly); the
 *                        trailing pair may be omitted by families that have none (zero-initialized to "none").
 *  @param regression_count Number of @p regressions.
 */
struct utf8_segment_corpora_t {
    char const *family_name;
    sz::string_view const *motifs;
    std::size_t motif_count;
    std::vector<std::string> (*dense_runs)(std::mt19937 &rng);
    std::vector<std::string> (*straddles)(std::mt19937 &rng, std::size_t gap);
    sz::string_view const *regressions;
    std::size_t regression_count;
};

#pragma endregion // Shared types

#pragma region Shared helpers

/** @brief Prints one labeled hex dump line to `stderr`; used by the malformed-input safety sweep. */
inline void print_utf8_test_bytes_(char const *label, char const *bytes, std::size_t length) {
    std::fprintf(stderr, "  %s (%zu bytes): ", label, length);
    for (std::size_t index = 0; index < length; ++index) std::fprintf(stderr, "%02X ", (unsigned char)bytes[index]);
    std::fprintf(stderr, "\n");
}

/** @brief Drive any boundary finder one-shot over @p text and return the emitted segments as byte strings. */
inline std::vector<std::string> utf8_segments_(sz_utf8_segmenter_t finder, sz::string_view text) {
    std::vector<sz_size_t> starts(text.size() + 1), lengths(text.size() + 1);
    sz_size_t consumed = 0;
    sz_size_t const count = finder(text.data(), text.size(), starts.data(), lengths.data(), text.size() + 1, &consumed);
    std::vector<std::string> segments;
    for (sz_size_t index = 0; index != count; ++index)
        segments.push_back(std::string(text.data() + starts[index], lengths[index]));
    return segments;
}

/**
 *  @brief SMP/astral fixtures the pure-BMP random corpora miss (Regional-Indicator pairs, ZWJ sequences, lone
 *         astral codepoints). Borrowed `sz::string_view` table — reused by every family's safety + differential.
 */
static sz::string_view const utf8_astral_fixtures[] = {
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"_sv,                                         // RI(U) RI(S) flag pair
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7"_sv,         // two flags
    "\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA7"_sv, // family ZWJ sequence
    "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD"_sv,                                         // emoji + skin-tone modifier
    "\xF0\x9F\x98\x80\xF0\x9F\x98\x81"_sv,                                         // two astral emoji
    "a\xF0\x9F\x98\x80\x62"_sv,                                                    // ASCII around an astral codepoint
    "\xF0\x90\x80\x80\xF0\x90\x80\x81"_sv,                                         // U+10000 U+10001 plain astral pair
    "\xF0\x9F\x87\xBA\x61\xF0\x9F\x87\xB8"_sv,                                     // RI ASCII RI - RI parity must reset
};
static constexpr std::size_t utf8_astral_fixtures_count = sizeof(utf8_astral_fixtures) /
                                                          sizeof(utf8_astral_fixtures[0]);

/** @brief Append one codepoint to @p text as UTF-8 via `sz_rune_export` (silently skips invalid runes). */
inline void append_codepoint_(std::string &text, sz_rune_t codepoint) {
    sz_u8_t bytes[4];
    sz_rune_length_t const length = sz_rune_export(codepoint, bytes);
    if (length == sz_utf8_invalid_k) return;
    text.append((char const *)bytes, (std::size_t)length);
}

/** @brief @p link_count Regional-Indicator codepoints in a row (GB12/13 / WB15/16 pairing parity). Cross-family. */
inline std::string utf8_dense_regional_indicators_(std::mt19937 &rng, std::size_t link_count) {
    std::string text;
    std::uniform_int_distribution<sz_rune_t> indicator(0x1F1E6, 0x1F1FF); // U+1F1E6..U+1F1FF
    for (std::size_t index = 0; index != link_count; ++index) append_codepoint_(text, indicator(rng));
    return text;
}

/**
 *  @brief Append one randomly chosen malformed-UTF-8 class to @p text: overlong encodings, surrogates, lone
 *         continuations, invalid leads, truncated tails, out-of-range leads, and noncharacters. Drawn from @p rng.
 */
inline void append_malformed_class_(std::string &text, std::mt19937 &rng) {
    static char const *const malformed[] = {
        "\xC0\x80",         // overlong 2-byte encoding of NUL
        "\xC1\xBF",         // overlong 2-byte encoding of U+007F
        "\xE0\x80\x80",     // overlong 3-byte encoding of NUL
        "\xE0\x9F\xBF",     // overlong 3-byte encoding of U+07FF
        "\xF0\x80\x80\x80", // overlong 4-byte encoding of NUL
        "\xF0\x8F\xBF\xBF", // overlong 4-byte encoding of U+FFFF
        "\xED\xA0\x80",     // surrogate U+D800 in 3-byte form
        "\xED\xBF\xBF",     // surrogate U+DFFF in 3-byte form
        "\x80",             // lone continuation (low)
        "\xBF",             // lone continuation (high)
        "\xFE",             // invalid lead 0xFE
        "\xFF",             // invalid lead 0xFF
        "\xC2",             // truncated 2-byte tail (mid-string)
        "\xE2\x80",         // truncated 3-byte tail (mid-string)
        "\xF0\x9F\x98",     // truncated 4-byte tail (mid-string)
        "\xF5\x80\x80\x80", // out-of-range lead >U+10FFFF (0xF5)
        "\xEF\xB7\x90",     // noncharacter U+FDD0
        "\xEF\xB7\xAF",     // noncharacter U+FDEF
        "\xEF\xBF\xBE",     // plane-ender noncharacter U+FFFE
        "\xEF\xBF\xBF",     // plane-ender noncharacter U+FFFF
    };
    std::size_t const count = sizeof(malformed) / sizeof(malformed[0]);
    std::uniform_int_distribution<std::size_t> pick(0, count - 1);
    text.append(malformed[pick(rng)]);
}

/**
 *  @brief Apply structural mutation passes to @p text: NUL injection (10%), random byte-swap (10%), a
 *         truncate-last-codepoint pass, and a stray-continuation insertion. All randomness flows through @p rng.
 */
inline void apply_mutation_passes_(std::string &text, std::mt19937 &rng) {
    if (text.empty()) return;
    std::uniform_int_distribution<std::size_t> byte_index(0, text.size() - 1);
    std::size_t const to_corrupt = text.size() / 10;
    for (std::size_t index = 0; index != to_corrupt; ++index) text[byte_index(rng)] = '\0';
    for (std::size_t index = 0; index != to_corrupt; ++index) std::swap(text[byte_index(rng)], text[byte_index(rng)]);

    // Truncate the last codepoint: drop a 1-3 byte trailing run so a multi-byte sequence loses its tail.
    std::uniform_int_distribution<std::size_t> truncate_distribution(1, 3);
    std::size_t const truncate_by = std::min((std::size_t)truncate_distribution(rng), text.size());
    text.resize(text.size() - truncate_by);

    // Insert a stray continuation byte at a random position, breaking codepoint alignment.
    if (!text.empty()) {
        std::uniform_int_distribution<std::size_t> insert_at(0, text.size());
        std::uniform_int_distribution<int> continuation(0x80, 0xBF);
        text.insert(text.begin() + insert_at(rng), (char)continuation(rng));
    }
}

/**
 *  @brief Build a random UTF-8 corpus from a weighted corner-case alphabet (all four byte lengths, byte-length
 *         boundary codepoints, BOM / ZWJ / VS16, the astral fixtures) plus the family's @p motifs, until at least
 *         @p min_length bytes. The @p malformed_k flavor mixes malformed classes in; @p valid_k stays well-formed.
 */
inline std::string utf8_random_segmentation_corpus_(std::size_t min_length, utf8_corpus_flavor_t flavor,
                                                    sz::string_view const *motifs, std::size_t motif_count,
                                                    std::mt19937 &rng) {
    static sz_rune_t const boundary_codepoints[] = {
        0x007F, 0x0080, 0x07FF, 0x0800,  0xFFFF, 0x10000, 0x10FFFF, // byte-length boundaries
        0xFEFF,                                                     // BOM U+FEFF
        0x200D,                                                     // ZWJ U+200D
        0xFE0F,                                                     // VS16 U+FE0F
        0x0041, 0x00DF, 0x4E2D, 0x1F600,                            // one codepoint per byte length
    };
    static char const *const snippets[] = {
        "a",
        "Hello, world! ",
        "don't ",
        "3,14 ",
        "\xE4\xBD\xA0\xE5\xA5\xBD",
        "\r\n",
        "one. two. ",
        "A! B? ",
        "e\xCC\x81",
        "\xEA\xB0\x80",
        "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7",
        "line\nbreak ",
        "soft wrap here ",
        "\xE2\x80\xA8",
        "\xE2\x80\xA9",
    };
    std::size_t const boundary_count = sizeof(boundary_codepoints) / sizeof(boundary_codepoints[0]);
    std::size_t const snippet_count = sizeof(snippets) / sizeof(snippets[0]);

    // Weighted choice: snippets are the bulk, boundary codepoints / astral / family motifs sprinkled in, and
    // (for the malformed flavor) a malformed class injected with a small probability.
    std::uniform_int_distribution<int> weight(0, 99);
    std::uniform_int_distribution<std::size_t> boundary_pick(0, boundary_count - 1);
    std::uniform_int_distribution<std::size_t> snippet_pick(0, snippet_count - 1);
    std::uniform_int_distribution<std::size_t> astral_pick(0, utf8_astral_fixtures_count - 1);
    std::uniform_int_distribution<std::size_t> motif_pick(0, motif_count ? motif_count - 1 : 0);

    std::string text;
    while (text.size() < min_length) {
        int const roll = weight(rng);
        if (flavor == utf8_corpus_flavor_t::malformed_k && roll < 10) append_malformed_class_(text, rng);
        else if (roll < 30) append_codepoint_(text, boundary_codepoints[boundary_pick(rng)]);
        else if (roll < 45) {
            sz::string_view const fixture = utf8_astral_fixtures[astral_pick(rng)];
            text.append(fixture.data(), fixture.size());
        }
        else if (roll < 65 && motif_count) {
            sz::string_view const motif = motifs[motif_pick(rng)];
            text.append(motif.data(), motif.size());
        }
        else text.append(snippets[snippet_pick(rng)]);
    }
    return text;
}

/**
 *  @brief Metamorphic invariant: the emitted segments must exactly tile `[0, length)` with no gaps or overlaps.
 *         For @p valid_k corpora every non-empty segment additionally starts on a codepoint boundary; malformed
 *         input may legitimately split mid-byte, so that alignment check is gated on the flavor.
 */
inline void assert_segment_invariants_(std::vector<sz_size_t> const &offsets, std::vector<sz_size_t> const &lengths,
                                       sz_cptr_t data, sz_size_t length, utf8_corpus_flavor_t flavor) {
    sz_size_t running_cursor = 0;
    for (std::size_t index = 0; index != offsets.size(); ++index) {
        assert(offsets[index] == running_cursor && "segments do not tile the input contiguously");
        if (flavor == utf8_corpus_flavor_t::valid_k && lengths[index] != 0)
            assert((((sz_u8_t)data[offsets[index]]) & 0xC0u) != 0x80u && "segment starts mid-codepoint");
        running_cursor += lengths[index];
    }
    assert(running_cursor == length && "segments do not cover the whole input");
}

#pragma endregion // Shared helpers

#pragma region Unit driver

/**
 *  @brief Drive one segmentation backend (dispatched / serial / ISA) over the hand-checked golden vectors,
 *         asserting the forward segmentation matches. The caller invokes it once per backend (so a wrong constant
 *         shared by serial and SIMD is still caught against external ground truth).
 */
inline void check_utf8_segment_unit_(char const *family, sz_utf8_segmenter_t forward, utf8_unit_case_t const *cases,
                                     std::size_t case_count) {
    for (std::size_t case_index = 0; case_index != case_count; ++case_index) {
        utf8_unit_case_t const &one = cases[case_index];
        std::vector<std::string> const got = utf8_segments_(forward, one.text);
        bool matches = got.size() == one.expected.size();
        std::size_t segment_index = 0;
        for (sz::string_view const expected_segment : one.expected) {
            if (!matches) break;
            matches = got[segment_index].size() == expected_segment.size() &&
                      std::memcmp(got[segment_index].data(), expected_segment.data(), expected_segment.size()) == 0;
            ++segment_index;
        }
        assert(matches && family && "segment forward != golden");
    }
}

#pragma endregion // Unit driver

#pragma region Safety sweep

/**
 *  @brief Feed the full adversarial-byte battery (4 named shapes, the astral fixtures, all 256 single bytes, all
 *         65 536 byte pairs, and random garbage at every sub-cache-line offset) to @p callback as `(bytes, length)`.
 *         Factored so each family's `_safety` runs one shared sweep instead of three copies of the 65 536-pair loop.
 */
template <typename callback_type_>
inline void for_each_adversarial_utf8_input_(std::mt19937 &rng, std::size_t random_input_count,
                                             callback_type_ &&callback) {
    char input[utf8_unit_capacity_k];

    // Named adversarial shapes.
    callback("\x80", (std::size_t)1);              // lone continuation byte
    callback("\xC0\x80", (std::size_t)2);          // overlong encoding of NUL
    callback("\xED\xA0\x80", (std::size_t)3);      // surrogate-encoded codepoint (U+D800)
    callback("hello\xF0\x9F\x98", (std::size_t)8); // truncated 4-byte sequence at the very end

    // The SMP/astral fixtures the random corpora miss (RI parity, ZWJ, astral).
    for (std::size_t index = 0; index != utf8_astral_fixtures_count; ++index)
        callback(utf8_astral_fixtures[index].data(), utf8_astral_fixtures[index].size());

    // All 256 single bytes.
    for (std::size_t byte = 0; byte != 256; ++byte) { input[0] = (char)byte, callback(input, (std::size_t)1); }

    // All 65,536 byte pairs.
    for (std::size_t first_byte = 0; first_byte != 256; ++first_byte)
        for (std::size_t second_byte = 0; second_byte != 256; ++second_byte) {
            input[0] = (char)first_byte, input[1] = (char)second_byte;
            callback(input, (std::size_t)2);
        }

    // Random garbage at every sub-cache-line alignment.
    std::uniform_int_distribution<std::size_t> length_distribution(1, utf8_unit_capacity_k);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    for (std::size_t iteration = 0; iteration != random_input_count; ++iteration) {
        std::size_t const input_length = length_distribution(rng);
        for (std::size_t index = 0; index != input_length; ++index) input[index] = (char)byte_distribution(rng);
        for_each_cacheline_offset_(input_length, [&](sz_ptr_t buffer, std::size_t /*offset*/) {
            std::memcpy(buffer, input, input_length);
            callback((char const *)buffer, input_length);
        });
    }
}

/**
 *  @brief Feed the adversarial battery through one boundary finder, asserting it survives, every emitted segment is
 *         in-bounds, and `bytes_consumed <= length`. The caller invokes it once per backend.
 */
inline void check_utf8_segment_safety_(char const *family, sz_utf8_segmenter_t forward,
                                       std::size_t random_input_count = scale_iterations(10000)) {
    sz_size_t offsets[utf8_unit_capacity_k + 1], lengths[utf8_unit_capacity_k + 1];
    auto probe = [&](char const *input, std::size_t input_length) {
        sz_size_t bytes_consumed = 0;
        sz_size_t const found = forward(input, (sz_size_t)input_length, offsets, lengths,
                                        (sz_size_t)(utf8_unit_capacity_k + 1), &bytes_consumed);
        assert(bytes_consumed <= input_length && "segment finder consumed past the input");
        for (sz_size_t index = 0; index != found; ++index) {
            if (offsets[index] + lengths[index] <= input_length) continue;
            std::fprintf(stderr, "%s emitted out-of-bounds segment (offset=%zu len=%zu, input=%zu)\n", family,
                         (std::size_t)offsets[index], (std::size_t)lengths[index], input_length);
            print_utf8_test_bytes_("input", input, input_length);
            assert(false && "segment finder emitted a span outside the input");
        }
    };
    std::mt19937 &rng = global_random_generator();
    for_each_adversarial_utf8_input_(rng, random_input_count, probe);
}

#pragma endregion // Safety sweep

#pragma region Differential driver

/**
 *  @brief Differential fuzz of any ISA finder against the serial reference, enumerating segments via repeated
 *         streaming calls across the 6 capacities {len+64,65,63,16,3,1}, over random UTF-8 (valid, mutated, and
 *         malformed), the family's high-density runs, its long-range straddles (gap-swept + phase-shifted), and an
 *         alignment sweep. Asserts tiling/alignment invariants and capacity-independence.
 */
inline void test_utf8_segment_equivalence_(sz_utf8_segmenter_t reference, sz_utf8_segmenter_t candidate,
                                           utf8_segment_corpora_t const &corpora,
                                           std::size_t iterations = scale_iterations(2000)) {
    std::printf("  - testing %s serial-vs-ISA differential...\n", corpora.family_name);

    // Hoisted batch scratch reused across every streaming call (resized once per `compare` to the max capacity).
    std::vector<sz_size_t> batch_offsets, batch_lengths;

    // Enumerate every segment via repeated streaming calls (resuming from `bytes_consumed`), so a small `capacity`
    // exercises the window loop, its resume logic, and the serial tail handoff.
    auto enumerate = [&](sz_utf8_segmenter_t matcher, sz_cptr_t data, sz_size_t length, sz_size_t capacity,
                         std::vector<sz_size_t> &offsets, std::vector<sz_size_t> &lengths) {
        offsets.clear(), lengths.clear();
        sz_size_t base = 0;
        while (base < length) {
            sz_size_t consumed = 0;
            sz_size_t const got = matcher(data + base, length - base, batch_offsets.data(), batch_lengths.data(),
                                          capacity, &consumed);
            for (sz_size_t index = 0; index != got; ++index)
                offsets.push_back(base + batch_offsets[index]), lengths.push_back(batch_lengths[index]);
            if (got == 0 && consumed == 0) break;
            base += consumed;
        }
    };

    std::vector<sz_size_t> reference_offsets, reference_lengths, candidate_offsets, candidate_lengths;
    std::vector<sz_size_t> first_offsets, first_lengths;
    auto check = [&](sz_cptr_t data, sz_size_t length, utf8_corpus_flavor_t flavor) {
        sz_size_t const capacities[] = {length + 64, 65, 63, 16, 3, 1};
        sz_size_t max_capacity = 1;
        for (sz_size_t capacity : capacities) max_capacity = std::max(max_capacity, capacity);
        if (batch_offsets.size() < max_capacity) batch_offsets.resize(max_capacity), batch_lengths.resize(max_capacity);
        bool first = true;
        for (sz_size_t capacity : capacities) {
            if (capacity == 0) continue;
            enumerate(reference, data, length, capacity, reference_offsets, reference_lengths);
            enumerate(candidate, data, length, capacity, candidate_offsets, candidate_lengths);
            assert(reference_offsets == candidate_offsets && reference_lengths == candidate_lengths &&
                   "segment offsets/lengths mismatch");
            assert_segment_invariants_(candidate_offsets, candidate_lengths, data, length, flavor);
            // Resume-identity / capacity-independence: every capacity reproduces the capacity[0] enumeration.
            if (first) first_offsets = candidate_offsets, first_lengths = candidate_lengths, first = false;
            else
                assert(candidate_offsets == first_offsets && candidate_lengths == first_lengths &&
                       "capacity-dependent segmentation");
        }
    };

    // Fixed regression inputs (hand-found seam bugs, each > one window): must agree serial-vs-ISA exactly.
    for (std::size_t index = 0; index != corpora.regression_count; ++index)
        check((sz_cptr_t)corpora.regressions[index].data(), corpora.regressions[index].size(),
              utf8_corpus_flavor_t::valid_k);

    std::mt19937 &rng = global_random_generator();
    for (std::size_t iteration = 0; iteration != iterations; ++iteration) {
        // Valid corpus: serial-vs-ISA agreement plus tiling/alignment invariants on well-formed input.
        std::string const valid = utf8_random_segmentation_corpus_(400, utf8_corpus_flavor_t::valid_k, corpora.motifs,
                                                                   corpora.motif_count, rng);
        check(valid.data(), valid.size(), utf8_corpus_flavor_t::valid_k);

        // Multi-window size tier: on a fraction of iterations, blow the corpus up to ~4096 bytes so the window loop
        // and its carry are exercised across dozens of 64-byte windows, not just six or seven.
        if ((iteration & 0x7u) == 0) {
            std::string const wide = utf8_random_segmentation_corpus_(4096, utf8_corpus_flavor_t::valid_k,
                                                                      corpora.motifs, corpora.motif_count, rng);
            check(wide.data(), wide.size(), utf8_corpus_flavor_t::valid_k);
        }

        // High-density homogeneous runs: one long single-rule blob per family, stressing per-window parity/carry.
        if (corpora.dense_runs)
            for (std::string const &dense : corpora.dense_runs(rng))
                check(dense.data(), dense.size(), utf8_corpus_flavor_t::valid_k);

        // Long-range window-straddling: the decisive rule context sits well past one window from the boundary. Sweep
        // the gap so the seam lands at every phase 0..63 and crosses >=2/>=3 windows, with a leading ASCII filler
        // that additionally shifts the content phase relative to the window.
        if (corpora.straddles) {
            static std::size_t const straddle_gaps[] = {64, 65, 95, 96, 127, 128, 129, 191, 192, 256};
            std::uniform_int_distribution<std::size_t> filler_length(0, 63);
            for (std::size_t gap : straddle_gaps) {
                std::string const filler(filler_length(rng), 'x');
                for (std::string const &straddle : corpora.straddles(rng, gap)) {
                    std::string const shifted = filler + straddle;
                    check(shifted.data(), shifted.size(), utf8_corpus_flavor_t::valid_k);
                }
            }
        }

        // Mutated valid corpus: NUL/swap/truncate/stray-continuation passes break codepoints, yet both backends must
        // still agree (alignment invariant relaxed via the malformed flavor).
        std::string mutated = valid;
        apply_mutation_passes_(mutated, rng);
        check(mutated.data(), mutated.size(), utf8_corpus_flavor_t::malformed_k);

        // Malformed corpus: malformed classes mixed directly into otherwise-valid text.
        std::string const malformed = utf8_random_segmentation_corpus_(400, utf8_corpus_flavor_t::malformed_k,
                                                                       corpora.motifs, corpora.motif_count, rng);
        check(malformed.data(), malformed.size(), utf8_corpus_flavor_t::malformed_k);
    }

    // Alignment sweep: drive a fixed corpus at every sub-cache-line offset so the load alignment is swept.
    std::string const probe = utf8_random_segmentation_corpus_(256, utf8_corpus_flavor_t::valid_k, corpora.motifs,
                                                               corpora.motif_count, rng);
    for_each_cacheline_offset_(probe.size(), [&](sz_ptr_t buffer, std::size_t /*offset*/) {
        std::memcpy(buffer, probe.data(), probe.size());
        check((sz_cptr_t)buffer, probe.size(), utf8_corpus_flavor_t::valid_k);
    });
}

#pragma endregion // Differential driver

#endif // STRINGZILLA_TEST_UTF8_HPP_
