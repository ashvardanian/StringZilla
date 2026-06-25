/**
 *  @brief  Shared harness for the UTF-8 segmentation family tests (words / graphemes / sentences / linewraps).
 *  @file   scripts/test_utf8.hpp
 *  @author Ash Vardanian
 *
 *  Each segmentation family lives in its own translation unit (`test_utf8_<family>.cpp`) and pulls its substrate
 *  from here. The three layers are:
 *    - known-answer goldens (`check_utf8_segment_unit_`), compared lazily against expected literals;
 *    - the malformed-input safety sweep (`check_utf8_segment_safety_`);
 *    - the serial-vs-ISA differential (`test_utf8_segment_equivalence_`), a short orchestrator over named
 *      deterministic and randomized stressors.
 *
 *  Two segmentation backends are compared by STREAMING them in lockstep through @ref utf8_segment_cursor_t (a
 *  fixed-capacity batch pull with `bytes_consumed` resume) and asserting each emitted segment agrees — no
 *  `std::vector<std::string>` is ever materialized, and the comparison stops at the first divergence with a full
 *  reproduction dump (seed + iteration + stressor + capacity + hex). Random corpora are produced from a per-family
 *  @ref utf8_corpus_alphabet_t (named weighted categories via `std::discrete_distribution`), and the family's
 *  high-density / long-range generators emit runs through a @ref utf8_run_sink_t callback rather than returning
 *  containers.
 *
 *  Everything here is `inline` (the header is included into four TUs) and C++11-clean (uses `sz::string_view`,
 *  never `std::string_view`/`std::span`).
 */
#ifndef STRINGZILLA_TEST_UTF8_HPP_
#define STRINGZILLA_TEST_UTF8_HPP_

#include <cassert> // `assert`
#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::fprintf`
#include <cstring> // `std::memcpy`, `std::strcmp`, `std::strlen`

#include <initializer_list> // `std::initializer_list`
#include <random>           // `std::mt19937`, `std::uniform_int_distribution`, `std::discrete_distribution`
#include <string>           // `std::string`

#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // `sz::string_view`

#include "stringzilla.hpp" // `global_random_generator`, `scale_iterations`, `for_each_cacheline_offset_`

namespace sz = ashvardanian::stringzilla;
using sz::scripts::for_each_cacheline_offset_; // alignment sweep used by the safety + differential drivers
using sz::scripts::global_random_generator;    // shared seeded RNG (honors `SZ_TESTS_SEED`)
using sz::scripts::global_random_seed;         // the active seed (printed in failure repro)
using sz::scripts::scale_iterations;           // scales fuzz counts by `SZ_TESTS_MULTIPLIER`
using sz::literals::operator""_sv;

#pragma region Shared constants and types

/** @brief The 64-byte window every SIMD backend processes; phase sweeps and gaps are sized from it. */
static constexpr sz_size_t utf8_window_k = 64;
/** @brief Upper bound on the inputs the unit/safety layers feed (<=70 bytes, so <=70 segments). */
static constexpr sz_size_t utf8_unit_capacity_k = 70;
/** @brief Fixed lazy-cursor batch buffer; also the largest swept caller capacity (the "whole-input" pass). */
static constexpr sz_size_t utf8_segment_batch_k = 128;

/** @brief Whether a generated corpus must stay well-formed UTF-8 or may have malformed classes mixed in. */
enum class utf8_corpus_flavor_t { valid_k, malformed_k };

/** @brief One hand-checked segmentation golden vector: the source text and its expected segment list (borrowed). */
struct utf8_unit_case_t {
    sz::string_view text;
    std::initializer_list<sz::string_view> expected;
};

/** @brief Named weighted categories the random corpus draws from (a `std::discrete_distribution` over the weights). */
enum utf8_corpus_category_t {
    utf8_corpus_snippet_k = 0,   /**< family-biased snippet strings */
    utf8_corpus_boundary_k,      /**< single byte-length-boundary / format codepoints */
    utf8_corpus_astral_k,        /**< the shared SMP/astral fixtures (RI / ZWJ / emoji) */
    utf8_corpus_motif_k,         /**< the family's own corner-case motifs */
    utf8_corpus_malformed_k,     /**< a malformed UTF-8 class (only for the malformed flavor) */
    utf8_corpus_category_count_k /**< category count (weight-array length) */
};

/** @brief A family's random-corpus alphabet: its own snippet table, boundary codepoints, and category weights so
 *         each family biases generation toward its own rules instead of one shared grab-bag. */
struct utf8_corpus_alphabet_t {
    char const *const *snippets;
    std::size_t snippet_count;
    sz_rune_t const *boundary_codepoints;
    std::size_t boundary_count;
    int category_weights[utf8_corpus_category_count_k]; // snippet / boundary / astral / motif / malformed
};

/** @brief Sink invoked once per generated corpus run with its bytes (a reused scratch buffer the generator owns),
 *         so the high-density / long-range generators stream runs instead of returning `std::vector<std::string>`. */
typedef void (*utf8_run_sink_t)(void *context, sz_cptr_t data, sz_size_t length);

/**
 *  @brief A family's corpora, passed to the shared differential driver. Each family TU fills this with its own
 *         corner-case motifs, its high-density / long-range run generators (visitor style), and its alphabet.
 */
struct utf8_segment_corpora_t {
    char const *family_name; /**< human label printed by the driver (e.g. "word") */
    sz::string_view const *motifs;
    std::size_t motif_count;
    /** Streams the family's high-density homogeneous runs (each spans several 64-byte windows) to @p sink. */
    void (*dense_runs)(std::mt19937 &rng, utf8_run_sink_t sink, void *context);
    /** Streams the family's long-range straddling constructions for a given @p gap to @p sink. */
    void (*straddles)(std::mt19937 &rng, std::size_t gap, utf8_run_sink_t sink, void *context);
    sz::string_view const *regressions; /**< optional fixed hand-found regression inputs (may be null) */
    std::size_t regression_count;
    utf8_corpus_alphabet_t const *alphabet; /**< per-family random-corpus alphabet (null -> the shared default) */
};

/** @brief Identifies a fuzz case for replay: which family, which stressor, the iteration, capacity and flavor. */
struct utf8_repro_t {
    char const *family;
    char const *stressor;
    std::size_t iteration;
    sz_size_t capacity;
    utf8_corpus_flavor_t flavor;
};

#pragma endregion // Shared constants and types

#pragma region Shared helpers

/** @brief Prints one labeled hex dump line to `stderr`; used by the safety sweep and the divergence repro. */
inline void print_utf8_test_bytes_(char const *label, char const *bytes, std::size_t length) {
    std::fprintf(stderr, "  %s (%zu bytes): ", label, length);
    for (std::size_t index = 0; index < length; ++index) std::fprintf(stderr, "%02X ", (unsigned char)bytes[index]);
    std::fprintf(stderr, "\n");
}

/** @brief Append one codepoint to @p text as UTF-8 via `sz_rune_encode` (silently skips invalid runes). */
inline void append_codepoint_(std::string &text, sz_rune_t codepoint) {
    sz_u8_t bytes[4];
    sz_rune_length_t const length = sz_rune_encode(codepoint, bytes);
    if (length == sz_rune_invalid_k) return;
    text.append((char const *)bytes, (std::size_t)length);
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

/** @brief Append @p link_count Regional-Indicator codepoints to @p out (cleared first); cross-family run builder. */
inline void utf8_dense_regional_indicators_(std::string &out, std::mt19937 &rng, std::size_t link_count) {
    out.clear();
    std::uniform_int_distribution<sz_rune_t> indicator(0x1F1E6, 0x1F1FF); // U+1F1E6..U+1F1FF
    for (std::size_t index = 0; index != link_count; ++index) append_codepoint_(out, indicator(rng));
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

/** @brief Byte-length boundaries, BOM, ZWJ, VS16, and one codepoint per byte length — the shared default boundaries. */
static sz_rune_t const utf8_default_boundary_codepoints[] = {
    0x007F, 0x0080, 0x07FF, 0x0800,  0xFFFF, 0x10000, 0x10FFFF, // byte-length boundaries
    0xFEFF,                                                     // BOM U+FEFF
    0x200D,                                                     // ZWJ U+200D
    0xFE0F,                                                     // VS16 U+FE0F
    0x0041, 0x00DF, 0x4E2D, 0x1F600,                            // one codepoint per byte length
};

/** @brief A generic mixed-script snippet grab-bag — the shared default a family uses when it has no own alphabet. */
static char const *const utf8_default_snippets[] = {
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

/** @brief The shared default alphabet (weights mirror the legacy snippet/boundary/astral/motif/malformed mix). */
static utf8_corpus_alphabet_t const utf8_default_alphabet = {
    utf8_default_snippets,
    sizeof(utf8_default_snippets) / sizeof(utf8_default_snippets[0]),
    utf8_default_boundary_codepoints,
    sizeof(utf8_default_boundary_codepoints) / sizeof(utf8_default_boundary_codepoints[0]),
    {35, 20, 15, 20, 10}, // snippet, boundary, astral, motif, malformed
};

/**
 *  @brief Build a random UTF-8 corpus into @p out (cleared first) from @p alphabet's weighted categories plus the
 *         family's @p motifs, until at least @p min_length bytes. The malformed category is muted for @p valid_k;
 *         empty tables are muted so `std::discrete_distribution` never selects an unpopulated category.
 */
inline void utf8_random_segmentation_corpus_(std::string &out, std::size_t min_length, utf8_corpus_flavor_t flavor,
                                             utf8_corpus_alphabet_t const &alphabet, sz::string_view const *motifs,
                                             std::size_t motif_count, std::mt19937 &rng) {
    out.clear();
    double weights[utf8_corpus_category_count_k];
    for (int category = 0; category != utf8_corpus_category_count_k; ++category)
        weights[category] = (double)alphabet.category_weights[category];
    if (flavor == utf8_corpus_flavor_t::valid_k) weights[utf8_corpus_malformed_k] = 0.0;
    if (!motif_count) weights[utf8_corpus_motif_k] = 0.0;
    if (!alphabet.snippet_count) weights[utf8_corpus_snippet_k] = 0.0;
    if (!alphabet.boundary_count) weights[utf8_corpus_boundary_k] = 0.0;

    std::discrete_distribution<int> category(weights, weights + utf8_corpus_category_count_k);
    std::uniform_int_distribution<std::size_t> boundary_pick(0,
                                                             alphabet.boundary_count ? alphabet.boundary_count - 1 : 0);
    std::uniform_int_distribution<std::size_t> snippet_pick(0, alphabet.snippet_count ? alphabet.snippet_count - 1 : 0);
    std::uniform_int_distribution<std::size_t> astral_pick(0, utf8_astral_fixtures_count - 1);
    std::uniform_int_distribution<std::size_t> motif_pick(0, motif_count ? motif_count - 1 : 0);

    while (out.size() < min_length) {
        switch ((utf8_corpus_category_t)category(rng)) {
        case utf8_corpus_malformed_k: append_malformed_class_(out, rng); break;
        case utf8_corpus_boundary_k: append_codepoint_(out, alphabet.boundary_codepoints[boundary_pick(rng)]); break;
        case utf8_corpus_astral_k: {
            sz::string_view const fixture = utf8_astral_fixtures[astral_pick(rng)];
            out.append(fixture.data(), fixture.size());
        } break;
        case utf8_corpus_motif_k: {
            sz::string_view const motif = motifs[motif_pick(rng)];
            out.append(motif.data(), motif.size());
        } break;
        case utf8_corpus_snippet_k:
        default: out.append(alphabet.snippets[snippet_pick(rng)]); break;
        }
    }
}

#pragma endregion // Shared helpers

#pragma region Lazy streaming comparison

/**
 *  @brief Streams one backend at a fixed @ref capacity, yielding one (absolute start, length) segment at a time and
 *         refilling from the C finder (`bytes_consumed` resume). Bounded memory: a single batch buffer, no heap.
 */
struct utf8_segment_cursor_t {
    sz_utf8_segmenter_t finder;
    sz_cptr_t data;
    sz_size_t length;
    sz_size_t capacity;   // caller capacity passed to the finder (<= utf8_segment_batch_k)
    sz_size_t next_base;  // absolute offset where the NEXT batch refill starts
    sz_size_t batch_base; // absolute base of the CURRENT batch (for absolute segment offsets)
    sz_size_t count;      // segments buffered in the current batch
    sz_size_t index;      // next segment within the current batch
    sz_bool_t exhausted;
    sz_size_t starts[utf8_segment_batch_k];
    sz_size_t lengths[utf8_segment_batch_k];
};

/** @brief Make a cursor over @p finder, clamping the caller @p capacity to the batch buffer. */
inline utf8_segment_cursor_t utf8_segment_cursor_make_(sz_utf8_segmenter_t finder, sz_cptr_t data, sz_size_t length,
                                                       sz_size_t capacity) {
    utf8_segment_cursor_t cursor;
    cursor.finder = finder, cursor.data = data, cursor.length = length;
    cursor.capacity = capacity < utf8_segment_batch_k ? capacity : utf8_segment_batch_k;
    cursor.next_base = 0, cursor.batch_base = 0, cursor.count = 0, cursor.index = 0, cursor.exhausted = sz_false_k;
    return cursor;
}

/** @brief Pull the next segment; returns sz_false_k at end of stream, else fills @p out_start (absolute) / @p out_length. */
inline sz_bool_t utf8_segment_cursor_next_(utf8_segment_cursor_t &cursor, sz_size_t &out_start, sz_size_t &out_length) {
    while (cursor.index == cursor.count) {
        if (cursor.exhausted || cursor.next_base >= cursor.length) return sz_false_k;
        sz_size_t consumed = 0;
        cursor.batch_base = cursor.next_base;
        cursor.count = cursor.finder(cursor.data + cursor.batch_base, cursor.length - cursor.batch_base, cursor.starts,
                                     cursor.lengths, cursor.capacity, &consumed);
        cursor.index = 0;
        if (cursor.count == 0 && consumed == 0) {
            cursor.exhausted = sz_true_k;
            return sz_false_k;
        }
        cursor.next_base = cursor.batch_base + consumed;
    }
    out_start = cursor.batch_base + cursor.starts[cursor.index];
    out_length = cursor.lengths[cursor.index];
    ++cursor.index;
    return sz_true_k;
}

/** @brief Emit a full reproduction record to `stderr` then abort; called on the first segment divergence. */
inline void utf8_report_divergence_(utf8_repro_t const &repro, sz_cptr_t data, sz_size_t length,
                                    std::size_t segment_index, sz_bool_t reference_more, sz_size_t reference_start,
                                    sz_size_t reference_length, sz_bool_t candidate_more, sz_size_t candidate_start,
                                    sz_size_t candidate_length) {
    unsigned const seed = (unsigned)global_random_seed();
    std::fprintf(stderr, "\nUTF-8 %s divergence in stressor '%s': seed=%u iteration=%zu capacity=%zu flavor=%s\n",
                 repro.family, repro.stressor, seed, repro.iteration, (std::size_t)repro.capacity,
                 repro.flavor == utf8_corpus_flavor_t::valid_k ? "valid" : "malformed");
    std::fprintf(stderr, "  rerun: SZ_TESTS_SEED=%u <test binary>\n", seed);
    std::fprintf(stderr, "  first divergence at segment %zu:\n", segment_index);
    if (reference_more)
        std::fprintf(stderr, "    reference: start=%zu length=%zu\n", (std::size_t)reference_start,
                     (std::size_t)reference_length);
    else std::fprintf(stderr, "    reference: <end of stream>\n");
    if (candidate_more)
        std::fprintf(stderr, "    candidate: start=%zu length=%zu\n", (std::size_t)candidate_start,
                     (std::size_t)candidate_length);
    else std::fprintf(stderr, "    candidate: <end of stream>\n");
    print_utf8_test_bytes_("input", data, length);
    assert(false && "UTF-8 segmentation backends diverged (see stderr for the reproduction record)");
}

/**
 *  @brief Stream @p reference and @p candidate in lockstep; assert identical (start,length) per segment and
 *         (flavor-gated) that segments tile `[0, length)` and start on codepoint boundaries. On the first divergence
 *         emit a full repro and abort. No heap; stops at the first mismatch. @p reference and @p candidate may be the
 *         same finder at different capacities (capacity-independence) or two backends at the same capacity.
 */
inline void utf8_compare_streams_(utf8_repro_t const &repro, sz_utf8_segmenter_t reference,
                                  sz_size_t reference_capacity, sz_utf8_segmenter_t candidate,
                                  sz_size_t candidate_capacity, sz_cptr_t data, sz_size_t length,
                                  utf8_corpus_flavor_t flavor) {
    utf8_segment_cursor_t reference_cursor = utf8_segment_cursor_make_(reference, data, length, reference_capacity);
    utf8_segment_cursor_t candidate_cursor = utf8_segment_cursor_make_(candidate, data, length, candidate_capacity);
    sz_size_t running_cursor = 0;
    std::size_t segment_index = 0;
    for (;;) {
        sz_size_t reference_start = 0, reference_length = 0, candidate_start = 0, candidate_length = 0;
        sz_bool_t const reference_more = utf8_segment_cursor_next_(reference_cursor, reference_start, reference_length);
        sz_bool_t const candidate_more = utf8_segment_cursor_next_(candidate_cursor, candidate_start, candidate_length);
        if (!reference_more && !candidate_more) break; // both streams ended together: success
        sz_bool_t const agree = (sz_bool_t)(reference_more && candidate_more && reference_start == candidate_start &&
                                            reference_length == candidate_length);
        if (!agree)
            utf8_report_divergence_(repro, data, length, segment_index, reference_more, reference_start,
                                    reference_length, candidate_more, candidate_start, candidate_length);
        assert(candidate_start == running_cursor && "segments do not tile the input contiguously");
        if (flavor == utf8_corpus_flavor_t::valid_k && candidate_length != 0)
            assert((((sz_u8_t)data[candidate_start]) & 0xC0u) != 0x80u && "segment starts mid-codepoint");
        running_cursor += candidate_length;
        ++segment_index;
    }
    assert(running_cursor == length && "segments do not cover the whole input");
}

#pragma endregion // Lazy streaming comparison

#pragma region Unit driver

/**
 *  @brief Drive one segmentation backend over the hand-checked goldens, streaming its segments and comparing each
 *         lazily against the next expected literal (no materialized container). The caller invokes it once per
 *         backend, so a wrong constant shared by serial and SIMD is still caught against external ground truth.
 */
inline void check_utf8_segment_unit_(char const *family, sz_utf8_segmenter_t forward, utf8_unit_case_t const *cases,
                                     std::size_t case_count) {
    for (std::size_t case_index = 0; case_index != case_count; ++case_index) {
        utf8_unit_case_t const &golden = cases[case_index];
        utf8_segment_cursor_t cursor = utf8_segment_cursor_make_(forward, golden.text.data(), golden.text.size(),
                                                                 utf8_segment_batch_k);
        sz_size_t start = 0, segment_length = 0;
        for (sz::string_view const expected : golden.expected) {
            sz_bool_t const more = utf8_segment_cursor_next_(cursor, start, segment_length);
            assert(more && family && "segment forward emitted fewer segments than the golden");
            assert(segment_length == expected.size() &&
                   std::memcmp(golden.text.data() + start, expected.data(), expected.size()) == 0 && family &&
                   "segment forward bytes != golden");
        }
        sz_size_t extra_start = 0, extra_length = 0;
        assert(!utf8_segment_cursor_next_(cursor, extra_start, extra_length) && family &&
               "segment forward emitted more segments than the golden");
    }
}

#pragma endregion // Unit driver

#pragma region Rule coverage

/** @brief Whether a rule-coverage motif fires the rule in its break or its no-break (join) direction. */
enum utf8_rule_direction_t { utf8_rule_breaks_k, utf8_rule_joins_k };

/** @brief One UAX rule-coverage motif: a short input that exercises a named spec rule in a given direction. */
struct utf8_rule_case_t {
    char const *rule_id;             /**< UAX rule id this motif exercises, e.g. "WB6", "GB9c", "SB8", "LB21a". */
    utf8_rule_direction_t direction; /**< whether the motif demonstrates the rule breaking or joining. */
    sz::string_view text;            /**< short input that fires the rule. */
};

/**
 *  @brief Per-family rule-coverage gate. Two obligations (CONTRIBUTING-KERNELS.md §6.12):
 *         (1) every motif segments identically on @p reference and @p candidate, one-shot AND re-anchored at the
 *             window-edge phases 61/62/63 so a rule firing across the 64-byte boundary is exercised too; and
 *         (2) every id in @p required_rule_ids is exercised by at least one motif, so no spec rule is left untested.
 */
inline void check_utf8_rule_coverage_(char const *family, sz_utf8_segmenter_t reference, sz_utf8_segmenter_t candidate,
                                      utf8_rule_case_t const *cases, std::size_t case_count,
                                      char const *const *required_rule_ids, std::size_t required_count) {
    static sz_size_t const window_phases[] = {0, 61, 62, 63};
    std::string probe;
    for (std::size_t case_index = 0; case_index != case_count; ++case_index) {
        sz::string_view const text = cases[case_index].text;
        for (sz_size_t phase : window_phases) {
            probe.assign((std::size_t)phase, 'a');
            probe.append(text.data(), text.size());
            probe.append(8, 'a');
            utf8_repro_t const repro = {family, "rule-coverage", case_index, utf8_segment_batch_k,
                                        utf8_corpus_flavor_t::valid_k};
            utf8_compare_streams_(repro, reference, utf8_segment_batch_k, candidate, utf8_segment_batch_k, probe.data(),
                                  probe.size(), utf8_corpus_flavor_t::valid_k);
        }
    }
    for (std::size_t required_index = 0; required_index != required_count; ++required_index) {
        bool covered = false;
        for (std::size_t case_index = 0; case_index != case_count && !covered; ++case_index)
            covered = std::strcmp(cases[case_index].rule_id, required_rule_ids[required_index]) == 0;
        assert(covered && family && "UAX rule id not exercised by any coverage motif");
    }
}

#pragma endregion // Rule coverage

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

#pragma region Differential stressors

/** @brief Caller capacities swept per differential input: small values exercise the window-loop resume at every
 *         phase; the window-adjacent values (32/33/63/64/65) catch resume-seam bugs; the batch value is the
 *         whole-input single-shot. */
static sz_size_t const utf8_sweep_capacities[] = {1, 2, 3, 16, 17, 32, 33, 63, 64, 65, utf8_segment_batch_k};

/** @brief Long-range straddle gaps: each places the decisive rule context past one window from the boundary, sweeping
 *         the seam across >=2 / >=3 windows at every phase. */
static std::size_t const utf8_straddle_gaps[] = {64, 65, 95, 96, 127, 128, 129, 191, 192, 256};

/** @brief Marathon run length cap: every rule-critical unit is repeated past several 64-byte windows. */
static constexpr std::size_t utf8_marathon_length_k = 320;

/**
 *  @brief Long homogeneous carry units (the cross-window register-carry stressor): a single rule-critical unit
 *         repeated far past one 64-byte window. Family-agnostic — each stresses some family's carry (RI parity,
 *         ZWJ/Extend chains, NU runs, SATerm shadows, SP/ZW/QU runs, Hebrew-Hyphen, Hangul).
 */
static char const *const utf8_marathon_units[] = {
    ".",
    ":",
    ",",
    "'",
    "\xE2\x80\x99", // Mid / MidNumLetQ (word bridge), ATerm (".")
    ". ",
    ") ",
    "a.",
    "9,",                           // shadow / numeric-mid contexts
    "\xF0\x9F\x87\xBA",             // Regional_Indicator (GB12/13, WB15/16 parity)
    "\xF0\x9F\x98\x80\xE2\x80\x8D", // emoji + ZWJ (GB11 chain)
    "\xCC\x81",
    "\xEF\xB8\x8F",             // Extend / VS16 (grapheme, WB4 transparency)
    "\xE0\xA4\x95\xE0\xA5\x8D", // Indic consonant + linker (GB9c)
    " ",                        // long SP run (LB7/LB18 SP, WB3d WSegSpace, SB Sp shadow)
    "1",                        // long NU run (LB25 numeric, WB8)
    "\xE2\x80\x8B",             // ZWSP / ZW run (LB8 "ZW SP* /")
    "\"",                       // QU run (LB19 East-Asian quote, LB15)
    "\xD7\x90\x2D",             // Hebrew_Letter + Hyphen (LB21a "HL (HY|HH) x" across windows)
    "\xEA\xB0\x80",             // Hangul LV syllable run (LB26/LB27, GB6/7)
};
static char const *const utf8_marathon_prefixes[] = {"", "x", "5", "\xD7\x90"}; // none / letter / digit / Hebrew
static char const *const utf8_marathon_terminators[] = {"", "A", "b", "9", ".", "\n"};

/** @brief Malformed fragments dropped exactly at a rule-critical seam (not standalone noise). */
static char const *const utf8_malformed_seam_fragments[] = {
    "\xC0\x80",         // overlong NUL
    "\xED\xA0\x80",     // surrogate U+D800
    "\xE2\x80",         // truncated 3-byte lead
    "\xF0\x9F\x98",     // truncated 4-byte lead
    "\x80",             // stray continuation
    "\xF5\x80\x80\x80", // out-of-range lead > U+10FFFF
};
/** @brief (prefix, suffix) hosts placing the fragment at a rule-critical seam across the families. */
static char const *const utf8_malformed_seam_prefixes[] = {"ab'", "a ", "a.", "a", "1,", "\xD7\x90-", "(\xC2\xA0"};
static char const *const utf8_malformed_seam_suffixes[] = {"cd", " b", " B", "\xCC\x81", "2", "a", ")"};
static sz_size_t const utf8_malformed_seam_phases[] = {0, 60, 61, 62, 63};

/** @brief The differential's shared state: the two backends, the corpora, the RNG, and a reused corpus scratch. */
struct utf8_differential_context_t {
    sz_utf8_segmenter_t reference;
    sz_utf8_segmenter_t candidate;
    utf8_segment_corpora_t const *corpora;
    std::mt19937 *rng;
    std::string scratch;
};

/** @brief The family's alphabet, or the shared default when it supplies none. */
inline utf8_corpus_alphabet_t const &utf8_context_alphabet_(utf8_differential_context_t const &context) {
    return context.corpora->alphabet ? *context.corpora->alphabet : utf8_default_alphabet;
}

/**
 *  @brief The per-input check: compare serial-vs-ISA across the capacity sweep, and (capacity-independence) the
 *         candidate at each capacity against the candidate at full batch. This is the unit of all stressors below.
 */
inline void utf8_differential_input_(utf8_differential_context_t &context, char const *stressor, std::size_t iteration,
                                     sz_cptr_t data, sz_size_t length, utf8_corpus_flavor_t flavor) {
    char const *family = context.corpora->family_name;
    for (sz_size_t capacity : utf8_sweep_capacities) {
        utf8_repro_t const repro = {family, stressor, iteration, capacity, flavor};
        utf8_compare_streams_(repro, context.reference, capacity, context.candidate, capacity, data, length, flavor);
        if (capacity != utf8_segment_batch_k)
            utf8_compare_streams_(repro, context.candidate, capacity, context.candidate, utf8_segment_batch_k, data,
                                  length, flavor);
    }
}

/** @brief Visitor context for the dense-run / straddle sinks; forwards each produced run to `utf8_differential_input_`. */
struct utf8_sink_context_t {
    utf8_differential_context_t *context;
    char const *stressor;
    std::size_t iteration;
    std::size_t filler; // straddle phase-shift prefix length (unused by the plain dense sink)
    std::string buffer; // straddle prefix+run scratch (unused by the plain dense sink)
};

/** @brief Plain dense-run sink: compare the produced run directly. */
inline void utf8_sink_dense_run_(void *context, sz_cptr_t data, sz_size_t length) {
    utf8_sink_context_t *sink = (utf8_sink_context_t *)context;
    utf8_differential_input_(*sink->context, sink->stressor, sink->iteration, data, length,
                             utf8_corpus_flavor_t::valid_k);
}

/** @brief Straddle sink: prepend an ASCII filler so the run's content lands at a shifted window phase, then compare. */
inline void utf8_sink_straddle_(void *context, sz_cptr_t data, sz_size_t length) {
    utf8_sink_context_t *sink = (utf8_sink_context_t *)context;
    sink->buffer.assign(sink->filler, 'x');
    sink->buffer.append(data, length);
    utf8_differential_input_(*sink->context, sink->stressor, sink->iteration, sink->buffer.data(), sink->buffer.size(),
                             utf8_corpus_flavor_t::valid_k);
}

/** @brief Fixed regression inputs (hand-found seam bugs, each > one window): must agree serial-vs-ISA exactly. */
inline void utf8_differential_regressions_(utf8_differential_context_t &context) {
    for (std::size_t index = 0; index != context.corpora->regression_count; ++index)
        utf8_differential_input_(context, "regression", index, (sz_cptr_t)context.corpora->regressions[index].data(),
                                 context.corpora->regressions[index].size(), utf8_corpus_flavor_t::valid_k);
}

/** @brief Randomized fuzz: per iteration a 400-byte valid corpus, an occasional ~4096-byte wide tier, a mutated copy,
 *         and a malformed corpus — each compared serial-vs-ISA across the capacity sweep. */
inline void utf8_differential_fuzz_corpus_(utf8_differential_context_t &context, std::size_t iterations) {
    std::printf("  - fuzzing %s random corpus (serial-vs-ISA)...\n", context.corpora->family_name);
    utf8_corpus_alphabet_t const &alphabet = utf8_context_alphabet_(context);
    sz::string_view const *motifs = context.corpora->motifs;
    std::size_t const motif_count = context.corpora->motif_count;
    std::string mutated;
    for (std::size_t iteration = 0; iteration != iterations; ++iteration) {
        utf8_random_segmentation_corpus_(context.scratch, 400, utf8_corpus_flavor_t::valid_k, alphabet, motifs,
                                         motif_count, *context.rng);
        utf8_differential_input_(context, "fuzz-corpus", iteration, context.scratch.data(), context.scratch.size(),
                                 utf8_corpus_flavor_t::valid_k);

        if ((iteration & 0x7u) == 0) {
            utf8_random_segmentation_corpus_(context.scratch, 4096, utf8_corpus_flavor_t::valid_k, alphabet, motifs,
                                             motif_count, *context.rng);
            utf8_differential_input_(context, "fuzz-corpus-wide", iteration, context.scratch.data(),
                                     context.scratch.size(), utf8_corpus_flavor_t::valid_k);
        }

        utf8_random_segmentation_corpus_(context.scratch, 400, utf8_corpus_flavor_t::valid_k, alphabet, motifs,
                                         motif_count, *context.rng);
        mutated.assign(context.scratch.data(), context.scratch.size());
        apply_mutation_passes_(mutated, *context.rng);
        utf8_differential_input_(context, "fuzz-mutated", iteration, mutated.data(), mutated.size(),
                                 utf8_corpus_flavor_t::malformed_k);

        utf8_random_segmentation_corpus_(context.scratch, 400, utf8_corpus_flavor_t::malformed_k, alphabet, motifs,
                                         motif_count, *context.rng);
        utf8_differential_input_(context, "fuzz-malformed", iteration, context.scratch.data(), context.scratch.size(),
                                 utf8_corpus_flavor_t::malformed_k);
    }
}

/** @brief Randomized fuzz: the family's high-density homogeneous runs (one long single-rule blob per run). */
inline void utf8_differential_fuzz_dense_runs_(utf8_differential_context_t &context, std::size_t iterations) {
    if (!context.corpora->dense_runs) return;
    std::printf("  - fuzzing %s dense runs...\n", context.corpora->family_name);
    for (std::size_t iteration = 0; iteration != iterations; ++iteration) {
        utf8_sink_context_t sink;
        sink.context = &context, sink.stressor = "dense-run", sink.iteration = iteration, sink.filler = 0;
        context.corpora->dense_runs(*context.rng, utf8_sink_dense_run_, &sink);
    }
}

/** @brief Randomized fuzz: the family's long-range straddles, gap-swept and phase-shifted by a random ASCII filler. */
inline void utf8_differential_fuzz_straddles_(utf8_differential_context_t &context, std::size_t iterations) {
    if (!context.corpora->straddles) return;
    std::printf("  - fuzzing %s long-range straddles...\n", context.corpora->family_name);
    std::uniform_int_distribution<std::size_t> filler_length(0, utf8_window_k - 1);
    for (std::size_t iteration = 0; iteration != iterations; ++iteration)
        for (std::size_t gap : utf8_straddle_gaps) {
            utf8_sink_context_t sink;
            sink.context = &context, sink.stressor = "straddle", sink.iteration = iteration;
            sink.filler = filler_length(*context.rng);
            context.corpora->straddles(*context.rng, gap, utf8_sink_straddle_, &sink);
        }
}

/** @brief Deterministic: a fixed corpus driven at every sub-cache-line offset so the load alignment is swept. */
inline void utf8_differential_alignment_sweep_(utf8_differential_context_t &context) {
    std::printf("  - testing %s alignment sweep...\n", context.corpora->family_name);
    utf8_corpus_alphabet_t const &alphabet = utf8_context_alphabet_(context);
    utf8_random_segmentation_corpus_(context.scratch, 256, utf8_corpus_flavor_t::valid_k, alphabet,
                                     context.corpora->motifs, context.corpora->motif_count, *context.rng);
    std::string const probe = context.scratch; // stable source copied into each aligned buffer
    for_each_cacheline_offset_(probe.size(), [&](sz_ptr_t buffer, std::size_t /*offset*/) {
        std::memcpy(buffer, probe.data(), probe.size());
        utf8_differential_input_(context, "alignment-sweep", 0, (sz_cptr_t)buffer, probe.size(),
                                 utf8_corpus_flavor_t::valid_k);
    });
}

/** @brief Deterministic: each marathon unit repeated past several windows, behind every prefix, closed by every
 *         terminator — the shape that exposes open-bridge / parity / shadow / pending carry bugs. */
inline void utf8_differential_marathon_runs_(utf8_differential_context_t &context) {
    std::printf("  - testing %s marathon carry runs...\n", context.corpora->family_name);
    std::size_t iteration = 0;
    for (char const *unit : utf8_marathon_units) {
        std::size_t const unit_length = std::strlen(unit);
        for (char const *prefix : utf8_marathon_prefixes)
            for (char const *terminator : utf8_marathon_terminators) {
                context.scratch.assign(prefix);
                while (context.scratch.size() + unit_length <= utf8_marathon_length_k)
                    context.scratch.append(unit, unit_length);
                context.scratch.append(terminator);
                utf8_differential_input_(context, "marathon", iteration++, context.scratch.data(),
                                         context.scratch.size(), utf8_corpus_flavor_t::valid_k);
            }
    }
}

/** @brief Deterministic: place each family motif at every byte offset 0..63 within ASCII filler so it straddles the
 *         64-byte window edge at every alignment (phase 0 lands the motif at true start-of-text). */
inline void utf8_differential_phase_sweep_(utf8_differential_context_t &context) {
    std::printf("  - testing %s all-phase straddle sweep...\n", context.corpora->family_name);
    for (std::size_t motif_index = 0; motif_index != context.corpora->motif_count; ++motif_index) {
        sz::string_view const motif = context.corpora->motifs[motif_index];
        for (std::size_t phase = 0; phase != utf8_window_k; ++phase) {
            context.scratch.assign(phase, 'a');
            context.scratch.append(motif.data(), motif.size());
            context.scratch.append(80, 'a');
            utf8_differential_input_(context, "phase-sweep", motif_index, context.scratch.data(),
                                     context.scratch.size(), utf8_corpus_flavor_t::valid_k);
        }
    }
}

/** @brief Deterministic: drop a malformed fragment at a rule-critical seam (after MidLetter, inside an SP-run, after
 *         an ATerm, between a base and a combining mark, ...) at the window-edge phases. Both backends apply the same
 *         U+FFFD substitution (§6.2), so they must still agree (malformed flavor relaxes the alignment invariant). */
inline void utf8_differential_malformed_seams_(utf8_differential_context_t &context) {
    std::printf("  - testing %s malformed-at-seam injection...\n", context.corpora->family_name);
    std::size_t const host_count = sizeof(utf8_malformed_seam_prefixes) / sizeof(utf8_malformed_seam_prefixes[0]);
    for (char const *fragment : utf8_malformed_seam_fragments)
        for (std::size_t host = 0; host != host_count; ++host)
            for (sz_size_t phase : utf8_malformed_seam_phases) {
                context.scratch.assign((std::size_t)phase, 'a');
                context.scratch.append(utf8_malformed_seam_prefixes[host]);
                context.scratch.append(fragment);
                context.scratch.append(utf8_malformed_seam_suffixes[host]);
                utf8_differential_input_(context, "malformed-seam", host, context.scratch.data(),
                                         context.scratch.size(), utf8_corpus_flavor_t::malformed_k);
            }
}

/**
 *  @brief Deterministic exhaustive window-edge byte partition: place every byte value of a (possibly multi-byte,
 *         possibly truncated) lead and its continuations at the TOP lanes of a full 64-byte window — exactly where a
 *         declared multi-byte span crosses the window edge — and assert serial-vs-ISA agreement. Guarantees coverage
 *         of the edge handling the random fuzz only hits probabilistically. Compared at full batch (single call).
 */
inline void utf8_differential_byte_edge_exhaustive_(utf8_differential_context_t &context) {
    std::printf("  - testing %s window-edge byte partition (exhaustive)...\n", context.corpora->family_name);
    unsigned char buffer[96];
    std::size_t iteration = 0;
    auto compare_one = [&](sz_size_t length) {
        utf8_repro_t const repro = {context.corpora->family_name, "byte-edge", iteration++, utf8_segment_batch_k,
                                    utf8_corpus_flavor_t::malformed_k};
        utf8_compare_streams_(repro, context.reference, utf8_segment_batch_k, context.candidate, utf8_segment_batch_k,
                              (sz_cptr_t)buffer, length, utf8_corpus_flavor_t::malformed_k);
    };
    // Two-byte edge: every (first, second) pair at the last two lanes, swept across the loaded boundary (62..66).
    for (sz_size_t length = 62; length <= 66; ++length)
        for (int first_byte = 0; first_byte != 256; ++first_byte)
            for (int second_byte = 0; second_byte != 256; ++second_byte) {
                std::memset(buffer, 'a', length);
                buffer[length - 2] = (unsigned char)first_byte, buffer[length - 1] = (unsigned char)second_byte;
                compare_one(length);
            }
    // Three-byte edge: a lead at lane 61, a continuation candidate at lane 62, an edge byte at lane 63 — the declared
    // third byte then falls at lane 64, past the full window.
    static int const edge_bytes[] = {0x80, 0xBF, 0x41, 0x00, 0xE0};
    for (int lead_byte = 0xC0; lead_byte != 256; ++lead_byte)
        for (int continuation_byte = 0; continuation_byte != 256; ++continuation_byte)
            for (int edge_byte : edge_bytes) {
                std::memset(buffer, 'a', 64);
                buffer[61] = (unsigned char)lead_byte, buffer[62] = (unsigned char)continuation_byte,
                buffer[63] = (unsigned char)edge_byte;
                compare_one(64);
            }
}

#pragma endregion // Differential stressors

#pragma region Differential driver

/**
 *  @brief Differential of any ISA finder against the serial reference: a short orchestrator over the randomized fuzz
 *         stressors and the deterministic exhaustive stressors. Every stressor drives each input through the capacity
 *         sweep (`utf8_differential_input_`), asserting serial≡ISA, capacity-independence, and tiling/alignment, and
 *         aborts with a full reproduction record at the first divergence.
 */
inline void test_utf8_segment_equivalence_(sz_utf8_segmenter_t reference, sz_utf8_segmenter_t candidate,
                                           utf8_segment_corpora_t const &corpora,
                                           std::size_t iterations = scale_iterations(5000)) {
    std::printf("  - testing %s serial-vs-ISA differential...\n", corpora.family_name);
    utf8_differential_context_t context;
    context.reference = reference, context.candidate = candidate, context.corpora = &corpora,
    context.rng = &global_random_generator();

    utf8_differential_regressions_(context);
    utf8_differential_fuzz_corpus_(context, iterations);
    utf8_differential_fuzz_dense_runs_(context, iterations);
    utf8_differential_fuzz_straddles_(context, iterations);
    utf8_differential_alignment_sweep_(context);
    utf8_differential_marathon_runs_(context);
    utf8_differential_phase_sweep_(context);
    utf8_differential_malformed_seams_(context);
    utf8_differential_byte_edge_exhaustive_(context);
}

#pragma endregion // Differential driver

#endif // STRINGZILLA_TEST_UTF8_HPP_
