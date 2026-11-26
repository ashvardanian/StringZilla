/**
 *  @file test/ngrams.cpp
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief N-grams tests: known answers, the query accessors, and the dispatched engine's overlaps
 *      and sketches against the oracles in @c cross.hpp.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#undef STRINGZILLA_DEBUG
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

#include <set>    // `std::set`, the fox's distinct N-grams
#include <string> // Baseline
#include <vector> // `std::vector`

#include "stringzilla/stringzilla.h"   // Primary C API
#include "stringzilla/stringzilla.hpp" // C++ string class replacement

#include "cross.hpp"   // `ngrams_backend_t`, `check_ngrams_overlaps_`, `check_ngrams_sketches_`
#include "harness.hpp" // `strings_from_`, `case_stream_key`

namespace ashvardanian::stringzilla::test {

#pragma region Helpers

/** The dispatched engine constructor over the CPU capabilities, in the shape of its capability
 *  kernels, whose mask sits before the allocator. */
static sz_status_t ngrams_engine_init_dispatched_(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                  sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                  sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                  sz_stream_t stream) {
    return sz_ngrams_engine_init(engine, queries, window_widths, window_widths_count, candidates_budget,
                                 sz::default_capabilities(), allocator, stream);
}

/** The dispatched constructor and the verbs that run whatever capability it prepared for; the
 *  per-text kernels have no dispatch point. */
static ngrams_backend_t const ngrams_dispatched {
    "dispatched", ngrams_engine_init_dispatched_, sz_ngrams_overlaps, sz_ngrams_sketches, nullptr, nullptr};

/** One pair through the dispatched engine at one width, asserting the literal @p expected count. */
static void check_ngrams_pair_(std::string const &query, std::string const &candidate, std::size_t width,
                               sz_u32_t expected) {
    handle_checked_heap_t heap;
    std::vector<std::string> const queries = {query}, candidates = {candidate};
    sz_strings_view_t const query_strings = strings_from_(queries);
    sz_strings_view_t const candidate_strings = strings_from_(candidates);
    sz_ngrams_engine_t engine {};
    expect_equal(sz_ngrams_engine_init(&engine, &query_strings, &width, 1, 0, sz::default_capabilities(),
                                       &heap.allocator, nullptr),
                 sz_success_k);
    sz_u32_t matches = 0xFFFFFFFFu;
    expect_equal(sz_ngrams_overlaps(&engine, &candidate_strings, 1.0f, &matches, 1, 1, nullptr), sz_success_k);
    sz_ngrams_engine_free(&engine, nullptr);
    expect_equal(heap.live_allocations, 0);
    expect_equal(matches, expected);
}

#pragma endregion Helpers

#pragma region Unit

/** Known answers: capacities, strides, counts readable off the texts, and the accessors a kit
 *  probes a host engine's tables with. */
void test_ngrams_unit() {
    // Identical texts share every distinct window, disjoint alphabets share none, and a candidate
    // narrower than the width has no N-grams to share; "the" holds three, two and one of the
    // query's N-grams below that.
    std::string const fox = "the quick brown fox jumps over the lazy dog";
    std::vector<std::string> const queries = {fox};
    std::vector<std::string> const candidates = {fox, "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ", "the", ""};
    std::vector<std::size_t> const widths = {1, 2, 3, 4, 6, 8, 43};
    handle_checked_heap_t heap;
    sz_strings_view_t const query_strings = strings_from_(queries);
    sz_strings_view_t const candidate_strings = strings_from_(candidates);
    sz_ngrams_engine_t engine {};
    expect_equal(sz_ngrams_engine_init(&engine, &query_strings, widths.data(), widths.size(), 0,
                                       sz::default_capabilities(), &heap.allocator, nullptr),
                 sz_success_k);
    sz_u32_t matches[4 * 7];
    expect_equal(sz_ngrams_overlaps(&engine, &candidate_strings, 1.0f, matches, 4 * 7, 7, nullptr), sz_success_k);
    for (std::size_t width_index = 0; width_index != widths.size(); ++width_index) {
        std::set<std::string> distinct;
        for (std::size_t first = 0; first + widths[width_index] <= fox.size(); ++first)
            distinct.insert(fox.substr(first, widths[width_index]));
        expect_equal(matches[0 * 7 + width_index], distinct.size());
        expect_equal(engine.distinct_counts[width_index], distinct.size());
        expect_equal(matches[1 * 7 + width_index], 0);
        expect_equal(matches[3 * 7 + width_index], 0);
    }
    expect_true(matches[2 * 7 + 0] == 3 && matches[2 * 7 + 1] == 2 && matches[2 * 7 + 2] == 1);
    for (std::size_t width_index = 3; width_index != widths.size(); ++width_index)
        expect_equal(matches[2 * 7 + width_index], 0);

    // A candidate stride wider than the widths leaves the gap it names alone, and every count lands
    // where it says, read this time from a tape of the same candidates.
    std::string payload;
    std::vector<sz_u32_t> offsets;
    sz_strings_view_t const candidate_tape = strings_tape_from_(candidates, payload, offsets);
    sz_u32_t padded[4 * 9];
    for (sz_u32_t &slot : padded) slot = 0xFFFFFFFFu;
    expect_equal(sz_ngrams_overlaps(&engine, &candidate_tape, 1.0f, padded, 4 * 9, 9, nullptr), sz_success_k);
    for (std::size_t candidate = 0; candidate != candidates.size(); ++candidate) {
        for (std::size_t width_index = 0; width_index != widths.size(); ++width_index)
            expect_equal(padded[candidate * 9 + width_index], matches[candidate * 7 + width_index]);
        expect_true(padded[candidate * 9 + 7] == 0xFFFFFFFFu && padded[candidate * 9 + 8] == 0xFFFFFFFFu);
    }
    sz_ngrams_engine_free(&engine, nullptr);

    // "aaaa" holds one distinct window at width one, which "ab" holds; "ab" holds two, of which
    // "aaaa" holds one.
    check_ngrams_pair_("aaaa", "ab", 1, 1);
    check_ngrams_pair_("ab", "aaaa", 1, 1);

    // A candidate holding the whole query holds all 9 of its distinct N-grams at width three and
    // all 7 at five.
    check_ngrams_pair_("quick brown", "the quick brown fox", 3, 9);
    check_ngrams_pair_("quick brown", "the quick brown fox", 5, 7);

    // "abcabcabc" repeats 3 distinct N-grams "abc", "bca", "cab": "abc" holds 1, "xabcax" holds
    // "abc" and "bca", and "abcabcabc" as a candidate against "abc" counts its one window once.
    check_ngrams_pair_("abcabcabc", "abc", 3, 1);
    check_ngrams_pair_("abcabcabc", "xabcax", 3, 2);
    check_ngrams_pair_("abc", "abcabcabc", 3, 1);

    // An empty side has no N-grams, on either side or both.
    check_ngrams_pair_("", "abc", 1, 0);
    check_ngrams_pair_("abc", "", 1, 0);
    check_ngrams_pair_("", "", 1, 0);

    // The scalar helpers against their definitions.
    std::uint64_t power = 1;
    for (std::size_t width = 0; width != 70; ++width, power = power * 256 % sz_ngrams_modulus_k)
        expect_equal(sz_ngrams_power(width), power);
    sz_size_t const counted[] = {1, 0, 4, 9};
    expect_equal(sz_ngrams_count(8, counted, 4), 8 + 5 + 0);
    expect_equal(sz_ngrams_count(0, counted, 4), 0);
    expect_equal(sz_ngrams_sketch_word_distance(0x11223344u, 0x11223344u), 0);
    expect_equal(sz_ngrams_sketch_word_distance(0x80000001u, 0x00000000u), 2);
    expect_equal(sz_ngrams_sketch_word_distance(0xFF00FF00u, 0x00FF00FFu), 4);
    expect_equal(sz_ngrams_sketch_word_distance(0x01FF7F80u, 0x00FF7F80u), 1);

    // Every window of a query with repeats finds its slot, which names its first byte and whether
    // the query repeats it, and a window the query lacks finds none.
    std::vector<std::string> const repeating = {"abracadabra, abracadabra!", "", "xyz"};
    std::vector<std::size_t> const probed_widths = {1, 4, 11, 30};
    sz_strings_view_t const repeating_strings = strings_from_(repeating);
    expect_equal(sz_ngrams_engine_init(&engine, &repeating_strings, probed_widths.data(), probed_widths.size(), 0,
                                       sz::default_capabilities(), &heap.allocator, nullptr),
                 sz_success_k);
    for (std::size_t index = 0; index != repeating.size(); ++index) {
        std::string const &text = repeating[index];
        sz_ngrams_query_t const query = sz_ngrams_engine_query(&engine, index);
        expect_equal(query.length, text.size());
        expect_equal(query.widths_count, probed_widths.size());
        std::size_t ngrams = 0, distinct = 0;
        for (std::size_t width_index = 0; width_index != probed_widths.size(); ++width_index) {
            std::size_t const width = probed_widths[width_index];
            expect_equal(query.widths[width_index], width);
            expect_equal(query.powers[width_index], sz_ngrams_power(width));
            distinct += ngrams_reference_set_(text, width).size();
            for (std::size_t first = 0; first + width <= text.size(); ++first, ++ngrams) {
                std::string const window = text.substr(first, width);
                sz_u64_t const slot = sz_ngrams_query_probe(&query, ngrams_reference_residue_(window), width_index);
                expect_true(slot != 0);
                expect_equal(sz_ngrams_slot_first(slot), text.find(window));
                expect_equal(sz_ngrams_slot_repeated(slot) == sz_true_k,
                             text.find(window, text.find(window) + 1) != std::string::npos);
            }
            for (std::string const absent : {"q", "zzzz", "abracadabrq"})
                if (absent.size() == width)
                    expect_equal(sz_ngrams_query_probe(&query, ngrams_reference_residue_(absent), width_index), 0);
        }
        expect_equal(query.ngrams, ngrams);
        expect_equal(query.distinct, distinct);
    }

    // An engine of no queries only sketches, and frees what both verbs grew.
    sz_ngrams_engine_t sketching {};
    expect_equal(sz_ngrams_engine_init(&sketching, nullptr, probed_widths.data(), probed_widths.size(), 0,
                                       sz::default_capabilities(), &heap.allocator, nullptr),
                 sz_success_k);
    expect_equal(sketching.count, 0);
    std::vector<sz_u8_t> sketches(repeating.size() * 8);
    expect_equal(sz_ngrams_sketches(&sketching, &repeating_strings, 8, sketches.data(), 8, nullptr), sz_success_k);
    std::vector<sz_u8_t> from_queries(repeating.size() * 8);
    expect_equal(sz_ngrams_sketches(&engine, &repeating_strings, 8, from_queries.data(), 8, nullptr), sz_success_k);
    expect_true(sketches == from_queries);
    sz_ngrams_engine_free(&sketching, nullptr);
    sz_ngrams_engine_free(&engine, nullptr);
    expect_equal(heap.live_allocations, 0);
}

#pragma endregion Unit

#pragma region Drivers

/** Degenerate inputs for the ngrams family, asserting survival and the stated refusals. */
void test_ngrams_safety() { check_ngrams_safety_(ngrams_dispatched); }

/** Holds the dispatched engine and every CPU capability's engine this build compiled to the oracles
 *  and to serial. */
void test_ngrams_all() {
    split_mix_generator_t generator {case_stream_key()};
    check_ngrams_overlaps_(generator, ngrams_dispatched);
    check_ngrams_sketches_(generator, ngrams_dispatched);
    check_ngrams_jaccard_(generator, ngrams_dispatched);
    for (sz_capability_t capability = sz_cap_serial_k; capability & sz_cap_cpus_k; capability <<= 1) {
        sz_kernel_punned_t init = nullptr, overlaps = nullptr, sketches = nullptr;
        sz_capability_t picked = 0;
        // A capability without a kernel of its own would check the one below it again.
        if (!(capability & sz::default_capabilities()) ||
            sz_ngrams_find_kernel(sz_kernel_ngrams_engine_init_k, capability, &init, &picked) != sz_success_k ||
            picked != capability ||
            sz_ngrams_find_kernel(sz_kernel_ngrams_overlaps_k, capability, &overlaps, &picked) != sz_success_k ||
            sz_ngrams_find_kernel(sz_kernel_ngrams_sketches_k, capability, &sketches, &picked) != sz_success_k)
            continue;
        char name[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
        sz_capabilities_name(capability, name, sizeof(name));
        ngrams_backend_t const found {name,
                                      reinterpret_cast<sz_kernel_ngrams_engine_init_t>(init),
                                      reinterpret_cast<sz_kernel_ngrams_overlaps_t>(overlaps),
                                      reinterpret_cast<sz_kernel_ngrams_sketches_t>(sketches),
                                      nullptr,
                                      nullptr};
        check_ngrams_sketches_(generator, found);
    }
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test
