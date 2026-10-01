/**
 *  @file test/overlap.cpp
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief Window overlap tests: known answers, integer and ordered-set oracles, and the dispatched
 *      engine against them.
 */
#undef NDEBUG // ! Enable all assertions for testing

/** The Visual C++ run-time library detects incorrect iterator use, and asserts and displays a
 *  dialog box at run time on Windows. */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Make sure to include the StringZilla headers before anything else, to intercept missing
 *  `#include` directives and other issues. */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include <cstdint> // `std::uint64_t`

#include <string> // Baseline
#include <vector> // `std::vector`

#include "cross.hpp"   // `overlap_backend_t`, `check_overlap_equivalence_`, `check_overlap_safety_`
#include "harness.hpp" // `sequence_from_`, `test_context_t`

namespace ashvardanian::stringzilla::test {

#pragma region Helpers

/** The dispatched engine constructor over the CPU capabilities, in the shape of its capability
 *  kernels, whose mask sits before the ordinal rather than the stream. */
static sz_status_t overlap_engine_init_dispatched_(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                   sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                   sz_size_t candidates_budget, sz_size_t ordinal,
                                                   sz_memory_allocator_t *allocator, void *stream) {
    return sz_overlap_engine_init(engine, queries, window_widths, window_widths_count, candidates_budget,
                                  sz::default_capabilities(), ordinal, allocator, stream);
}

/** The dispatched constructor and the round that scores whatever capability it prepared for. */
static overlap_backend_t const overlap_dispatched {"dispatched", overlap_engine_init_dispatched_, sz_overlap_scores};

/** One pair through the dispatched engine at one width, asserting the literal @p expected share. */
static void check_overlap_pair_(std::string const &query, std::string const &candidate, std::size_t width,
                                sz_f32_t expected) {
    handle_checked_heap_t heap;
    std::vector<std::string> const queries = {query}, candidates = {candidate};
    sz_sequence_t const query_sequence = sequence_from_(queries);
    sz_sequence_t const candidate_sequence = sequence_from_(candidates);
    sz_overlap_engine_t engine {};
    verify(sz_overlap_engine_init(&engine, &query_sequence, &width, 1, 0, sz::default_capabilities(), 0,
                                  &heap.allocator, nullptr) == sz_success_k);
    sz_f32_t share = -1.0f;
    verify(sz_overlap_scores(&engine, &candidate_sequence, &share, 1, 1, nullptr) == sz_success_k);
    sz_overlap_engine_free(&engine);
    verify(heap.live_allocations == 0);
    verify(share == expected);
}

#pragma endregion Helpers

#pragma region Unit

/** Known answers: the constants, the capacities, the strides, and shares readable off the texts. */
void test_overlap_unit() {
    // The modulus is prime, by trial division up to its root.
    std::uint64_t const prime = static_cast<std::uint64_t>(sz_overlap_modulus_k);
    for (std::uint64_t divisor = 2; divisor * divisor <= prime; ++divisor) verify(prime % divisor != 0);
    for (std::size_t exponent = 0; exponent != 9; ++exponent)
        verify(sz_overlap_powers_of_256_k[exponent] == sz_overlap_window_power(exponent));

    verify(sz_overlap_btree_sorted_capacity(0) == 64);
    verify(sz_overlap_btree_sorted_capacity(1) == 64);
    verify(sz_overlap_btree_sorted_capacity(64) == 64);
    verify(sz_overlap_btree_sorted_capacity(65) == 128);
    verify(sz_overlap_btree_sorted_capacity(117) == 128);

    // Identical texts share every window, disjoint alphabets share none, and a candidate narrower than the width
    // has no windows to share; "the" holds three, two and one windows below that, over the query's 43 bytes.
    std::string const fox = "the quick brown fox jumps over the lazy dog";
    std::vector<std::string> const queries = {fox};
    std::vector<std::string> const candidates = {fox, "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ", "the", ""};
    std::vector<std::size_t> const widths = {1, 2, 3, 4, 6, 8, 43};
    handle_checked_heap_t heap;
    sz_sequence_t const query_sequence = sequence_from_(queries);
    sz_sequence_t const sequence = sequence_from_(candidates);
    sz_overlap_engine_t engine {};
    verify(sz_overlap_engine_init(&engine, &query_sequence, widths.data(), widths.size(), 0, sz::default_capabilities(),
                                  0, &heap.allocator, nullptr) == sz_success_k);
    sz_f32_t shares[4 * 7];
    verify(sz_overlap_scores(&engine, &sequence, shares, 4 * 7, 7, nullptr) == sz_success_k);
    for (std::size_t width_index = 0; width_index != widths.size(); ++width_index) {
        verify(shares[0 * 7 + width_index] == 1.0f);
        verify(shares[1 * 7 + width_index] == 0.0f);
        verify(shares[3 * 7 + width_index] == 0.0f);
    }
    verify(shares[2 * 7 + 0] == static_cast<sz_f32_t>(3.0 / 43.0));
    verify(shares[2 * 7 + 1] == static_cast<sz_f32_t>(2.0 / 42.0));
    verify(shares[2 * 7 + 2] == static_cast<sz_f32_t>(1.0 / 41.0));
    for (std::size_t width_index = 3; width_index != widths.size(); ++width_index)
        verify(shares[2 * 7 + width_index] == 0.0f);

    // A candidate stride wider than the widths leaves the gap it names alone, and every share lands where it says.
    sz_f32_t padded[4 * 9];
    for (sz_f32_t &slot : padded) slot = -1.0f;
    verify(sz_overlap_scores(&engine, &sequence, padded, 4 * 9, 9, nullptr) == sz_success_k);
    for (std::size_t candidate = 0; candidate != candidates.size(); ++candidate) {
        for (std::size_t width_index = 0; width_index != widths.size(); ++width_index)
            verify(padded[candidate * 9 + width_index] == shares[candidate * 7 + width_index]);
        verify(padded[candidate * 9 + 7] == -1.0f && padded[candidate * 9 + 8] == -1.0f);
    }

    // A stride under the axis it spans would write one row into its neighbour's, so it is refused.
    verify(sz_overlap_scores(&engine, &sequence, shares, 4 * 7, 6, nullptr) == sz_unexpected_dimensions_k);
    verify(sz_overlap_scores(&engine, &sequence, shares, 4 * 7 - 1, 7, nullptr) == sz_unexpected_dimensions_k);
    sz_overlap_engine_free(&engine);

    // Zero widths answer nothing, and are refused before anything is prepared.
    sz_overlap_engine_t refused {};
    verify(sz_overlap_engine_init(&refused, &query_sequence, widths.data(), 0, 0, sz::default_capabilities(), 0,
                                  &heap.allocator, nullptr) == sz_unexpected_dimensions_k);
    verify(heap.live_allocations == 0);

    // "aaaa" holds one distinct window at width one, so "ab" finds one in four; the other way finds all four.
    check_overlap_pair_("aaaa", "ab", 1, 0.25f);
    check_overlap_pair_("ab", "aaaa", 1, 1.0f);

    // An empty side has no windows, on either side or both.
    check_overlap_pair_("", "abc", 1, 0.0f);
    check_overlap_pair_("abc", "", 1, 0.0f);
    check_overlap_pair_("", "", 1, 0.0f);
}

#pragma endregion Unit

#pragma region Drivers

/** Degenerate inputs for the window-overlap family, asserting survival and the stated refusals.
 *  Answers are not the subject here: empties and narrow candidates are accepted, zero widths and a
 *  refused allocation are reported, and no failure writes an output. */
void test_overlap_safety(test_context_t &) { check_overlap_safety_(overlap_dispatched); }

/** Holds the dispatched engine to the @c std::set oracle and to serial's answers, bit for bit. */
void test_overlap_all(test_context_t &context) { check_overlap_equivalence_(context, overlap_dispatched); }

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test
