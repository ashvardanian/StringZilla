/**
 *  @file test/overlap.cpp
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief Window overlap tests: known answers, integer and ordered-set oracles, and every
 *      backend against them.
 */
#undef NDEBUG // ! Enable all assertions for testing

/** The Visual C++ run-time library detects incorrect iterator use, and asserts and displays a
 *  dialog box at run time on Windows. */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

/*  ! Overload the following with caution. Those parameters must never be explicitly set during
 *  releases, but they come handy during development to validate ISA-specific implementations.
 *
 *  #define SZ_USE_HASWELL 0
 *  #define SZ_USE_SKYLAKE 0 */
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Make sure to include the StringZilla headers before anything else, to intercept missing
 *  `#include` directives and other issues. */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include <cmath>   // `std::fabs`
#include <cstdint> // `std::uint64_t`
#include <cstring> // `std::memcmp`, `std::strcmp`

#include <algorithm> // `std::sort`, `std::unique`, `std::binary_search`
#include <random>    // `std::uniform_int_distribution`
#include <set>       // Window-overlap oracle
#include <string>    // Baseline
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "stringzilla.hpp" // `global_random_generator`, `random_string`, `refusing_allocator_`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;

#pragma region Helpers

using overlap_prefix_hash_step_t = sz_f64_t (*)(sz_f64_t, sz_cptr_t, sz_f64_t *);
using overlap_prefix_hash_step_tail_t = sz_f64_t (*)(sz_f64_t, sz_cptr_t, sz_size_t, sz_f64_t *);
using overlap_window_hash_step_t = void (*)(sz_f64_t const *, sz_f64_t const *, sz_f64_t, sz_u32_t *);
using overlap_window_hash_step_tail_t = void (*)(sz_f64_t const *, sz_f64_t const *, sz_f64_t, sz_size_t, sz_u32_t *);
using overlap_btree_sort_t = sz_size_t (*)(sz_u32_t *, sz_size_t);
using overlap_btree_probe_t = sz_size_t (*)(sz_overlap_btree_t const *, sz_u32_t const *, sz_size_t);

/** One backend's step verbs, with no dispatcher; @c positions_per_step is one full step's reach. */
struct overlap_step_backend_t {
    char const *name;
    std::size_t positions_per_step;
    overlap_prefix_hash_step_t prefix_hash_step;
    overlap_prefix_hash_step_tail_t prefix_hash_step_tail;
    overlap_window_hash_step_t window_hash_step;
    overlap_window_hash_step_tail_t window_hash_step_tail;
    overlap_btree_sort_t btree_sort;
    overlap_btree_probe_t btree_probe;
};

using overlap_engine_init_t = sz_status_t (*)(sz_sequence_t const *, sz_size_t const *, sz_size_t,
                                              sz_memory_allocator_t *, sz_overlap_engine_t *);

/** One backend's engine constructor and the round it then scores. */
struct overlap_backend_t {
    char const *name;
    overlap_engine_init_t init;
    sz_overlap_scores_t scores;
};

/** Every step-verb backend compiled into this translation unit. */
static overlap_step_backend_t const overlap_step_backends[] = {
    {"serial", sz_overlap_serial_f64x1_positions_per_step_k, sz_overlap_f64x1_prefix_hash_step_serial,
     sz_overlap_f64x1_prefix_hash_step_tail_serial, sz_overlap_f64x1_window_hash_step_serial,
     sz_overlap_f64x1_window_hash_step_tail_serial, sz_overlap_u32x1_btree_sort_serial,
     sz_overlap_u32x1_btree_probe_serial},
#if SZ_USE_HASWELL
    {"haswell", sz_overlap_haswell_f64x4_positions_per_step_k, sz_overlap_f64x4_prefix_hash_step_haswell,
     sz_overlap_f64x4_prefix_hash_step_tail_haswell, sz_overlap_f64x4_window_hash_step_haswell,
     sz_overlap_f64x4_window_hash_step_tail_haswell, sz_overlap_u32x8_btree_sort_haswell,
     sz_overlap_u32x8_btree_probe_haswell},
#endif
#if SZ_USE_SKYLAKE
    {"skylake", sz_overlap_skylake_f64x8_positions_per_step_k, sz_overlap_f64x8_prefix_hash_step_skylake,
     sz_overlap_f64x8_prefix_hash_step_tail_skylake, sz_overlap_f64x8_window_hash_step_skylake,
     sz_overlap_f64x8_window_hash_step_tail_skylake, sz_overlap_u32x16_btree_sort_skylake,
     sz_overlap_u32x16_btree_probe_skylake},
#endif
};

/** Every whole-verb backend compiled into this translation unit, dispatched first. */
static overlap_backend_t const overlap_backends[] = {
    {"dispatched", sz_overlap_engine_init_cpu, sz_overlap_scores},
    {"serial", sz_overlap_engine_init_serial, sz_overlap_scores_serial},
#if SZ_USE_HASWELL
    {"haswell", sz_overlap_engine_init_haswell, sz_overlap_scores_haswell},
#endif
#if SZ_USE_SKYLAKE
    {"skylake", sz_overlap_engine_init_skylake, sz_overlap_scores_skylake},
#endif
};

/** The first @p end bytes of @p text as a big-endian base-256 number mod p, in exact integers. */
static sz_u32_t overlap_reference_prefix_hash_(std::string const &text, std::size_t end) {
    std::uint64_t const prime = static_cast<std::uint64_t>(sz_overlap_modulus_k);
    std::uint64_t residue = 0;
    for (std::size_t offset = 0; offset != end; ++offset)
        residue = (residue * 256 + static_cast<std::uint8_t>(text[offset])) % prime;
    return static_cast<sz_u32_t>(residue);
}

/** One window hash: its big-endian value mod p, hashed directly, not as a prefix difference. */
static sz_u32_t overlap_reference_window_hash_(std::string const &text, std::size_t start, std::size_t width) {
    std::uint64_t const prime = static_cast<std::uint64_t>(sz_overlap_modulus_k);
    std::uint64_t residue = 0;
    for (std::size_t offset = 0; offset != width; ++offset)
        residue = (residue * 256 + static_cast<std::uint8_t>(text[start + offset])) % prime;
    return static_cast<sz_u32_t>(residue);
}

/** The share of @p candidate windows among the distinct @p query windows, through @c std::set. */
static sz_f32_t overlap_reference_score_(std::string const &query, std::string const &candidate, std::size_t width) {
    std::size_t const query_windows = width <= query.size() ? query.size() - width + 1 : 0;
    std::size_t const candidate_windows = width <= candidate.size() ? candidate.size() - width + 1 : 0;
    std::set<std::string> present;
    for (std::size_t window = 0; window != query_windows; ++window) present.insert(query.substr(window, width));
    std::size_t matches = 0;
    for (std::size_t window = 0; window != candidate_windows; ++window)
        matches += present.count(candidate.substr(window, width));
    std::size_t const longer = std::max(query_windows, candidate_windows);
    return longer ? static_cast<sz_f32_t>(static_cast<double>(matches) / static_cast<double>(longer)) : 0.0f;
}

/** One backend's prefix chain over the whole text, through its full and tail steps. */
static void overlap_prefix_hashes_(overlap_step_backend_t const &backend, std::string const &text,
                                   std::vector<sz_f64_t> &prefix_hashes) {
    std::size_t const step = backend.positions_per_step;
    prefix_hashes.assign(text.size() + 1, 0.0);
    sz_f64_t prior = 0.0;
    std::size_t position = 0;
    for (; position + step <= text.size(); position += step)
        prior = backend.prefix_hash_step(prior, text.data() + position, prefix_hashes.data() + position + 1);
    if (position != text.size())
        backend.prefix_hash_step_tail(prior, text.data() + position, text.size() - position,
                                      prefix_hashes.data() + position + 1);
}

/** One backend's window hashes at one width, through its full and tail steps. */
static void overlap_window_hashes_(overlap_step_backend_t const &backend, std::vector<sz_f64_t> const &prefix_hashes,
                                   std::size_t width, std::vector<sz_u32_t> &window_hashes) {
    std::size_t const step = backend.positions_per_step;
    std::size_t const windows = prefix_hashes.size() - width;
    sz_f64_t const power = sz_overlap_window_power(width);
    window_hashes.assign(windows, 0);
    std::size_t window = 0;
    for (; window + step <= windows; window += step)
        backend.window_hash_step(prefix_hashes.data() + window, prefix_hashes.data() + window + width, power,
                                 window_hashes.data() + window);
    if (window != windows)
        backend.window_hash_step_tail(prefix_hashes.data() + window, prefix_hashes.data() + window + width, power,
                                      windows - window, window_hashes.data() + window);
}

/** Full-width keys with repeats: a fifth of them restate an earlier one. */
static std::vector<sz_u32_t> overlap_repeating_keys_(std::size_t count) {
    std::size_t const distinct_span = count - count / 5;
    std::vector<sz_u32_t> keys(count);
    for (std::size_t index = 0; index != count; ++index)
        keys[index] = static_cast<sz_u32_t>(index % (distinct_span ? distinct_span : 1)) * 2654435761u %
                      sz_overlap_modulus_k;
    return keys;
}

/** One backend's engine over @p queries, scoring @p candidates into a packed @b [Q,C,W] tensor. */
static std::vector<sz_f32_t> overlap_tensor_(overlap_backend_t const &backend, std::vector<std::string> const &queries,
                                             std::vector<std::string> const &candidates,
                                             std::vector<std::size_t> const &widths) {
    handle_checked_heap_t heap;
    sz_sequence_t const query_sequence = sequence_from_(queries);
    sz_sequence_t const candidate_sequence = sequence_from_(candidates);
    sz_overlap_engine_t engine {};
    if (backend.init(&query_sequence, widths.data(), widths.size(), &heap.allocator, &engine) != sz_success_k)
        fail_backend_(backend.name, "the engine refused a well-formed batch of queries");
    std::vector<sz_f32_t> scores(queries.size() * candidates.size() * widths.size(), -1.0f);
    if (backend.scores(&engine, &candidate_sequence, scores.data(), candidates.size() * widths.size(), widths.size()) !=
        sz_success_k)
        fail_backend_(backend.name, "the engine refused a well-formed batch of candidates");
    sz_overlap_engine_free(&engine);
    verify(heap.live_allocations == 0);
    return scores;
}

/** Runs @p queries against @p candidates through @p backend at @p widths, asserting each share
 *  matches the oracle to within one rounding. */
static void check_overlap_scores_(overlap_backend_t const &backend, std::vector<std::string> const &queries,
                                  std::vector<std::string> const &candidates, std::vector<std::size_t> const &widths) {
    std::vector<sz_f32_t> const computed = overlap_tensor_(backend, queries, candidates, widths);
    std::size_t const candidate_stride = widths.size(), query_stride = candidates.size() * widths.size();
    for (std::size_t query = 0; query != queries.size(); ++query)
        for (std::size_t candidate = 0; candidate != candidates.size(); ++candidate)
            for (std::size_t width_index = 0; width_index != widths.size(); ++width_index) {
                sz_f32_t const expected = overlap_reference_score_(queries[query], candidates[candidate],
                                                                   widths[width_index]);
                sz_f32_t const produced = computed[query * query_stride + candidate * candidate_stride + width_index];
                if (std::fabs(produced - expected) > 1e-6f)
                    fail_backend_(backend.name, "a share differs from the std::set oracle by more than one rounding");
            }
}

/** One pair through the dispatched engine at one width, asserting the literal @p expected share. */
static void check_overlap_pair_(std::string const &query, std::string const &candidate, std::size_t width,
                                sz_f32_t expected) {
    handle_checked_heap_t heap;
    std::vector<std::string> const queries = {query}, candidates = {candidate};
    sz_sequence_t const query_sequence = sequence_from_(queries);
    sz_sequence_t const candidate_sequence = sequence_from_(candidates);
    sz_overlap_engine_t engine {};
    verify(sz_overlap_engine_init_cpu(&query_sequence, &width, 1, &heap.allocator, &engine) == sz_success_k);
    sz_f32_t share = -1.0f;
    verify(sz_overlap_scores(&engine, &candidate_sequence, &share, 1, 1) == sz_success_k);
    sz_overlap_engine_free(&engine);
    verify(heap.live_allocations == 0);
    verify(share == expected);
}

/** One backend's chain and window hashes of @p text at widths to 12, against the integer oracle. */
static void check_overlap_steps_(overlap_step_backend_t const &backend, std::string const &text) {
    std::vector<sz_f64_t> prefix_hashes;
    std::vector<sz_u32_t> window_hashes;
    overlap_prefix_hashes_(backend, text, prefix_hashes);
    for (std::size_t end = 0; end <= text.size(); ++end)
        if (static_cast<sz_u32_t>(prefix_hashes[end]) != overlap_reference_prefix_hash_(text, end))
            fail_backend_(backend.name, "the prefix chain differs from the integer oracle");
    for (std::size_t width = 1; width <= 12 && width <= text.size(); ++width) {
        overlap_window_hashes_(backend, prefix_hashes, width, window_hashes);
        for (std::size_t window = 0; window != window_hashes.size(); ++window)
            if (window_hashes[window] != overlap_reference_window_hash_(text, window, width))
                fail_backend_(backend.name, "a window hash differs from the integer oracle");
    }
}

/** One backend's sort of @p count repeating keys against @c std::sort and @c std::unique. */
static void check_overlap_sort_(overlap_step_backend_t const &backend, std::size_t count) {
    std::vector<sz_u32_t> expected = overlap_repeating_keys_(count);
    std::vector<sz_u32_t> keys(sz_overlap_btree_sorted_capacity(count), 0);
    std::copy(expected.begin(), expected.end(), keys.begin());
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    std::size_t const distinct = backend.btree_sort(keys.data(), count);
    if (distinct != expected.size())
        fail_backend_(backend.name, "the sort counted a different number of distinct keys than std::unique");
    for (std::size_t index = 0; index != distinct; ++index)
        if (keys[index] != expected[index]) fail_backend_(backend.name, "the sorted keys differ from std::sort");
}

/** Probes @p stream through @p backend in shifting chunk lengths, against @c std::binary_search. */
static void check_overlap_probes_(overlap_step_backend_t const &backend, sz_overlap_btree_t const &btree,
                                  std::vector<sz_u32_t> const &present, std::vector<sz_u32_t> const &stream) {
    std::size_t const chunks[] = {0, 1, 3, 7, 8, 9, 15, 16, 17, 33, 100};
    std::size_t start = 0;
    for (std::size_t chunk_index = 0; start != stream.size(); ++chunk_index) {
        std::size_t const count = std::min(chunks[chunk_index % 11], stream.size() - start);
        std::size_t expected = 0;
        for (std::size_t offset = 0; offset != count; ++offset)
            expected += std::binary_search(present.begin(), present.end(), stream[start + offset]);
        if (backend.btree_probe(&btree, stream.data() + start, count) != expected)
            fail_backend_(backend.name, "a probe counted differently from std::binary_search");
        start += count;
    }
}

/** One backend's sort, layout and probe of @p count repeating keys, on hit- and miss-heavy runs. */
static void check_overlap_btree_(overlap_step_backend_t const &backend, std::size_t count) {
    std::mt19937 &generator = global_random_generator();
    std::uniform_int_distribution<sz_u32_t> below_modulus(0, sz_overlap_modulus_k - 1);
    std::vector<sz_u32_t> const raw = overlap_repeating_keys_(count);
    std::vector<sz_u32_t> nodes(sz_overlap_btree_entries(count), 0);
    std::copy(raw.begin(), raw.end(), nodes.begin());
    std::size_t const distinct = backend.btree_sort(nodes.data(), count);
    std::vector<sz_u32_t> const present(nodes.begin(), nodes.begin() + distinct);
    sz_overlap_btree_t btree {};
    verify(sz_overlap_btree_prepare(nodes.data(), distinct, &btree) == sz_success_k &&
           "The B-tree layout must accept every sorted key count");

    // The padding key itself sits outside the key domain, since a residue never reaches it; its neighbour is in.
    std::vector<sz_u32_t> const boundaries = {0u,
                                              sz_overlap_sign_flip_k - 1,
                                              sz_overlap_sign_flip_k,
                                              sz_overlap_modulus_k - 1,
                                              sz_overlap_modulus_k,
                                              sz_overlap_padding_key_k - 1};
    std::vector<sz_u32_t> hit_heavy = boundaries, miss_heavy = boundaries;
    for (std::size_t index = 0; index != raw.size(); ++index) {
        hit_heavy.push_back(raw[index]);
        if (index % 8 == 0) hit_heavy.push_back(below_modulus(generator));
        miss_heavy.push_back(below_modulus(generator));
        if (index % 8 == 0) miss_heavy.push_back(raw[index]);
    }
    for (sz_u32_t const key : present)
        for (sz_u32_t const neighbour : {key - 1, key + 1})
            if (neighbour != sz_overlap_padding_key_k) miss_heavy.push_back(neighbour);
    check_overlap_probes_(backend, btree, present, hit_heavy);
    check_overlap_probes_(backend, btree, present, miss_heavy);
}

#pragma endregion Helpers

#pragma region Unit

/** Known answers: the constants, the capacities, the strides, and shares readable off the texts. */
void test_overlap_unit() {
    fmt::println("  - testing window-overlap known-answer vectors...");

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
    verify(sz_overlap_engine_init_cpu(&query_sequence, widths.data(), widths.size(), &heap.allocator, &engine) ==
           sz_success_k);
    sz_f32_t shares[4 * 7];
    verify(sz_overlap_scores(&engine, &sequence, shares, 4 * 7, 7) == sz_success_k);
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
    verify(sz_overlap_scores(&engine, &sequence, padded, 4 * 9, 9) == sz_success_k);
    for (std::size_t candidate = 0; candidate != candidates.size(); ++candidate) {
        for (std::size_t width_index = 0; width_index != widths.size(); ++width_index)
            verify(padded[candidate * 9 + width_index] == shares[candidate * 7 + width_index]);
        verify(padded[candidate * 9 + 7] == -1.0f && padded[candidate * 9 + 8] == -1.0f);
    }

    // A stride under the axis it spans would write one row into its neighbour's, so it is refused.
    verify(sz_overlap_scores(&engine, &sequence, shares, 4 * 7, 6) == sz_unexpected_dimensions_k);
    verify(sz_overlap_scores(&engine, &sequence, shares, 4 * 7 - 1, 7) == sz_unexpected_dimensions_k);
    sz_overlap_engine_free(&engine);

    // Zero widths answer nothing, and are refused before anything is prepared.
    sz_overlap_engine_t refused {};
    verify(sz_overlap_engine_init_cpu(&query_sequence, widths.data(), 0, &heap.allocator, &refused) ==
           sz_unexpected_dimensions_k);
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

#pragma region Safety

/** One backend's engine on degenerate inputs: empties on either side, both, and none survive, as
 *  does a candidate narrower than the width; zero widths, a refused allocation and a stride under
 *  its own axis are reported without touching the outputs. */
static void check_overlap_safety_(overlap_backend_t const &backend) {
    handle_checked_heap_t heap;
    sz_memory_allocator_t refusing = refusing_allocator_();
    std::vector<std::string> const words = {"sitting", "kitten"};
    std::vector<std::string> const kitten = {"kitten"};
    std::vector<std::string> const empty = {""};
    std::vector<std::string> const narrow = {"ab"};
    std::vector<std::string> const none;
    std::vector<std::size_t> const widths = {3, 8};

    check_overlap_scores_(backend, empty, words, widths);
    check_overlap_scores_(backend, kitten, empty, widths);
    check_overlap_scores_(backend, empty, empty, widths);
    check_overlap_scores_(backend, kitten, none, widths);
    check_overlap_scores_(backend, none, words, widths);
    check_overlap_scores_(backend, kitten, narrow, widths);

    sz_sequence_t const kitten_sequence = sequence_from_(kitten);
    sz_sequence_t const words_sequence = sequence_from_(words);
    sz_overlap_engine_t engine {};
    if (backend.init(&kitten_sequence, widths.data(), 0, &heap.allocator, &engine) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "the engine accepted zero widths");
    if (backend.init(&kitten_sequence, widths.data(), widths.size(), &refusing, &engine) != sz_bad_alloc_k)
        fail_backend_(backend.name, "the engine did not report the refused allocation");
    if (backend.init(&kitten_sequence, widths.data(), widths.size(), &heap.allocator, &engine) != sz_success_k)
        fail_backend_(backend.name, "the engine refused a well-formed batch of queries");

    sz_f32_t refused[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    if (backend.scores(&engine, &words_sequence, refused, words.size() * widths.size(), widths.size() - 1) !=
        sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "the engine accepted a candidate stride under its widths");
    if (backend.scores(&engine, &words_sequence, refused, words.size() * widths.size() - 1, widths.size()) !=
        sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "the engine accepted a query stride under its candidates");
    for (sz_f32_t const untouched : refused)
        if (untouched != -1.0f) fail_backend_(backend.name, "a refused call still wrote a score");
    sz_overlap_engine_free(&engine);
    verify(heap.live_allocations == 0);
}

/** Degenerate inputs for the window-overlap family, asserting survival and the stated refusals.
 *  Answers are not the subject here: empties and narrow candidates are accepted, zero widths and a
 *  refused allocation are reported, and no failure writes an output. */
void test_overlap_safety() {
    fmt::println("  - testing degenerate inputs and refused allocations of the window-overlap kernels...");
    for (overlap_backend_t const &backend : overlap_backends) check_overlap_safety_(backend);
}

#pragma endregion Safety

#pragma region Drivers

/** One backend's step verbs against oracles over generated corpora: sort against @c std::sort, tree
 *  against @c std::binary_search, and chain and window hashes against integer hashing. */
static void check_overlap_step_oracles_(overlap_step_backend_t const &backend) {
    std::mt19937 &generator = global_random_generator();
    std::size_t const key_counts[] = {0, 1, 9, 63, 64, 65, 117, 512, 1000, 8187};
    for (std::size_t const count : key_counts) check_overlap_sort_(backend, count);
    for (std::size_t const count : key_counts) check_overlap_btree_(backend, count);

    std::size_t const lengths[] = {0, 1, 2, 7, 8, 9, 15, 16, 17, 64, 127, 293, 1024};
    for (std::size_t const length : lengths) {
        std::string text(length, '\0');
        randomize_string(&text[0], text.size());
        check_overlap_steps_(backend, text);
    }
    for (std::size_t round = 0; round != scale_iterations(8); ++round) {
        std::string query(std::uniform_int_distribution<std::size_t>(0, 700)(generator), '\0');
        randomize_string(&query[0], query.size());
        check_overlap_steps_(backend, query);
    }
}

/** One backend's engine against the @c std::set oracle over generated corpora, and one engine over
 *  every query against one engine per query, since batched and lone rows walk different forests. */
static void check_overlap_score_oracles_(overlap_backend_t const &backend) {
    std::mt19937 &generator = global_random_generator();
    std::vector<std::size_t> const widths = {1, 3, 4, 6, 8, 11, 32};
    for (std::size_t round = 0; round != scale_iterations(8); ++round) {
        std::size_t const query_length = std::uniform_int_distribution<std::size_t>(0, 700)(generator);
        std::string query(query_length, '\0');
        randomize_string(&query[0], query.size());

        // Full bytes rarely repeat a window past width three, so half the candidates are cut from the query itself
        // and the rest come from a two-letter alphabet where every width repeats.
        std::vector<std::string> candidates;
        randomize_strings(fuzzy_config_t("ab", 12, 0, 900), candidates);
        candidates.push_back(query);
        candidates.push_back(query.substr(query_length / 3));
        candidates.push_back(query + query);
        std::vector<std::string> const queries = {query, random_string(query_length, "ab", 2), std::string()};
        check_overlap_scores_(backend, queries, candidates, widths);

        std::vector<sz_f32_t> const batched = overlap_tensor_(backend, queries, candidates, widths);
        std::size_t const query_stride = candidates.size() * widths.size();
        for (std::size_t index = 0; index != queries.size(); ++index) {
            std::vector<std::string> const alone = {queries[index]};
            std::vector<sz_f32_t> const lone = overlap_tensor_(backend, alone, candidates, widths);
            if (std::memcmp(batched.data() + index * query_stride, lone.data(), query_stride * sizeof(sz_f32_t)) != 0)
                fail_backend_(backend.name, "a query's row differs between a batched engine and its own");
        }
    }
}

/** One backend's engine against the serial backend's, bit for bit, over random query batches and
 *  two-letter candidate batches at every width. */
static void check_overlap_equivalence_(overlap_backend_t const &reference, overlap_backend_t const &candidate,
                                       std::size_t rounds) {
    std::mt19937 &generator = global_random_generator();
    std::vector<std::size_t> const widths = {1, 3, 4, 6, 8, 11, 32};
    std::vector<std::string> candidates;
    for (std::size_t round = 0; round != rounds; ++round) {
        std::string query(std::uniform_int_distribution<std::size_t>(0, 700)(generator), '\0');
        randomize_string(&query[0], query.size());
        randomize_strings(fuzzy_config_t("ab", 12, 0, 900), candidates);
        candidates.push_back(query);
        candidates.push_back(query.substr(query.size() / 3));
        std::vector<std::string> const queries = {query, query.substr(query.size() / 2), std::string()};

        std::vector<sz_f32_t> const from_reference = overlap_tensor_(reference, queries, candidates, widths);
        std::vector<sz_f32_t> const from_candidate = overlap_tensor_(candidate, queries, candidates, widths);
        if (std::memcmp(from_reference.data(), from_candidate.data(), from_reference.size() * sizeof(sz_f32_t)) != 0)
            fail_backend_(candidate.name, "the engine's scores disagreed with serial");
    }
}

/** Drives the oracles and the serial-versus-SIMD differential across every backend compiled here:
 *  the step verbs against their integer and @c std:: oracles, the engines against @c std::set, and
 *  every engine against serial's answers bit for bit. */
void test_overlap_all() {
    for (overlap_step_backend_t const &backend : overlap_step_backends) check_overlap_step_oracles_(backend);
    for (overlap_backend_t const &backend : overlap_backends) check_overlap_score_oracles_(backend);

    // Serial is the reference for everything, itself included.
    overlap_backend_t const &reference = backend_named_(overlap_backends, "serial");
    for (overlap_backend_t const &candidate : overlap_backends)
        check_overlap_equivalence_(reference, candidate, scale_iterations(8));
}

#pragma endregion Drivers
