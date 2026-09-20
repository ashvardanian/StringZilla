/**
 *  @brief  Levenshtein distances on the GPU: the device backend against serial's answers, the memory contract the
 *          two verbs keep, and the query bound the per-thread verticals impose.
 *  @file   test/levenshtein.cu
 *  @author Ash Vardanian
 *  @date   September 15, 2026
 *
 *  @c test/levenshtein.cpp defines @c test_levenshtein_all and @c test_levenshtein_safety over the CPU backend
 *  table, and this file defines them over the CUDA one. No target links both - @c stringzilla_test_cpp20 takes
 *  the first and @c stringzilla_test_cu20 the second - a CMake invariant rather than a language one.
 *
 *  The sibling @c test/levenshtein.cpp drives the step primitives and the CPU backends; nothing here repeats that.
 *  These are the cases a host translation unit cannot express: device-reachable memory, a device-bound sequence,
 *  a caller's own stream, and the refusals that keep a host pointer from reaching a kernel as an address.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::printf`

#include <array>  // `std::array`
#include <string> // `std::string`
#include <vector> // `std::vector`

#include <stringzilla/levenshtein.h> // `sz_levenshtein_*`
#include <stringzilla/stringzilla.h> // Primary C API

#include "stringzilla.hpp" // `randomize_string`, `verify`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;

#pragma region Helpers

/**
 *  @brief A corpus both sides address: one arena of near-duplicates, the views into it, and room for the distances.
 *
 *  The candidates are edits of the query, so the distances are small and every word of the Myers state moves.
 *  Unified storage is readable from the host, so the serial reference runs against these very bytes.
 */
struct levenshtein_cuda_corpus_t {
    unified_vector<char> query;             /**< The pattern every candidate is scored against. */
    unified_vector<char> arena;             /**< Every candidate's bytes, back to back. */
    unified_vector<sz_string_view_t> views; /**< One view per candidate; its size is the candidate count. */
    unified_vector<sz_size_t> distances;    /**< @b [candidates], written by whichever verb ran. */
    sz_sequence_t device_candidates {};     /**< Accessors a kernel calls, as the strict verb requires. */
    sz_sequence_t host_candidates {};       /**< Accessors the serial reference calls, over the same views. */

    levenshtein_cuda_corpus_t(std::size_t count, std::size_t query_length)
        : query(query_length), views(count), distances(count) {
        randomize_string(query.data(), query.size());

        std::string edited;
        for (std::size_t index = 0; index != count; ++index) {
            edited.assign(query.data(), query.size());
            for (std::size_t edit = 0, edits = index % 17; edit != edits && edited.size() > 1; ++edit) {
                std::size_t const at = (index * 31 + edit * 7) % edited.size();
                if (edit % 2) { edited.erase(at, 1); }
                else { edited[at] = (char)('a' + (edit + index) % 26); }
            }
            views[index].length = edited.size();
            arena.insert(arena.end(), edited.begin(), edited.end());
        }
        // The arena's address is only final once it has stopped growing, so the starts are filled afterwards.
        std::size_t written = 0;
        for (sz_string_view_t &view : views) view.start = arena.data() + written, written += view.length;
        verify(sz_sequence_from_string_views_cuda(views.data(), views.size(), &device_candidates) ==
               sz_success_k);
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
    }
};

/** The serial backend's answers for the same corpus, read off the very bytes the device reads. */
static std::vector<sz_size_t> levenshtein_serial_reference_(levenshtein_cuda_corpus_t const &corpus) {
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);
    std::vector<sz_size_t> expected(corpus.distances.size());
    verify(sz_levenshtein_distances_serial(corpus.query.data(), corpus.query.size(), &corpus.host_candidates, &host,
                                           expected.data()) == sz_success_k);
    return expected;
}

#pragma endregion Helpers

#pragma region Backends

/** @brief One CUDA backend's one-to-many verb, and the device it needs. */
struct levenshtein_cuda_backend_t {
    char const *name;         /**< The row's spelling, for @ref fail_backend_ and the log. */
    sz_capability_t required; /**< The tier bits @ref sz_capabilities must carry for this row to run. */
    sz_levenshtein_distances_t distances; /**< The verb itself, which stages whatever the device cannot reach. */
};

/**
 *  @brief Every CUDA backend compiled into this translation unit, dispatched first.
 *
 *  Unlike the CPU tables, whose rows a `#if` selects, every CUDA tier is compiled into one fatbin and chosen at
 *  run time - so a row states the tier bits it needs and the drivers skip it on a device that lacks them.
 */
static levenshtein_cuda_backend_t const levenshtein_cuda_backends[] = {
    {"dispatched", sz_cap_cuda_k, sz_levenshtein_distances},
    {"cuda", sz_cap_cuda_k, sz_levenshtein_distances_cuda},
};

#pragma endregion Backends

#pragma region Checks

/** One backend's distances against serial's, on a corpus that never leaves the device. */
static void check_levenshtein_cuda_equivalence_(levenshtein_cuda_backend_t const &backend) {
    sz_memory_allocator_t unified;
    sz_memory_allocator_init_unified(&unified);

    // One word, several words, and the last query the per-thread verticals hold.
    for (std::size_t query_length : {1u, 64u, 65u, 300u, (unsigned)(sz_levenshtein_cuda_words_max_k * 64)})
        for (std::size_t count : {1u, 129u, 2048u}) {
            levenshtein_cuda_corpus_t corpus(count, query_length);
            std::vector<sz_size_t> const expected = levenshtein_serial_reference_(corpus);

            if (backend.distances(corpus.query.data(), corpus.query.size(), &corpus.device_candidates, &unified,
                                  corpus.distances.data()) != sz_success_k)
                fail_backend_(backend.name, "a device-resident batch was refused");
            for (std::size_t index = 0; index != expected.size(); ++index)
                if (corpus.distances[index] != expected[index])
                    fail_backend_(backend.name, "a distance differs from serial");
        }
}

/** One backend staging host memory it cannot reach, and answering what serial answers. */
static void check_levenshtein_cuda_memory_safety_(levenshtein_cuda_backend_t const &backend) {
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);

    std::string query(200, '\0');
    randomize_string(&query[0], query.size());
    std::string arena(660, '\0');
    randomize_string(&arena[0], arena.size());
    std::array<sz_string_view_t, 4> const views {sz_string_view_t {arena.data(), 1},
                                                 {arena.data() + 1, 60},
                                                 {arena.data() + 61, 199},
                                                 {arena.data() + 260, 400}};
    sz_sequence_t candidates {};
    sz_sequence_from_string_views(views.data(), views.size(), &candidates);

    std::vector<sz_size_t> expected(views.size()), produced(views.size());
    verify(sz_levenshtein_distances_serial(query.data(), query.size(), &candidates, &host, expected.data()) ==
           sz_success_k);
    if (backend.distances(query.data(), query.size(), &candidates, &host, produced.data()) != sz_success_k)
        fail_backend_(backend.name, "host memory was refused by the verb that must stage it");
    if (produced != expected) fail_backend_(backend.name, "a staged round differs from serial");
}

/** One backend's refusal of a query past what the per-thread verticals hold. */
static void check_levenshtein_cuda_query_safety_(levenshtein_cuda_backend_t const &backend) {
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);
    std::string query(sz_levenshtein_cuda_words_max_k * 64 + 1, '\0');
    randomize_string(&query[0], query.size());
    sz_string_view_t const view {query.data(), query.size()};
    sz_sequence_t candidates {};
    sz_sequence_from_string_views(&view, 1, &candidates);

    sz_size_t distance = 0;
    if (backend.distances(query.data(), query.size(), &candidates, &host, &distance) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "a query past the verticals was not refused");
}

/** The verb a caller schedules on its own stream refuses host memory rather than staging it. */
static void check_levenshtein_cuda_scheduled_refusal_() {
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);
    std::string query(200, '\0');
    randomize_string(&query[0], query.size());
    sz_string_view_t const view {query.data(), query.size()};
    sz_sequence_t candidates {};
    sz_sequence_from_string_views(&view, 1, &candidates);
    sz_size_t distance = 0;
    verify(sz_levenshtein_distances_scheduled_cuda(query.data(), query.size(), &candidates, &host, &distance,
                                                   SZ_NULL) == sz_device_memory_mismatch_k);
}

#pragma endregion Checks

#pragma region Drivers

/** @brief Every CUDA backend this device carries, against serial, over generated corpora. */
void test_levenshtein_all() {
    std::printf("  - testing the CUDA edit distances against the serial backend...\n");
    for (levenshtein_cuda_backend_t const &backend : levenshtein_cuda_backends) {
        if ((sz_capabilities() & backend.required) != backend.required) continue;
        check_levenshtein_cuda_equivalence_(backend);
    }
}

/** @brief Degenerate inputs, stated refusals, and the bound the per-thread verticals impose. */
void test_levenshtein_safety() {
    std::printf("  - testing degenerate inputs and refused batches of the CUDA edit-distance kernels...\n");
    for (levenshtein_cuda_backend_t const &backend : levenshtein_cuda_backends) {
        if ((sz_capabilities() & backend.required) != backend.required) continue;
        check_levenshtein_cuda_memory_safety_(backend);
        check_levenshtein_cuda_query_safety_(backend);
    }
    check_levenshtein_cuda_scheduled_refusal_();
}

#pragma endregion Drivers
