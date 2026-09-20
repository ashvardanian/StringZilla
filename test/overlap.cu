/**
 *  @brief  Window overlap on the GPU: the device backend against serial's answers, the memory contract the two
 *          verbs keep, and the width bound the per-thread ring imposes.
 *  @file   test/overlap.cu
 *  @author Ash Vardanian
 *  @date   September 15, 2026
 *
 *  @c test/overlap.cpp defines @c test_overlap_all and @c test_overlap_safety over the CPU backend table, and
 *  this file defines them over the CUDA one. No target links both - @c stringzilla_test_cpp20 takes the first
 *  and @c stringzilla_test_cu20 the second - which is a CMake invariant rather than a language one.
 *
 *  The sibling @c test/overlap.cpp drives the step primitives and the CPU backends; nothing here repeats that.
 *  These are the cases a host translation unit cannot express: device-reachable memory, a device-bound sequence,
 *  a caller's own stream, and the refusals that keep a host pointer from reaching a kernel as an address.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::printf`

#include <array>  // `std::array`
#include <string> // `std::string`
#include <vector> // `std::vector`

#include <stringzilla/overlap.h>     // `sz_overlap_*`
#include <stringzilla/stringzilla.h> // Primary C API

#include "stringzilla.hpp" // `randomize_string`, `verify`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;

#pragma region Helpers

/**
 *  @brief A corpus both sides address: one arena, the views into it, and the room for one round's scores.
 *
 *  Unified storage is readable from the host, so the serial reference runs against these very bytes rather than
 *  a second copy of them, and the two sequences differ only in whose accessors they carry.
 */
struct overlap_cuda_corpus_t {
    unified_vector<char> query;             /**< The text whose windows are sorted into the tree. */
    unified_vector<char> arena;             /**< Every candidate's bytes, back to back. */
    unified_vector<sz_string_view_t> views; /**< One view per candidate; its size is the candidate count. */
    unified_vector<sz_f32_t> scores;        /**< @b [candidates,window_widths], written by whichever verb ran. */
    sz_sequence_t device_candidates {};     /**< Accessors a kernel calls, as the strict verb requires. */
    sz_sequence_t host_candidates {};       /**< Accessors the serial reference calls, over the same views. */

    overlap_cuda_corpus_t(std::size_t count, std::size_t query_length, std::size_t widths_count)
        : query(query_length), views(count), scores(count * widths_count) {
        randomize_string(query.data(), query.size());
        for (std::size_t index = 0; index != count; ++index) arena.resize(arena.size() + 1 + (index * 37) % 900);
        randomize_string(arena.data(), arena.size());

        std::size_t written = 0;
        for (std::size_t index = 0; index != count; ++index) {
            std::size_t const length = 1 + (index * 37) % 900;
            views[index].start = arena.data() + written, views[index].length = length;
            written += length;
        }
        verify(sz_sequence_from_string_views_cuda(views.data(), views.size(), &device_candidates) == sz_success_k);
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
    }

    std::size_t widths_count() const noexcept { return scores.size() / views.size(); }
};

/** The serial backend's answers for the same corpus, read off the very bytes the device reads. */
static std::vector<sz_f32_t> overlap_serial_reference_(overlap_cuda_corpus_t const &corpus,
                                                       sz::span<sz_size_t const> widths) {
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);
    std::vector<sz_f32_t> expected(corpus.scores.size());
    verify(sz_overlap_scores_serial(corpus.query.data(), corpus.query.size(), &corpus.host_candidates, widths.data(),
                                    widths.size(), &host, expected.data()) == sz_success_k);
    return expected;
}

#pragma endregion Helpers

#pragma region Backends

/** @brief One CUDA backend's one-to-many verb, and the device it needs. */
struct overlap_cuda_backend_t {
    char const *name;         /**< The row's spelling, for @ref fail_backend_ and the log. */
    sz_capability_t required; /**< The tier bits @ref sz_capabilities must carry for this row to run. */
    sz_overlap_scores_t scores; /**< The verb itself, which stages whatever the device cannot reach. */
};

/**
 *  @brief Every CUDA backend compiled into this translation unit, dispatched first.
 *
 *  Unlike the CPU tables, whose rows a `#if` selects, every CUDA tier is compiled into one fatbin and chosen at
 *  run time - so a row states the tier bits it needs and the drivers skip it on a device that lacks them.
 */
static overlap_cuda_backend_t const overlap_cuda_backends[] = {
    {"dispatched", sz_cap_cuda_k, sz_overlap_scores},
    {"cuda", sz_cap_cuda_k, sz_overlap_scores_cuda},
};

#pragma endregion Backends

#pragma region Checks

/** One backend's scores against serial's, on a corpus that never leaves the device. */
static void check_overlap_cuda_equivalence_(overlap_cuda_backend_t const &backend) {
    std::array<sz_size_t, 3> const widths {4, 6, 8};
    sz_memory_allocator_t unified;
    sz_memory_allocator_init_unified(&unified);

    for (std::size_t count : {1u, 129u, 4096u})
        for (std::size_t query_length : {12u, 777u}) {
            overlap_cuda_corpus_t corpus(count, query_length, widths.size());
            std::vector<sz_f32_t> const expected = overlap_serial_reference_(corpus, {widths.data(), widths.size()});

            if (backend.scores(corpus.query.data(), corpus.query.size(), &corpus.device_candidates, widths.data(),
                               widths.size(), &unified, corpus.scores.data()) != sz_success_k)
                fail_backend_(backend.name, "a device-resident batch was refused");
            for (std::size_t slot = 0; slot != expected.size(); ++slot)
                if (corpus.scores[slot] != expected[slot]) fail_backend_(backend.name, "a share differs from serial");
        }
}

/** One backend staging host memory it cannot reach, and answering what serial answers. */
static void check_overlap_cuda_memory_safety_(overlap_cuda_backend_t const &backend) {
    std::array<sz_size_t, 2> const widths {4, 6};
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);

    std::string query(333, '\0');
    randomize_string(&query[0], query.size());
    std::string arena(541, '\0');
    randomize_string(&arena[0], arena.size());
    std::array<sz_string_view_t, 3> const views {
        sz_string_view_t {arena.data(), 1}, {arena.data() + 1, 40}, {arena.data() + 41, 500}};
    sz_sequence_t candidates {};
    sz_sequence_from_string_views(views.data(), views.size(), &candidates);

    std::vector<sz_f32_t> expected(views.size() * widths.size()), produced(views.size() * widths.size());
    verify(sz_overlap_scores_serial(query.data(), query.size(), &candidates, widths.data(), widths.size(), &host,
                                    expected.data()) == sz_success_k);
    if (backend.scores(query.data(), query.size(), &candidates, widths.data(), widths.size(), &host,
                       produced.data()) != sz_success_k)
        fail_backend_(backend.name, "host memory was refused by the verb that must stage it");
    if (produced != expected) fail_backend_(backend.name, "a staged round differs from serial");
}

/** One backend's widest window, and the refusal one past it. */
static void check_overlap_cuda_width_safety_(overlap_cuda_backend_t const &backend) {
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);
    std::string query(256, '\0');
    randomize_string(&query[0], query.size());
    sz_string_view_t const view {query.data(), query.size()};
    sz_sequence_t candidates {};
    sz_sequence_from_string_views(&view, 1, &candidates);

    std::array<sz_size_t, 1> const widest {sz_overlap_cuda_widest_window_k};
    std::array<sz_size_t, 1> const past {sz_overlap_cuda_widest_window_k + 1};
    sz_f32_t score = 0.0f;
    if (backend.scores(query.data(), query.size(), &candidates, widest.data(), widest.size(), &host, &score) !=
        sz_success_k)
        fail_backend_(backend.name, "the widest window the ring holds was refused");
    if (score != 1.0f) fail_backend_(backend.name, "a text does not fully overlap itself");
    if (backend.scores(query.data(), query.size(), &candidates, past.data(), past.size(), &host, &score) !=
        sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "a width past the ring was not refused");
}

/** The verb a caller schedules on its own stream refuses host memory rather than staging it. */
static void check_overlap_cuda_scheduled_refusal_() {
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);
    std::array<sz_size_t, 2> const widths {4, 6};
    std::string query(333, '\0');
    randomize_string(&query[0], query.size());
    sz_string_view_t const view {query.data(), query.size()};
    sz_sequence_t candidates {};
    sz_sequence_from_string_views(&view, 1, &candidates);
    std::array<sz_f32_t, 2> produced {};
    verify(sz_overlap_scores_scheduled_cuda(query.data(), query.size(), &candidates, widths.data(), widths.size(),
                                            &host, produced.data(), SZ_NULL) == sz_device_memory_mismatch_k);
}

#pragma endregion Checks

#pragma region Drivers

/** @brief Every CUDA backend this device carries, against serial, over generated corpora. */
void test_overlap_all() {
    std::printf("  - testing the CUDA window-overlap scores against the serial backend...\n");
    for (overlap_cuda_backend_t const &backend : overlap_cuda_backends) {
        if ((sz_capabilities() & backend.required) != backend.required) continue;
        check_overlap_cuda_equivalence_(backend);
    }
}

/** @brief Degenerate inputs, stated refusals, and the bound the per-thread ring imposes. */
void test_overlap_safety() {
    std::printf("  - testing degenerate inputs and refused batches of the CUDA window-overlap kernels...\n");
    for (overlap_cuda_backend_t const &backend : overlap_cuda_backends) {
        if ((sz_capabilities() & backend.required) != backend.required) continue;
        check_overlap_cuda_memory_safety_(backend);
        check_overlap_cuda_width_safety_(backend);
    }
    check_overlap_cuda_scheduled_refusal_();
}

#pragma endregion Drivers
