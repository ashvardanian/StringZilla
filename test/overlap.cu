/**
 *  @brief  Window overlap on the GPU: the device engine against serial's answers, the memory contract the scoring
 *          verb keeps, the width bounds the per-thread ring imposes, and the asynchrony the verb promises.
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

#include <array>  // `std::array`
#include <string> // `std::string`
#include <vector> // `std::vector`

#include <fmt/format.h>

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
 *  a second copy of them, and the two candidate sequences differ only in whose accessors they carry. The queries
 *  stay plain host strings, because an engine's builder reads them on the host and no kernel ever sees them.
 */
struct overlap_cuda_corpus_t {
    std::vector<std::string> queries;       /**< The texts whose windows are sorted into the forest. */
    unified_vector<char> arena;             /**< Every candidate's bytes, back to back. */
    unified_vector<sz_string_view_t> views; /**< One view per candidate; its size is the candidate count. */
    unified_vector<sz_f32_t> scores;        /**< @b [queries,candidates,widths], written by whichever verb ran. */
    sz_sequence_t query_sequence {};        /**< Host accessors, which is what an engine's builder calls. */
    sz_sequence_t device_candidates {};     /**< Accessors a kernel calls, as the scoring verb requires. */
    sz_sequence_t host_candidates {};       /**< Accessors the serial reference calls, over the same views. */

    overlap_cuda_corpus_t(std::size_t queries_count, std::size_t count, std::size_t query_length,
                          std::size_t widths_count)
        : views(count), scores(queries_count * count * widths_count) {
        for (std::size_t index = 0; index != queries_count; ++index) {
            std::string text(query_length ? query_length - index % query_length : 0, '\0');
            if (!text.empty()) randomize_string(&text[0], text.size());
            queries.push_back(text);
        }
        for (std::size_t index = 0; index != count; ++index) arena.resize(arena.size() + 1 + (index * 37) % 900);
        randomize_string(arena.data(), arena.size());

        std::size_t written = 0;
        for (std::size_t index = 0; index != count; ++index) {
            std::size_t const length = 1 + (index * 37) % 900;
            views[index].start = arena.data() + written, views[index].length = length;
            written += length;
        }
        query_sequence = sequence_from_(queries);
        verify(sz_sequence_from_string_views_cuda(views.data(), views.size(), &device_candidates) == sz_success_k);
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
    }

    std::size_t candidate_stride() const noexcept { return scores.size() / (queries.size() * views.size()); }
    std::size_t query_stride() const noexcept { return views.size() * candidate_stride(); }
};

/** The serial backend's answers for the same corpus, read off the very bytes the device reads. */
static std::vector<sz_f32_t> overlap_serial_reference_(overlap_cuda_corpus_t const &corpus,
                                                       sz::span<sz_size_t const> widths) {
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);
    sz_overlap_engine_t engine {};
    verify(sz_overlap_engine_init_serial(&corpus.query_sequence, widths.data(), widths.size(), &host, &engine) ==
           sz_success_k);
    std::vector<sz_f32_t> expected(corpus.scores.size(), -1.0f);
    verify(sz_overlap_scores_serial(&engine, &corpus.host_candidates, expected.data(), corpus.query_stride(),
                                    corpus.candidate_stride()) == sz_success_k);
    sz_overlap_engine_free(&engine);
    return expected;
}

#pragma endregion Helpers

#pragma region Backends

/** @brief One CUDA backend's scoring verb, and the device it needs. */
struct overlap_cuda_backend_t {
    char const *name;           /**< The row's spelling, for @ref fail_backend_ and the log. */
    sz_capability_t required;   /**< The tier bits @ref sz_capabilities must carry for this row to run. */
    sz_overlap_scores_t scores; /**< The verb itself, which every row reaches through a device-built engine. */
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

    for (std::size_t queries_count : {1u, 3u})
        for (std::size_t count : {1u, 129u, 4096u})
            for (std::size_t query_length : {12u, 777u}) {
                overlap_cuda_corpus_t corpus(queries_count, count, query_length, widths.size());
                std::vector<sz_f32_t> const expected = overlap_serial_reference_(corpus,
                                                                                 {widths.data(), widths.size()});
                sz_overlap_engine_t engine {};
                if (sz_overlap_engine_init_gpu(&corpus.query_sequence, widths.data(), widths.size(), SZ_NULL, SZ_NULL,
                                               &engine) != sz_success_k)
                    fail_backend_(backend.name, "a device engine was refused for a well-formed batch");
                if (backend.scores(&engine, &corpus.device_candidates, corpus.scores.data(), corpus.query_stride(),
                                   corpus.candidate_stride()) != sz_success_k)
                    fail_backend_(backend.name, "a device-resident batch was refused");
                verify(cudaStreamSynchronize(SZ_NULL) == cudaSuccess);
                sz_overlap_engine_free(&engine);
                for (std::size_t slot = 0; slot != expected.size(); ++slot)
                    if (corpus.scores[slot] != expected[slot])
                        fail_backend_(backend.name, "a share differs from serial");
            }
}

/** One backend refusing the host memory no kernel can address, rather than reaching it as an invalid pointer. */
static void check_overlap_cuda_memory_safety_(overlap_cuda_backend_t const &backend) {
    std::array<sz_size_t, 2> const widths {4, 6};
    overlap_cuda_corpus_t corpus(1, 8, 333, widths.size());
    sz_overlap_engine_t engine {};
    if (sz_overlap_engine_init_gpu(&corpus.query_sequence, widths.data(), widths.size(), SZ_NULL, SZ_NULL, &engine) !=
        sz_success_k)
        fail_backend_(backend.name, "a device engine was refused for a well-formed batch");

    // The corpus's own views are unified, so a genuinely host-resident sequence needs its own plain storage.
    std::string text(64, '\0');
    randomize_string(&text[0], text.size());
    std::array<sz_string_view_t, 1> const host_views {sz_string_view_t {text.data(), text.size()}};
    sz_sequence_t host_candidates {};
    sz_sequence_from_string_views(host_views.data(), host_views.size(), &host_candidates);

    std::vector<sz_f32_t> host_scores(corpus.scores.size(), -1.0f);
    if (backend.scores(&engine, &corpus.device_candidates, host_scores.data(), corpus.query_stride(),
                       corpus.candidate_stride()) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "host scores were not refused");
    if (backend.scores(&engine, &host_candidates, corpus.scores.data(), corpus.query_stride(),
                       corpus.candidate_stride()) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "a host-resident sequence handle was not refused");
    for (sz_f32_t const untouched : host_scores)
        if (untouched != -1.0f) fail_backend_(backend.name, "a refused call still wrote a score");
    sz_overlap_engine_free(&engine);
}

/** The widest window the per-thread ring holds, and the two refusals one step past each of its bounds. */
static void check_overlap_cuda_width_safety_(overlap_cuda_backend_t const &backend) {
    std::array<sz_size_t, 1> const widest {sz_overlap_cuda_widest_window_k};
    std::array<sz_size_t, 1> const past {sz_overlap_cuda_widest_window_k + 1};
    std::array<sz_size_t, sz_overlap_cuda_widths_max_k + 1> too_many {};
    for (std::size_t index = 0; index != too_many.size(); ++index) too_many[index] = index + 1;

    // The one candidate is the query's own bytes in unified storage, so the text scores against itself.
    overlap_cuda_corpus_t corpus(1, 1, 256, widest.size());
    corpus.arena.assign(corpus.queries.front().begin(), corpus.queries.front().end());
    corpus.views[0].start = corpus.arena.data(), corpus.views[0].length = corpus.arena.size();

    sz_overlap_engine_t engine {};
    if (sz_overlap_engine_init_gpu(&corpus.query_sequence, past.data(), past.size(), SZ_NULL, SZ_NULL, &engine) !=
        sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "a width past the ring was not refused");
    if (sz_overlap_engine_init_gpu(&corpus.query_sequence, too_many.data(), too_many.size(), SZ_NULL, SZ_NULL,
                                   &engine) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "more widths than the registers hold were not refused");
    if (sz_overlap_engine_init_gpu(&corpus.query_sequence, widest.data(), widest.size(), SZ_NULL, SZ_NULL, &engine) !=
        sz_success_k)
        fail_backend_(backend.name, "the widest window the ring holds was refused");
    if (backend.scores(&engine, &corpus.device_candidates, corpus.scores.data(), corpus.query_stride(),
                       corpus.candidate_stride()) != sz_success_k)
        fail_backend_(backend.name, "a device-resident batch was refused");
    verify(cudaStreamSynchronize(SZ_NULL) == cudaSuccess);
    sz_overlap_engine_free(&engine);
    if (corpus.scores[0] != 1.0f) fail_backend_(backend.name, "a text does not fully overlap itself");
}

/** The scoring verb enqueues and returns, so a round big enough to outlive the call is still running after it. */
static void check_overlap_cuda_asynchrony_() {
    std::array<sz_size_t, 3> const widths {4, 6, 8};
    overlap_cuda_corpus_t corpus(8, 4096, 777, widths.size());
    cudaStream_t stream = SZ_NULL;
    verify(cudaStreamCreate(&stream) == cudaSuccess);

    sz_overlap_engine_t engine {};
    verify(sz_overlap_engine_init_gpu(&corpus.query_sequence, widths.data(), widths.size(), SZ_NULL, stream, &engine) ==
           sz_success_k);
    verify(sz_overlap_scores(&engine, &corpus.device_candidates, corpus.scores.data(), corpus.query_stride(),
                             corpus.candidate_stride()) == sz_success_k);
    verify(cudaStreamQuery(stream) == cudaErrorNotReady && "the scoring verb joined the stream it enqueued on");
    verify(cudaStreamSynchronize(stream) == cudaSuccess);
    sz_overlap_engine_free(&engine);
    verify(cudaStreamDestroy(stream) == cudaSuccess);
}

#pragma endregion Checks

#pragma region Drivers

/** @brief Every CUDA backend this device carries, against serial, over generated corpora. */
void test_overlap_all() {
    fmt::println("  - testing the CUDA window-overlap scores against the serial backend...");
    for (overlap_cuda_backend_t const &backend : overlap_cuda_backends) {
        if ((sz_capabilities() & backend.required) != backend.required) continue;
        check_overlap_cuda_equivalence_(backend);
    }
}

/** @brief Degenerate inputs, stated refusals, the bounds the per-thread ring imposes, and the asynchrony promised. */
void test_overlap_safety() {
    fmt::println("  - testing degenerate inputs and refused batches of the CUDA window-overlap kernels...");
    for (overlap_cuda_backend_t const &backend : overlap_cuda_backends) {
        if ((sz_capabilities() & backend.required) != backend.required) continue;
        check_overlap_cuda_memory_safety_(backend);
        check_overlap_cuda_width_safety_(backend);
    }
    if ((sz_capabilities() & sz_cap_cuda_k) == sz_cap_cuda_k) check_overlap_cuda_asynchrony_();
}

#pragma endregion Drivers
