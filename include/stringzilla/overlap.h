/**
 *  @file include/stringzilla/overlap.h
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief Window overlap between a batch of prepared queries and a batch of candidates, at one or
 *      more n-gram widths.
 *
 *  A window is a fixed-width byte n-gram, and the overlap of two texts at that width is the count
 *  of one's windows that occur in the other, over whichever of the two has more. Every query is
 *  hashed once into one B-tree over every width's window hashes, and every candidate then streams
 *  through the forest, hashing its own windows and probing. Nothing is stored per candidate, so the
 *  candidates may change round to round while the queries and the widths stay.
 *
 *  Window hashes come from a prefix difference, H(i, w) = P(i + w) − P(i) × bʷ mod p, which costs
 *  one modular multiply-add at any width and any offset; widths need not form a doubling chain. The
 *  modulus is a 32-bit prime chosen for collision weight, so a window hash is full-width.
 *
 *  @section overlap_api Public API
 *
 *  - @ref sz_overlap_engine_init_cpu → prepares a batch of queries on the host and fixes the tier;
 *  - @ref sz_overlap_engine_init_gpu → prepares the same batch on a device, on the caller's stream;
 *  - @ref sz_overlap_scores → the @b [queries,candidates,widths] shares of one round;
 *  - @ref sz_overlap_engine_free → returns both of the engine's blocks.
 *
 *  Every register-level step behind these - the prefix chain, the window hashing, the key sort and
 *  the B-tree probe - is public per backend in @c overlap/serial.h and its SIMD backends, named by
 *  positions per step, such as @c sz_overlap_f64x4_prefix_hash_step_haswell, so a caller driving
 *  its own loops binds those steps and skips these verbs entirely. The tree layout is the same on
 *  every backend, so any of them may probe it.
 */
#ifndef STRINGZILLA_OVERLAP_H_
#define STRINGZILLA_OVERLAP_H_

#include "stringzilla/types.h"

#include "stringzilla/overlap/serial.h"
#include "stringzilla/overlap/haswell.h"
#include "stringzilla/overlap/skylake.h"
#include "stringzilla/overlap/cuda.cuh"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Prepares @p queries on the host into one block from @p alloc, resolving the tier once.
 *
 *  Every width's window hashes of one query share that query's B-tree, so its sort runs once per
 *  batch rather than once per round, and the probe depth grows only logarithmically as widths are
 *  added to the batch.
 *
 *  @param[in] queries The texts whose windows are sorted into the forest; read on the host.
 *  @param[in] window_widths Window widths in bytes; a width past a text scores zero for its pairs.
 *  @param[in] window_widths_count Number of widths, which is the last axis of every output.
 *  @param[in] alloc Source of the forest's block, or @c STRINGZILLA_NULL for the default host
 *      allocator; stored by value, so @ref sz_overlap_engine_free needs no allocator of its own.
 *  @param[out] engine Left untouched unless the call succeeds.
 *  @return @c sz_success_k, @c sz_unexpected_dimensions_k for zero widths, or @c sz_bad_alloc_k
 *      when the forest's block cannot be taken.
 *  @note Selects the fastest implementation at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH, and records it, so a round never resolves a tier again.
 *  @sa sz_overlap_engine_init_serial, sz_overlap_engine_init_haswell
 *  @sa sz_overlap_engine_init_skylake
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_overlap_engine_init_cpu(sz_sequence_t const *queries,
                                                               sz_size_t const *window_widths,
                                                               sz_size_t window_widths_count,
                                                               sz_memory_allocator_t *alloc,
                                                               sz_overlap_engine_t *engine);

/**
 *  @brief Prepares @p queries on @p stream 's device, resolving the launch geometry once.
 *
 *  @param[in] alloc Unified and device-reachable, or @c STRINGZILLA_NULL to derive
 *      one from context.
 *  @param[in] stream A @c cudaStream_t the caller owns and keeps, or zero for the default stream.
 *  @param[out] engine Left untouched unless the call succeeds.
 *  @return @c sz_success_k; @c sz_unexpected_dimensions_k for zero widths or a width the device
 *      ring cannot hold; @c sz_device_memory_mismatch_k when @p alloc hands back memory the device
 *      cannot reach; or @c sz_device_code_mismatch_k when no GPU runtime is compiled in.
 *
 *  The device backend's per-thread ring bounds the widths, see @ref sz_overlap_cuda_widest_window_k
 *  and @ref sz_overlap_cuda_widths_max_k.
 *
 *  @note May join @p stream; no scoring verb ever does.
 *  @sa sz_overlap_engine_init_cuda
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_overlap_engine_init_gpu(sz_sequence_t const *queries,
                                                               sz_size_t const *window_widths,
                                                               sz_size_t window_widths_count,
                                                               sz_memory_allocator_t *alloc, void *stream,
                                                               sz_overlap_engine_t *engine);

/** Returns both of @p engine 's blocks to the allocator that built them, and leaves it empty. */
STRINGZILLA_API_RUNTIME void sz_overlap_engine_free(sz_overlap_engine_t *engine);

/**
 *  @brief Window overlap of every prepared query with every candidate, at every width of the batch.
 *
 *  @param[in] engine A batch prepared by @ref sz_overlap_engine_init_cpu or
 *      @ref sz_overlap_engine_init_gpu, which also picked the tier this runs on; its round block
 *      grows here and is never shrunk.
 *  @param[in] candidates The texts whose windows are probed against the forest's.
 *  @param[out] scores Shares in [0, 1] with a unit-strided width axis, laid out as shown below.
 *  @param[in] scores_query_stride Entries from one query's plane to the next, at least the
 *      candidates times @p scores_candidate_stride.
 *  @param[in] scores_candidate_stride Entries between candidate rows, at least the engine's widths.
 *  @return @c sz_success_k, @c sz_unexpected_dimensions_k when either stride is under the axis it
 *      spans, or @c sz_bad_alloc_k when the round's block cannot be grown.
 *  @note Asymmetric: a candidate's window occurrences count against a query's distinct windows, so
 *      swapping sides changes the score when either repeats a window.
 *  @note On a device engine this enqueues and returns; the caller joins the stream before reading
 *      @p scores.
 *  @sa sz_overlap_scores_serial, sz_overlap_scores_haswell, sz_overlap_scores_skylake
 *  @sa sz_overlap_scores_cuda
 *
 *  The share of query q, candidate c and width w sits at:
 *
 *  @verbatim
 *  scores[q * scores_query_stride + c * scores_candidate_stride + w]
 *  @endverbatim
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_overlap_scores(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                      sz_f32_t *scores, sz_size_t scores_query_stride,
                                                      sz_size_t scores_candidate_stride);

#pragma endregion Core API

/*  Pick the right implementation for the window overlap algorithms. To override this behavior and
 *  precompile all backends - set @c STRINGZILLA_RUNTIME_DISPATCH to 1. */
#pragma region Compile Time Dispatching
#if !STRINGZILLA_RUNTIME_DISPATCH

STRINGZILLA_API_RUNTIME sz_status_t sz_overlap_engine_init_cpu(sz_sequence_t const *queries,
                                                               sz_size_t const *window_widths,
                                                               sz_size_t window_widths_count,
                                                               sz_memory_allocator_t *alloc,
                                                               sz_overlap_engine_t *engine) {
#if STRINGZILLA_TARGET_SKYLAKE
    return sz_overlap_engine_init_skylake(queries, window_widths, window_widths_count, alloc, engine);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_overlap_engine_init_haswell(queries, window_widths, window_widths_count, alloc, engine);
#else
    return sz_overlap_engine_init_serial(queries, window_widths, window_widths_count, alloc, engine);
#endif
}

STRINGZILLA_API_RUNTIME sz_status_t sz_overlap_engine_init_gpu(sz_sequence_t const *queries,
                                                               sz_size_t const *window_widths,
                                                               sz_size_t window_widths_count,
                                                               sz_memory_allocator_t *alloc, void *stream,
                                                               sz_overlap_engine_t *engine) {
#if STRINGZILLA_TARGET_CUDA
    return sz_overlap_engine_init_cuda(queries, window_widths, window_widths_count, alloc, stream, engine);
#else
    sz_unused_(queries), sz_unused_(window_widths), sz_unused_(window_widths_count);
    sz_unused_(alloc), sz_unused_(stream), sz_unused_(engine);
    return sz_device_code_mismatch_k;
#endif
}

STRINGZILLA_API_RUNTIME void sz_overlap_engine_free(sz_overlap_engine_t *engine) { sz_overlap_engine_close_(engine); }

STRINGZILLA_API_RUNTIME sz_status_t sz_overlap_scores(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                      sz_f32_t *scores, sz_size_t scores_query_stride,
                                                      sz_size_t scores_candidate_stride) {
#if STRINGZILLA_TARGET_CUDA
    if (engine->capability & sz_caps_cuda_k)
        return sz_overlap_scores_cuda(engine, candidates, scores, scores_query_stride, scores_candidate_stride);
#endif
#if STRINGZILLA_TARGET_SKYLAKE
    if (engine->capability & sz_cap_skylake_k)
        return sz_overlap_scores_skylake(engine, candidates, scores, scores_query_stride, scores_candidate_stride);
#endif
#if STRINGZILLA_TARGET_HASWELL
    if (engine->capability & sz_cap_haswell_k)
        return sz_overlap_scores_haswell(engine, candidates, scores, scores_query_stride, scores_candidate_stride);
#endif
    return sz_overlap_scores_serial(engine, candidates, scores, scores_query_stride, scores_candidate_stride);
}

#endif // !STRINGZILLA_RUNTIME_DISPATCH
#pragma endregion Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_OVERLAP_H_
