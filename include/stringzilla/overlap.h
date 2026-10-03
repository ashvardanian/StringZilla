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
 *  - @ref sz_overlap_engine_init → prepares a batch of queries for the best capability of a mask,
 *    on the host or on one device;
 *  - @ref sz_overlap_scores → the @b [queries,candidates,widths] shares of one round, on the
 *    capability its engine was prepared for;
 *  - @ref sz_overlap_engine_free → returns both of the engine's blocks;
 *  - @ref sz_overlap_find_kernel → the kernel either would run, for a caller that keeps it.
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
#include "stringzilla/capabilities.h" // `sz_capability_t`
#include "stringzilla/cuda.cuh"       // Ahead of `extern "C"`, as the GPU runtimes' headers declare templates
#include "stringzilla/rocm.cuh"
#include "stringzilla/metal.h"
#include "stringzilla/overlap/serial.h" // `sz_overlap_engine_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Prepares @p queries into one block from @p allocator, for the best capability of
 *      @p capabilities that has an init kernel.
 *
 *  Every width's window hashes of one query share that query's B-tree, so its sort runs once per
 *  batch rather than once per round, and the probe depth grows only logarithmically as widths are
 *  added to the batch.
 *
 *  @param[out] engine Left untouched unless the call succeeds.
 *  @param[in] queries The texts whose windows are sorted into the forest; read on the host.
 *  @param[in] window_widths Window widths in bytes; a width past a text scores zero for its pairs.
 *  @param[in] window_widths_count Number of widths, which is the last axis of every output.
 *  @param[in] candidates_budget Candidates one round may carry, for a backend whose rounds keep
 *      state per candidate, zero asking for its default; one that keeps none ignores it.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled or
 *      @c sz_metal_capabilities_enabled report; its group picks the CPU or a GPU vendor.
 *  @param[in] allocator Source of the forest's block, or @c STRINGZILLA_NULL for
 *      @ref sz_allocator_init_unified_best of @p capabilities; stored by value, so
 *      @ref sz_overlap_engine_free needs no allocator of its own.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, also naming the device:
 *      a @c cudaStream_t, a @c hipStream_t, or an @c id<MTLCommandQueue>; null for the default.
 *  @return @c sz_success_k; @c sz_missing_kernel_k when no capability of the mask has an init;
 *      @c sz_unexpected_dimensions_k for zero widths, or on a device a width its ring cannot hold;
 *      @c sz_bad_alloc_k when the forest's block cannot be taken; or on a device
 *      @c sz_device_memory_mismatch_k for memory or a @p stream it cannot use, @c sz_missing_gpu_k
 *      when it doesn't answer, or @c sz_device_code_mismatch_k when a kernel fails to build.
 *  @note May join @p stream; no scoring verb ever does.
 */
STRINGZILLA_API sz_status_t sz_overlap_engine_init(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                   sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                   sz_size_t candidates_budget, sz_capability_t capabilities,
                                                   sz_allocator_t *allocator, void *stream);

/** Returns both of @p engine 's blocks to the allocator that built them, once the work queued on
 *  @p stream is done with them, and leaves it empty. */
STRINGZILLA_API void sz_overlap_engine_free(sz_overlap_engine_t *engine, void *stream);

/**
 *  @brief Window overlap of every prepared query with every candidate, at every width of the batch,
 *      by the kernel of the capability @p engine was prepared for.
 *
 *  @param[in] engine The prepared forest; a host round block grows here and is never shrunk. A
 *      device engine keeps no stream, so on CUDA any number of streams may score it at once; on
 *      Metal its one round block orders the rounds on the device's one queue.
 *  @param[in] candidates The texts whose windows are probed against the forest's, on the residency
 *      the engine was built for.
 *  @param[out] scores Shares in [0, 1] with a unit-strided width axis, laid out as shown below.
 *  @param[in] scores_query_stride Entries from one query's plane to the next, at least the
 *      candidates times @p scores_candidate_stride.
 *  @param[in] scores_candidate_stride Entries between candidate rows, at least the engine's widths.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, which also names the device
 *      the round runs on; null for the default.
 *  @return @c sz_success_k, on a device once enqueued; @c sz_unexpected_dimensions_k when either
 *      stride is under the axis it spans; @c sz_bad_alloc_k when a host round's block cannot be
 *      grown; @c sz_missing_kernel_k for an empty engine; or on a device
 *      @c sz_device_memory_mismatch_k for an engine, an output, a sequence or a @p stream the
 *      device of @p stream cannot use, and @c sz_device_code_mismatch_k for a launch it refused.
 *  @note Asymmetric: a candidate's window occurrences count against a query's distinct windows, so
 *      swapping sides changes the score when either repeats a window.
 *  @note Never joins, so the caller joins @p stream before reading @p scores.
 *
 *  The share of query q, candidate c and width w sits at:
 *
 *  @verbatim
 *  scores[q * scores_query_stride + c * scores_candidate_stride + w]
 *  @endverbatim
 */
STRINGZILLA_API sz_status_t sz_overlap_scores(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                              sz_f32_t *scores, sz_size_t scores_query_stride,
                                              sz_size_t scores_candidate_stride, void *stream);

/** @copydoc sz_overlap_engine_init */
STRINGZILLA_API sz_status_t sz_overlap_engine_init_serial(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                          sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                          sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                          void *stream);
/** @copydoc sz_overlap_scores */
STRINGZILLA_API sz_status_t sz_overlap_scores_serial(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                     sz_f32_t *scores, sz_size_t scores_query_stride,
                                                     sz_size_t scores_candidate_stride, void *stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_overlap_engine_init */
STRINGZILLA_API sz_status_t sz_overlap_engine_init_haswell(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                           sz_size_t const *window_widths,
                                                           sz_size_t window_widths_count, sz_size_t candidates_budget,
                                                           sz_allocator_t *allocator, void *stream);
/** @copydoc sz_overlap_scores */
STRINGZILLA_API sz_status_t sz_overlap_scores_haswell(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                      sz_f32_t *scores, sz_size_t scores_query_stride,
                                                      sz_size_t scores_candidate_stride, void *stream);
#endif

#if STRINGZILLA_TARGET_SKYLAKE
/** @copydoc sz_overlap_engine_init */
STRINGZILLA_API sz_status_t sz_overlap_engine_init_skylake(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                           sz_size_t const *window_widths,
                                                           sz_size_t window_widths_count, sz_size_t candidates_budget,
                                                           sz_allocator_t *allocator, void *stream);
/** @copydoc sz_overlap_scores */
STRINGZILLA_API sz_status_t sz_overlap_scores_skylake(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                      sz_f32_t *scores, sz_size_t scores_query_stride,
                                                      sz_size_t scores_candidate_stride, void *stream);
#endif

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_overlap_engine_init */
STRINGZILLA_API sz_status_t sz_overlap_engine_init_cuda(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        void *stream);
/** @copydoc sz_overlap_scores */
STRINGZILLA_API sz_status_t sz_overlap_scores_cuda(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                   sz_f32_t *scores, sz_size_t scores_query_stride,
                                                   sz_size_t scores_candidate_stride, void *stream);
#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_overlap_engine_init */
STRINGZILLA_API sz_status_t sz_overlap_engine_init_rocm(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        void *stream);
/** @copydoc sz_overlap_scores */
STRINGZILLA_API sz_status_t sz_overlap_scores_rocm(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                   sz_f32_t *scores, sz_size_t scores_query_stride,
                                                   sz_size_t scores_candidate_stride, void *stream);
#endif

#if STRINGZILLA_TARGET_METAL
/** @copydoc sz_overlap_engine_init */
STRINGZILLA_API sz_status_t sz_overlap_engine_init_metal(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                         sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                         sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                         void *stream);
/** @copydoc sz_overlap_scores */
STRINGZILLA_API sz_status_t sz_overlap_scores_metal(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                    sz_f32_t *scores, sz_size_t scores_query_stride,
                                                    sz_size_t scores_candidate_stride, void *stream);
#endif

/**
 *  @brief Finds the overlap kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_overlap_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                   sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/overlap/haswell.h"
#include "stringzilla/overlap/skylake.h"
#include "stringzilla/overlap/metal.h"
#include "stringzilla/overlap/cuda.cuh"
#include "stringzilla/overlap/rocm.cuh"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_overlap_engine_init(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                   sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                   sz_size_t candidates_budget, sz_capability_t capabilities,
                                                   sz_allocator_t *allocator, void *stream) {
    sz_unused_(engine), sz_unused_(queries), sz_unused_(window_widths), sz_unused_(window_widths_count),
        sz_unused_(candidates_budget), sz_unused_(capabilities), sz_unused_(allocator), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API void sz_overlap_engine_free(sz_overlap_engine_t *engine, void *stream) {
    sz_overlap_engine_close_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_overlap_scores(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                              sz_f32_t *scores, sz_size_t scores_query_stride,
                                              sz_size_t scores_candidate_stride, void *stream) {
    sz_unused_(engine), sz_unused_(candidates), sz_unused_(scores), sz_unused_(scores_query_stride),
        sz_unused_(scores_candidate_stride), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_overlap_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                   sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_OVERLAP_H_
