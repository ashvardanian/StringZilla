/**
 *  @file include/stringzilla/levenshtein.h
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Hardware-accelerated Levenshtein edit distances under unit costs.
 *
 *  Includes core APIs with hardware-specific backends:
 *
 *  - @c sz_levenshtein_engine_init - prepares a batch of queries for the best capability of a mask,
 *    on the host or on one device.
 *  - @c sz_levenshtein_engine_free - returns an engine's blocks to the allocator that built them.
 *  - @c sz_levenshtein_distances - the @b [queries, candidates] edit distances of one batch, on the
 *    capability its engine was prepared for.
 *  - @c sz_levenshtein_find_kernel - the kernel any of them would run, for a caller that keeps it.
 *
 *  Myers' bit-parallel algorithm treats every query as a pattern, packed 64 symbols per machine
 *  word, and every candidate streams one symbol per step. An engine prepares the whole batch's
 *  match masks once and advances several candidates per step - one per scalar state, four per YMM,
 *  eight per ZMM, one per thread on a device - so the tables are built once per batch rather than
 *  once per pair. A byte is its own mask class, while a rune takes the class its query assigned it
 *  or class zero when the query lacks it, which @ref sz_levenshtein_symbol_t picks between. Strings
 *  of any length are accepted on either side; device queries are bounded by their length index.
 *  Long byte pairs on a device use an internal tiled wavefront for intra-pair parallelism.
 *
 *  The building blocks are public as well, for callers that own the loop nest: the query
 *  preparation and the transposes on the query side, and per backend a state, a vertical, and the
 *  init, step, any-active and score verbs over them, named by candidates per step, such as
 *  @c sz_levenshtein_u64x4_step_haswell. Scoring itself has one entry point per backend, such as
 *  @c sz_levenshtein_distances_haswell, which takes the caller's order and a bound past which a
 *  distance need not be exact.
 */
#ifndef STRINGZILLA_LEVENSHTEIN_H_
#define STRINGZILLA_LEVENSHTEIN_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`
#include "stringzilla/metal.h"
#include "stringzilla/levenshtein/serial.h" // `sz_levenshtein_engine_t`, `sz_levenshtein_symbol_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Prepares @p queries into one block from @p allocator, for the best capability of
 *      @p capabilities that has an init kernel.
 *
 *  @param[out] engine Untouched unless the call succeeds; @ref sz_levenshtein_engine_free frees it.
 *  @param[in] queries The patterns every candidate is scored against, in any layout over
 *      host-readable texts, whose symbols and classes size the block.
 *  @param[in] symbol Whether a distance counts bytes or UTF-8 runes.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu or
 *      @c sz_capabilities_enabled_cuda report; its group picks the CPU or a GPU vendor.
 *  @param[in] allocator Source of the engine's blocks, or @c STRINGZILLA_NULL for
 *      @ref sz_allocator_init_unified_best of @p capabilities.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, also naming the device:
 *      a @c cudaStream_t, a @c hipStream_t, or an @c id<MTLCommandQueue>; null for the default.
 *
 *  @note May join @p stream; no scoring verb ever does.
 *
 *  @par Statuses:
 *  - @c sz_missing_kernel_k No capability of the mask has an init.
 *  - @c sz_unexpected_dimensions_k On a device, an unrepresentable query length.
 *  - @c sz_device_memory_mismatch_k On a device, memory or a @p stream the device cannot use.
 */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init(sz_levenshtein_engine_t *engine,
                                                       sz_strings_view_t const *queries, sz_levenshtein_symbol_t symbol,
                                                       sz_capability_t capabilities, sz_allocator_t *allocator,
                                                       sz_stream_t stream);

/** Returns both of @p engine 's blocks to the allocator that built them, once the work queued on
 *  @p stream is done with them, and leaves it empty. */
STRINGZILLA_API void sz_levenshtein_engine_free(sz_levenshtein_engine_t *engine, sz_stream_t stream);

/**
 *  @brief Edit distances from every prepared query to every candidate, by the kernel of the
 *      capability @p engine was prepared for, exact up to @p bound.
 *
 *  A query prepared once scores any number of candidate batches: the engine's scratch only grows,
 *  so batches no larger than an earlier one allocate nothing. A host sweep advances a few
 *  candidates together and costs as long as the longest of them, so an @p order grouping similar
 *  lengths keeps it short, and it stops early once every candidate it holds is past @p bound.
 *
 *  @param[in] engine The prepared batch; its round scratch may grow. It keeps no stream and no
 *      round state on a device, so any number of that device's streams may score it at once.
 *  @param[in] candidates The texts, on the residency @p engine was built for: any layout on the
 *      CPU, and on a device a tape @ref sz_strings_to_tape_best wrote into a unified allocator.
 *  @param[in] symbol_counts Each candidate's length in the engine's symbols by candidate index, as
 *      @ref sz_levenshtein_utf8_runes counts runes, or null to count them as the sweep goes.
 *  @param[in] order The visiting order, a permutation of the candidate indices, or null for the
 *      engine's own. Neither it nor @p symbol_counts changes a distance, and a device engine, which
 *      orders and counts candidates itself, reads neither.
 *  @param[in] bound The largest distance reported exactly; a candidate past it reports
 *      @p bound + 1 on every backend. @c STRINGZILLA_SIZE_MAX keeps every distance exact.
 *  @param[out] distances The @b [count, candidates] distances, query @c q at
 *      <tt>distances[q * stride + c]</tt>, device-reachable on a device.
 *  @param[in] distances_stride Entries from one query's row to the next, at least the count.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, which also names the device
 *      the round runs on; null for the default.
 *
 *  @note Grows the engine's host scratch when a round needs more than the last one did; never
 *      joins, so the caller joins @p stream before reading @p distances.
 *
 *  @par Statuses:
 *  - @c sz_unexpected_dimensions_k @p distances_stride is under the candidate count.
 *  - @c sz_missing_kernel_k An empty engine.
 *  - @c sz_device_memory_mismatch_k On a device, an engine, an output, an accessor, a tape or a
 *    stream the device of @p stream cannot use.
 */
STRINGZILLA_API sz_status_t sz_levenshtein_distances(sz_levenshtein_engine_t *engine,
                                                     sz_strings_view_t const *candidates,
                                                     sz_size_t const *symbol_counts, sz_size_t const *order,
                                                     sz_size_t bound, sz_size_t *distances, sz_size_t distances_stride,
                                                     sz_stream_t stream);

/**
 *  @brief Prepares @p queries into one block from @p allocator, for the serial kernels.
 *  @param[out] engine Untouched unless the call succeeds; @ref sz_levenshtein_engine_free frees it.
 *  @param[in] queries The patterns every candidate is scored against, in any layout over
 *      host-readable texts, whose symbols and classes size the block.
 *  @param[in] symbol Whether a distance counts bytes or UTF-8 runes.
 *  @param[in] allocator Source of the engine's blocks, or @c STRINGZILLA_NULL for
 *      @c sz_allocator_init_unified_serial.
 *  @param[in] stream Null on the CPU.
 *
 *  @par Statuses:
 *  - @c sz_unexpected_dimensions_k An unrepresentable query length.
 */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_serial(sz_levenshtein_engine_t *engine,
                                                              sz_strings_view_t const *queries,
                                                              sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                              sz_stream_t stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_serial(sz_levenshtein_engine_t *engine,
                                                            sz_strings_view_t const *candidates,
                                                            sz_size_t const *symbol_counts, sz_size_t const *order,
                                                            sz_size_t bound, sz_size_t *distances,
                                                            sz_size_t distances_stride, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_haswell(sz_levenshtein_engine_t *engine,
                                                               sz_strings_view_t const *queries,
                                                               sz_levenshtein_symbol_t symbol,
                                                               sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_haswell(sz_levenshtein_engine_t *engine,
                                                             sz_strings_view_t const *candidates,
                                                             sz_size_t const *symbol_counts, sz_size_t const *order,
                                                             sz_size_t bound, sz_size_t *distances,
                                                             sz_size_t distances_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SKYLAKE
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_skylake(sz_levenshtein_engine_t *engine,
                                                               sz_strings_view_t const *queries,
                                                               sz_levenshtein_symbol_t symbol,
                                                               sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_skylake(sz_levenshtein_engine_t *engine,
                                                             sz_strings_view_t const *candidates,
                                                             sz_size_t const *symbol_counts, sz_size_t const *order,
                                                             sz_size_t bound, sz_size_t *distances,
                                                             sz_size_t distances_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_icelake(sz_levenshtein_engine_t *engine,
                                                               sz_strings_view_t const *queries,
                                                               sz_levenshtein_symbol_t symbol,
                                                               sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_icelake(sz_levenshtein_engine_t *engine,
                                                             sz_strings_view_t const *candidates,
                                                             sz_size_t const *symbol_counts, sz_size_t const *order,
                                                             sz_size_t bound, sz_size_t *distances,
                                                             sz_size_t distances_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_neon(sz_levenshtein_engine_t *engine,
                                                            sz_strings_view_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                            sz_stream_t stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_neon(sz_levenshtein_engine_t *engine,
                                                          sz_strings_view_t const *candidates,
                                                          sz_size_t const *symbol_counts, sz_size_t const *order,
                                                          sz_size_t bound, sz_size_t *distances,
                                                          sz_size_t distances_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_cuda(sz_levenshtein_engine_t *engine,
                                                            sz_strings_view_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                            sz_stream_t stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_cuda(sz_levenshtein_engine_t *engine,
                                                          sz_strings_view_t const *candidates,
                                                          sz_size_t const *symbol_counts, sz_size_t const *order,
                                                          sz_size_t bound, sz_size_t *distances,
                                                          sz_size_t distances_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_BLACKWELL
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_blackwell(sz_levenshtein_engine_t *engine,
                                                                 sz_strings_view_t const *queries,
                                                                 sz_levenshtein_symbol_t symbol,
                                                                 sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_blackwell(sz_levenshtein_engine_t *engine,
                                                               sz_strings_view_t const *candidates,
                                                               sz_size_t const *symbol_counts, sz_size_t const *order,
                                                               sz_size_t bound, sz_size_t *distances,
                                                               sz_size_t distances_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_rocm(sz_levenshtein_engine_t *engine,
                                                            sz_strings_view_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                            sz_stream_t stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_rocm(sz_levenshtein_engine_t *engine,
                                                          sz_strings_view_t const *candidates,
                                                          sz_size_t const *symbol_counts, sz_size_t const *order,
                                                          sz_size_t bound, sz_size_t *distances,
                                                          sz_size_t distances_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_metal(sz_levenshtein_engine_t *engine,
                                                             sz_strings_view_t const *queries,
                                                             sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                             sz_stream_t stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_metal(sz_levenshtein_engine_t *engine,
                                                           sz_strings_view_t const *candidates,
                                                           sz_size_t const *symbol_counts, sz_size_t const *order,
                                                           sz_size_t bound, sz_size_t *distances,
                                                           sz_size_t distances_stride, sz_stream_t stream);
#endif

/**
 *  @brief Finds the Levenshtein kernel of @p kind, from the best of @p capabilities.
 *  @param[in] kind The verb whose kernel to find.
 *  @param[in] capabilities The mask to pick the best capability from.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *
 *  @par Statuses:
 *  - @c sz_missing_kernel_k
 *  - @c sz_missing_library_k Header-only.
 */
STRINGZILLA_API sz_status_t sz_levenshtein_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                       sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/levenshtein/haswell.h"
#include "stringzilla/levenshtein/skylake.h"
#include "stringzilla/levenshtein/icelake.h"
#include "stringzilla/levenshtein/neon.h"
#include "stringzilla/levenshtein/cuda.cuh"
#include "stringzilla/levenshtein/blackwell.cuh"
#include "stringzilla/levenshtein/rocm.cuh"
#include "stringzilla/levenshtein/metal.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init(sz_levenshtein_engine_t *engine,
                                                       sz_strings_view_t const *queries, sz_levenshtein_symbol_t symbol,
                                                       sz_capability_t capabilities, sz_allocator_t *allocator,
                                                       sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(queries), sz_unused_(symbol), sz_unused_(capabilities), sz_unused_(allocator),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API void sz_levenshtein_engine_free(sz_levenshtein_engine_t *engine, sz_stream_t stream) {
    sz_levenshtein_engine_free_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances(sz_levenshtein_engine_t *engine,
                                                     sz_strings_view_t const *candidates,
                                                     sz_size_t const *symbol_counts, sz_size_t const *order,
                                                     sz_size_t bound, sz_size_t *distances, sz_size_t distances_stride,
                                                     sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(candidates), sz_unused_(symbol_counts), sz_unused_(order), sz_unused_(bound),
        sz_unused_(distances), sz_unused_(distances_stride), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_levenshtein_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                       sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_LEVENSHTEIN_H_
