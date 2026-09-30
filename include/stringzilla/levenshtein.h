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
 *  - @c sz_levenshtein_distance_tiled_best - one long pair through the GPU wavefront.
 *  - @c sz_levenshtein_find_kernel - the kernel any of them would run, for a caller that keeps it.
 *
 *  All run Myers' bit-parallel algorithm: every query is a pattern, packed 64 symbols per machine
 *  word, and every candidate streams one symbol per step. An engine prepares the whole batch's
 *  match masks once and advances several candidates per step - one per scalar state, four per YMM,
 *  eight per ZMM, one per thread on a device - so the tables are built once per batch rather than
 *  once per pair. A byte is its own mask class, while a rune takes the class its query assigned it
 *  or class zero when the query lacks it, which @ref sz_levenshtein_symbol_t picks between. Strings
 *  of any length are accepted on either side.
 *
 *  The building blocks are public as well, for callers that own the loop nest: the query
 *  preparation and the transposes on the query side, and per backend a state, a vertical, and the
 *  init, step, any-active and score verbs over them, named by candidates per step, such as
 *  @c sz_levenshtein_u64x4_step_haswell.
 */
#ifndef STRINGZILLA_LEVENSHTEIN_H_
#define STRINGZILLA_LEVENSHTEIN_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`
#include "stringzilla/types.cuh"      // Ahead of `extern "C"`, as the GPU runtimes' headers declare templates
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
 *  @param[in] queries The patterns every candidate is scored against, read through host-callable
 *      accessors over host-readable texts, whose symbols and classes size the block.
 *  @param[in] symbol Whether a distance counts bytes or UTF-8 runes.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled or
 *      @c sz_cuda_capabilities_enabled report; its group picks the CPU or a GPU vendor.
 *  @param[in] ordinal The device of that vendor, as its runtime numbers them, or zero on the CPU.
 *  @param[in] allocator Source of the engine's blocks, or @c STRINGZILLA_NULL for the default one
 *      of that device: the host heap, unified memory, or on Metal the arena of @p stream.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on, which on Metal
 *      is the @ref sz_metal_device_t opened on it.
 *  @return @c sz_success_k; @c sz_missing_kernel_k when no capability of the mask has an init;
 *      @c sz_bad_alloc_k when a block could not be allocated; or on a device
 *      @c sz_unexpected_dimensions_k for an empty query or one past
 *      @ref sz_levenshtein_simt_words_max_k words, @c sz_device_memory_mismatch_k for a
 *      @p stream of another device, @c sz_missing_gpu_k when the device doesn't answer, or
 *      @c sz_device_code_mismatch_k when a launch fails.
 *  @note May join @p stream; no scoring verb ever does.
 */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init(sz_levenshtein_engine_t *engine, sz_sequence_t const *queries,
                                                       sz_levenshtein_symbol_t symbol, sz_capability_t capabilities,
                                                       sz_size_t ordinal, sz_memory_allocator_t *allocator,
                                                       void *stream);

/** Returns both of @p engine 's blocks to the allocator that built them, and leaves it empty. */
STRINGZILLA_API void sz_levenshtein_engine_free(sz_levenshtein_engine_t *engine);

/**
 *  @brief Edit distances from every prepared query to every candidate, by the kernel of the
 *      capability @p engine was prepared for.
 *
 *  @param[in] engine The prepared batch; its round scratch may grow. It keeps no stream and no
 *      round state on a device, so any number of that device's streams may score it at once.
 *  @param[in] candidates The collection of texts, on the residency @p engine was built for: host
 *      accessors on the CPU, device accessors from @c sz_cuda_sequence_from_string_views on CUDA,
 *      and a view array in the arena from @c sz_sequence_from_string_views on Metal.
 *  @param[out] distances The @b [count, candidates] distances, query @c q at
 *      `distances[q * stride + c]`, device-reachable on a device.
 *  @param[in] distances_stride Entries from one query's row to the next, at least the count.
 *  @param[in] stream Null on the CPU, or the GPU stream of the engine's device to queue on,
 *      which on Metal is the engine's own @ref sz_metal_device_t.
 *  @return @c sz_success_k, on a device once enqueued; @c sz_unexpected_dimensions_k when
 *      @p distances_stride is under the candidate count; @c sz_bad_alloc_k when a host round's
 *      verticals could not be allocated; @c sz_missing_kernel_k for an empty engine; or on a device
 *      @c sz_device_memory_mismatch_k for an output, a sequence or a @p stream the engine's device
 *      cannot use, and @c sz_device_code_mismatch_k for a launch the device refused.
 *  @note Grows the engine's host scratch when a round needs more than the last one did; never
 *      joins, so the caller joins @p stream before reading @p distances.
 */
STRINGZILLA_API sz_status_t sz_levenshtein_distances(sz_levenshtein_engine_t *engine, sz_sequence_t const *candidates,
                                                     sz_size_t *distances, sz_size_t distances_stride, void *stream);

/**
 *  @brief One pair's distance through the GPU's tiled wavefront, enqueued on @p stream of the
 *      caller's current device.
 *
 *  Myers parallelizes over candidates and over the query's words, and a pair is one candidate, so
 *  an engine of one query hands it a single lane. The wavefront parallelizes over the long text's
 *  tile-columns instead, which is the axis the bit-parallel recurrence cannot touch, and is the
 *  entry point for a pair too long for it.
 *
 *  @param[in] a First text, device-reachable.
 *  @param[in] a_length Its length in bytes.
 *  @param[in] b Second text, device-reachable.
 *  @param[in] b_length Its length in bytes.
 *  @param[in] scratch The frontier, device-reachable and at least
 *      @ref sz_levenshtein_distance_tiled_scratch_bytes bytes, owned by the caller.
 *  @param[out] distance Device-reachable slot the distance lands in once @p stream is joined.
 *  @param[in] capabilities One device's capabilities; only CUDA and ROCm have the wavefront.
 *  @param[in] stream The GPU stream of the current device to queue on, or null for its default.
 *  @return @c sz_success_k once enqueued, @c sz_missing_kernel_k for any other capability,
 *      @c sz_unexpected_dimensions_k when either text is longer than the kernel indexes,
 *      @c sz_device_memory_mismatch_k when a text, the scratch or the distance is not memory the
 *      device reaches, or @c sz_missing_gpu_k when no device answers.
 *  @note Allocates nothing and joins nothing.
 */
STRINGZILLA_API sz_status_t sz_levenshtein_distance_tiled_best(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length, void *scratch, sz_size_t *distance,
                                                               sz_capability_t capabilities, void *stream);

/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_serial(sz_levenshtein_engine_t *engine,
                                                              sz_sequence_t const *queries,
                                                              sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                              sz_memory_allocator_t *allocator, void *stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_serial(sz_levenshtein_engine_t *engine,
                                                            sz_sequence_t const *candidates, sz_size_t *distances,
                                                            sz_size_t distances_stride, void *stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_haswell(sz_levenshtein_engine_t *engine,
                                                               sz_sequence_t const *queries,
                                                               sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                               sz_memory_allocator_t *allocator, void *stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_haswell(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *candidates, sz_size_t *distances,
                                                             sz_size_t distances_stride, void *stream);
#endif

#if STRINGZILLA_TARGET_SKYLAKE
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_skylake(sz_levenshtein_engine_t *engine,
                                                               sz_sequence_t const *queries,
                                                               sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                               sz_memory_allocator_t *allocator, void *stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_skylake(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *candidates, sz_size_t *distances,
                                                             sz_size_t distances_stride, void *stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_icelake(sz_levenshtein_engine_t *engine,
                                                               sz_sequence_t const *queries,
                                                               sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                               sz_memory_allocator_t *allocator, void *stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_icelake(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *candidates, sz_size_t *distances,
                                                             sz_size_t distances_stride, void *stream);
#endif

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_cuda(sz_levenshtein_engine_t *engine,
                                                            sz_sequence_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                            sz_memory_allocator_t *allocator, void *stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_cuda(sz_levenshtein_engine_t *engine,
                                                          sz_sequence_t const *candidates, sz_size_t *distances,
                                                          sz_size_t distances_stride, void *stream);
/** @copydoc sz_levenshtein_distance_tiled_best */
STRINGZILLA_API sz_status_t sz_levenshtein_distance_tiled_cuda(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length, void *scratch, sz_size_t *distance,
                                                               void *stream);
#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_rocm(sz_levenshtein_engine_t *engine,
                                                            sz_sequence_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                            sz_memory_allocator_t *allocator, void *stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_rocm(sz_levenshtein_engine_t *engine,
                                                          sz_sequence_t const *candidates, sz_size_t *distances,
                                                          sz_size_t distances_stride, void *stream);
/** @copydoc sz_levenshtein_distance_tiled_best */
STRINGZILLA_API sz_status_t sz_levenshtein_distance_tiled_rocm(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length, void *scratch, sz_size_t *distance,
                                                               void *stream);
#endif

#if STRINGZILLA_TARGET_METAL
/** @copydoc sz_levenshtein_engine_init */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_metal(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *queries,
                                                             sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                             sz_memory_allocator_t *allocator, void *stream);
/** @copydoc sz_levenshtein_distances */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_metal(sz_levenshtein_engine_t *engine,
                                                           sz_sequence_t const *candidates, sz_size_t *distances,
                                                           sz_size_t distances_stride, void *stream);
#endif

/**
 *  @brief Finds the Levenshtein kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_levenshtein_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                       sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/levenshtein/haswell.h"
#include "stringzilla/levenshtein/skylake.h"
#include "stringzilla/levenshtein/icelake.h"
#include "stringzilla/levenshtein/simt.h"
#include "stringzilla/levenshtein/simt.cuh"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init(sz_levenshtein_engine_t *engine, sz_sequence_t const *queries,
                                                       sz_levenshtein_symbol_t symbol, sz_capability_t capabilities,
                                                       sz_size_t ordinal, sz_memory_allocator_t *allocator,
                                                       void *stream) {
    sz_unused_(engine), sz_unused_(queries), sz_unused_(symbol), sz_unused_(capabilities), sz_unused_(ordinal),
        sz_unused_(allocator), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API void sz_levenshtein_engine_free(sz_levenshtein_engine_t *engine) {
    sz_levenshtein_engine_free_(engine);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances(sz_levenshtein_engine_t *engine, sz_sequence_t const *candidates,
                                                     sz_size_t *distances, sz_size_t distances_stride, void *stream) {
    sz_unused_(engine), sz_unused_(candidates), sz_unused_(distances), sz_unused_(distances_stride), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_levenshtein_distance_tiled_best(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length, void *scratch, sz_size_t *distance,
                                                               sz_capability_t capabilities, void *stream) {
    sz_unused_(a), sz_unused_(a_length), sz_unused_(b), sz_unused_(b_length), sz_unused_(scratch), sz_unused_(distance),
        sz_unused_(capabilities), sz_unused_(stream);
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
