/**
 *  @brief Hardware-accelerated Levenshtein edit distances under unit costs.
 *  @file include/stringzilla/levenshtein.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs with hardware-specific backends:
 *
 *  - @c sz_levenshtein_engine_init_cpu - prepares a batch of queries on the host and fixes the tier that scores it.
 *  - @c sz_levenshtein_engine_init_gpu - prepares the same batch on a device, on a stream the caller owns.
 *  - @c sz_levenshtein_engine_free - returns both of an engine's blocks to the allocator that built them.
 *  - @c sz_levenshtein_distances - the @b [queries, candidates] edit distances of one prepared batch.
 *
 *  All run Myers' bit-parallel algorithm: every query is a pattern, packed 64 symbols per machine word, and every
 *  candidate streams one symbol per step. An engine prepares the whole batch's match masks once and advances
 *  several candidates per step - one per scalar state, four per YMM, eight per ZMM, one per thread on a device -
 *  so the tables are built once per batch rather than once per pair. A byte is its own mask class, while a rune
 *  takes the class its query assigned it or class zero when the query lacks it, which @ref sz_levenshtein_symbol_t
 *  picks between. Strings of any length are accepted on either side.
 *
 *  The building blocks are public as well, for callers that own the loop nest: the query preparation and the
 *  transposes on the query side, and per backend a @c state, a @c vertical, and the @c init / @c step / @c any_active
 *  / @c score verbs over them, named by candidates per step - @c sz_levenshtein_u64x4_step_haswell and so on.
 */
#ifndef STRINGZILLA_LEVENSHTEIN_H_
#define STRINGZILLA_LEVENSHTEIN_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/levenshtein/haswell.h"
#include "stringzilla/levenshtein/skylake.h"
#include "stringzilla/levenshtein/icelake.h"
#include "stringzilla/levenshtein/cuda.cuh"

#pragma region Core API

/**
 *  @brief Prepares @p queries on the host into one block @p alloc hands back, and resolves the CPU tier once.
 *
 *  @param[in] queries The patterns every candidate is scored against, read through host-callable accessors.
 *  @param[in] symbol Whether a distance counts bytes or UTF-8 runes, which picks the tier as well as the layout.
 *  @param[in] alloc Where both of the engine's blocks come from, or @c SZ_NULL for the default host allocator.
 *  @param[out] engine Left untouched unless the call succeeds, and released by @ref sz_levenshtein_engine_free.
 *
 *  @retval sz_bad_alloc_k when the batch's block or the sizing pass's scratch could not be allocated.
 *  @note Ice Lake's byte lanes have no rune arm, so a rune batch resolves to Skylake however capable the machine is.
 *  @sa sz_levenshtein_tier_for
 */
SZ_API_RUNTIME sz_status_t sz_levenshtein_engine_init_cpu(sz_sequence_t const *queries,
                                                          sz_levenshtein_symbol_t symbol,
                                                          sz_memory_allocator_t *alloc,
                                                          sz_levenshtein_engine_t *engine);

/**
 *  @brief Prepares @p queries on @p stream 's device, resolving the launch geometry once.
 *
 *  @param[in] queries The patterns every candidate is scored against, read through @b host-callable accessors
 *      over @b host-readable texts, since this side counts their symbols and their classes to size the block.
 *  @param[in] symbol Whether a distance counts bytes or UTF-8 runes.
 *  @param[in] alloc Unified and bound to @p stream, or @c SZ_NULL to have a unified one derived from it.
 *  @param[in] stream A @c cudaStream_t the caller owns and keeps, or zero for the current device's default one.
 *  @param[out] engine Left untouched unless the call succeeds.
 *
 *  @retval sz_unexpected_dimensions_k for an empty query, or one past @ref sz_levenshtein_cuda_words_max_k words.
 *  @retval sz_device_code_mismatch_k when no GPU runtime is compiled in, or a launch itself fails.
 *  @note May join @p stream; no scoring verb ever does.
 */
SZ_API_RUNTIME sz_status_t sz_levenshtein_engine_init_gpu(sz_sequence_t const *queries,
                                                          sz_levenshtein_symbol_t symbol,
                                                          sz_memory_allocator_t *alloc, void *stream,
                                                          sz_levenshtein_engine_t *engine);

/** @brief Returns both of @p engine 's blocks to the allocator they were built with, and leaves it empty. */
SZ_API_RUNTIME void sz_levenshtein_engine_free(sz_levenshtein_engine_t *engine);

/**
 *  @brief Edit distances from every prepared query to every candidate, on the tier @p engine was built for.
 *
 *  @param[in] engine The batch @c _init_cpu or @c _init_gpu prepared, whose round scratch this may grow.
 *  @param[in] candidates The collection of texts, on the residency @p engine was built for.
 *  @param[out] distances The @b [count, candidates] distances; query @c q at @c distances[q*stride + c].
 *  @param[in] distances_stride Entries from one query's row to the next, in @c sz_size_t, at least the count.
 *
 *  @retval sz_unexpected_dimensions_k when @p distances_stride is under @c candidates->count.
 *  @retval sz_bad_alloc_k when the round's verticals could not be allocated.
 *  @note Grows @c engine->scratch when a round needs more than the last one did; never joins.
 *  @sa sz_levenshtein_distances_serial, sz_levenshtein_distances_haswell, sz_levenshtein_distances_skylake,
 *      sz_levenshtein_distances_icelake, sz_levenshtein_distances_cuda
 */
SZ_API_RUNTIME sz_status_t sz_levenshtein_distances(sz_levenshtein_engine_t *engine, sz_sequence_t const *candidates,
                                                    sz_size_t *distances, sz_size_t distances_stride);

/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_serial(sz_levenshtein_engine_t *engine,
                                                            sz_sequence_t const *candidates, sz_size_t *distances,
                                                            sz_size_t distances_stride);

#if SZ_USE_HASWELL
/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_haswell(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *candidates, sz_size_t *distances,
                                                             sz_size_t distances_stride);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_skylake(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *candidates, sz_size_t *distances,
                                                             sz_size_t distances_stride);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_icelake(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *candidates, sz_size_t *distances,
                                                             sz_size_t distances_stride);
#endif

#if SZ_USE_CUDA
/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_cuda(sz_levenshtein_engine_t *engine,
                                                          sz_sequence_t const *candidates, sz_size_t *distances,
                                                          sz_size_t distances_stride);

/** @copydoc sz_levenshtein_engine_init_gpu */
SZ_API_COMPTIME sz_status_t sz_levenshtein_engine_init_cuda(sz_sequence_t const *queries,
                                                            sz_levenshtein_symbol_t symbol,
                                                            sz_memory_allocator_t *alloc, void *stream,
                                                            sz_levenshtein_engine_t *engine);

/**
 *  @brief One pair's Levenshtein distance through the tiled wavefront, on texts the device already reaches.
 *
 *  Myers parallelizes over candidates and over the query's words, and a pair is one candidate, so an engine of
 *  one query hands it a single lane. The wavefront parallelizes over the long text's tile-columns instead, which
 *  is the axis the bit-parallel recurrence cannot touch, and is the entry point for a pair too long for it.
 *
 *  @param[in] a First text, device-reachable.
 *  @param[in] a_length Its length in bytes.
 *  @param[in] b Second text, device-reachable.
 *  @param[in] b_length Its length in bytes.
 *  @param[in] alloc Hands back the frontier scratch, which has to be memory the device reaches.
 *  @param[out] distance Host-readable slot receiving the distance.
 *  @param[in] stream The @c cudaStream_t to schedule on, or @c SZ_NULL for the default one.
 *  @retval sz_unexpected_dimensions_k when either text is longer than the kernel indexes.
 *  @retval sz_device_memory_mismatch_k when a text or the scratch is not memory the device reaches.
 *  @note Joins @p stream before it answers, which is what carries the distance back.
 */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distance_tiled_cuda(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                               sz_size_t *distance, void *stream);
#endif

#pragma endregion Core API

/*  Pick the right implementation for the edit-distance algorithms.
 *  To override this behavior and precompile all backends - set @c SZ_DYNAMIC_DISPATCH to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_API_RUNTIME sz_status_t sz_levenshtein_engine_init_cpu(sz_sequence_t const *queries,
                                                          sz_levenshtein_symbol_t symbol,
                                                          sz_memory_allocator_t *alloc,
                                                          sz_levenshtein_engine_t *engine) {
    // Only the tiers this build compiled in can answer, so the widest of them is the machine's answer too.
    return sz_levenshtein_engine_init_cpu_(queries, symbol, sz_caps_cpus_k, alloc, engine);
}

SZ_API_RUNTIME sz_status_t sz_levenshtein_engine_init_gpu(sz_sequence_t const *queries,
                                                          sz_levenshtein_symbol_t symbol,
                                                          sz_memory_allocator_t *alloc, void *stream,
                                                          sz_levenshtein_engine_t *engine) {
#if SZ_USE_CUDA
    return sz_levenshtein_engine_init_cuda(queries, symbol, alloc, stream, engine);
#else
    sz_unused_(queries), sz_unused_(symbol), sz_unused_(alloc), sz_unused_(stream), sz_unused_(engine);
    return sz_device_code_mismatch_k;
#endif
}

SZ_API_RUNTIME void sz_levenshtein_engine_free(sz_levenshtein_engine_t *engine) {
    sz_levenshtein_engine_free_(engine);
}

SZ_API_RUNTIME sz_status_t sz_levenshtein_distances(sz_levenshtein_engine_t *engine, sz_sequence_t const *candidates,
                                                    sz_size_t *distances, sz_size_t distances_stride) {
#if SZ_USE_CUDA
    if ((engine->capability & sz_caps_cuda_k) != 0)
        return sz_levenshtein_distances_cuda(engine, candidates, distances, distances_stride);
#endif
#if SZ_USE_ICELAKE
    if ((engine->capability & sz_cap_icelake_k) != 0)
        return sz_levenshtein_distances_icelake(engine, candidates, distances, distances_stride);
#endif
#if SZ_USE_SKYLAKE
    if ((engine->capability & sz_cap_skylake_k) != 0)
        return sz_levenshtein_distances_skylake(engine, candidates, distances, distances_stride);
#endif
#if SZ_USE_HASWELL
    if ((engine->capability & sz_cap_haswell_k) != 0)
        return sz_levenshtein_distances_haswell(engine, candidates, distances, distances_stride);
#endif
    return sz_levenshtein_distances_serial(engine, candidates, distances, distances_stride);
}

#endif // !SZ_DYNAMIC_DISPATCH
#pragma endregion Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_LEVENSHTEIN_H_
