/**
 *  @brief Hardware-accelerated Levenshtein edit distances under unit costs.
 *  @file include/stringzilla/levenshtein.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs with hardware-specific backends:
 *
 *  - @c sz_levenshtein_distance - the edit distance between one pair of byte strings.
 *  - @c sz_levenshtein_distances - the edit distances from one query to every string of a collection.
 *  - @c sz_levenshtein_distance_utf8 - the same pair distance counted in runes, not bytes.
 *  - @c sz_levenshtein_distances_utf8 - the same one-to-many distances counted in runes.
 *
 *  All run Myers' bit-parallel algorithm: the query is the pattern, packed 64 symbols per machine word, and every
 *  candidate streams one symbol per step. The one-to-many forms prepare the query's match masks once and advance
 *  several candidates per step - one per scalar state, four per YMM, eight per ZMM - so the table is built once
 *  per query rather than once per pair; the one-to-one forms fill a single candidate and run the serial walk on
 *  every backend. A byte is its own mask class; a rune takes the class the query assigned it, or class zero when
 *  the query lacks it. Strings of any length are accepted on either side.
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

#pragma region Core API

/**
 *  @brief Computes the Levenshtein edit distance between two byte strings under unit costs.
 *
 *  @param[in] a First string.
 *  @param[in] a_length Number of bytes in the first string.
 *  @param[in] b Second string.
 *  @param[in] b_length Number of bytes in the second string.
 *  @param[in] alloc Memory allocator for the temporary match-mask table and word states.
 *  @param[out] distance The edit distance.
 *
 *  @retval @c sz_success_k if the distance was computed.
 *  @retval @c sz_bad_alloc_k if the temporary storage could not be allocated.
 *  @note Selects the fastest implementation at compile- or run-time based on @c SZ_DYNAMIC_DISPATCH.
 *  @sa sz_levenshtein_distance_serial
 */
SZ_API_RUNTIME sz_status_t sz_levenshtein_distance(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                                   sz_memory_allocator_t *alloc, sz_size_t *distance);

/**
 *  @brief Computes the Levenshtein edit distances from one query to every string of @p candidates.
 *
 *  @param[in] query The pattern every candidate is scored against.
 *  @param[in] query_length Number of bytes in the query.
 *  @param[in] candidates The collection of texts.
 *  @param[in] alloc Memory allocator for the temporary match-mask table and word states.
 *  @param[out] distances One edit distance per candidate, in the collection's order.
 *
 *  @retval @c sz_success_k if every distance was computed.
 *  @retval @c sz_bad_alloc_k if the temporary storage could not be allocated.
 *  @pre The @p distances array must fit at least @c candidates->count entries.
 *  @note Selects the fastest implementation at compile- or run-time based on @c SZ_DYNAMIC_DISPATCH.
 *  @sa sz_levenshtein_distances_serial, sz_levenshtein_distances_haswell, sz_levenshtein_distances_skylake,
 *      sz_levenshtein_distances_icelake
 */
SZ_API_RUNTIME sz_status_t sz_levenshtein_distances(sz_cptr_t query, sz_size_t query_length,
                                                    sz_sequence_t const *candidates, sz_memory_allocator_t *alloc,
                                                    sz_size_t *distances);

/**
 *  @brief Computes the Levenshtein edit distance between two UTF-8 strings, counting runes rather than bytes.
 *
 *  An ill-formed byte decodes to @c U+FFFD and counts as one rune, so every byte string has a distance.
 *
 *  @param[in] a First string.
 *  @param[in] a_length Number of bytes in the first string.
 *  @param[in] b Second string.
 *  @param[in] b_length Number of bytes in the second string.
 *  @param[in] alloc Memory allocator for the temporary match-mask table, rune classes, and word states.
 *  @param[out] distance The edit distance in runes.
 *
 *  @retval @c sz_success_k if the distance was computed.
 *  @retval @c sz_bad_alloc_k if the temporary storage could not be allocated.
 *  @note Selects the fastest implementation at compile- or run-time based on @c SZ_DYNAMIC_DISPATCH.
 *  @sa sz_levenshtein_distance_utf8_serial
 */
SZ_API_RUNTIME sz_status_t sz_levenshtein_distance_utf8(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                        sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                        sz_size_t *distance);

/**
 *  @brief Computes the Levenshtein edit distances from one UTF-8 query to every string of @p candidates,
 *      counting runes rather than bytes.
 *
 *  @param[in] query The pattern every candidate is scored against.
 *  @param[in] query_length Number of bytes in the query.
 *  @param[in] candidates The collection of texts.
 *  @param[in] alloc Memory allocator for the temporary match-mask table, rune classes, and word states.
 *  @param[out] distances One edit distance in runes per candidate, in the collection's order.
 *
 *  @retval @c sz_success_k if every distance was computed.
 *  @retval @c sz_bad_alloc_k if the temporary storage could not be allocated.
 *  @pre The @p distances array must fit at least @c candidates->count entries.
 *  @note Selects the fastest implementation at compile- or run-time based on @c SZ_DYNAMIC_DISPATCH.
 *  @sa sz_levenshtein_distances_utf8_serial, sz_levenshtein_distances_utf8_haswell,
 *      sz_levenshtein_distances_utf8_skylake
 */
SZ_API_RUNTIME sz_status_t sz_levenshtein_distances_utf8(sz_cptr_t query, sz_size_t query_length,
                                                         sz_sequence_t const *candidates, sz_memory_allocator_t *alloc,
                                                         sz_size_t *distances);

/** @copydoc sz_levenshtein_distance */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distance_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                           sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                           sz_size_t *distance);
/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_serial(sz_cptr_t query, sz_size_t query_length,
                                                            sz_sequence_t const *candidates,
                                                            sz_memory_allocator_t *alloc, sz_size_t *distances);
/** @copydoc sz_levenshtein_distance_utf8 */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distance_utf8_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                                sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                                sz_size_t *distance);
/** @copydoc sz_levenshtein_distances_utf8 */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_utf8_serial(sz_cptr_t query, sz_size_t query_length,
                                                                 sz_sequence_t const *candidates,
                                                                 sz_memory_allocator_t *alloc, sz_size_t *distances);

#if SZ_USE_HASWELL
/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_haswell(sz_cptr_t query, sz_size_t query_length,
                                                             sz_sequence_t const *candidates,
                                                             sz_memory_allocator_t *alloc, sz_size_t *distances);
/** @copydoc sz_levenshtein_distances_utf8 */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_utf8_haswell(sz_cptr_t query, sz_size_t query_length,
                                                                  sz_sequence_t const *candidates,
                                                                  sz_memory_allocator_t *alloc, sz_size_t *distances);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_skylake(sz_cptr_t query, sz_size_t query_length,
                                                             sz_sequence_t const *candidates,
                                                             sz_memory_allocator_t *alloc, sz_size_t *distances);
/** @copydoc sz_levenshtein_distances_utf8 */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_utf8_skylake(sz_cptr_t query, sz_size_t query_length,
                                                                  sz_sequence_t const *candidates,
                                                                  sz_memory_allocator_t *alloc, sz_size_t *distances);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_icelake(sz_cptr_t query, sz_size_t query_length,
                                                             sz_sequence_t const *candidates,
                                                             sz_memory_allocator_t *alloc, sz_size_t *distances);
#endif

#if SZ_USE_CUDA
/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_cuda(sz_cptr_t query, sz_size_t query_length,
                                                          sz_sequence_t const *candidates, sz_memory_allocator_t *alloc,
                                                          sz_size_t *distances);

/**
 *  @copydoc sz_levenshtein_distances
 *  @note Takes the CPU backends' arguments exactly, so it binds wherever they do, and asks nothing of the caller
 *      about where its memory lives: whatever the device cannot reach is staged into memory it can, scored, and
 *      read back before returning. A caller already holding its texts and distances on the device pays none of
 *      that - the round then runs in place.
 *  @sa sz_levenshtein_distances_scheduled_cuda to run on a chosen stream, where staging is refused.
 */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distance_cuda(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                         sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                         sz_size_t *distance);

/**
 *  @brief Levenshtein distances from @p query to every candidate on one GPU, scheduled on the caller's stream.
 *
 *  Nothing crosses the bus: @p candidates ' texts are read where they already are, and @p distances and whatever
 *  @p alloc hands back are written the same way, so the round costs one launch rather than a round trip. Host
 *  memory is refused rather than copied - page-locked host memory counts as host memory here.
 *
 *  @param[in] stream A @c cudaStream_t the caller owns and keeps, or zero for the current device's default one.
 *      The round is synchronous either way - @p distances is filled before returning - but only this stream is
 *      waited on, so the caller's other work on the device keeps running.
 *  @param[in] alloc Where the query's match masks come from; must itself reach the device, freed before returning.
 *  @param[out] distances The @b [candidates] edit distances, in the collection's order.
 *
 *  @retval sz_unexpected_dimensions_k for an empty query, or one past @ref sz_levenshtein_cuda_words_max_k words.
 *  @retval sz_device_memory_mismatch_k when the distances, the sequence handle, or the masks are host memory.
 *  @retval sz_device_code_mismatch_k when the launch itself fails, the stream reporting it at the join.
 *  @pre @p candidates carries @b device accessors, as @ref sz_sequence_from_string_views_cuda binds
 *      them, because the kernel is what calls them. Only the handle can be checked from this side.
 */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_scheduled_cuda(sz_cptr_t query, sz_size_t query_length,
                                                                    sz_sequence_t const *candidates,
                                                                    sz_memory_allocator_t *alloc, sz_size_t *distances,
                                                                    void *stream);
#endif

#pragma endregion Core API

#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/levenshtein/haswell.h"
#include "stringzilla/levenshtein/icelake.h"
#include "stringzilla/levenshtein/cuda.cuh"

/*  Pick the right implementation for the edit-distance algorithms.
 *  To override this behavior and precompile all backends - set @c SZ_DYNAMIC_DISPATCH to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_API_RUNTIME sz_status_t sz_levenshtein_distance(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                                   sz_memory_allocator_t *alloc, sz_size_t *distance) {
    return sz_levenshtein_distance_serial(a, a_length, b, b_length, alloc, distance);
}

SZ_API_RUNTIME sz_status_t sz_levenshtein_distances(sz_cptr_t query, sz_size_t query_length,
                                                    sz_sequence_t const *candidates, sz_memory_allocator_t *alloc,
                                                    sz_size_t *distances) {
#if SZ_USE_CUDA
    return sz_levenshtein_distances_cuda(query, query_length, candidates, alloc, distances);
#elif SZ_USE_ICELAKE
    return sz_levenshtein_distances_icelake(query, query_length, candidates, alloc, distances);
#elif SZ_USE_SKYLAKE
    return sz_levenshtein_distances_skylake(query, query_length, candidates, alloc, distances);
#elif SZ_USE_HASWELL
    return sz_levenshtein_distances_haswell(query, query_length, candidates, alloc, distances);
#else
    return sz_levenshtein_distances_serial(query, query_length, candidates, alloc, distances);
#endif
}

SZ_API_RUNTIME sz_status_t sz_levenshtein_distance_utf8(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                        sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                        sz_size_t *distance) {
    return sz_levenshtein_distance_utf8_serial(a, a_length, b, b_length, alloc, distance);
}

SZ_API_RUNTIME sz_status_t sz_levenshtein_distances_utf8(sz_cptr_t query, sz_size_t query_length,
                                                         sz_sequence_t const *candidates, sz_memory_allocator_t *alloc,
                                                         sz_size_t *distances) {
#if SZ_USE_ICELAKE
    return sz_levenshtein_distances_utf8_icelake(query, query_length, candidates, alloc, distances);
#elif SZ_USE_HASWELL
    return sz_levenshtein_distances_utf8_haswell(query, query_length, candidates, alloc, distances);
#else
    return sz_levenshtein_distances_utf8_serial(query, query_length, candidates, alloc, distances);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH
#pragma endregion Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_LEVENSHTEIN_H_
