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
 *  stripes on the query side, and per backend a @c state, a @c vertical, and the @c init / @c step / @c any_active
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
 *  @sa sz_levenshtein_distances_serial, sz_levenshtein_distances_haswell, sz_levenshtein_distances_icelake
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
 *      sz_levenshtein_distances_utf8_icelake
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

#if SZ_USE_ICELAKE
/** @copydoc sz_levenshtein_distances */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_icelake(sz_cptr_t query, sz_size_t query_length,
                                                             sz_sequence_t const *candidates,
                                                             sz_memory_allocator_t *alloc, sz_size_t *distances);
/** @copydoc sz_levenshtein_distances_utf8 */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_utf8_icelake(sz_cptr_t query, sz_size_t query_length,
                                                                  sz_sequence_t const *candidates,
                                                                  sz_memory_allocator_t *alloc, sz_size_t *distances);
#endif

#pragma endregion Core API

#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/levenshtein/haswell.h"
#include "stringzilla/levenshtein/icelake.h"

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
#if SZ_USE_ICELAKE
    return sz_levenshtein_distances_icelake(query, query_length, candidates, alloc, distances);
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
