/**
 *  @file include/stringzilla/intersect/neonaes.h
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief NEON + AES backend for set intersection.
 *
 *  @sa include/stringzilla/intersect.h
 */
#ifndef STRINGZILLA_INTERSECT_NEONAES_H_
#define STRINGZILLA_INTERSECT_NEONAES_H_

#include "stringzilla/types.h"
#include "stringzilla/intersect/serial.h" // `sz_sequence_intersect_serial_`
#include "stringzilla/hash/neonaes.h"     // `sz_hash_neonaes_`

#ifdef __cplusplus
extern "C" {
#endif

/*  Only a kernel lives here, so the whole body follows the kernel's own target: a region with no
 *  function in it would leave the target attribute unused. */
#if STRINGZILLA_TARGET_NEONAES
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd+crypto+aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd+crypto+aes")
#endif

STRINGZILLA_API sz_status_t sz_sequence_intersect_neonaes(                     //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,   //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions, void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    // Batching four hashes over SVE2-AES measured no faster on Neoverse V3: table probes dominate.
    return sz_sequence_intersect_serial_(first_sequence, second_sequence, allocator, seed, intersection_count,
                                         first_positions, second_positions, sz_hash_neonaes_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_NEONAES

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_INTERSECT_NEONAES_H_
