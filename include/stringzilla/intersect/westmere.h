/**
 *  @file include/stringzilla/intersect/westmere.h
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Westmere (SSE4.2+AES-NI) backend for set intersection.
 *
 *  @sa include/stringzilla/intersect.h
 */
#ifndef STRINGZILLA_INTERSECT_WESTMERE_H_
#define STRINGZILLA_INTERSECT_WESTMERE_H_

#include "stringzilla/types.h"
#include "stringzilla/intersect/serial.h" // `sz_sequence_intersect_serial_`
#include "stringzilla/hash/westmere.h"    // `sz_hash_westmere_`

#ifdef __cplusplus
extern "C" {
#endif

/*  Only a kernel lives here, so the whole body follows the kernel's own target: a region with no
 *  function in it would leave the target attribute unused. */
#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_ARCH_X8664_WESTMERE_
#if STRINGZILLA_TARGET_WESTMERE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse4.2,aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("sse4.2", "aes")
#endif

STRINGZILLA_API sz_status_t sz_sequence_intersect_westmere(                    //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,   //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_sequence_intersect_serial_(first_sequence, second_sequence, allocator, seed, intersection_count,
                                         first_positions, second_positions, sz_hash_westmere_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_WESTMERE
#endif // STRINGZILLA_ARCH_X8664_WESTMERE_
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_INTERSECT_WESTMERE_H_
