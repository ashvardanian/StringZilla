/**
 *  @file include/stringzilla/intersect/neonaes.h
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief NEON + AES backend for set intersection.
 *
 *  @sa c/stringzilla/intersect.h
 */
#ifndef STRINGZILLA_INTERSECT_NEONAES_H_
#define STRINGZILLA_INTERSECT_NEONAES_H_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_NEONAES_

#include "stringzilla/intersect/serial.h" // `sz_intersect_serial_`
#include "stringzilla/hash/neonaes.h"     // `sz_hash_neonaes_`

#if defined(__cplusplus)
extern "C" {
#endif

#if STRINGZILLA_TARGET_NEONAES // Kernels only: an empty pragma region would fail under -Werror
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd+crypto+aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd+crypto+aes")
#endif

STRINGZILLA_API sz_status_t sz_intersect_neonaes(                                    //
    sz_strings_view_t const *first_strings, sz_strings_view_t const *second_strings, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,         //
    sz_size_t *first_positions, sz_size_t *second_positions, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    // Batching four hashes over SVE2-AES measured no faster on Neoverse V3: table probes dominate.
    return sz_intersect_serial_(first_strings, second_strings, allocator, seed, intersection_count, first_positions,
                                second_positions, sz_hash_neonaes_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_NEONAES

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // STRINGZILLA_ARCH_ARM64_NEONAES_
#endif // STRINGZILLA_ARCH_ARM64_

#endif // STRINGZILLA_INTERSECT_NEONAES_H_
