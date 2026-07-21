/**
 *  @brief SVE backend for string hashing and checksums.
 *  @file include/stringzilla/hash/sve.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_SVE_H_
#define STRINGZILLA_HASH_SVE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

SZ_API_COMPTIME sz_u64_t sz_bytesum_sve(sz_cptr_t text, sz_size_t length) {
    sz_size_t progress = 0;
    sz_size_t const vector_length = svcntb();
    // Base SVE lacks the `svaddwb`/`svaddwt` widening accumulators that the SVE2 sibling uses, but `UDOT`
    // (`svdot_u32`) is available: it widens u8 -> u32, summing four byte-products into each u32 lane. With a
    // multiplier of 1 it becomes a pure widening byte-sum, so the per-iteration horizontal reduction is gone -
    // a single `svaddv_u32` at the very end suffices. A u32 lane sums at most 4 bytes per `UDOT` and only grows
    // by `<= 255` per iteration, so it cannot overflow within any addressable buffer; no epoch boundary is needed.
    // The predicated load zeroes inactive tail lanes, so they contribute nothing to the dot product.
    svuint8_t const ones_u8x = svdup_n_u8(1);
    svuint32_t sum_u32x = svdup_n_u32(0);
    for (; progress < length; progress += vector_length) {
        svbool_t progress_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)length);
        svuint8_t text_u8x = svld1_u8(progress_b8x, (sz_u8_t const *)(text + progress));
        sum_u32x = svdot_u32(sum_u32x, text_u8x, ones_u8x);
    }
    return svaddv_u32(svptrue_b32(), sum_u32x);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_SVE_H_
