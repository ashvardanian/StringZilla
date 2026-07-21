/**
 *  @brief SVE2 backend for string hashing and checksums.
 *  @file include/stringzilla/hash/sve2.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_SVE2_H_
#define STRINGZILLA_HASH_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2")
#endif

SZ_API_COMPTIME sz_u64_t sz_bytesum_sve2(sz_cptr_t text, sz_size_t length) {
    sz_u64_t sum = 0;
    sz_size_t progress = 0;
    sz_size_t const vector_length = svcntb();
    // In SVE2 we have an instruction, that can add 8-bit elements in one operand to 16-bit elements in another.
    // Assuming the size mismatch, there 2 such instructions - for the top and bottom elements in each 8-bit pair.
    //
    // We can use that kind of logic to accelerate the inner loop, but we still need to reduce the 64-bit results.
    while (progress < length) {
        svuint16_t sum_top_u16x = svdup_n_u16(0);
        svuint16_t sum_bottom_u16x = svdup_n_u16(0);
        // Assuming `u16` has a 256x wider range than `u8`, we can aggregate up to 256 lanes in each value.
        for (sz_size_t loop_index = 0; progress < length && loop_index < 256; progress += vector_length, ++loop_index) {
            svbool_t progress_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)length);
            svuint8_t text_u8x = svld1_u8(progress_b8x, (sz_u8_t const *)(text + progress));
            sum_top_u16x = svaddwb_u16(sum_top_u16x, text_u8x);
            sum_bottom_u16x = svaddwt_u16(sum_bottom_u16x, text_u8x);
        }
        sum += svaddv_u16(svptrue_b16(), sum_top_u16x);
        sum += svaddv_u16(svptrue_b16(), sum_bottom_u16x);
    }

    return sum;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_SVE2_H_
