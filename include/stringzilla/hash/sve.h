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

SZ_PUBLIC sz_u64_t sz_bytesum_sve(sz_cptr_t text, sz_size_t length) {
    sz_u64_t sum = 0;
    sz_size_t progress = 0;
    sz_size_t const vector_length = svcntb();
    // SVE doesn't have widening accumulation, so we reduce across each loaded vector
    for (; progress < length; progress += vector_length) {
        svbool_t progress_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)length);
        svuint8_t text_u8x = svld1_u8(progress_b8x, (sz_u8_t const *)(text + progress));
        sum += svaddv_u8(progress_b8x, text_u8x);
    }
    return sum;
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
