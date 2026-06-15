/**
 *  @brief WebAssembly relaxed-SIMD backend for the single-pass Unicode normalizer (level above SIMD128).
 *  @file include/stringzilla/utf8_norm/v128relaxed.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  The scanner is built from `wasm_u8x16` range compares and the 4x16 `wasm_i8x16_swizzle` 64-entry
 *  gather. Relaxed-SIMD's only relevant op is `wasm_i8x16_relaxed_swizzle`, whose sole difference from
 *  the baseline swizzle is the out-of-range behaviour for indices >= 16; the gather already masks every
 *  index to [0, 15] before the swizzle, so a relaxed swizzle would be byte-for-byte identical. With no
 *  win on offer, both public entry points delegate to the baseline SIMD128 implementation.
 */
#ifndef STRINGZILLA_UTF8_NORM_V128RELAXED_H_
#define STRINGZILLA_UTF8_NORM_V128RELAXED_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"
#include "stringzilla/utf8_norm/v128.h" // baseline SIMD128 fallbacks

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

#pragma region relaxed_simd

SZ_PUBLIC sz_size_t sz_utf8_norm_v128relaxed(sz_cptr_t source, sz_size_t length, sz_normal_form_t form,
                                             sz_ptr_t destination) {
    return sz_utf8_norm_v128(source, length, form, destination);
}

SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_v128relaxed(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_norm_violation_v128(source, length, form);
}

#pragma endregion relaxed_simd

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_V128RELAXED_H_
