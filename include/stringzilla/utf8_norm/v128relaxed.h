/**
 *  @file include/stringzilla/utf8_norm/v128relaxed.h
 *  @author Ash Vardanian
 *  @date June 15, 2026
 *  @brief WebAssembly relaxed-SIMD backend for the single-pass normalizer, a level above SIMD128.
 *
 *  The scanner is built from @c wasm_u8x16 range compares and the 4x16 @c wasm_i8x16_swizzle
 *  64-entry gather. Relaxed-SIMD's only relevant op is @c wasm_i8x16_relaxed_swizzle, whose sole
 *  difference from the baseline swizzle is the out-of-range behaviour for indices ≥ 16; the gather
 *  already masks every index to [0, 15] before the swizzle, so a relaxed swizzle would be
 *  byte-for-byte identical. With no win on offer, both public entry points delegate to the baseline
 *  SIMD128 implementation.
 *
 *  @sa c/stringzilla/utf8_norm.h
 */
#ifndef STRINGZILLA_UTF8_NORM_V128RELAXED_H_
#define STRINGZILLA_UTF8_NORM_V128RELAXED_H_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_WASM_
#if STRINGZILLA_ARCH_WASM_V128RELAXED_

#include "stringzilla/utf8_norm/serial.h"
#include "stringzilla/utf8_norm/v128.h" // baseline SIMD128 fallbacks

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region relaxed_simd

#if STRINGZILLA_TARGET_V128RELAXED // Kernels only: an empty pragma region would fail under -Werror
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

STRINGZILLA_API sz_status_t sz_utf8_norm_v128relaxed(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                                     sz_ptr_t target, sz_size_t *target_length, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *target_length = sz_utf8_norm_engine_(source, source_length, form, target, &sz_utf8_norm_classify_v128_,
                                          &sz_utf8_uncased_fold_v128_);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_v128relaxed(sz_cptr_t source, sz_size_t source_length,
                                                                  sz_normal_form_t form, sz_cptr_t *match,
                                                                  sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_utf8_find_denormalized_engine_(source, source_length, form, &sz_utf8_norm_classify_v128_,
                                               &sz_utf8_uncased_fold_v128_);
    return sz_success_k;
}

#pragma endregion relaxed_simd

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // STRINGZILLA_TARGET_V128RELAXED

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // STRINGZILLA_ARCH_WASM_V128RELAXED_
#endif // STRINGZILLA_ARCH_WASM_

#endif // STRINGZILLA_UTF8_NORM_V128RELAXED_H_
