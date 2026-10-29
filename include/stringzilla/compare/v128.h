/**
 *  @file include/stringzilla/compare/v128.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief WebAssembly SIMD128 backend for compare.
 *
 *  @sa c/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_V128_H_
#define STRINGZILLA_COMPARE_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  Branch-light partial load/store of 0..16 bytes — the shared building block for WASM tails, so
 *  the kernels never fall back to a scalar byte loop and never over-read/over-write past the
 *  buffer. WASM lane load/store need compile-time-constant lane indices, so the 0..7 byte assembly
 *  is a small constant-lane switch; the 8..15 case loads the low 8 with @c load64_zero and folds
 *  the remaining bytes in with one constant `i8x16.shuffle`. They live in this lowest @c v128
 *  header, which includes no other family, so every @c v128 backend reaches them without cycles. */
#if STRINGZILLA_ARCH_WASM_
#if STRINGZILLA_ARCH_WASM_V128_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

STRINGZILLA_INLINE v128_t sz_load_partial_lo8_v128_(sz_u8_t const *source_pointer, sz_size_t remainder) {
    v128_t result_u8x16 = wasm_u64x2_splat(0);
    switch (remainder) {
    case 1: result_u8x16 = wasm_v128_load8_lane(source_pointer, result_u8x16, 0); break;
    case 2: result_u8x16 = wasm_v128_load16_lane(source_pointer, result_u8x16, 0); break;
    case 3:
        result_u8x16 = wasm_v128_load16_lane(source_pointer, result_u8x16, 0);
        result_u8x16 = wasm_v128_load8_lane(source_pointer + 2, result_u8x16, 2);
        break;
    case 4: result_u8x16 = wasm_v128_load32_lane(source_pointer, result_u8x16, 0); break;
    case 5:
        result_u8x16 = wasm_v128_load32_lane(source_pointer, result_u8x16, 0);
        result_u8x16 = wasm_v128_load8_lane(source_pointer + 4, result_u8x16, 4);
        break;
    case 6:
        result_u8x16 = wasm_v128_load32_lane(source_pointer, result_u8x16, 0);
        result_u8x16 = wasm_v128_load16_lane(source_pointer + 4, result_u8x16, 2);
        break;
    case 7:
        result_u8x16 = wasm_v128_load32_lane(source_pointer, result_u8x16, 0);
        result_u8x16 = wasm_v128_load16_lane(source_pointer + 4, result_u8x16, 2);
        result_u8x16 = wasm_v128_load8_lane(source_pointer + 6, result_u8x16, 6);
        break;
    default: break;
    }
    return result_u8x16;
}

/** Load exactly @p length (0..16) bytes into a v128; remaining lanes zero; no over-read. */
STRINGZILLA_INLINE v128_t sz_load_partial_v128_(sz_cptr_t source, sz_size_t length) {
    sz_u8_t const *source_pointer = (sz_u8_t const *)source;
    if (length >= 16) return wasm_v128_load(source_pointer);
    if (length & 8) {
        v128_t low_half_u8x16 = wasm_v128_load64_zero(source_pointer);
        v128_t high_half_u8x16 = sz_load_partial_lo8_v128_(source_pointer + 8, length & 7);
        return wasm_i8x16_shuffle(low_half_u8x16, high_half_u8x16, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22,
                                  23);
    }
    return sz_load_partial_lo8_v128_(source_pointer, length);
}

STRINGZILLA_INLINE void sz_store_partial_lo8_v128_(sz_u8_t *target_pointer, v128_t data_u8x16, sz_size_t remainder) {
    switch (remainder) {
    case 1: wasm_v128_store8_lane(target_pointer, data_u8x16, 0); break;
    case 2: wasm_v128_store16_lane(target_pointer, data_u8x16, 0); break;
    case 3:
        wasm_v128_store16_lane(target_pointer, data_u8x16, 0);
        wasm_v128_store8_lane(target_pointer + 2, data_u8x16, 2);
        break;
    case 4: wasm_v128_store32_lane(target_pointer, data_u8x16, 0); break;
    case 5:
        wasm_v128_store32_lane(target_pointer, data_u8x16, 0);
        wasm_v128_store8_lane(target_pointer + 4, data_u8x16, 4);
        break;
    case 6:
        wasm_v128_store32_lane(target_pointer, data_u8x16, 0);
        wasm_v128_store16_lane(target_pointer + 4, data_u8x16, 2);
        break;
    case 7:
        wasm_v128_store32_lane(target_pointer, data_u8x16, 0);
        wasm_v128_store16_lane(target_pointer + 4, data_u8x16, 2);
        wasm_v128_store8_lane(target_pointer + 6, data_u8x16, 6);
        break;
    default: break;
    }
}

/** Store exactly @p length (0..16) bytes from a v128; no over-write past the buffer. */
STRINGZILLA_INLINE void sz_store_partial_v128_(sz_ptr_t target, v128_t data_u8x16, sz_size_t length) {
    sz_u8_t *target_pointer = (sz_u8_t *)target;
    if (length >= 16) {
        wasm_v128_store(target_pointer, data_u8x16);
        return;
    }
    if (length & 8) {
        wasm_v128_store64_lane(target_pointer, data_u8x16, 0);
        v128_t high_half_u8x16 = wasm_i8x16_shuffle(data_u8x16, data_u8x16, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4,
                                                    5, 6, 7);
        sz_store_partial_lo8_v128_(target_pointer + 8, high_half_u8x16, length & 7);
    }
    else { sz_store_partial_lo8_v128_(target_pointer, data_u8x16, length); }
}

STRINGZILLA_INLINE sz_ordering_t sz_order_v128_(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Lexicographic ordering needs the byte-reversed comparison of the first differing word,
    //! which has no cheap SIMD128 equivalent that beats the serial SWAR variant. Delegate.
    return sz_order_serial_(a, a_length, b, b_length);
}

STRINGZILLA_INLINE sz_bool_t sz_equal_v128_(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    if (length < 16) return sz_equal_serial_(a, b, length);

    // Per block we `xor` the inputs and test with `wasm_v128_any_true` (a single cheap reduction the
    // engines lower to one `ptest`), instead of `wasm_i8x16_eq` + `wasm_i8x16_all_true` (a lane
    // minimum). Four 16-byte differences are OR-ed so a single `any_true` covers a 64-byte stride.
    sz_size_t offset = 0;
    while (offset + 64 <= length) {
        v128_t diff_at_0_u8x16 = wasm_v128_xor(wasm_v128_load(a + offset + 0), wasm_v128_load(b + offset + 0));
        v128_t diff_at_16_u8x16 = wasm_v128_xor(wasm_v128_load(a + offset + 16), wasm_v128_load(b + offset + 16));
        v128_t diff_at_32_u8x16 = wasm_v128_xor(wasm_v128_load(a + offset + 32), wasm_v128_load(b + offset + 32));
        v128_t diff_at_48_u8x16 = wasm_v128_xor(wasm_v128_load(a + offset + 48), wasm_v128_load(b + offset + 48));
        v128_t any_diff_u8x16 = wasm_v128_or(wasm_v128_or(diff_at_0_u8x16, diff_at_16_u8x16),
                                             wasm_v128_or(diff_at_32_u8x16, diff_at_48_u8x16));
        if (wasm_v128_any_true(any_diff_u8x16)) return sz_false_k;
        offset += 64;
    }
    while (offset + 16 <= length) {
        if (wasm_v128_any_true(wasm_v128_xor(wasm_v128_load(a + offset), wasm_v128_load(b + offset))))
            return sz_false_k;
        offset += 16;
    }

    // Final overlapping check on the last 16-byte window.
    if (wasm_v128_any_true(wasm_v128_xor(wasm_v128_load(a + length - 16), wasm_v128_load(b + length - 16))))
        return sz_false_k;
    return sz_true_k;
}

#if STRINGZILLA_TARGET_V128

STRINGZILLA_API sz_status_t sz_order_v128(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                          sz_ordering_t *ordering, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *ordering = sz_order_v128_(a, a_length, b, b_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_equal_v128(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                          sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *equal = sz_equal_v128_(a, b, length);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_V128

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // STRINGZILLA_ARCH_WASM_V128_
#endif // STRINGZILLA_ARCH_WASM_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_V128_H_
