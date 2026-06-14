/**
 *  @brief WebAssembly SIMD128 case-insensitive UTF-8 search, comparison & invariance backend.
 *  @file include/stringzilla/utf8_case_insensitive/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_case_insensitive.h
 */
#ifndef STRINGZILLA_UTF8_CASE_INSENSITIVE_V128_H_
#define STRINGZILLA_UTF8_CASE_INSENSITIVE_V128_H_

#include "stringzilla/utf8_case_insensitive/serial.h"
#include "stringzilla/memory/v128.h" // `sz_load_partial_v128_`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/*  Case @b invariance is vectorized: it is a pure scan with no length-changing folds, so it ports
 *  directly from the NEON kernel — wasm has a TRUE `wasm_i8x16_bitmask`, so the two-register movemask is
 *  just `bitmask(low) | bitmask(high) << 16`. Case-insensitive @b find and @b order route through the
 *  value-exact serial scaffolding (the vectorized scripted-driver search is a separate effort). */

#pragma region Case Invariance

/** @brief 32-bit movemask of two 16-byte registers: bit `i` = lane `i` of `low`, bit `16+i` = lane `i` of `high`. */
SZ_INTERNAL sz_u32_t sz_utf8_ci_movemask_v128x2_(v128_t low, v128_t high) {
    return (sz_u32_t)(sz_u16_t)wasm_i8x16_bitmask(low) | ((sz_u32_t)(sz_u16_t)wasm_i8x16_bitmask(high) << 16);
}

/** @brief 0xFF where `(byte - start)` is unsigned-`< length`, i.e. `byte` in `[start, start+length)`. */
SZ_INTERNAL v128_t sz_utf8_ci_in_range_v128_(v128_t bytes, sz_u8_t start, sz_u8_t length) {
    return wasm_u8x16_lt(wasm_i8x16_sub(bytes, wasm_i8x16_splat((sz_i8_t)start)), wasm_i8x16_splat((sz_i8_t)length));
}

SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_v128(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *text_cursor = (sz_u8_t const *)str;

    // Advance by min(length, 29) per block; the 3-byte slack keeps every checked lead's continuation
    // bytes inside the same 32-byte load. Lanes past the string are zero — never letters or leads.
    while (length) {
        sz_size_t block_length = length < 29 ? length : 29;
        sz_u32_t lead_mask = block_length >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << block_length) - 1);
        v128_t low, high;
        if (length >= 32) {
            low = wasm_v128_load(text_cursor), high = wasm_v128_load(text_cursor + 16);
        }
        else {
            low = sz_load_partial_v128_((sz_cptr_t)text_cursor, length < 16 ? length : 16);
            high = length > 16 ? sz_load_partial_v128_((sz_cptr_t)(text_cursor + 16), length - 16) : wasm_i8x16_splat(0);
        }

        // 1. ASCII letters (A-Z, a-z) are never case-invariant.
        sz_u32_t is_upper = sz_utf8_ci_movemask_v128x2_(sz_utf8_ci_in_range_v128_(low, 'A', 26),
                                                        sz_utf8_ci_in_range_v128_(high, 'A', 26));
        sz_u32_t is_lower = sz_utf8_ci_movemask_v128x2_(sz_utf8_ci_in_range_v128_(low, 'a', 26),
                                                        sz_utf8_ci_in_range_v128_(high, 'a', 26));
        if (is_upper | is_lower) return sz_false_k;

        // 2. Any non-ASCII lead needs a per-family check.
        v128_t x80 = wasm_i8x16_splat((sz_i8_t)0x80);
        sz_u32_t is_non_ascii = sz_utf8_ci_movemask_v128x2_(wasm_u8x16_ge(low, x80), wasm_u8x16_ge(high, x80)) &
                                lead_mask;
        if (is_non_ascii) {
            v128_t xe0 = wasm_i8x16_splat((sz_i8_t)0xE0), xf0 = wasm_i8x16_splat((sz_i8_t)0xF0);
            v128_t xc0 = wasm_i8x16_splat((sz_i8_t)0xC0), xf8 = wasm_i8x16_splat((sz_i8_t)0xF8);
            sz_u32_t is_two = sz_utf8_ci_movemask_v128x2_(wasm_i8x16_eq(wasm_v128_and(low, xe0), xc0),
                                                          wasm_i8x16_eq(wasm_v128_and(high, xe0), xc0)) &
                              lead_mask;
            sz_u32_t is_three = sz_utf8_ci_movemask_v128x2_(wasm_i8x16_eq(wasm_v128_and(low, xf0), xe0),
                                                            wasm_i8x16_eq(wasm_v128_and(high, xf0), xe0)) &
                                lead_mask;
            sz_u32_t is_four = sz_utf8_ci_movemask_v128x2_(wasm_i8x16_eq(wasm_v128_and(low, xf8), xf0),
                                                           wasm_i8x16_eq(wasm_v128_and(high, xf8), xf0)) &
                               lead_mask;

            // 3. 4-byte bicameral scripts (SMP): F0 with second byte 90/91/96/9D/9E.
            if (is_four) {
                static sz_u8_t const seconds[5] = {0x90, 0x91, 0x96, 0x9D, 0x9E};
                sz_u32_t hit = 0;
                for (int value = 0; value < 5; ++value) {
                    v128_t b = wasm_i8x16_splat((sz_i8_t)seconds[value]);
                    hit |= sz_utf8_ci_movemask_v128x2_(wasm_i8x16_eq(low, b), wasm_i8x16_eq(high, b));
                }
                if ((is_four << 1) & hit) return sz_false_k;
            }

            // 4. 2-byte bicameral leads C3-D6, plus the C2 B5 micro sign.
            if (is_two) {
                sz_u32_t is_bicameral = sz_utf8_ci_movemask_v128x2_(sz_utf8_ci_in_range_v128_(low, 0xC3, 0x14),
                                                                    sz_utf8_ci_in_range_v128_(high, 0xC3, 0x14));
                v128_t xc2 = wasm_i8x16_splat((sz_i8_t)0xC2), xb5 = wasm_i8x16_splat((sz_i8_t)0xB5);
                sz_u32_t is_c2 = sz_utf8_ci_movemask_v128x2_(wasm_i8x16_eq(low, xc2), wasm_i8x16_eq(high, xc2)) & is_two;
                if (is_c2) {
                    sz_u32_t is_b5 = sz_utf8_ci_movemask_v128x2_(wasm_i8x16_eq(low, xb5), wasm_i8x16_eq(high, xb5));
                    if ((is_c2 << 1) & is_b5) return sz_false_k;
                }
                if (is_bicameral & is_two) return sz_false_k;
            }

            // 5. 3-byte bicameral sequences.
            if (is_three) {
                v128_t xe1 = wasm_i8x16_splat((sz_i8_t)0xE1), xef = wasm_i8x16_splat((sz_i8_t)0xEF);
                sz_u32_t is_e1 = sz_utf8_ci_movemask_v128x2_(wasm_i8x16_eq(low, xe1), wasm_i8x16_eq(high, xe1));
                if (is_e1 & is_three) return sz_false_k;
                sz_u32_t is_ef = sz_utf8_ci_movemask_v128x2_(wasm_i8x16_eq(low, xef), wasm_i8x16_eq(high, xef));
                if (is_ef & is_three) return sz_false_k;

                // E2: safe only for second byte 80-83.
                v128_t xe2 = wasm_i8x16_splat((sz_i8_t)0xE2);
                sz_u32_t is_e2 = sz_utf8_ci_movemask_v128x2_(wasm_i8x16_eq(low, xe2), wasm_i8x16_eq(high, xe2)) &
                                 is_three;
                if (is_e2) {
                    sz_u32_t e2_safe = sz_utf8_ci_movemask_v128x2_(sz_utf8_ci_in_range_v128_(low, 0x80, 0x04),
                                                                  sz_utf8_ci_in_range_v128_(high, 0x80, 0x04));
                    if ((is_e2 << 1) & ~e2_safe) return sz_false_k;
                }

                // EA: bicameral second bytes 99-9F, AC-AE.
                v128_t xea = wasm_i8x16_splat((sz_i8_t)0xEA);
                sz_u32_t is_ea = sz_utf8_ci_movemask_v128x2_(wasm_i8x16_eq(low, xea), wasm_i8x16_eq(high, xea)) &
                                 is_three;
                if (is_ea) {
                    sz_u32_t is_99 = sz_utf8_ci_movemask_v128x2_(sz_utf8_ci_in_range_v128_(low, 0x99, 0x07),
                                                                 sz_utf8_ci_in_range_v128_(high, 0x99, 0x07));
                    sz_u32_t is_ac = sz_utf8_ci_movemask_v128x2_(sz_utf8_ci_in_range_v128_(low, 0xAC, 0x03),
                                                                 sz_utf8_ci_in_range_v128_(high, 0xAC, 0x03));
                    if ((is_ea << 1) & (is_99 | is_ac)) return sz_false_k;
                }
            }
        }

        text_cursor += block_length;
        length -= block_length;
    }

    return sz_true_k;
}

#pragma endregion // Case Invariance

SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_v128( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length,          //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                matched_length);
}

SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_v128(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                            sz_size_t b_length) {
    return sz_utf8_case_insensitive_order_serial(a, a_length, b, b_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CASE_INSENSITIVE_V128_H_
