/**
 *  @brief WebAssembly relaxed-SIMD backend for find (level above SIMD128).
 *  @file include/stringzilla/find/v128relaxed.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_V128RELAXED_H_
#define STRINGZILLA_FIND_V128RELAXED_H_

#include "stringzilla/types.h"
#include "stringzilla/find/serial.h"
#include "stringzilla/find/v128.h" // baseline SIMD128 fallbacks

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/*  Byte and substring search are `wasm_i8x16_eq` + `wasm_i8x16_bitmask` + ctz. The relaxed-SIMD
 *  instruction set — `relaxed_swizzle`, `relaxed_laneselect`, `relaxed_madd`/`nmadd`, the integer/`q15`
 *  `relaxed_dot`, `relaxed_min`/`max`, `relaxed_trunc` — operates on table lookups, lane muxing, fused
 *  multiply-add and float math. NONE of them act on an equality mask or movemask, so there is no relaxed
 *  way to make this search faster: `find_byte`, `rfind_byte`, `find`, and `rfind` delegate to the
 *  baseline. `relaxed_swizzle` is the one find-family win, and it is already used by the byteset kernels
 *  below (their bit-table index is provably in `[0, 7]`, so relaxed equals strict). */

SZ_API_COMPTIME sz_cptr_t sz_find_byte_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    return sz_find_byte_v128(haystack, haystack_length, needle);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    return sz_rfind_byte_v128(haystack, haystack_length, needle);
}

SZ_API_COMPTIME sz_cptr_t sz_find_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_size_t needle_length) {
    return sz_find_v128(haystack, haystack_length, needle, needle_length);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_size_t needle_length) {
    return sz_rfind_v128(haystack, haystack_length, needle, needle_length);
}

/*  Relaxed byteset probe: the per-byte bit mask `1 << (c & 7)` is fetched with the cheaper
 *  `wasm_i8x16_relaxed_swizzle` — its index `c & 7` is always in `[0, 7]`, so it can never hit the
 *  strict variant's out-of-range zeroing path and the result is identical to serial.
 *
 *  The two SET lookups (`set_top` / `set_bottom`) MUST stay on the strict `wasm_i8x16_swizzle`: they
 *  deliberately rely on the strict guarantee that out-of-range indices return 0 to implement the
 *  32-byte table split across two 16-byte halves. Relaxed swizzle leaves those lanes UNDEFINED, so
 *  using it there would break correctness. */

/**
 *  @brief Compute a per-lane match mask using relaxed swizzle for the bit-table lookup.
 *  @param haystack_vec The 16-byte input register.
 *  @param set_top_vec Top half of the byteset (byte indices 0..15).
 *  @param set_bottom_vec Bottom half of the byteset (byte indices 16..31).
 *  @return 0xFF per lane where the byte belongs to the set, 0x00 otherwise.
 */
SZ_HELPER_AUTO v128_t sz_find_byteset_match_v128relaxed_(v128_t haystack_vec, v128_t set_top_vec,
                                                         v128_t set_bottom_vec) {
    v128_t byte_index_vec = wasm_u8x16_shr(haystack_vec, 3); // c >> 3, in [0, 31]
    v128_t bit_table_vec = wasm_i8x16_make(1, 2, 4, 8, 16, 32, 64, (sz_i8_t)128, 0, 0, 0, 0, 0, 0, 0, 0);
    // Index `c & 7` is in [0, 7] -> always in range -> relaxed swizzle is exact.
    v128_t byte_mask_vec = wasm_i8x16_relaxed_swizzle(bit_table_vec, wasm_v128_and(haystack_vec, wasm_i8x16_splat(7)));
    // Strict swizzle below: out-of-range indices must zero (table split across two halves).
    v128_t matches_top_vec = wasm_i8x16_swizzle(set_top_vec, byte_index_vec);
    v128_t matches_bottom_vec = wasm_i8x16_swizzle(set_bottom_vec,
                                                   wasm_i8x16_sub(byte_index_vec, wasm_i8x16_splat(16)));
    v128_t matches_vec = wasm_v128_or(matches_top_vec, matches_bottom_vec);
    return wasm_i8x16_ne(wasm_v128_and(matches_vec, byte_mask_vec), wasm_i8x16_splat(0));
}

SZ_API_COMPTIME sz_cptr_t sz_find_byteset_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length,
                                                      sz_byteset_t const *set) {
    v128_t set_top_vec = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_vec = wasm_v128_load(&set->_u8s[16]);

    while (haystack_length >= 64) {
        v128_t match_at_0_vec = sz_find_byteset_match_v128relaxed_(wasm_v128_load(haystack + 0), set_top_vec,
                                                                   set_bottom_vec);
        v128_t match_at_16_vec = sz_find_byteset_match_v128relaxed_(wasm_v128_load(haystack + 16), set_top_vec,
                                                                    set_bottom_vec);
        v128_t match_at_32_vec = sz_find_byteset_match_v128relaxed_(wasm_v128_load(haystack + 32), set_top_vec,
                                                                    set_bottom_vec);
        v128_t match_at_48_vec = sz_find_byteset_match_v128relaxed_(wasm_v128_load(haystack + 48), set_top_vec,
                                                                    set_bottom_vec);
        v128_t any_match_vec = wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec),
                                            wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_first_v128_(match_at_0_vec, haystack + 0))) return found;
            if ((found = sz_locate_first_v128_(match_at_16_vec, haystack + 16))) return found;
            if ((found = sz_locate_first_v128_(match_at_32_vec, haystack + 32))) return found;
            if ((found = sz_locate_first_v128_(match_at_48_vec, haystack + 48))) return found;
        }
        haystack += 64, haystack_length -= 64;
    }

    for (; haystack_length >= 16; haystack += 16, haystack_length -= 16) {
        v128_t match_vec = sz_find_byteset_match_v128relaxed_(wasm_v128_load(haystack), set_top_vec, set_bottom_vec);
        sz_cptr_t found = sz_locate_first_v128_(match_vec, haystack);
        if (found) return found;
    }

    return sz_find_byteset_serial(haystack, haystack_length, set);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length,
                                                       sz_byteset_t const *set) {
    v128_t set_top_vec = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_vec = wasm_v128_load(&set->_u8s[16]);

    while (haystack_length >= 64) {
        sz_cptr_t window_start = haystack + haystack_length - 64;
        v128_t match_at_0_vec = sz_find_byteset_match_v128relaxed_(wasm_v128_load(window_start + 0), set_top_vec,
                                                                   set_bottom_vec);
        v128_t match_at_16_vec = sz_find_byteset_match_v128relaxed_(wasm_v128_load(window_start + 16), set_top_vec,
                                                                    set_bottom_vec);
        v128_t match_at_32_vec = sz_find_byteset_match_v128relaxed_(wasm_v128_load(window_start + 32), set_top_vec,
                                                                    set_bottom_vec);
        v128_t match_at_48_vec = sz_find_byteset_match_v128relaxed_(wasm_v128_load(window_start + 48), set_top_vec,
                                                                    set_bottom_vec);
        v128_t any_match_vec = wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec),
                                            wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_last_v128_(match_at_48_vec, window_start + 48))) return found;
            if ((found = sz_locate_last_v128_(match_at_32_vec, window_start + 32))) return found;
            if ((found = sz_locate_last_v128_(match_at_16_vec, window_start + 16))) return found;
            if ((found = sz_locate_last_v128_(match_at_0_vec, window_start + 0))) return found;
        }
        haystack_length -= 64;
    }

    for (; haystack_length >= 16; haystack_length -= 16) {
        sz_cptr_t window_start = haystack + haystack_length - 16;
        v128_t match_vec = sz_find_byteset_match_v128relaxed_(wasm_v128_load(window_start), set_top_vec,
                                                              set_bottom_vec);
        sz_cptr_t found = sz_locate_last_v128_(match_vec, window_start);
        if (found) return found;
    }

    return sz_rfind_byteset_serial(haystack, haystack_length, set);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_V128RELAXED_H_
