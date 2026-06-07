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
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("relaxed-simd")
#endif

/*  Byte and substring search are dominated by `wasm_i8x16_eq` + `wasm_i8x16_bitmask`; relaxed-simd
 *  adds nothing there, so `find_byte`, `rfind_byte`, `find`, and `rfind` delegate to the baseline.
 *  Only the byteset kernels benefit, via `wasm_i8x16_relaxed_swizzle` on the small bit table. */

SZ_PUBLIC sz_cptr_t sz_find_byte_v128relaxed(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    return sz_find_byte_v128(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_v128relaxed(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    return sz_rfind_byte_v128(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_find_v128relaxed(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    return sz_find_v128(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_v128relaxed(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    return sz_rfind_v128(h, h_length, n, n_length);
}

/*  Relaxed byteset probe: the per-byte bit mask `1 << (c & 7)` is fetched with the cheaper
 *  `wasm_i8x16_relaxed_swizzle` — its index `c & 7` is always in `[0, 7]`, so it can never hit the
 *  strict variant's out-of-range zeroing path and the result is identical to serial.
 *
 *  The two SET lookups (`set_top` / `set_bottom`) MUST stay on the strict `wasm_i8x16_swizzle`: they
 *  deliberately rely on the strict guarantee that out-of-range indices return 0 to implement the
 *  32-byte table split across two 16-byte halves. Relaxed swizzle leaves those lanes UNDEFINED, so
 *  using it there would break correctness. */
SZ_INTERNAL sz_u32_t sz_find_byteset_wasm_relaxed_register_(v128_t h_vec, v128_t set_top_vec, v128_t set_bottom_vec) {
    v128_t byte_index_vec = wasm_u8x16_shr(h_vec, 3); // c >> 3, in [0, 31]
    v128_t bit_table_vec = wasm_i8x16_make(1, 2, 4, 8, 16, 32, 64, (sz_i8_t)128, 0, 0, 0, 0, 0, 0, 0, 0);
    // Index `c & 7` is in [0, 7] -> always in range -> relaxed swizzle is exact.
    v128_t byte_mask_vec = wasm_i8x16_relaxed_swizzle(bit_table_vec, wasm_v128_and(h_vec, wasm_i8x16_splat(7)));
    // Strict swizzle below: out-of-range indices must zero (table split across two halves).
    v128_t matches_top_vec = wasm_i8x16_swizzle(set_top_vec, byte_index_vec);
    v128_t matches_bottom_vec =
        wasm_i8x16_swizzle(set_bottom_vec, wasm_i8x16_sub(byte_index_vec, wasm_i8x16_splat(16)));
    v128_t matches_vec = wasm_v128_or(matches_top_vec, matches_bottom_vec);
    matches_vec = wasm_i8x16_ne(wasm_v128_and(matches_vec, byte_mask_vec), wasm_i8x16_splat(0));
    return (sz_u32_t)wasm_i8x16_bitmask(matches_vec);
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_v128relaxed(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    sz_u128_vec_t h_vec;
    v128_t set_top_vec = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_vec = wasm_v128_load(&set->_u8s[16]);

    for (; h_length >= 16; h += 16, h_length -= 16) {
        h_vec.v128 = wasm_v128_load(h);
        sz_u32_t matches = sz_find_byteset_wasm_relaxed_register_(h_vec.v128, set_top_vec, set_bottom_vec);
        if (matches) return h + __builtin_ctz(matches);
    }

    return sz_find_byteset_serial(h, h_length, set);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_v128relaxed(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    sz_u128_vec_t h_vec;
    v128_t set_top_vec = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_vec = wasm_v128_load(&set->_u8s[16]);

    for (; h_length >= 16; h_length -= 16) {
        h_vec.v128 = wasm_v128_load(h + h_length - 16);
        sz_u32_t matches = sz_find_byteset_wasm_relaxed_register_(h_vec.v128, set_top_vec, set_bottom_vec);
        if (matches) return h + h_length - 1 - (__builtin_clz(matches) - 16);
    }

    return sz_rfind_byteset_serial(h, h_length, set);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_V128RELAXED_H_
