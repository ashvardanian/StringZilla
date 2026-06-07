/**
 *  @brief WebAssembly SIMD128 backend for find.
 *  @file include/stringzilla/find/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_V128_H_
#define STRINGZILLA_FIND_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/find/serial.h"
#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128

/*  WebAssembly SIMD128 has a TRUE movemask via `wasm_i8x16_bitmask`, producing one bit per byte
 *  (the most significant bit of each lane). Combined with `__builtin_ctz`/`__builtin_clz` we get
 *  the SSE/Westmere-style search. The fixed register width is 16 bytes, like Arm NEON. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("simd128")
#endif

SZ_PUBLIC sz_cptr_t sz_find_byte_v128(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u128_vec_t h_vec, n_vec, matches_vec;
    n_vec.v128 = wasm_i8x16_splat(*(sz_i8_t const *)n);

    while (h_length >= 16) {
        h_vec.v128 = wasm_v128_load(h);
        matches_vec.v128 = wasm_i8x16_eq(h_vec.v128, n_vec.v128);
        sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(matches_vec.v128);
        if (matches) return h + __builtin_ctz(matches);
        h += 16, h_length -= 16;
    }

    return sz_find_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_v128(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u128_vec_t h_vec, n_vec, matches_vec;
    n_vec.v128 = wasm_i8x16_splat(*(sz_i8_t const *)n);

    while (h_length >= 16) {
        h_vec.v128 = wasm_v128_load(h + h_length - 16);
        matches_vec.v128 = wasm_i8x16_eq(h_vec.v128, n_vec.v128);
        sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(matches_vec.v128);
        // `matches` occupies the low 16 bits; the highest set bit is the last match in the window.
        if (matches) return h + h_length - 1 - (__builtin_clz(matches) - 16);
        h_length -= 16;
    }

    return sz_rfind_byte_serial(h, h_length, n);
}

SZ_INTERNAL sz_u32_t sz_find_byteset_wasm_register_(v128_t h_vec, v128_t set_top_vec, v128_t set_bottom_vec) {
    // Serial equivalent per byte `c`: `(set->_u8s[c >> 3] & (1u << (c & 7u))) != 0`.
    v128_t byte_index_vec = wasm_u8x16_shr(h_vec, 3); // c >> 3, in [0, 31]
    // The bit mask `1 << (c & 7)` is produced via a swizzle into a tiny power-of-two table.
    v128_t bit_table = wasm_i8x16_make(1, 2, 4, 8, 16, 32, 64, (sz_i8_t)128, 0, 0, 0, 0, 0, 0, 0, 0);
    v128_t byte_mask_vec = wasm_i8x16_swizzle(bit_table, wasm_v128_and(h_vec, wasm_i8x16_splat(7)));
    // `swizzle` returns 0 for indices >= 16 (or with the high bit set). For the bottom half we shift
    // the index down by 16: values originally < 16 underflow into [240, 256) and yield zeros there.
    v128_t matches_top_vec = wasm_i8x16_swizzle(set_top_vec, byte_index_vec);
    v128_t matches_bottom_vec =
        wasm_i8x16_swizzle(set_bottom_vec, wasm_i8x16_sub(byte_index_vec, wasm_i8x16_splat(16)));
    v128_t matches_vec = wasm_v128_or(matches_top_vec, matches_bottom_vec);
    // Keep a lane only if its selected bit is set.
    matches_vec = wasm_i8x16_ne(wasm_v128_and(matches_vec, byte_mask_vec), wasm_i8x16_splat(0));
    return (sz_u32_t)wasm_i8x16_bitmask(matches_vec);
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_v128(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    sz_u128_vec_t h_vec;
    v128_t set_top_vec = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_vec = wasm_v128_load(&set->_u8s[16]);

    for (; h_length >= 16; h += 16, h_length -= 16) {
        h_vec.v128 = wasm_v128_load(h);
        sz_u32_t matches = sz_find_byteset_wasm_register_(h_vec.v128, set_top_vec, set_bottom_vec);
        if (matches) return h + __builtin_ctz(matches);
    }

    return sz_find_byteset_serial(h, h_length, set);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_v128(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    sz_u128_vec_t h_vec;
    v128_t set_top_vec = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_vec = wasm_v128_load(&set->_u8s[16]);

    for (; h_length >= 16; h_length -= 16) {
        h_vec.v128 = wasm_v128_load(h + h_length - 16);
        sz_u32_t matches = sz_find_byteset_wasm_register_(h_vec.v128, set_top_vec, set_bottom_vec);
        if (matches) return h + h_length - 1 - (__builtin_clz(matches) - 16);
    }

    return sz_rfind_byteset_serial(h, h_length, set);
}

SZ_PUBLIC sz_cptr_t sz_find_v128(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_v128(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
    n_first_vec.v128 = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_first]);
    n_mid_vec.v128 = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_mid]);
    n_last_vec.v128 = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_last]);

    for (; h_length >= n_length + 16; h += 16, h_length -= 16) {
        h_first_vec.v128 = wasm_v128_load(h + offset_first);
        h_mid_vec.v128 = wasm_v128_load(h + offset_mid);
        h_last_vec.v128 = wasm_v128_load(h + offset_last);
        matches_vec.v128 = wasm_v128_and(                          //
            wasm_v128_and(                                         //
                wasm_i8x16_eq(h_first_vec.v128, n_first_vec.v128), //
                wasm_i8x16_eq(h_mid_vec.v128, n_mid_vec.v128)),
            wasm_i8x16_eq(h_last_vec.v128, n_last_vec.v128));
        sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(matches_vec.v128);
        while (matches) {
            int potential_offset = __builtin_ctz(matches);
            if (sz_equal_v128(h + potential_offset, n, n_length)) return h + potential_offset;
            matches &= matches - 1;
        }
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_v128(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_rfind_byte_v128(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
    n_first_vec.v128 = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_first]);
    n_mid_vec.v128 = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_mid]);
    n_last_vec.v128 = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_last]);

    sz_cptr_t h_reversed;
    for (; h_length >= n_length + 16; h_length -= 16) {
        h_reversed = h + h_length - n_length - 16 + 1;
        h_first_vec.v128 = wasm_v128_load(h_reversed + offset_first);
        h_mid_vec.v128 = wasm_v128_load(h_reversed + offset_mid);
        h_last_vec.v128 = wasm_v128_load(h_reversed + offset_last);
        matches_vec.v128 = wasm_v128_and(                          //
            wasm_v128_and(                                         //
                wasm_i8x16_eq(h_first_vec.v128, n_first_vec.v128), //
                wasm_i8x16_eq(h_mid_vec.v128, n_mid_vec.v128)),
            wasm_i8x16_eq(h_last_vec.v128, n_last_vec.v128));
        sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(matches_vec.v128);
        while (matches) {
            // The highest set bit (within the low 16) is the latest offset in this window.
            int potential_offset = __builtin_clz(matches) - 16;
            if (sz_equal_v128(h + h_length - n_length - potential_offset, n, n_length))
                return h + h_length - n_length - potential_offset;
            matches &= ~(1u << (15 - potential_offset));
        }
    }

    return sz_rfind_serial(h, h_length, n, n_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_V128_H_
