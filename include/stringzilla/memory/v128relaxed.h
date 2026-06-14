/**
 *  @brief WebAssembly relaxed-SIMD backend for memory (level above SIMD128).
 *  @file include/stringzilla/memory/v128relaxed.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_V128RELAXED_H_
#define STRINGZILLA_MEMORY_V128RELAXED_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"
#include "stringzilla/memory/v128.h" // baseline SIMD128 fallbacks

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/*  `copy`, `move`, and `fill` are pure load/store streams with no shuffle or arithmetic, so
 *  relaxed-simd offers nothing — delegate to the baseline SIMD128 kernels. */

SZ_PUBLIC void sz_copy_v128relaxed(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_copy_v128(target, source, length);
}

SZ_PUBLIC void sz_move_v128relaxed(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_move_v128(target, source, length);
}

SZ_PUBLIC void sz_fill_v128relaxed(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    sz_fill_v128(target, length, value);
}

/*  `sz_lookup_v128relaxed` replaces every `wasm_i8x16_swizzle` with `wasm_i8x16_relaxed_swizzle`.
 *  The relaxed variant skips the out-of-range-index zeroing of the strict swizzle, which is cheaper
 *  on most engines (~21% faster in audit). The selector here is always the LOW NIBBLE of a byte, so
 *  indices are in `[0, 15]` and never trigger the strict variant's zeroing path — the result is
 *  therefore byte-for-byte identical to `sz_lookup_serial`. */
SZ_PUBLIC void sz_lookup_v128relaxed(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                     char const lut[sz_at_least_(256)]) {

    // For tiny inputs the SIMD setup isn't worth it. Match the baseline heuristic.
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    // Load the 256-byte table into 16 vectors of 16 bytes each.
    v128_t lut_vecs[16];
    for (sz_size_t sub_table_index = 0; sub_table_index < 16; ++sub_table_index)
        lut_vecs[sub_table_index] = wasm_v128_load(lut + sub_table_index * 16);

    v128_t const low_nibble_mask_vec = wasm_i8x16_splat(0x0F);
    v128_t const bit4_vec = wasm_i8x16_splat(0x10);
    v128_t const bit5_vec = wasm_i8x16_splat(0x20);
    v128_t const bit6_vec = wasm_i8x16_splat(0x40);
    v128_t const bit7_vec = wasm_i8x16_splat((sz_i8_t)0x80);

    while (length >= 16) {
        v128_t source_vec = wasm_v128_load(source);
        // The low nibble of each byte selects within a 16-byte sub-table (indices always in [0, 15]).
        v128_t low_nibble_index_vec = wasm_v128_and(source_vec, low_nibble_mask_vec);

        // Index each of the 16 sub-tables by the low nibble with the cheaper relaxed swizzle.
        v128_t sub_table_vecs[16];
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; ++sub_table_index)
            sub_table_vecs[sub_table_index] = wasm_i8x16_relaxed_swizzle(lut_vecs[sub_table_index],
                                                                         low_nibble_index_vec);

        // Tree reduction selecting the right sub-table by the four high bits of the source.
        v128_t select_bit4_vec = wasm_i8x16_eq(wasm_v128_and(source_vec, bit4_vec), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 2)
            sub_table_vecs[sub_table_index] = wasm_v128_bitselect(sub_table_vecs[sub_table_index],
                                                                  sub_table_vecs[sub_table_index + 1], select_bit4_vec);
        v128_t select_bit5_vec = wasm_i8x16_eq(wasm_v128_and(source_vec, bit5_vec), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 4)
            sub_table_vecs[sub_table_index] = wasm_v128_bitselect(sub_table_vecs[sub_table_index],
                                                                  sub_table_vecs[sub_table_index + 2], select_bit5_vec);
        v128_t select_bit6_vec = wasm_i8x16_eq(wasm_v128_and(source_vec, bit6_vec), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 8)
            sub_table_vecs[sub_table_index] = wasm_v128_bitselect(sub_table_vecs[sub_table_index],
                                                                  sub_table_vecs[sub_table_index + 4], select_bit6_vec);
        v128_t select_bit7_vec = wasm_i8x16_eq(wasm_v128_and(source_vec, bit7_vec), wasm_i8x16_splat(0));
        v128_t result_vec = wasm_v128_bitselect(sub_table_vecs[0], sub_table_vecs[8], select_bit7_vec);

        wasm_v128_store(target, result_vec);
        source += 16, target += 16, length -= 16;
    }

    // Handle the tail with serial code.
    if (length) sz_lookup_serial(target, length, source, lut);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_MEMORY_V128RELAXED_H_
