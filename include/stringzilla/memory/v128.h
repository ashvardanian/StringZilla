/**
 *  @brief WebAssembly SIMD128 backend for memory.
 *  @file include/stringzilla/memory/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_V128_H_
#define STRINGZILLA_MEMORY_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("simd128")
#endif

SZ_PUBLIC void sz_copy_v128(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    for (; length >= 16; target += 16, source += 16, length -= 16) wasm_v128_store(target, wasm_v128_load(source));
    if (length) sz_copy_serial(target, source, length);
}

SZ_PUBLIC void sz_move_v128(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target < source || target >= source + length) {
        // Non-overlapping (or `target` precedes `source`): copy forward.
        sz_copy_v128(target, source, length);
    }
    else {
        // Overlapping with `target` after `source`: copy backward.
        target += length;
        source += length;
        while (length >= 16) {
            target -= 16, source -= 16, length -= 16;
            wasm_v128_store(target, wasm_v128_load(source));
        }
        while (length) {
            target -= 1, source -= 1, length -= 1;
            *target = *source;
        }
    }
}

SZ_PUBLIC void sz_fill_v128(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    v128_t fill_vec = wasm_i8x16_splat((sz_i8_t)value);
    while (length >= 16) {
        wasm_v128_store(target, fill_vec);
        target += 16;
        length -= 16;
    }
    if (length) sz_fill_serial(target, length, value);
}

SZ_PUBLIC void sz_lookup_v128(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]) {

    // For tiny inputs the SIMD setup isn't worth it. Match the NEON heuristic.
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    // Load the 256-byte table into 16 vectors of 16 bytes each.
    v128_t lut_vecs[16];
    for (sz_size_t i = 0; i < 16; ++i) lut_vecs[i] = wasm_v128_load(lut + i * 16);

    v128_t const low_nibble_mask_vec = wasm_i8x16_splat(0x0F);
    v128_t const bit4_vec = wasm_i8x16_splat(0x10);
    v128_t const bit5_vec = wasm_i8x16_splat(0x20);
    v128_t const bit6_vec = wasm_i8x16_splat(0x40);
    v128_t const bit7_vec = wasm_i8x16_splat((sz_i8_t)0x80);

    while (length >= 16) {
        v128_t source_vec = wasm_v128_load(source);
        // The low nibble of each byte selects within a 16-byte sub-table via swizzle (== pshufb).
        v128_t low_nibble_idx_vec = wasm_v128_and(source_vec, low_nibble_mask_vec);

        // Index each of the 16 sub-tables by the low nibble.
        v128_t sub_table_vecs[16];
        for (sz_size_t i = 0; i < 16; ++i) sub_table_vecs[i] = wasm_i8x16_swizzle(lut_vecs[i], low_nibble_idx_vec);

        // Tree reduction selecting the right sub-table by the four high bits of the source.
        // Bit 4 (0x10): select between adjacent pairs.
        v128_t select_bit4_vec = wasm_i8x16_eq(wasm_v128_and(source_vec, bit4_vec), wasm_i8x16_splat(0));
        for (sz_size_t i = 0; i < 16; i += 2)
            sub_table_vecs[i] = wasm_v128_bitselect(sub_table_vecs[i], sub_table_vecs[i + 1], select_bit4_vec);
        // Bit 5 (0x20).
        v128_t select_bit5_vec = wasm_i8x16_eq(wasm_v128_and(source_vec, bit5_vec), wasm_i8x16_splat(0));
        for (sz_size_t i = 0; i < 16; i += 4)
            sub_table_vecs[i] = wasm_v128_bitselect(sub_table_vecs[i], sub_table_vecs[i + 2], select_bit5_vec);
        // Bit 6 (0x40).
        v128_t select_bit6_vec = wasm_i8x16_eq(wasm_v128_and(source_vec, bit6_vec), wasm_i8x16_splat(0));
        for (sz_size_t i = 0; i < 16; i += 8)
            sub_table_vecs[i] = wasm_v128_bitselect(sub_table_vecs[i], sub_table_vecs[i + 4], select_bit6_vec);
        // Bit 7 (0x80).
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
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_MEMORY_V128_H_
