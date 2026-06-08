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
#endif

/*  Branch-light partial load/store of 0..16 bytes — the shared building block for WASM tails, so the
 *  kernels never fall back to a scalar byte loop and never over-read/over-write past the buffer. WASM
 *  lane load/store need compile-time-constant lane indices, so the 0..7 byte assembly is a small
 *  constant-lane switch; the 8..15 case loads the low 8 with `load64_zero` and folds the remaining
 *  bytes in with one constant `i8x16.shuffle`. Other `v128` backends `#include` this header to reuse
 *  these (hash short-string loads, `fill_random` tails, …). */
SZ_INTERNAL v128_t sz_load_partial_lo8_v128_(sz_u8_t const *source_pointer, sz_size_t remainder) {
    v128_t result = wasm_u64x2_splat(0);
    switch (remainder) {
    case 1: result = wasm_v128_load8_lane(source_pointer, result, 0); break;
    case 2: result = wasm_v128_load16_lane(source_pointer, result, 0); break;
    case 3:
        result = wasm_v128_load16_lane(source_pointer, result, 0);
        result = wasm_v128_load8_lane(source_pointer + 2, result, 2);
        break;
    case 4: result = wasm_v128_load32_lane(source_pointer, result, 0); break;
    case 5:
        result = wasm_v128_load32_lane(source_pointer, result, 0);
        result = wasm_v128_load8_lane(source_pointer + 4, result, 4);
        break;
    case 6:
        result = wasm_v128_load32_lane(source_pointer, result, 0);
        result = wasm_v128_load16_lane(source_pointer + 4, result, 2);
        break;
    case 7:
        result = wasm_v128_load32_lane(source_pointer, result, 0);
        result = wasm_v128_load16_lane(source_pointer + 4, result, 2);
        result = wasm_v128_load8_lane(source_pointer + 6, result, 6);
        break;
    default: break;
    }
    return result;
}

/** @brief Load exactly `length` (0..16) bytes into a v128; remaining lanes zero; no over-read. */
SZ_INTERNAL v128_t sz_load_partial_v128_(sz_cptr_t source, sz_size_t length) {
    sz_u8_t const *source_pointer = (sz_u8_t const *)source;
    if (length >= 16) return wasm_v128_load(source_pointer);
    if (length & 8) {
        v128_t low_half = wasm_v128_load64_zero(source_pointer);
        v128_t high_half = sz_load_partial_lo8_v128_(source_pointer + 8, length & 7);
        return wasm_i8x16_shuffle(low_half, high_half, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
    }
    return sz_load_partial_lo8_v128_(source_pointer, length);
}

SZ_INTERNAL void sz_store_partial_lo8_v128_(sz_u8_t *target_pointer, v128_t data, sz_size_t remainder) {
    switch (remainder) {
    case 1: wasm_v128_store8_lane(target_pointer, data, 0); break;
    case 2: wasm_v128_store16_lane(target_pointer, data, 0); break;
    case 3:
        wasm_v128_store16_lane(target_pointer, data, 0);
        wasm_v128_store8_lane(target_pointer + 2, data, 2);
        break;
    case 4: wasm_v128_store32_lane(target_pointer, data, 0); break;
    case 5:
        wasm_v128_store32_lane(target_pointer, data, 0);
        wasm_v128_store8_lane(target_pointer + 4, data, 4);
        break;
    case 6:
        wasm_v128_store32_lane(target_pointer, data, 0);
        wasm_v128_store16_lane(target_pointer + 4, data, 2);
        break;
    case 7:
        wasm_v128_store32_lane(target_pointer, data, 0);
        wasm_v128_store16_lane(target_pointer + 4, data, 2);
        wasm_v128_store8_lane(target_pointer + 6, data, 6);
        break;
    default: break;
    }
}

/** @brief Store exactly `length` (0..16) bytes from a v128; no over-write past the buffer. */
SZ_INTERNAL void sz_store_partial_v128_(sz_ptr_t target, v128_t data, sz_size_t length) {
    sz_u8_t *target_pointer = (sz_u8_t *)target;
    if (length >= 16) {
        wasm_v128_store(target_pointer, data);
        return;
    }
    if (length & 8) {
        wasm_v128_store64_lane(target_pointer, data, 0);
        v128_t high_half = wasm_i8x16_shuffle(data, data, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
        sz_store_partial_lo8_v128_(target_pointer + 8, high_half, length & 7);
    }
    else { sz_store_partial_lo8_v128_(target_pointer, data, length); }
}

SZ_PUBLIC void sz_copy_v128(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    for (; length >= 16; target += 16, source += 16, length -= 16) wasm_v128_store(target, wasm_v128_load(source));
    if (length) sz_store_partial_v128_(target, sz_load_partial_v128_(source, length), length);
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
    if (length) sz_store_partial_v128_(target, fill_vec, length);
}

SZ_PUBLIC void sz_lookup_v128(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]) {

    // For tiny inputs the SIMD setup isn't worth it. Match the NEON heuristic.
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
        // The low nibble of each byte selects within a 16-byte sub-table via swizzle (== pshufb).
        v128_t low_nibble_index_vec = wasm_v128_and(source_vec, low_nibble_mask_vec);

        // Index each of the 16 sub-tables by the low nibble.
        v128_t sub_table_vecs[16];
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; ++sub_table_index)
            sub_table_vecs[sub_table_index] = wasm_i8x16_swizzle(lut_vecs[sub_table_index], low_nibble_index_vec);

        // Tree reduction selecting the right sub-table by the four high bits of the source.
        // Bit 4 (0x10): select between adjacent pairs.
        v128_t select_bit4_vec = wasm_i8x16_eq(wasm_v128_and(source_vec, bit4_vec), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 2)
            sub_table_vecs[sub_table_index] = wasm_v128_bitselect(sub_table_vecs[sub_table_index],
                                                                  sub_table_vecs[sub_table_index + 1], select_bit4_vec);
        // Bit 5 (0x20).
        v128_t select_bit5_vec = wasm_i8x16_eq(wasm_v128_and(source_vec, bit5_vec), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 4)
            sub_table_vecs[sub_table_index] = wasm_v128_bitselect(sub_table_vecs[sub_table_index],
                                                                  sub_table_vecs[sub_table_index + 2], select_bit5_vec);
        // Bit 6 (0x40).
        v128_t select_bit6_vec = wasm_i8x16_eq(wasm_v128_and(source_vec, bit6_vec), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 8)
            sub_table_vecs[sub_table_index] = wasm_v128_bitselect(sub_table_vecs[sub_table_index],
                                                                  sub_table_vecs[sub_table_index + 4], select_bit6_vec);
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
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_MEMORY_V128_H_
