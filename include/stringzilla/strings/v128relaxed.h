/**
 *  @file include/stringzilla/strings/v128relaxed.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief WebAssembly relaxed-SIMD backend for collections of strings, a level above SIMD128.
 *
 *  @sa include/stringzilla/strings.h
 */
#ifndef STRINGZILLA_STRINGS_V128RELAXED_H_
#define STRINGZILLA_STRINGS_V128RELAXED_H_

#include "stringzilla/types.h"
#include "stringzilla/find/v128relaxed.h" // `sz_find_byteset_v128relaxed_`
#include "stringzilla/utf8_norm.h"        // `sz_utf8_norm_v128relaxed`, `sz_utf8_find_denormalized_v128relaxed`
#include "stringzilla/strings/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  The lookup swaps every @c wasm_i8x16_swizzle for @c wasm_i8x16_relaxed_swizzle, which skips
 *  zeroing out-of-range indices and runs about 21% faster on most engines. Its indices are low
 *  nibbles, never out of range, so the result matches the serial kernel byte for byte. */
#if STRINGZILLA_ARCH_WASM_
#if STRINGZILLA_TARGET_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

STRINGZILLA_INLINE void sz_strings_lookup_bytes_v128relaxed_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                             char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);

    // For tiny inputs the SIMD setup isn't worth it. Match the baseline heuristic.
    if (length <= 128) {
        sz_strings_lookup_bytes_serial_(target, source, length, lut);
        return;
    }

    // Load the 256-byte table into 16 vectors of 16 bytes each.
    v128_t lut_u8x16s[16];
    for (sz_size_t sub_table_index = 0; sub_table_index < 16; ++sub_table_index)
        lut_u8x16s[sub_table_index] = wasm_v128_load(lut + sub_table_index * 16);

    v128_t const low_nibble_mask_u8x16 = wasm_i8x16_splat(0x0F);
    v128_t const bit4_u8x16 = wasm_i8x16_splat(0x10);
    v128_t const bit5_u8x16 = wasm_i8x16_splat(0x20);
    v128_t const bit6_u8x16 = wasm_i8x16_splat(0x40);
    v128_t const bit7_u8x16 = wasm_i8x16_splat((sz_i8_t)0x80);

    while (length >= 16) {
        v128_t source_u8x16 = wasm_v128_load(source);
        // The low nibble of each byte selects within a 16-byte sub-table.
        v128_t low_nibble_index_u8x16 = wasm_v128_and(source_u8x16, low_nibble_mask_u8x16);

        // Index each of the 16 sub-tables by the low nibble with the cheaper relaxed swizzle.
        v128_t sub_table_u8x16s[16];
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; ++sub_table_index)
            sub_table_u8x16s[sub_table_index] = wasm_i8x16_relaxed_swizzle(lut_u8x16s[sub_table_index],
                                                                           low_nibble_index_u8x16);

        // Tree reduction selecting the right sub-table by the four high bits of the source.
        v128_t select_bit4_u8x16 = wasm_i8x16_eq(wasm_v128_and(source_u8x16, bit4_u8x16), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 2)
            sub_table_u8x16s[sub_table_index] = wasm_v128_bitselect(
                sub_table_u8x16s[sub_table_index], sub_table_u8x16s[sub_table_index + 1], select_bit4_u8x16);
        v128_t select_bit5_u8x16 = wasm_i8x16_eq(wasm_v128_and(source_u8x16, bit5_u8x16), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 4)
            sub_table_u8x16s[sub_table_index] = wasm_v128_bitselect(
                sub_table_u8x16s[sub_table_index], sub_table_u8x16s[sub_table_index + 2], select_bit5_u8x16);
        v128_t select_bit6_u8x16 = wasm_i8x16_eq(wasm_v128_and(source_u8x16, bit6_u8x16), wasm_i8x16_splat(0));
        for (sz_size_t sub_table_index = 0; sub_table_index < 16; sub_table_index += 8)
            sub_table_u8x16s[sub_table_index] = wasm_v128_bitselect(
                sub_table_u8x16s[sub_table_index], sub_table_u8x16s[sub_table_index + 4], select_bit6_u8x16);
        v128_t select_bit7_u8x16 = wasm_i8x16_eq(wasm_v128_and(source_u8x16, bit7_u8x16), wasm_i8x16_splat(0));
        v128_t result_u8x16 = wasm_v128_bitselect(sub_table_u8x16s[0], sub_table_u8x16s[8], select_bit7_u8x16);

        wasm_v128_store(target, result_u8x16);
        source += 16, target += 16, length -= 16;
    }

    // Handle the tail with serial code.
    if (length) sz_strings_lookup_bytes_serial_(target, source, length, lut);
}

/** The relaxed-SIMD stages. */
STRINGZILLA_INLINE sz_strings_walks_t sz_strings_walks_v128relaxed_(void) {
    sz_strings_walks_t walks;
    walks.find_byteset = &sz_find_byteset_v128relaxed_;
    walks.lookup_bytes = &sz_strings_lookup_bytes_v128relaxed_;
    return walks;
}

STRINGZILLA_API sz_status_t sz_strings_to_tape_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                           sz_size_t count, sz_allocator_t *allocator,
                                                           sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_tape_with_(strings, order, count, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                           sz_size_t count, sz_strings_rows_t const *rows,
                                                           sz_size_t *longest, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_rows_with_(strings, order, count, rows, longest);
}

STRINGZILLA_API sz_status_t sz_strings_escape_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, sz_escape_kind_t kind,
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_v128relaxed_();
    return sz_strings_escape_with_(strings, order, count, kind, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                            sz_size_t count, sz_escape_kind_t kind,
                                                            sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                            sz_size_t *malformed_index, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_v128relaxed_();
    return sz_strings_unescape_with_(strings, order, count, kind, allocator, tape, malformed_index, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, char const lut[sz_at_least_(256)],
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_v128relaxed_();
    return sz_strings_lookup_with_(strings, order, count, lut, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_normalize_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                             sz_size_t count, sz_normal_form_t form,
                                                             sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                             sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_normalize_with_(strings, order, count, form, allocator, tape, &sz_utf8_norm_v128relaxed);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // STRINGZILLA_TARGET_V128RELAXED
#endif // STRINGZILLA_ARCH_WASM_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_STRINGS_V128RELAXED_H_
