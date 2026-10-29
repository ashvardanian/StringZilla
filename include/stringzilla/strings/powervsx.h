/**
 *  @file include/stringzilla/strings/powervsx.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief IBM Power VSX backend for collections of strings.
 *
 *  @sa c/stringzilla/strings.h
 */
#ifndef STRINGZILLA_STRINGS_POWERVSX_H_
#define STRINGZILLA_STRINGS_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/find/powervsx.h" // `sz_find_byteset_powervsx_`
#include "stringzilla/utf8_norm.h"     // `sz_utf8_norm_powervsx`, `sz_utf8_find_denormalized_powervsx`
#include "stringzilla/strings/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_PPC64_
#if STRINGZILLA_TARGET_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

STRINGZILLA_INLINE void sz_strings_lookup_bytes_powervsx_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                          char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);
    // Small inputs aren't worth the SIMD setup cost — defer to the serial path.
    if (length <= 128) {
        sz_strings_lookup_bytes_serial_(target, source, length, lut);
        return;
    }

    // `vec_perm` indexes 16-byte chunks by the low nibble, and a 4-level `vec_sel` tree on the high
    // nibble picks the chunk, as the v128relaxed backend does, instead of a 16-way compare chain.
    __vector unsigned char lut_u8x16[16];
    for (sz_size_t sub_table_index = 0; sub_table_index < 16; ++sub_table_index)
        lut_u8x16[sub_table_index] = vec_xl(0, (unsigned char const *)(lut + sub_table_index * 16));

    __vector unsigned char const low_nibble_mask_u8x16 = vec_splats((unsigned char)0x0F);
    __vector unsigned char const bit4_mask_u8x16 = vec_splats((unsigned char)0x10);
    __vector unsigned char const bit5_mask_u8x16 = vec_splats((unsigned char)0x20);
    __vector unsigned char const bit6_mask_u8x16 = vec_splats((unsigned char)0x40);
    __vector unsigned char const bit7_mask_u8x16 = vec_splats((unsigned char)0x80);
    sz_size_t byte_index = 0;
    for (; byte_index + 16 <= length; byte_index += 16) {
        __vector unsigned char source_u8x16 = vec_xl(0, (unsigned char const *)(source + byte_index));
        __vector unsigned char low_nibble_index_u8x16 = vec_and(source_u8x16, low_nibble_mask_u8x16);

        // Permute all 16 sub-tables by the low nibble of each source byte.
        __vector unsigned char permuted_u8x16[16];
        for (sz_size_t chunk_index = 0; chunk_index < 16; ++chunk_index)
            permuted_u8x16[chunk_index] = vec_perm(lut_u8x16[chunk_index], lut_u8x16[chunk_index],
                                                   low_nibble_index_u8x16);

        // `vec_sel(a, b, sel)` keeps `b` where the bit under test is set, halving the candidates.
        __vector unsigned char select_bit4_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(source_u8x16, bit4_mask_u8x16), bit4_mask_u8x16);
        for (sz_size_t chunk_index = 0; chunk_index < 16; chunk_index += 2)
            permuted_u8x16[chunk_index] = vec_sel(permuted_u8x16[chunk_index], permuted_u8x16[chunk_index + 1],
                                                  select_bit4_u8x16);
        __vector unsigned char select_bit5_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(source_u8x16, bit5_mask_u8x16), bit5_mask_u8x16);
        for (sz_size_t chunk_index = 0; chunk_index < 16; chunk_index += 4)
            permuted_u8x16[chunk_index] = vec_sel(permuted_u8x16[chunk_index], permuted_u8x16[chunk_index + 2],
                                                  select_bit5_u8x16);
        __vector unsigned char select_bit6_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(source_u8x16, bit6_mask_u8x16), bit6_mask_u8x16);
        for (sz_size_t chunk_index = 0; chunk_index < 16; chunk_index += 8)
            permuted_u8x16[chunk_index] = vec_sel(permuted_u8x16[chunk_index], permuted_u8x16[chunk_index + 4],
                                                  select_bit6_u8x16);
        __vector unsigned char select_bit7_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(source_u8x16, bit7_mask_u8x16), bit7_mask_u8x16);
        __vector unsigned char result_u8x16 = vec_sel(permuted_u8x16[0], permuted_u8x16[8], select_bit7_u8x16);
        vec_xst(result_u8x16, 0, (unsigned char *)(target + byte_index));
    }

    // Handle the remaining tail serially.
    if (byte_index < length)
        sz_strings_lookup_bytes_serial_(target + byte_index, source + byte_index, length - byte_index, lut);
}

/** The POWER VSX stages. */
STRINGZILLA_INLINE sz_strings_walks_t sz_strings_walks_powervsx_(void) {
    sz_strings_walks_t walks;
    walks.find_byteset = &sz_find_byteset_powervsx_;
    walks.lookup_bytes = &sz_strings_lookup_bytes_powervsx_;
    return walks;
}

STRINGZILLA_API sz_status_t sz_strings_to_tape_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_allocator_t *allocator,
                                                        sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_tape_with_(strings, order, count, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_strings_rows_t const *rows,
                                                        sz_size_t *longest, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_rows_with_(strings, order, count, rows, longest);
}

STRINGZILLA_API sz_status_t sz_strings_escape_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_escape_kind_t kind,
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_powervsx_();
    return sz_strings_escape_with_(strings, order, count, kind, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                         sz_size_t count, sz_escape_kind_t kind,
                                                         sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                         sz_size_t *malformed_index, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_powervsx_();
    return sz_strings_unescape_with_(strings, order, count, kind, allocator, tape, malformed_index, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, char const lut[sz_at_least_(256)],
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_powervsx_();
    return sz_strings_lookup_with_(strings, order, count, lut, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_normalize_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, sz_normal_form_t form,
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_normalize_with_(strings, order, count, form, allocator, tape, &sz_utf8_norm_powervsx);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_POWERVSX
#endif // STRINGZILLA_ARCH_PPC64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_STRINGS_POWERVSX_H_
