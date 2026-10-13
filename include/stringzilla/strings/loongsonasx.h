/**
 *  @file include/stringzilla/strings/loongsonasx.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief LoongArch LASX (256-bit) backend for collections of strings.
 *
 *  @sa include/stringzilla/strings.h
 */
#ifndef STRINGZILLA_STRINGS_LOONGSONASX_H_
#define STRINGZILLA_STRINGS_LOONGSONASX_H_

#include "stringzilla/types.h"
#include "stringzilla/find/loongsonasx.h" // `sz_find_byteset_loongsonasx_`
#include "stringzilla/utf8_norm.h"        // `sz_utf8_norm_loongsonasx`, `sz_utf8_find_denormalized_loongsonasx`
#include "stringzilla/strings/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_LOONGARCH64_
#if STRINGZILLA_TARGET_LOONGSONASX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("lasx"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("lasx")
#endif

/**
 *  @brief Broadcast a 16-byte slice of the 256-entry lookup table into both 128-bit lanes of a YMM.
 *
 *  @param[in] lut 256-byte lookup table.
 *  @param[in] offset Byte offset into @p lut at which to start the 16-byte slice.
 *  @return A 256-bit LASX register with the 16-byte slice duplicated into both 128-bit lanes.
 */
STRINGZILLA_INLINE __m256i sz_strings_lookup_slice_loongsonasx_(char const lut[sz_at_least_(256)], sz_size_t offset) {
    sz_u8_t lut_pairs[32];
    for (sz_size_t lane_index = 0; lane_index < 16; ++lane_index)
        lut_pairs[lane_index] = lut_pairs[lane_index + 16] = (sz_u8_t)lut[offset + lane_index];
    return __lasx_xvld(lut_pairs, 0);
}

STRINGZILLA_INLINE void sz_strings_lookup_bytes_loongsonasx_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                             char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);

    // The setup cost only pays off for larger inputs.
    if (length <= 128) {
        sz_strings_lookup_bytes_serial_(target, source, length, lut);
        return;
    }

    // Pull the 256-entry lookup table into 16 LASX registers, each holding a 16-byte slice
    // broadcast into both 128-bit lanes, so `__lasx_xvshuf_b` can index it with the same nibble in
    // either lane. Unlike AVX2's `_mm256_shuffle_epi8`, `__lasx_xvshuf_b(a, b, index)` picks
    // `b[index & 15]` within a lane when `index & 31 < 16`, matching pshufb for nibble indices.
    sz_u256_vec_t lut_0_to_15_vec, lut_16_to_31_vec, lut_32_to_47_vec, lut_48_to_63_vec, //
        lut_64_to_79_vec, lut_80_to_95_vec, lut_96_to_111_vec, lut_112_to_127_vec,       //
        lut_128_to_143_vec, lut_144_to_159_vec, lut_160_to_175_vec, lut_176_to_191_vec,  //
        lut_192_to_207_vec, lut_208_to_223_vec, lut_224_to_239_vec, lut_240_to_255_vec;

    lut_0_to_15_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 0);
    lut_16_to_31_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 16);
    lut_32_to_47_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 32);
    lut_48_to_63_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 48);
    lut_64_to_79_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 64);
    lut_80_to_95_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 80);
    lut_96_to_111_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 96);
    lut_112_to_127_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 112);
    lut_128_to_143_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 128);
    lut_144_to_159_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 144);
    lut_160_to_175_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 160);
    lut_176_to_191_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 176);
    lut_192_to_207_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 192);
    lut_208_to_223_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 208);
    lut_224_to_239_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 224);
    lut_240_to_255_vec.lasx = sz_strings_lookup_slice_loongsonasx_(lut, 240);

    sz_u256_vec_t source_vec, source_bot_vec;
    sz_u256_vec_t blended_0_to_31_vec, blended_32_to_63_vec, blended_64_to_95_vec, blended_96_to_127_vec,
        blended_128_to_159_vec, blended_160_to_191_vec, blended_192_to_223_vec, blended_224_to_255_vec;

    __m256i const zero_u8x32 = __lasx_xvreplgr2vr_b(0);
    __m256i const nibble_mask_u8x32 = __lasx_xvreplgr2vr_b(0x0F);

    // `__lasx_xvbitsel_v(a, b, c)` selects bit-by-bit: result = (a & ~c) | (b & c).
    // Full-byte selector masks, 0xFF picking `b` and 0x00 picking `a`, come from testing one
    // source bit with `__lasx_xvslt_bu(0, byte & mask)`, which is 0xFF wherever the bit is set.
    while (length >= 32) {
        source_vec.lasx = __lasx_xvld(source, 0);
        source_bot_vec.lasx = __lasx_xvand_v(source_vec.lasx, nibble_mask_u8x32);

        // Round 1: select within each 32-entry pair using bit 4 (0x10).
        __m256i bit4_set_u8x32 = __lasx_xvslt_bu(zero_u8x32,
                                                 __lasx_xvand_v(source_vec.lasx, __lasx_xvreplgr2vr_b(0x10)));
        blended_0_to_31_vec.lasx = __lasx_xvbitsel_v(                                //
            __lasx_xvshuf_b(zero_u8x32, lut_0_to_15_vec.lasx, source_bot_vec.lasx),  //
            __lasx_xvshuf_b(zero_u8x32, lut_16_to_31_vec.lasx, source_bot_vec.lasx), //
            bit4_set_u8x32);
        blended_32_to_63_vec.lasx = __lasx_xvbitsel_v(                               //
            __lasx_xvshuf_b(zero_u8x32, lut_32_to_47_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_u8x32, lut_48_to_63_vec.lasx, source_bot_vec.lasx), //
            bit4_set_u8x32);
        blended_64_to_95_vec.lasx = __lasx_xvbitsel_v(                               //
            __lasx_xvshuf_b(zero_u8x32, lut_64_to_79_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_u8x32, lut_80_to_95_vec.lasx, source_bot_vec.lasx), //
            bit4_set_u8x32);
        blended_96_to_127_vec.lasx = __lasx_xvbitsel_v(                                //
            __lasx_xvshuf_b(zero_u8x32, lut_96_to_111_vec.lasx, source_bot_vec.lasx),  //
            __lasx_xvshuf_b(zero_u8x32, lut_112_to_127_vec.lasx, source_bot_vec.lasx), //
            bit4_set_u8x32);
        blended_128_to_159_vec.lasx = __lasx_xvbitsel_v(                               //
            __lasx_xvshuf_b(zero_u8x32, lut_128_to_143_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_u8x32, lut_144_to_159_vec.lasx, source_bot_vec.lasx), //
            bit4_set_u8x32);
        blended_160_to_191_vec.lasx = __lasx_xvbitsel_v(                               //
            __lasx_xvshuf_b(zero_u8x32, lut_160_to_175_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_u8x32, lut_176_to_191_vec.lasx, source_bot_vec.lasx), //
            bit4_set_u8x32);
        blended_192_to_223_vec.lasx = __lasx_xvbitsel_v(                               //
            __lasx_xvshuf_b(zero_u8x32, lut_192_to_207_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_u8x32, lut_208_to_223_vec.lasx, source_bot_vec.lasx), //
            bit4_set_u8x32);
        blended_224_to_255_vec.lasx = __lasx_xvbitsel_v(                               //
            __lasx_xvshuf_b(zero_u8x32, lut_224_to_239_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_u8x32, lut_240_to_255_vec.lasx, source_bot_vec.lasx), //
            bit4_set_u8x32);

        // Round 2: select using bit 5 (0x20).
        __m256i bit5_set_u8x32 = __lasx_xvslt_bu(zero_u8x32,
                                                 __lasx_xvand_v(source_vec.lasx, __lasx_xvreplgr2vr_b(0x20)));
        blended_0_to_31_vec.lasx = __lasx_xvbitsel_v(blended_0_to_31_vec.lasx, blended_32_to_63_vec.lasx,
                                                     bit5_set_u8x32);
        blended_64_to_95_vec.lasx = __lasx_xvbitsel_v(blended_64_to_95_vec.lasx, blended_96_to_127_vec.lasx,
                                                      bit5_set_u8x32);
        blended_128_to_159_vec.lasx = __lasx_xvbitsel_v(blended_128_to_159_vec.lasx, blended_160_to_191_vec.lasx,
                                                        bit5_set_u8x32);
        blended_192_to_223_vec.lasx = __lasx_xvbitsel_v(blended_192_to_223_vec.lasx, blended_224_to_255_vec.lasx,
                                                        bit5_set_u8x32);

        // Round 3: select using bit 6 (0x40).
        __m256i bit6_set_u8x32 = __lasx_xvslt_bu(zero_u8x32,
                                                 __lasx_xvand_v(source_vec.lasx, __lasx_xvreplgr2vr_b(0x40)));
        blended_0_to_31_vec.lasx = __lasx_xvbitsel_v(blended_0_to_31_vec.lasx, blended_64_to_95_vec.lasx,
                                                     bit6_set_u8x32);
        blended_128_to_159_vec.lasx = __lasx_xvbitsel_v(blended_128_to_159_vec.lasx, blended_192_to_223_vec.lasx,
                                                        bit6_set_u8x32);

        // Round 4: select using bit 7 (0x80).
        __m256i bit7_set_u8x32 = __lasx_xvslt_bu(zero_u8x32,
                                                 __lasx_xvand_v(source_vec.lasx, __lasx_xvreplgr2vr_b((char)0x80)));
        blended_0_to_31_vec.lasx = __lasx_xvbitsel_v(blended_0_to_31_vec.lasx, blended_128_to_159_vec.lasx,
                                                     bit7_set_u8x32);

        __lasx_xvst(blended_0_to_31_vec.lasx, target, 0);
        source += 32, target += 32, length -= 32;
    }

    if (length) sz_strings_lookup_bytes_serial_(target, source, length, lut);
}

/** The LASX stages. */
STRINGZILLA_INLINE sz_strings_walks_t sz_strings_walks_loongsonasx_(void) {
    sz_strings_walks_t walks;
    walks.find_byteset = &sz_find_byteset_loongsonasx_;
    walks.lookup_bytes = &sz_strings_lookup_bytes_loongsonasx_;
    return walks;
}

STRINGZILLA_API sz_status_t sz_strings_to_tape_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                           sz_size_t count, sz_allocator_t *allocator,
                                                           sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_tape_with_(strings, order, count, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                           sz_size_t count, sz_strings_rows_t const *rows,
                                                           sz_size_t *longest, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_rows_with_(strings, order, count, rows, longest);
}

STRINGZILLA_API sz_status_t sz_strings_escape_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, sz_escape_kind_t kind,
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_loongsonasx_();
    return sz_strings_escape_with_(strings, order, count, kind, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                            sz_size_t count, sz_escape_kind_t kind,
                                                            sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                            sz_size_t *malformed_index, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_loongsonasx_();
    return sz_strings_unescape_with_(strings, order, count, kind, allocator, tape, malformed_index, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, char const lut[sz_at_least_(256)],
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_loongsonasx_();
    return sz_strings_lookup_with_(strings, order, count, lut, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_normalize_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                             sz_size_t count, sz_normal_form_t form,
                                                             sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                             sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_normalize_with_(strings, order, count, form, allocator, tape, &sz_utf8_norm_loongsonasx);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_LOONGSONASX
#endif // STRINGZILLA_ARCH_LOONGARCH64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_STRINGS_LOONGSONASX_H_
