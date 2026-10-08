/**
 *  @file include/stringzilla/utf8_graphemes/neon.h
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief NEON UTF-8 grapheme boundaries.
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_NEON_H_
#define STRINGZILLA_UTF8_GRAPHEMES_NEON_H_
#include "stringzilla/types.h"
#include "stringzilla/utf8_graphemes/serial.h"
#include "stringzilla/utf8_runes/neon.h"
#ifdef __cplusplus
extern "C" {
#endif
#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_NEON_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

STRINGZILLA_INLINE void sz_grapheme_descriptors_neon_(sz_cptr_t text, sz_size_t length, sz_size_t base,
                                                      sz_u8_t *descriptors) {
    sz_u8_t tail[18];
    sz_u8_t const *bytes = (sz_u8_t const *)text + base;
    sz_size_t const remaining = length - base;
    if (remaining < sizeof(tail)) {
        for (sz_size_t i = 0; i != remaining; ++i) tail[i] = bytes[i];
        for (sz_size_t i = remaining; i != sizeof(tail); ++i) tail[i] = 0;
        bytes = tail;
    }
    uint8x16_t const lead_u8x16 = vld1q_u8(bytes);
    uint8x16_t const next1_u8x16 = vld1q_u8(bytes + 1), next2_u8x16 = vld1q_u8(bytes + 2);
    uint8x16_t const ascii_u8x16 = vcltq_u8(lead_u8x16, vdupq_n_u8(0x80));
    uint8x16_t descriptors_u8x16 = sz_utf8_rune_lut256_neon_(sz_utf8_grapheme_break_ascii_desc_lut_, lead_u8x16);
    if (vminvq_u8(ascii_u8x16) != 255) {
        uint8x16_t const three_u8x16 = vceqq_u8(vandq_u8(lead_u8x16, vdupq_n_u8(0xF0)), vdupq_n_u8(0xE0));
        uint8x16_t const two_high_u8x16 = vandq_u8(vshrq_n_u8(lead_u8x16, 2), vdupq_n_u8(7));
        uint8x16_t const two_low_u8x16 = vorrq_u8(vshlq_n_u8(lead_u8x16, 6), vandq_u8(next1_u8x16, vdupq_n_u8(0x3F)));
        uint8x16_t const three_high_u8x16 = vorrq_u8(vshlq_n_u8(vandq_u8(lead_u8x16, vdupq_n_u8(0xF)), 4),
                                                     vandq_u8(vshrq_n_u8(next1_u8x16, 2), vdupq_n_u8(0xF)));
        uint8x16_t const three_low_u8x16 = vorrq_u8(vshlq_n_u8(next1_u8x16, 6),
                                                    vandq_u8(next2_u8x16, vdupq_n_u8(0x3F)));
        uint8x16_t const high_u8x16 = vbslq_u8(ascii_u8x16, vdupq_n_u8(0),
                                               vbslq_u8(three_u8x16, three_high_u8x16, two_high_u8x16));
        uint8x16_t const low_u8x16 = vbslq_u8(ascii_u8x16, lead_u8x16,
                                              vbslq_u8(three_u8x16, three_low_u8x16, two_low_u8x16));
        descriptors_u8x16 = sz_utf8_rune_flat_lookup_neon_(sz_utf8_grapheme_break_bmp_page_lut_,
                                                           sz_utf8_grapheme_break_flat_bmp_,
                                                           sz_utf8_grapheme_break_flat_pages_k, high_u8x16, low_u8x16);
    }
    vst1q_u8(descriptors, descriptors_u8x16);
    for (sz_size_t i = 0; i != sz_min_of_two((sz_size_t)16, length - base); ++i)
        if (bytes[i] >= 0xF0) descriptors[i] = sz_grapheme_break_property_at_(text, length, base + i);
}

STRINGZILLA_INLINE sz_size_t sz_utf8_graphemes_neon_(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_lengths,
                                                     sz_size_t clusters_capacity) {
    if (length < 16) return sz_utf8_graphemes_serial_(text, length, cluster_lengths, clusters_capacity);
    if (!clusters_capacity) return 0;
    sz_u8_t descriptors[16];
    sz_grapheme_descriptors_neon_(text, length, 0, descriptors);
    sz_size_t base = 0, clusters = 0, cluster_start = 0;
    sz_grapheme_state_serial_t state = {0, sz_false_k, sz_false_k, sz_false_k, sz_false_k, sz_false_k};
    sz_grapheme_advance_serial_(&state, descriptors[0]);
    for (sz_size_t position = sz_grapheme_break_next_start_(text, length, 0); position < length;
         position = sz_grapheme_break_next_start_(text, length, position)) {
        if (position >= base + 16) {
            base = position;
            sz_grapheme_descriptors_neon_(text, length, base, descriptors);
        }
        sz_u8_t const after = descriptors[position - base];
        sz_size_t const before_start = sz_utf8_previous_rune_start_(text, position);
        sz_bool_t const boundary = (((sz_u8_t)text[before_start] & 0xC0u) == 0x80u)
                                       ? sz_true_k
                                       : sz_grapheme_boundary_serial_(&state, after);
        if (boundary) {
            cluster_lengths[clusters++] = position - cluster_start;
            if (clusters == clusters_capacity) return clusters;
            cluster_start = position;
        }
        sz_grapheme_advance_serial_(&state, after);
    }
    cluster_lengths[clusters] = length - cluster_start;
    return clusters + 1;
}

#if STRINGZILLA_TARGET_NEON
STRINGZILLA_API sz_status_t sz_utf8_graphemes_neon(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                   sz_size_t capacity, sz_size_t *count, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *count = sz_utf8_graphemes_neon_(text, length, lengths, capacity);
    sz_assert_(sz_utf8_segments_consistent_(length, capacity, *count, lengths));
    return sz_success_k;
}
#endif // STRINGZILLA_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_NEON_
#endif // STRINGZILLA_ARCH_ARM64_
#ifdef __cplusplus
}
#endif
#endif
