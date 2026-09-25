/**
 *  @file include/stringzilla/utf8_norm/haswell.h
 *  @author Ash Vardanian
 *  @date June 15, 2026
 *  @brief Haswell AVX2 backend for the single-pass Unicode normalizer, NFD / NFC / NFKD / NFKC.
 *
 *  This is the AVX2 sibling of the AVX-512 `skylake.h` scanner, mirrored at 32-byte YMM
 *  granularity. It overrides exactly one point of the shared engine: the scan primitive
 *  @c sz_utf8_norm_classify_haswell_, a 32-byte all-ASCII gate via @c _mm256_movemask_epi8 plus a
 *  lead-byte classify over the shared @c sz_utf8_norm_lead_lut_, then the shared cold per-codepoint
 *  verify, @c sz_utf8_norm_verify_block_.
 *
 *  AVX2 has no unsigned byte compare and no @c vpermb, so the continuation test rides the
 *  @c min_epu8 range idiom, and the 64-entry lead lookup is a nibble-split of four broadcast
 *  16-byte @c vpshufb quadrants selected by @c vpblendvb: the emulation Skylake uses, narrowed to
 *  one YMM lane pair.
 *
 *  @sa include/stringzilla/utf8_norm.h
 */
#ifndef STRINGZILLA_UTF8_NORM_HASWELL_H_
#define STRINGZILLA_UTF8_NORM_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2")
#endif

/**
 *  @brief 64-entry lead lookup without AVX-512 VBMI: four per-128-lane @c vpshufb over the
 *      broadcast LUT quadrants, selected by the high two index bits, then `families & flag` picks
 *      out the requested form.
 *  @return The flagged lanes, nonzero where a lead byte begins a candidate non-inert codepoint.
 */
STRINGZILLA_HELPER_INLINE __m256i sz_utf8_norm_lead_classify_shuffle_haswell_(__m256i bytes_u8x32,
                                                                              __m256i is_lead_u8x32,
                                                                              sz_u8_t form_flag) {
    __m256i index_u8x32 = _mm256_and_si256(bytes_u8x32, _mm256_set1_epi8(0x3F));
    __m256i low_nibble_u8x32 = _mm256_and_si256(index_u8x32, _mm256_set1_epi8(0x0F));
    // `srli_epi16` leaks the neighbouring byte's low bits into bits 4..7; index is in [0,63] so the high
    // two bits live in bits 0..1, and masking with 0x03 recovers `index >> 4` per byte.
    __m256i quadrant_u8x32 = _mm256_and_si256(_mm256_srli_epi16(index_u8x32, 4), _mm256_set1_epi8(0x03));
    __m256i table0_u8x32 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 0)));
    __m256i table1_u8x32 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 16)));
    __m256i table2_u8x32 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 32)));
    __m256i table3_u8x32 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 48)));
    __m256i families_u8x32 = _mm256_shuffle_epi8(table0_u8x32, low_nibble_u8x32);
    families_u8x32 = _mm256_blendv_epi8(families_u8x32, _mm256_shuffle_epi8(table1_u8x32, low_nibble_u8x32),
                                        _mm256_cmpeq_epi8(quadrant_u8x32, _mm256_set1_epi8(1)));
    families_u8x32 = _mm256_blendv_epi8(families_u8x32, _mm256_shuffle_epi8(table2_u8x32, low_nibble_u8x32),
                                        _mm256_cmpeq_epi8(quadrant_u8x32, _mm256_set1_epi8(2)));
    families_u8x32 = _mm256_blendv_epi8(families_u8x32, _mm256_shuffle_epi8(table3_u8x32, low_nibble_u8x32),
                                        _mm256_cmpeq_epi8(quadrant_u8x32, _mm256_set1_epi8(3)));
    // `families & flag != 0` per lane: invert the `== 0` compare to get an all-ones mask where a family bit is set.
    __m256i has_flag_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(families_u8x32, _mm256_set1_epi8((char)form_flag)),
                                               _mm256_setzero_si256());
    return _mm256_andnot_si256(has_flag_u8x32, is_lead_u8x32);
}

/**
 *  @brief Haswell scan primitive: finds the first byte starting a non-inert codepoint for @p form.
 *
 *  Mirrors @c sz_utf8_norm_classify_skylake_ at 32-byte granularity: a 32-byte all-ASCII gate, a
 *  @c vpshufb lead-classify, then the shared scalar verify on any block that survives the gate. The
 *  verify carries the combining class across blocks and reports order and quick-check violations
 *  exactly as they occur.
 *
 *  @return The first such byte, or NULL.
 */
STRINGZILLA_HELPER_NOINLINE sz_cptr_t sz_utf8_norm_classify_haswell_(sz_cptr_t text, sz_size_t length,
                                                                     sz_normal_form_t form) {
    sz_u8_t const *position = (sz_u8_t const *)text;
    sz_u8_t const *const end = position + length;
    sz_u8_t const form_flag = sz_utf8_norm_form_flag_(form);
    sz_u8_t previous_canonical_combining_class = 0;

    while (position + 32 <= end) {
        __m256i bytes_u8x32 = _mm256_loadu_si256((__m256i const *)position);
        if (_mm256_movemask_epi8(bytes_u8x32) == 0) { // all 32 bytes ASCII: inert
            position += 32, previous_canonical_combining_class = 0;
            continue;
        }
        // Lead bytes only: non-ASCII (high bit set) and not a 10xxxxxx continuation. AVX2 lacks unsigned
        // byte compares, so the continuation test rides the `min_epu8` range idiom: (byte - 0x80) < 0x40.
        __m256i non_ascii_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(bytes_u8x32, _mm256_set1_epi8((char)0x80)),
                                                    _mm256_set1_epi8((char)0x80));
        __m256i offsets_u8x32 = _mm256_sub_epi8(bytes_u8x32, _mm256_set1_epi8((char)0x80));
        __m256i continuation_u8x32 = _mm256_cmpeq_epi8(_mm256_min_epu8(offsets_u8x32, _mm256_set1_epi8(0x3F)),
                                                       offsets_u8x32);
        __m256i is_lead_u8x32 = _mm256_andnot_si256(continuation_u8x32, non_ascii_u8x32);
        __m256i flagged_u8x32 = sz_utf8_norm_lead_classify_shuffle_haswell_(bytes_u8x32, is_lead_u8x32, form_flag);
        if (_mm256_movemask_epi8(flagged_u8x32) == 0) { // 32 bytes inert for the form: skip, realign onto a boundary
            position += 32, previous_canonical_combining_class = 0;
            while (position < end && (*position & 0xC0) == 0x80) ++position;
            continue;
        }
        sz_cptr_t violation = sz_utf8_norm_verify_block_(&position, position + 32, end, form_flag,
                                                         &previous_canonical_combining_class);
        if (violation) return violation;
    }
    // Tail (< 32 bytes): the shared scalar verify carries the combining class across the final boundary.
    return sz_utf8_norm_verify_block_(&position, end, end, form_flag, &previous_canonical_combining_class);
}

STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_norm_haswell(sz_cptr_t source, sz_size_t length, sz_normal_form_t form,
                                                        sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_haswell_);
}

STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_find_denormalized_haswell(sz_cptr_t source, sz_size_t length,
                                                                     sz_normal_form_t form) {
    return sz_utf8_find_denormalized_engine_(source, length, form, &sz_utf8_norm_classify_haswell_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_HASWELL_H_
