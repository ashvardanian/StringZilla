/**
 *  @brief Haswell (AVX2) backend for the single-pass Unicode normalizer (NFD / NFC / NFKD / NFKC).
 *  @file include/stringzilla/utf8_norm/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  This is the AVX2 sibling of the AVX-512 `skylake.h` scanner, mirrored at 32-byte YMM granularity.
 *  It overrides exactly one point of the shared engine: the scan primitive `sz_utf8_norm_classify_haswell_`,
 *  a 32-byte all-ASCII gate (`_mm256_movemask_epi8`) plus a lead-byte classify over the shared
 *  `sz_utf8_norm_lead_lut_`, then the shared cold per-codepoint verify (`sz_utf8_norm_verify_block_`).
 *
 *  AVX2 has no unsigned byte compare and no `vpermb`, so the continuation test rides the `min_epu8`
 *  range idiom and the 64-entry lead lookup is a nibble-split of four broadcast 16-byte `vpshufb`
 *  quadrants selected by `vpblendvb` - exactly the emulation Skylake uses, narrowed to one YMM lane pair.
 */
#ifndef STRINGZILLA_UTF8_NORM_HASWELL_H_
#define STRINGZILLA_UTF8_NORM_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2")
#endif

/**
 *  @brief 64-entry lead lookup without AVX-512 VBMI: four per-128-lane `vpshufb` over the broadcast LUT
 *         quadrants, selected by the index's high two bits. `families & flag` then identifies the form.
 *  @return The vector of flagged lanes (nonzero where a lead byte begins a candidate-non-inert codepoint).
 */
SZ_INTERNAL __m256i sz_utf8_norm_lead_classify_shuffle_haswell_(__m256i bytes, __m256i is_lead, sz_u8_t form_flag) {
    __m256i index = _mm256_and_si256(bytes, _mm256_set1_epi8(0x3F));
    __m256i low_nibble = _mm256_and_si256(index, _mm256_set1_epi8(0x0F));
    // `srli_epi16` leaks the neighbouring byte's low bits into bits 4..7; index is in [0,63] so the high
    // two bits live in bits 0..1, and masking with 0x03 recovers `index >> 4` per byte.
    __m256i quadrant = _mm256_and_si256(_mm256_srli_epi16(index, 4), _mm256_set1_epi8(0x03));
    __m256i table0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 0)));
    __m256i table1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 16)));
    __m256i table2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 32)));
    __m256i table3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 48)));
    __m256i families = _mm256_shuffle_epi8(table0, low_nibble);
    families = _mm256_blendv_epi8(families, _mm256_shuffle_epi8(table1, low_nibble),
                                  _mm256_cmpeq_epi8(quadrant, _mm256_set1_epi8(1)));
    families = _mm256_blendv_epi8(families, _mm256_shuffle_epi8(table2, low_nibble),
                                  _mm256_cmpeq_epi8(quadrant, _mm256_set1_epi8(2)));
    families = _mm256_blendv_epi8(families, _mm256_shuffle_epi8(table3, low_nibble),
                                  _mm256_cmpeq_epi8(quadrant, _mm256_set1_epi8(3)));
    // `families & flag != 0` per lane: invert the `== 0` compare to get an all-ones mask where a family bit is set.
    __m256i has_flag = _mm256_cmpeq_epi8(_mm256_and_si256(families, _mm256_set1_epi8((char)form_flag)),
                                         _mm256_setzero_si256());
    return _mm256_andnot_si256(has_flag, is_lead);
}

/**
 *  @brief Scan primitive (Haswell): first byte that begins a non-inert codepoint for @p form, else NULL.
 *
 *  Mirrors `sz_utf8_norm_classify_skylake_` at 32-byte granularity: a 32-byte all-ASCII gate, a `vpshufb`
 *  lead-classify, then the shared scalar verify on any block that survives the gate. The verify carries
 *  the combining class across blocks and reports order / quick-check violations exactly.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_norm_classify_haswell_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    sz_u8_t const *position = (sz_u8_t const *)text;
    sz_u8_t const *const end = position + length;
    sz_u8_t const form_flag = sz_utf8_norm_form_flag_(form);
    sz_u8_t previous_canonical_combining_class = 0;

    while (position + 32 <= end) {
        __m256i bytes = _mm256_loadu_si256((__m256i const *)position);
        if (_mm256_movemask_epi8(bytes) == 0) { // all 32 bytes ASCII: inert
            position += 32, previous_canonical_combining_class = 0;
            continue;
        }
        // Lead bytes only: non-ASCII (high bit set) and not a 10xxxxxx continuation. AVX2 lacks unsigned
        // byte compares, so the continuation test rides the `min_epu8` range idiom: (byte - 0x80) < 0x40.
        __m256i non_ascii = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, _mm256_set1_epi8((char)0x80)),
                                              _mm256_set1_epi8((char)0x80));
        __m256i offsets = _mm256_sub_epi8(bytes, _mm256_set1_epi8((char)0x80));
        __m256i continuation = _mm256_cmpeq_epi8(_mm256_min_epu8(offsets, _mm256_set1_epi8(0x3F)), offsets);
        __m256i is_lead = _mm256_andnot_si256(continuation, non_ascii);
        __m256i flagged = sz_utf8_norm_lead_classify_shuffle_haswell_(bytes, is_lead, form_flag);
        if (_mm256_movemask_epi8(flagged) == 0) { // 32 bytes inert for the form: skip, then realign onto a boundary
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

SZ_PUBLIC sz_size_t sz_utf8_norm_haswell(sz_cptr_t source, sz_size_t length, sz_normal_form_t form,
                                         sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_haswell_);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_denormalized_haswell(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_find_denormalized_engine_(source, length, form, &sz_utf8_norm_classify_haswell_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_HASWELL_H_
