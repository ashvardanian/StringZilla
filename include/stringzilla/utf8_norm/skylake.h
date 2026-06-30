/**
 *  @brief Skylake-X (AVX-512 F/BW/VL baseline) backend for the single-pass Unicode normalizer.
 *  @file include/stringzilla/utf8_norm/skylake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  This is the AVX-512 baseline that Ice Lake reuses (never the reverse). It overrides exactly the
 *  scan primitive: a 64-byte all-ASCII gate (`_mm512_movepi8_mask`) plus a lead-byte classify over the
 *  shared `sz_utf8_norm_lead_lut_`, then the shared cold per-codepoint verify (`sz_utf8_norm_verify_block_`).
 *
 *  The scanner skeleton `sz_utf8_norm_classify_avx512_` is parameterized over the lead-classify step via a
 *  force-inlined function pointer (the same devirtualization idiom the engines use). Skylake instantiates
 *  it with the `vpshufb`-based 64-entry lookup, because Skylake lacks `vpermb` (AVX-512 VBMI); Ice Lake
 *  re-instantiates the same skeleton with a single `vpermb`. The scanner never compacts (it returns the
 *  first non-inert byte), so AVX-512 VBMI2 `vpcompressb` is irrelevant here.
 */
#ifndef STRINGZILLA_UTF8_NORM_SKYLAKE_H_
#define STRINGZILLA_UTF8_NORM_SKYLAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#endif

/** @brief Per-tier lead classifier: given a 64-byte vector, the lead mask, and the form flag, return the
 *         mask of lanes that begin a candidate-non-inert lead. The only step Ice Lake overrides. */
typedef __mmask64 (*sz_utf8_norm_lead_classify_avx512_t)(__m512i, __mmask64, sz_u8_t);

/**
 *  @brief 64-entry lead lookup without AVX-512 VBMI: four per-128-lane `vpshufb` over the broadcast LUT
 *         quadrants, selected by the index's high two bits. `families & flag` then identifies the form.
 */
SZ_HELPER_NOINLINE __mmask64 sz_utf8_norm_lead_classify_shuffle_skylake_(__m512i bytes, __mmask64 is_lead,
                                                                         sz_u8_t form_flag) {
    __m512i index = _mm512_and_si512(bytes, _mm512_set1_epi8(0x3F));
    __m512i low_nibble = _mm512_and_si512(index, _mm512_set1_epi8(0x0F));
    // `srli_epi16` leaks the neighbouring byte's low bits into bits 4..7; index is in [0,63] so the high
    // two bits live in bits 0..1, and masking with 0x03 recovers `index >> 4` per byte.
    __m512i quadrant = _mm512_and_si512(_mm512_srli_epi16(index, 4), _mm512_set1_epi8(0x03));
    __m512i table0 = _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 0)));
    __m512i table1 = _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 16)));
    __m512i table2 = _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 32)));
    __m512i table3 = _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i const *)(sz_utf8_norm_lead_lut_ + 48)));
    __m512i families = _mm512_shuffle_epi8(table0, low_nibble);
    families = _mm512_mask_mov_epi8(families, _mm512_cmpeq_epi8_mask(quadrant, _mm512_set1_epi8(1)),
                                    _mm512_shuffle_epi8(table1, low_nibble));
    families = _mm512_mask_mov_epi8(families, _mm512_cmpeq_epi8_mask(quadrant, _mm512_set1_epi8(2)),
                                    _mm512_shuffle_epi8(table2, low_nibble));
    families = _mm512_mask_mov_epi8(families, _mm512_cmpeq_epi8_mask(quadrant, _mm512_set1_epi8(3)),
                                    _mm512_shuffle_epi8(table3, low_nibble));
    __mmask64 has_flag = _mm512_test_epi8_mask(families, _mm512_set1_epi8((char)form_flag));
    return is_lead & has_flag;
}

/**
 *  @brief Shared AVX-512 scan skeleton: 64-byte all-ASCII gate, lead-classify via @p classify, then the
 *         shared scalar verify on any block that survives the gate. Ice Lake reuses this verbatim.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_norm_classify_avx512_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form,
                                                       sz_utf8_norm_lead_classify_avx512_t classify) {
    sz_u8_t const *position = (sz_u8_t const *)text;
    sz_u8_t const *const end = position + length;
    sz_u8_t const form_flag = sz_utf8_norm_form_flag_(form);
    sz_u8_t previous_canonical_combining_class = 0;

    while (position + 64 <= end) {
        __m512i bytes = _mm512_loadu_si512((void const *)position);
        __mmask64 non_ascii = _mm512_movepi8_mask(bytes);
        if (non_ascii == 0) { // all 64 bytes ASCII: inert
            position += 64, previous_canonical_combining_class = 0;
            continue;
        }
        __mmask64 continuation = _mm512_cmpeq_epi8_mask(_mm512_and_si512(bytes, _mm512_set1_epi8((char)0xC0)),
                                                        _mm512_set1_epi8((char)0x80));
        __mmask64 is_lead = non_ascii & ~continuation;
        __mmask64 flagged = classify(bytes, is_lead, form_flag);
        if (flagged == 0) { // 64 bytes inert for the form: skip, then realign onto a codepoint boundary
            position += 64, previous_canonical_combining_class = 0;
            while (position < end && (*position & 0xC0) == 0x80) ++position;
            continue;
        }
        sz_cptr_t violation = sz_utf8_norm_verify_block_(&position, position + 64, end, form_flag,
                                                         &previous_canonical_combining_class);
        if (violation) return violation;
    }
    // Tail (< 64 bytes): the shared scalar verify carries the combining class across the final boundary.
    return sz_utf8_norm_verify_block_(&position, end, end, form_flag, &previous_canonical_combining_class);
}

/** @brief Scan primitive (Skylake): first byte beginning a non-inert codepoint for @p form, else NULL. */
SZ_HELPER_NOINLINE sz_cptr_t sz_utf8_norm_classify_skylake_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_norm_classify_avx512_(text, length, form, &sz_utf8_norm_lead_classify_shuffle_skylake_);
}

SZ_API_COMPTIME sz_size_t sz_utf8_norm_skylake(sz_cptr_t source, sz_size_t length, sz_normal_form_t form,
                                               sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_skylake_);
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_find_denormalized_skylake(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_find_denormalized_engine_(source, length, form, &sz_utf8_norm_classify_skylake_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SKYLAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_SKYLAKE_H_
