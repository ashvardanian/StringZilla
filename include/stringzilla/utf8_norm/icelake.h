/**
 *  @brief Ice Lake (AVX-512 VBMI/VBMI2/GFNI) backend for the single-pass Unicode normalizer.
 *  @file include/stringzilla/utf8_norm/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  Ice Lake reuses the Skylake scan skeleton (`sz_utf8_norm_classify_avx512_`) and overrides only the
 *  lead classifier: a single `_mm512_permutexvar_epi8` (AVX-512 VBMI) reads the 64-byte
 *  `sz_utf8_norm_lead_lut_` verbatim, replacing Skylake's four `vpshufb` + select. The scanner returns the
 *  first non-inert byte (no compaction), so VBMI2 `vpcompressb` is unused; a GFNI affine classifier is a
 *  possible future micro-opt but a true 64-entry table is not affine, so `vpermb` is the right primitive.
 */
#ifndef STRINGZILLA_UTF8_NORM_ICELAKE_H_
#define STRINGZILLA_UTF8_NORM_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"
#include "stringzilla/utf8_norm/skylake.h" // reuses `sz_utf8_norm_classify_avx512_`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

/** @brief 64-entry lead lookup in one `vpermb` (AVX-512 VBMI), reading the shared LUT verbatim. */
SZ_INTERNAL __mmask64 sz_utf8_norm_lead_classify_vbmi_icelake_(__m512i bytes, __mmask64 is_lead, sz_u8_t form_flag) {
    __m512i index = _mm512_and_si512(bytes, _mm512_set1_epi8(0x3F));
    __m512i table = _mm512_loadu_si512((void const *)sz_utf8_norm_lead_lut_);
    __m512i families = _mm512_permutexvar_epi8(index, table);
    __mmask64 has_flag = _mm512_test_epi8_mask(families, _mm512_set1_epi8((char)form_flag));
    return is_lead & has_flag;
}

/** @brief Scan primitive (Ice Lake): the Skylake skeleton with the single-`vpermb` lead classifier. */
SZ_INTERNAL sz_cptr_t sz_utf8_norm_classify_icelake_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_norm_classify_avx512_(text, length, form, &sz_utf8_norm_lead_classify_vbmi_icelake_);
}

SZ_PUBLIC sz_size_t sz_utf8_norm_icelake(sz_cptr_t source, sz_size_t length, sz_normal_form_t form,
                                         sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_icelake_);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_denormalized_icelake(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_find_denormalized_engine_(source, length, form, &sz_utf8_norm_classify_icelake_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_ICELAKE_H_
