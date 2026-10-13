/**
 *  @file include/stringzilla/strings/icelake.h
 *  @author Ash Vardanian
 *  @date October 11, 2024
 *  @brief AVX-512 VBMI backend for collections of strings on Ice Lake and newer x86 CPUs.
 *
 *  @sa include/stringzilla/strings.h
 */
#ifndef STRINGZILLA_STRINGS_ICELAKE_H_
#define STRINGZILLA_STRINGS_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/find/icelake.h" // `sz_find_byteset_icelake_`
#include "stringzilla/utf8_norm.h"    // `sz_utf8_norm_icelake`, `sz_utf8_find_denormalized_icelake`
#include "stringzilla/strings/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_TARGET_ICELAKE
#if defined(__clang__) && STRINGZILLA_HAS_CLANG_EVEX512_
#pragma clang attribute push(                                                                      \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "bmi", "bmi2")
#endif

STRINGZILLA_INLINE void sz_strings_lookup_bytes_icelake_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                         char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);

    // Loading the table into registers only pays off once at least 3 cache lines are mapped.
    if (length <= 128) {
        sz_u8_t const *lut_u8 = (sz_u8_t const *)lut;
        sz_u8_t const *source_u8 = (sz_u8_t const *)source;
        sz_u8_t *target_u8 = (sz_u8_t *)target;
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
            target_u8[byte_index] = lut_u8[source_u8[byte_index]];
        return;
    }

    // Masked stores cover the unaligned head and tail, so every body store is cache-line aligned.
    sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
    sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
    __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
    __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);

    // We use VPERMI2B (`_mm512_permutex2var_epi8`) to perform 256-entry lookups efficiently.
    // VPERMI2B uses bit 6 of each index to select between two 64-byte tables, allowing us to
    // cover 128 entries per instruction (2 instructions for all 256 entries).
    //
    // For the high-bit (bit 7) selection, we use VPMOVB2M (`_mm512_movepi8_mask`) which extracts
    // the sign bit of each byte directly to a mask register. This goes to port 0 on Intel,
    // avoiding the port 5 bottleneck that VPTESTMB would cause.
    sz_u512_vec_t lut_0_to_63_vec, lut_64_to_127_vec, lut_128_to_191_vec, lut_192_to_255_vec;
    lut_0_to_63_vec.zmm = _mm512_loadu_si512((lut));
    lut_64_to_127_vec.zmm = _mm512_loadu_si512((lut + 64));
    lut_128_to_191_vec.zmm = _mm512_loadu_si512((lut + 128));
    lut_192_to_255_vec.zmm = _mm512_loadu_si512((lut + 192));

    __mmask64 high_bit_mask_m64;
    sz_u512_vec_t source_vec, low_half_vec, high_half_vec, result_vec;

    // Handling the head.
    if (head_length) {
        source_vec.zmm = _mm512_maskz_loadu_epi8(head_mask_m64, source);
        // VPERMI2B: bit 6 selects between the two tables, bits 0-5 index within each
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        // VPMOVB2M moves bit 7 of each byte into a mask on port 0, sparing the busy port 5
        high_bit_mask_m64 = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask_m64, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_mask_storeu_epi8(target, head_mask_m64, result_vec.zmm);
        source += head_length, target += head_length, length -= head_length;
    }

    // The body in 64-byte chunks, aligned to cache lines of `target`.
    while (length >= 64) {
        source_vec.zmm = _mm512_loadu_si512(source);
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        high_bit_mask_m64 = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask_m64, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_store_si512(target, result_vec.zmm); //! Aligned store, our main weapon!
        source += 64, target += 64, length -= 64;
    }

    // Handling the tail.
    if (tail_length) {
        source_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask_m64, source);
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        high_bit_mask_m64 = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask_m64, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_mask_storeu_epi8(target, tail_mask_m64, result_vec.zmm);
        source += tail_length, target += tail_length, length -= tail_length;
    }
}

/** The Ice Lake stages. */
STRINGZILLA_INLINE sz_strings_walks_t sz_strings_walks_icelake_(void) {
    sz_strings_walks_t walks;
    walks.find_byteset = &sz_find_byteset_icelake_;
    walks.lookup_bytes = &sz_strings_lookup_bytes_icelake_;
    return walks;
}

STRINGZILLA_API sz_status_t sz_strings_to_tape_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_allocator_t *allocator,
                                                       sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_tape_with_(strings, order, count, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_strings_rows_t const *rows,
                                                       sz_size_t *longest, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_rows_with_(strings, order, count, rows, longest);
}

STRINGZILLA_API sz_status_t sz_strings_escape_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_icelake_();
    return sz_strings_escape_with_(strings, order, count, kind, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_escape_kind_t kind,
                                                        sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                        sz_size_t *malformed_index, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_icelake_();
    return sz_strings_unescape_with_(strings, order, count, kind, allocator, tape, malformed_index, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, char const lut[sz_at_least_(256)],
                                                      sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                      sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_icelake_();
    return sz_strings_lookup_with_(strings, order, count, lut, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_normalize_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                         sz_size_t count, sz_normal_form_t form,
                                                         sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                         sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_normalize_with_(strings, order, count, form, allocator, tape, &sz_utf8_norm_icelake,
                                      &sz_utf8_find_denormalized_icelake);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_ICELAKE
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_STRINGS_ICELAKE_H_
