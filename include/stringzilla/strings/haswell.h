/**
 *  @file include/stringzilla/strings/haswell.h
 *  @author Ash Vardanian
 *  @date January 15, 2024
 *  @brief AVX2 backend for collections of strings on Haswell and newer x86 CPUs.
 *
 *  @sa include/stringzilla/strings.h
 */
#ifndef STRINGZILLA_STRINGS_HASWELL_H_
#define STRINGZILLA_STRINGS_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/find/haswell.h" // `sz_find_byteset_haswell_`
#include "stringzilla/utf8_norm.h"    // `sz_utf8_norm_haswell`, `sz_utf8_find_denormalized_haswell`
#include "stringzilla/strings/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_ARCH_X8664_HASWELL_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

STRINGZILLA_INLINE void sz_strings_lookup_bytes_haswell_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                         char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);

    // Loading the table into registers only pays off once at least 3 cache lines are mapped.
    if (length <= 128) {
        sz_strings_lookup_bytes_serial_(target, source, length, lut);
        return;
    }

    // AVX2 only shuffles within 128-bit lanes, so every 16-entry slice of the table is broadcast
    // into both lanes, and full YMM registers still halve the loop count over XMM.
    sz_u256_vec_t lut_0_to_15_vec, lut_16_to_31_vec, lut_32_to_47_vec, lut_48_to_63_vec, //
        lut_64_to_79_vec, lut_80_to_95_vec, lut_96_to_111_vec, lut_112_to_127_vec,       //
        lut_128_to_143_vec, lut_144_to_159_vec, lut_160_to_175_vec, lut_176_to_191_vec,  //
        lut_192_to_207_vec, lut_208_to_223_vec, lut_224_to_239_vec, lut_240_to_255_vec;

    lut_0_to_15_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut)));
    lut_16_to_31_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 16)));
    lut_32_to_47_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 32)));
    lut_48_to_63_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 48)));
    lut_64_to_79_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 64)));
    lut_80_to_95_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 80)));
    lut_96_to_111_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 96)));
    lut_112_to_127_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 112)));
    lut_128_to_143_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 128)));
    lut_144_to_159_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 144)));
    lut_160_to_175_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 160)));
    lut_176_to_191_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 176)));
    lut_192_to_207_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 192)));
    lut_208_to_223_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 208)));
    lut_224_to_239_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 224)));
    lut_240_to_255_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 240)));

    // Each shuffle reads 16 of the 256 entries, so four high bits pick among the slices.
    sz_u256_vec_t not_first_bit_vec, not_second_bit_vec, not_third_bit_vec, not_fourth_bit_vec;

    // Top and bottom nibbles of the source are used separately.
    sz_u256_vec_t source_vec, source_low_nibble_vec;
    sz_u256_vec_t blended_0_to_31_vec, blended_32_to_63_vec, blended_64_to_95_vec, blended_96_to_127_vec,
        blended_128_to_159_vec, blended_160_to_191_vec, blended_192_to_223_vec, blended_224_to_255_vec;

    // Handling the head.
    while (length >= 32) {
        // Load and separate the nibbles of each byte in the source.
        source_vec.ymm = _mm256_lddqu_si256((__m256i const *)source);
        source_low_nibble_vec.ymm = _mm256_and_si256(source_vec.ymm, _mm256_set1_epi8((char)0x0F));

        // In the first round, we select using the 4th bit.
        not_fourth_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x10), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8(                             //
            _mm256_shuffle_epi8(lut_16_to_31_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_0_to_15_vec.ymm, source_low_nibble_vec.ymm),  //
            not_fourth_bit_vec.ymm);
        blended_32_to_63_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_48_to_63_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_32_to_47_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_64_to_95_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_80_to_95_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_64_to_79_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_96_to_127_vec.ymm = _mm256_blendv_epi8(                             //
            _mm256_shuffle_epi8(lut_112_to_127_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_96_to_111_vec.ymm, source_low_nibble_vec.ymm),  //
            not_fourth_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_144_to_159_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_128_to_143_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_160_to_191_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_176_to_191_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_160_to_175_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_192_to_223_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_208_to_223_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_192_to_207_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_224_to_255_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_240_to_255_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_224_to_239_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);

        // A tree reduction of the 8 blended YMM registers, keyed on the source bits.
        // The first round selects using the 3rd bit.
        not_third_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x20), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( //
            blended_32_to_63_vec.ymm,                 //
            blended_0_to_31_vec.ymm,                  //
            not_third_bit_vec.ymm);
        blended_64_to_95_vec.ymm = _mm256_blendv_epi8( //
            blended_96_to_127_vec.ymm,                 //
            blended_64_to_95_vec.ymm,                  //
            not_third_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8( //
            blended_160_to_191_vec.ymm,                  //
            blended_128_to_159_vec.ymm,                  //
            not_third_bit_vec.ymm);
        blended_192_to_223_vec.ymm = _mm256_blendv_epi8( //
            blended_224_to_255_vec.ymm,                  //
            blended_192_to_223_vec.ymm,                  //
            not_third_bit_vec.ymm);

        // The second round selects using the 2nd bit.
        not_second_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x40), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( //
            blended_64_to_95_vec.ymm,                 //
            blended_0_to_31_vec.ymm,                  //
            not_second_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8( //
            blended_192_to_223_vec.ymm,                  //
            blended_128_to_159_vec.ymm,                  //
            not_second_bit_vec.ymm);

        // The third round selects using the 1st bit.
        not_first_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x80), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( //
            blended_128_to_159_vec.ymm,               //
            blended_0_to_31_vec.ymm,                  //
            not_first_bit_vec.ymm);

        // And dump the result into the target.
        _mm256_storeu_si256((__m256i *)target, blended_0_to_31_vec.ymm);
        source += 32, target += 32, length -= 32;
    }

    // Handle the tail.
    if (length) sz_strings_lookup_bytes_serial_(target, source, length, lut);
}

/** The Haswell stages. */
STRINGZILLA_INLINE sz_strings_walks_t sz_strings_walks_haswell_(void) {
    sz_strings_walks_t walks;
    walks.find_byteset = &sz_find_byteset_haswell_;
    walks.lookup_bytes = &sz_strings_lookup_bytes_haswell_;
    return walks;
}

#if STRINGZILLA_TARGET_HASWELL

STRINGZILLA_API sz_status_t sz_strings_to_tape_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_allocator_t *allocator,
                                                       sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_tape_with_(strings, order, count, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_strings_rows_t const *rows,
                                                       sz_size_t *longest, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_rows_with_(strings, order, count, rows, longest);
}

STRINGZILLA_API sz_status_t sz_strings_escape_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_haswell_();
    return sz_strings_escape_with_(strings, order, count, kind, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_escape_kind_t kind,
                                                        sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                        sz_size_t *malformed_index, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_haswell_();
    return sz_strings_unescape_with_(strings, order, count, kind, allocator, tape, malformed_index, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, char const lut[sz_at_least_(256)],
                                                      sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                      sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_haswell_();
    return sz_strings_lookup_with_(strings, order, count, lut, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_normalize_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                         sz_size_t count, sz_normal_form_t form,
                                                         sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                         sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_normalize_with_(strings, order, count, form, allocator, tape, &sz_utf8_norm_haswell);
}

#endif // STRINGZILLA_TARGET_HASWELL

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_X8664_HASWELL_
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_STRINGS_HASWELL_H_
