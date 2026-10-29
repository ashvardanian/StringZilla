/**
 *  @file include/stringzilla/strings/sve.h
 *  @author Ash Vardanian
 *  @date October 3, 2024
 *  @brief SVE backend for collections of strings on Arm v9 CPUs.
 *
 *  @sa c/stringzilla/strings.h
 */
#ifndef STRINGZILLA_STRINGS_SVE_H_
#define STRINGZILLA_STRINGS_SVE_H_

#include "stringzilla/types.h"
#include "stringzilla/find/neon.h" // `sz_find_byteset_neon_`
#include "stringzilla/utf8_norm.h" // `sz_utf8_norm_sve`, `sz_utf8_find_denormalized_sve`
#include "stringzilla/strings/serial.h"
#include "stringzilla/strings/neon.h" // `sz_strings_lookup_bytes_neon_`

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_SVE_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

STRINGZILLA_INLINE void sz_strings_lookup_bytes_sve_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                     char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);

    if (length <= 128) {
        sz_strings_lookup_bytes_serial_(target, source, length, lut);
        return;
    }

    sz_size_t const vector_length = svcntb();

    // Sixteen 16-byte rows, picked by the high nibble and indexed by the low one, keep every
    // `svtbl_u8` index below 16, so the lookup is correct at every vector length.
    svbool_t const row_predicate_b8x = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)16);
    svuint8_t const row_0_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x00));
    svuint8_t const row_1_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x10));
    svuint8_t const row_2_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x20));
    svuint8_t const row_3_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x30));
    svuint8_t const row_4_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x40));
    svuint8_t const row_5_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x50));
    svuint8_t const row_6_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x60));
    svuint8_t const row_7_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x70));
    svuint8_t const row_8_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x80));
    svuint8_t const row_9_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0x90));
    svuint8_t const row_10_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xA0));
    svuint8_t const row_11_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xB0));
    svuint8_t const row_12_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xC0));
    svuint8_t const row_13_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xD0));
    svuint8_t const row_14_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xE0));
    svuint8_t const row_15_u8x = svld1_u8(row_predicate_b8x, (sz_u8_t const *)(lut + 0xF0));

    // `svwhilelt` predicates the final partial vector, so the tail folds into the loop.
    for (sz_size_t byte_index = 0; byte_index < length; byte_index += vector_length) {
        svbool_t const active_b8x = svwhilelt_b8_u64(byte_index, length);
        svuint8_t const source_u8x = svld1_u8(active_b8x, (sz_u8_t const *)(source + byte_index));
        svuint8_t const column_u8x = svand_n_u8_x(active_b8x, source_u8x, 0x0F);
        svuint8_t const high_nibble_u8x = svlsr_n_u8_x(active_b8x, source_u8x, 4);

        // A flat 16-way select on the high nibble, starting from the row of nibble 0.
        svuint8_t result_u8x = svtbl_u8(row_0_u8x, column_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 1), svtbl_u8(row_1_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 2), svtbl_u8(row_2_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 3), svtbl_u8(row_3_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 4), svtbl_u8(row_4_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 5), svtbl_u8(row_5_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 6), svtbl_u8(row_6_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 7), svtbl_u8(row_7_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 8), svtbl_u8(row_8_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 9), svtbl_u8(row_9_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 10), svtbl_u8(row_10_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 11), svtbl_u8(row_11_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 12), svtbl_u8(row_12_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 13), svtbl_u8(row_13_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 14), svtbl_u8(row_14_u8x, column_u8x),
                              result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active_b8x, high_nibble_u8x, 15), svtbl_u8(row_15_u8x, column_u8x),
                              result_u8x);

        svst1_u8(active_b8x, (sz_u8_t *)(target + byte_index), result_u8x);
    }
}

/** The SVE stages, which search byte sets with NEON, as SVE before SVE2 has no byte-set match. */
STRINGZILLA_INLINE sz_strings_walks_t sz_strings_walks_sve_(void) {
    sz_strings_walks_t walks;
    walks.find_byteset = &sz_find_byteset_neon_;
    // Graviton 5: the scalable lookup only outruns NEON on registers wider than 128 bits.
    walks.lookup_bytes = svcntb() > 16 ? &sz_strings_lookup_bytes_sve_ : &sz_strings_lookup_bytes_neon_;
    return walks;
}

#if STRINGZILLA_TARGET_SVE

STRINGZILLA_API sz_status_t sz_strings_to_tape_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_tape_with_(strings, order, count, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                   sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_rows_with_(strings, order, count, rows, longest);
}

STRINGZILLA_API sz_status_t sz_strings_escape_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                  sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                  sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_escape_neon_(strings, order, count, kind, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                    sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                    sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_unescape_neon_(strings, order, count, kind, allocator, tape, malformed_index);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                  sz_size_t count, char const lut[sz_at_least_(256)],
                                                  sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                  sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_sve_();
    return sz_strings_lookup_with_(strings, order, count, lut, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_normalize_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_normalize_with_(strings, order, count, form, allocator, tape, &sz_utf8_norm_sve);
}

#endif // STRINGZILLA_TARGET_SVE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_SVE_
#endif // STRINGZILLA_ARCH_ARM64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_STRINGS_SVE_H_
