/**
 *  @file include/stringzilla/strings/rvv.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief RISC-V Vector (RVV 1.0) backend for collections of strings.
 *
 *  @sa c/stringzilla/strings.h
 */
#ifndef STRINGZILLA_STRINGS_RVV_H_
#define STRINGZILLA_STRINGS_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/find/rvv.h"  // `sz_find_byteset_rvv_`
#include "stringzilla/utf8_norm.h" // `sz_utf8_norm_rvv`, `sz_utf8_find_denormalized_rvv`
#include "stringzilla/strings/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  The lookup gathers `lut[index[i]]` from memory with @c vluxei8, so every byte is a valid index
 *  at any @c VLEN, and the @c vsetvl loop covers every tail, leaving no serial path. */
#if STRINGZILLA_ARCH_RISCV64_
#if STRINGZILLA_ARCH_RISCV64_RVV_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

STRINGZILLA_INLINE void sz_strings_lookup_bytes_rvv_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                     char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);
    sz_u8_t *target_cursor = (sz_u8_t *)target;
    sz_u8_t const *source_cursor = (sz_u8_t const *)source;
    sz_u8_t const *lut_u8 = (sz_u8_t const *)lut;
    while (length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(length);
        vuint8m8_t index_u8m8 = __riscv_vle8_v_u8m8(source_cursor, vector_length);
        vuint8m8_t result_u8m8 = __riscv_vluxei8_v_u8m8(lut_u8, index_u8m8, vector_length);
        __riscv_vse8_v_u8m8(target_cursor, result_u8m8, vector_length);
        target_cursor += vector_length, source_cursor += vector_length, length -= vector_length;
    }
}

/** The RVV stages. */
STRINGZILLA_INLINE sz_strings_walks_t sz_strings_walks_rvv_(void) {
    sz_strings_walks_t walks;
    walks.find_byteset = &sz_find_byteset_rvv_;
    walks.lookup_bytes = &sz_strings_lookup_bytes_rvv_;
    return walks;
}

#if STRINGZILLA_TARGET_RVV

STRINGZILLA_API sz_status_t sz_strings_to_tape_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_tape_with_(strings, order, count, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                   sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_rows_with_(strings, order, count, rows, longest);
}

STRINGZILLA_API sz_status_t sz_strings_escape_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                  sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                  sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_rvv_();
    return sz_strings_escape_with_(strings, order, count, kind, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                    sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                    sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_rvv_();
    return sz_strings_unescape_with_(strings, order, count, kind, allocator, tape, malformed_index, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                  sz_size_t count, char const lut[sz_at_least_(256)],
                                                  sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                  sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_rvv_();
    return sz_strings_lookup_with_(strings, order, count, lut, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_normalize_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_normalize_with_(strings, order, count, form, allocator, tape, &sz_utf8_norm_rvv);
}

#endif // STRINGZILLA_TARGET_RVV

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_RISCV64_RVV_
#endif // STRINGZILLA_ARCH_RISCV64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_STRINGS_RVV_H_
