/**
 *  @file include/stringzilla/compare/serial.h
 *  @author Ash Vardanian
 *  @date August 14, 2020
 *  @brief Serial backend for string comparison utilities.
 *
 *  @sa c/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_SERIAL_H_
#define STRINGZILLA_COMPARE_SERIAL_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

STRINGZILLA_INLINE sz_bool_t sz_equal_serial_(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
#if STRINGZILLA_ALLOW_MISALIGNED_LOADS
    if (length >= STRINGZILLA_SWAR_THRESHOLD) {
        sz_u64_vec_t a_vec, b_vec;
        for (; length >= 8; a += 8, b += 8, length -= 8) {
            a_vec = sz_u64_load(a);
            b_vec = sz_u64_load(b);
            if (a_vec.u64 != b_vec.u64) return sz_false_k;
        }
    }
#endif
    while (length && *a == *b) a++, b++, length--;
    return (sz_bool_t)(length == 0);
}

STRINGZILLA_INLINE sz_ordering_t sz_order_serial_(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_bool_t a_shorter = (sz_bool_t)(a_length < b_length);
    // Counting the length down keeps a null empty string defined, where `a + 0` on null is not.
    sz_size_t remaining = a_shorter ? a_length : b_length;
#if STRINGZILLA_ALLOW_MISALIGNED_LOADS && !STRINGZILLA_ARCH_BIG_ENDIAN_
    for (sz_u64_vec_t a_vec, b_vec; remaining >= 8; a += 8, b += 8, remaining -= 8) {
        a_vec = sz_u64_load(a);
        b_vec = sz_u64_load(b);
        if (a_vec.u64 != b_vec.u64)
            return sz_order_scalars_(sz_u64_bytes_reverse(a_vec.u64), sz_u64_bytes_reverse(b_vec.u64));
    }
#endif
    for (; remaining; ++a, ++b, --remaining)
        // Compare as `unsigned char` to match `memcmp`, `std::string`, and the byte-reversed
        // `u64` fast path above; this also preserves UTF-8 code-point order (bytes >= 0x80).
        if (*a != *b) return sz_order_scalars_((sz_u8_t)*a, (sz_u8_t)*b);

    // If the strings are equal up to the shorter one's length, then the shorter string is smaller
    return sz_order_scalars_(a_length, b_length);
}

#if STRINGZILLA_TARGET_SERIAL

STRINGZILLA_API sz_status_t sz_equal_serial(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                            sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *equal = sz_equal_serial_(a, b, length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                            sz_ordering_t *ordering, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *ordering = sz_order_serial_(a, a_length, b, b_length);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_SERIAL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_SERIAL_H_
