#include <stringzilla/stringzilla.h>

SZ_EXPORT sz_size_t sz_length_termainted(sz_cptr_t text) {
#ifdef __AVX512__
    return sz_length_termainted_avx512(text);
#else
    return sz_length_termainted_serial(text);
#endif
}

SZ_EXPORT sz_u32_t sz_crc32(sz_cptr_t text, sz_size_t length) {
#ifdef __ARM_FEATURE_CRC32
    return sz_crc32_arm(text, length);
#elif defined(__SSE4_2__)
    return sz_crc32_sse42(text, length);
#elif defined(__AVX512__)
    return sz_crc32_avx512(text, length);
#else
    return sz_crc32_serial(text, length);
#endif
}

SZ_EXPORT sz_order_t sz_order(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
#ifdef __AVX512__
    return sz_order_avx512(a, b, length);
#else
    return sz_order_serial(a, b, length);
#endif
}

SZ_EXPORT sz_order_t sz_order_terminated(sz_cptr_t a, sz_cptr_t b) {
#ifdef __AVX512__
    return sz_order_terminated_avx512(a, b);
#else
    return sz_order_terminated_serial(a, b);
#endif
}

SZ_EXPORT sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
#ifdef __AVX512__
    return sz_find_byte_avx512(haystack, h_length, needle);
#else
    return sz_find_byte_serial(haystack, h_length, needle);
#endif
}

SZ_EXPORT sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
#ifdef __AVX512__
    return sz_find_avx512(haystack, h_length, needle, n_length);
#elif defined(__AVX2__)
    return sz_find_avx2(haystack, h_length, needle, n_length);
#elif defined(__NEON__)
    return sz_find_neon(haystack, h_length, needle, n_length);
#else
    return sz_find_serial(haystack, h_length, needle, n_length);
#endif
}

SZ_EXPORT sz_cptr_t sz_find_terminated(sz_cptr_t haystack, sz_cptr_t needle) {
#ifdef __AVX512__
    return sz_find_terminated_avx512(haystack, needle);
#else
    return sz_find_terminated_serial(haystack, needle);
#endif
}

SZ_EXPORT sz_size_t sz_prefix_accepted(sz_cptr_t text, sz_cptr_t accepted) {
#ifdef __AVX512__
    return sz_prefix_accepted_avx512(text, accepted);
#else
    return sz_prefix_accepted_serial(text, accepted);
#endif
}

SZ_EXPORT sz_size_t sz_prefix_rejected(sz_cptr_t text, sz_cptr_t rejected) {
#ifdef __AVX512__
    return sz_prefix_rejected_avx512(text, rejected);
#else
    return sz_prefix_rejected_serial(text, rejected);
#endif
}

SZ_EXPORT void sz_tolower(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
#ifdef __AVX512__
    sz_tolower_avx512(text, length, result);
#else
    sz_tolower_serial(text, length, result);
#endif
}

SZ_EXPORT void sz_toupper(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
#ifdef __AVX512__
    sz_toupper_avx512(text, length, result);
#else
    sz_toupper_serial(text, length, result);
#endif
}

SZ_EXPORT void sz_toascii(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
#ifdef __AVX512__
    sz_toascii_avx512(text, length, result);
#else
    sz_toascii_serial(text, length, result);
#endif
}

SZ_EXPORT sz_size_t sz_levenshtein(  //
    sz_cptr_t a, sz_size_t a_length, //
    sz_cptr_t b, sz_size_t b_length, //
    sz_cptr_t buffer, sz_size_t bound) {
#ifdef __AVX512__
    return sz_levenshtein_avx512(a, a_length, b, b_length, buffer, bound);
#else
    return sz_levenshtein_serial(a, a_length, b, b_length, buffer, bound);
#endif
}

SZ_EXPORT sz_size_t sz_levenshtein_weighted(          //
    sz_cptr_t a, sz_size_t a_length,                  //
    sz_cptr_t b, sz_size_t b_length,                  //
    sz_error_cost_t gap, sz_error_cost_t const *subs, //
    sz_cptr_t buffer, sz_size_t bound) {

#ifdef __AVX512__
    return sz_levenshtein_weighted_avx512(a, a_length, b, b_length, gap, subs, buffer, bound);
#else
    return sz_levenshtein_weighted_serial(a, a_length, b, b_length, gap, subs, buffer, bound);
#endif
}
