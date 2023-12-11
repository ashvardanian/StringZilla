#include <stringzilla/stringzilla.h>

SZ_PUBLIC sz_size_t sz_length_termainted(sz_cptr_t text) { return sz_find_byte(text, ~0ull - 1ull, 0) - text; }

SZ_PUBLIC sz_u32_t sz_crc32(sz_cptr_t text, sz_size_t length) {
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

SZ_PUBLIC sz_ordering_t sz_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
#ifdef __AVX512__
    return sz_order_avx512(a, a_length, b, b_length);
#else
    return sz_order_serial(a, a_length, b, b_length);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
#ifdef __AVX512__
    return sz_find_byte_avx512(haystack, h_length, needle);
#else
    return sz_find_byte_serial(haystack, h_length, needle);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
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

SZ_PUBLIC sz_cptr_t sz_find_terminated(sz_cptr_t haystack, sz_cptr_t needle) {
    return sz_find(haystack, sz_length_termainted(haystack), needle, sz_length_termainted(needle));
}

SZ_PUBLIC sz_size_t sz_prefix_accepted(sz_cptr_t text, sz_size_t length, sz_cptr_t accepted, sz_size_t count) {
#ifdef __AVX512__
    return sz_prefix_accepted_avx512(text, length, accepted, count);
#else
    return sz_prefix_accepted_serial(text, length, accepted, count);
#endif
}

SZ_PUBLIC sz_size_t sz_prefix_rejected(sz_cptr_t text, sz_size_t length, sz_cptr_t rejected, sz_size_t count) {
#ifdef __AVX512__
    return sz_prefix_rejected_avx512(text, length, rejected, count);
#else
    return sz_prefix_rejected_serial(text, length, rejected, count);
#endif
}

SZ_PUBLIC void sz_tolower(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
#ifdef __AVX512__
    sz_tolower_avx512(text, length, result);
#else
    sz_tolower_serial(text, length, result);
#endif
}

SZ_PUBLIC void sz_toupper(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
#ifdef __AVX512__
    sz_toupper_avx512(text, length, result);
#else
    sz_toupper_serial(text, length, result);
#endif
}

SZ_PUBLIC void sz_toascii(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
#ifdef __AVX512__
    sz_toascii_avx512(text, length, result);
#else
    sz_toascii_serial(text, length, result);
#endif
}

SZ_PUBLIC sz_size_t sz_levenshtein(  //
    sz_cptr_t a, sz_size_t a_length, //
    sz_cptr_t b, sz_size_t b_length, //
    sz_cptr_t buffer, sz_size_t bound) {
#ifdef __AVX512__
    return sz_levenshtein_avx512(a, a_length, b, b_length, buffer, bound);
#else
    return sz_levenshtein_serial(a, a_length, b, b_length, buffer, bound);
#endif
}

SZ_PUBLIC sz_size_t sz_levenshtein_weighted(          //
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
