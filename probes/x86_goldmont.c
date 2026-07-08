/* StringZilla ISA probe: Goldmont (x86-64 SHA-NI), mirroring `include/stringzilla/hash/goldmont.h` */
#include <immintrin.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse3,ssse3,sse4.1,sha"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("sse3", "ssse3", "sse4.1", "sha")
#endif

static __m128i sz_probe_sha_rounds_(__m128i state0, __m128i state1, __m128i message) {
    return _mm_sha256rnds2_epu32(state0, state1, message);
}

int main(void) {
    __m128i state = sz_probe_sha_rounds_(_mm_set1_epi32(1), _mm_set1_epi32(2), _mm_set1_epi32(3));
    return _mm_extract_epi32(state, 0) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
