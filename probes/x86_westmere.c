/* StringZilla ISA probe: Westmere (x86-64 SSE4.2 + AES-NI), mirroring `include/stringzilla/{find,hash}/westmere.h` */
#include <immintrin.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse4.2,aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("sse4.2", "aes")
#endif

static __m128i sz_probe_aes_round_(__m128i state, __m128i key) { return _mm_aesenc_si128(state, key); }
static unsigned sz_probe_crc_(unsigned crc, unsigned long long word) { return (unsigned)_mm_crc32_u64(crc, word); }

int main(void) {
    __m128i mixed = sz_probe_aes_round_(_mm_set1_epi8(1), _mm_set1_epi8(2));
    unsigned crc = sz_probe_crc_(0u, 42ull);
    return (_mm_extract_epi32(mixed, 0) ^ (int)crc) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
