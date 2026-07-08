/* StringZilla ISA probe: Ice Lake (x86-64 AVX-512 VBMI/VNNI + VAES), mirroring `include/stringzilla/hash/icelake.h` */
#include <immintrin.h>

#if defined(__clang__)
#pragma clang attribute push( \
    __attribute__((           \
        target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,avx512vnni,bmi,bmi2,aes,vaes,sha"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", \
                   "avx512vnni", "bmi", "bmi2", "aes", "vaes", "sha")
#endif

/* The Ice Lake kernels split across two orthogonal AVX-512 sub-extensions - the hash/intersect cores use
 * VNNI dot-products, the find/UTF-8 cores use VBMI2 compress/expand - so the probe must exercise both. */
static __m512i sz_probe_permute_(__m512i table, __m512i indices) { return _mm512_permutexvar_epi8(indices, table); }
static __m512i sz_probe_compress_(__mmask64 mask, __m512i bytes) { return _mm512_maskz_compress_epi8(mask, bytes); }
static __m512i sz_probe_dot_(__m512i acc, __m512i a, __m512i b) { return _mm512_dpbusds_epi32(acc, a, b); }
static __m512i sz_probe_aes_rounds_(__m512i state, __m512i key) { return _mm512_aesenc_epi128(state, key); }

int main(void) {
    __m512i permuted = sz_probe_permute_(_mm512_set1_epi8(1), _mm512_set1_epi8(2));
    __m512i squeezed = sz_probe_compress_((__mmask64)0x5555555555555555ull, permuted);
    __m512i summed = sz_probe_dot_(_mm512_setzero_si512(), squeezed, permuted);
    __m512i mixed = sz_probe_aes_rounds_(summed, _mm512_set1_epi8(3));
    return _mm512_reduce_add_epi64(mixed) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
