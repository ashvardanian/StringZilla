/* StringZilla ISA probe: Skylake-X (x86-64 AVX-512 F/VL/BW), mirroring `include/stringzilla/find/skylake.h` */
#include <immintrin.h>

#if defined(__clang__) && (__clang_major__ >= 18 || (defined(__apple_build_version__) && __clang_major__ >= 17))
// LLVM 18+ (Apple Clang 17+, which is LLVM 19-based) splits `evex512` out of AVX-512:
// ZMM codegen in `target` attributes needs it named explicitly.
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,evex512"))), \
                             apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#endif

static __mmask64 sz_probe_match_(__m512i haystack, __m512i needle) { return _mm512_cmpeq_epi8_mask(haystack, needle); }
static __m512i sz_probe_blend_(__mmask64 mask, __m512i a, __m512i b) { return _mm512_mask_blend_epi8(mask, a, b); }

int main(void) {
    __mmask64 mask = sz_probe_match_(_mm512_set1_epi8(1), _mm512_set1_epi8(1));
    __m512i blended = sz_probe_blend_(mask, _mm512_setzero_si512(), _mm512_set1_epi8(3));
    return _mm512_reduce_add_epi64(blended) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
