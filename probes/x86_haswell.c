/* StringZilla ISA probe: Haswell (x86-64 AVX2 + BMI2), mirroring `include/stringzilla/find/haswell.h` */
#include <immintrin.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "popcnt")
#endif

static __m256i sz_probe_shuffle_(__m256i table, __m256i indices) { return _mm256_shuffle_epi8(table, indices); }
static unsigned long long sz_probe_deposit_(unsigned long long bits) { return _pdep_u64(bits, 0x5555555555555555ull); }

int main(void) {
    __m256i shuffled = sz_probe_shuffle_(_mm256_set1_epi8(1), _mm256_set1_epi8(2));
    unsigned long long spread = sz_probe_deposit_((unsigned long long)_mm256_movemask_epi8(shuffled));
    return spread != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
