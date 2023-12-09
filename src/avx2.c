#include <stringzilla/stringzilla.h>

#if defined(__AVX2__)
#include <x86intrin.h>

/**
 *  @brief  Substring-search implementation, leveraging x86 AVX2 intrinsics and speculative
 *          execution capabilities on modern CPUs. Performing 4 unaligned vector loads per cycle
 *          was practically more efficient than loading once and shifting around, as introduces
 *          less data dependencies.
 */
SZ_EXPORT sz_cptr_t sz_find_avx2(sz_cptr_t const haystack, sz_size_t const haystack_length, sz_cptr_t const needle,
                                 sz_size_t const needle_length) {

    // Precomputed constants
    sz_cptr_t const end = haystack + haystack_length;
    _sz_anomaly_t anomaly;
    _sz_anomaly_t mask;
    sz_export_prefix_u32(needle, needle_length, &anomaly, &mask);
    __m256i const anomalies = _mm256_set1_epi32(anomaly.u32);
    __m256i const masks = _mm256_set1_epi32(mask.u32);

    // Top level for-loop changes dramatically.
    // In sequential computing model for 32 offsets we would do:
    //  + 32 comparions.
    //  + 32 branches.
    // In vectorized computations models:
    //  + 4 vectorized comparisons.
    //  + 4 movemasks.
    //  + 3 bitwise ANDs.
    //  + 1 heavy (but very unlikely) branch.
    sz_cptr_t text = haystack;
    while (text + needle_length + 32 <= end) {

        // Performing many unaligned loads ends up being faster than loading once and shuffling around.
        __m256i texts0 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 0)), masks);
        int matches0 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(texts0, anomalies));
        __m256i texts1 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 1)), masks);
        int matches1 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(texts1, anomalies));
        __m256i text2 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 2)), masks);
        int matches2 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(text2, anomalies));
        __m256i texts3 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 3)), masks);
        int matches3 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(texts3, anomalies));

        if (matches0 | matches1 | matches2 | matches3) {
            int matches =                 //
                (matches0 & 0x11111111) | //
                (matches1 & 0x22222222) | //
                (matches2 & 0x44444444) | //
                (matches3 & 0x88888888);
            sz_size_t first_match_offset = sz_ctz64(matches);
            if (needle_length > 4) {
                if (sz_equal(text + first_match_offset + 4, needle + 4, needle_length - 4)) {
                    return text + first_match_offset;
                }
                else { text += first_match_offset + 1; }
            }
            else { return text + first_match_offset; }
        }
        else { text += 32; }
    }

    // Don't forget the last (up to 35) characters.
    return sz_find_serial(text, end - text, needle, needle_length);
}

#endif
