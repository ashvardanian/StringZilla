#include <stringzilla/stringzilla.h>

#if SZ_USE_X86_AVX512
#include <x86intrin.h>

SZ_EXPORT sz_cptr_t sz_find_byte_avx512(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {

    __m512i needle_vec = _mm512_set1_epi8(*needle);
    __m512i haystack_vec;

sz_find_byte_avx512_cycle:
    if (haystack_length < 64) {
        haystack_vec = _mm512_maskz_loadu_epi8(_bzhi_u64(0xFFFFFFFFFFFFFFFF, haystack_length), haystack);
        haystack_length = 0;
    }
    else {
        haystack_vec = _mm512_loadu_epi8(haystack);
        haystack_length -= 64;
    }

    // Match all loaded characters.
    __mmask64 matches = _mm512_cmp_epu8_mask(haystack_vec, needle_vec, _MM_CMPINT_EQ);
    if (matches != 0) return haystack + sz_ctz64(matches);

    // Jump forward, or exit if nothing is left.
    haystack += 64;
    if (haystack_length) goto sz_find_byte_avx512_cycle;
    return NULL;
}

SZ_EXPORT sz_cptr_t sz_find_2byte_avx512(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {

    // Shifting the bytes across the 512-register is quite expensive.
    // Instead we can simply load twice, and let the CPU do the heavy lifting.
    __m512i needle_vec = _mm512_set1_epi16(*(short const *)(needle));
    __m512i haystack0_vec, haystack1_vec;

sz_find_2byte_avx512_cycle:
    if (haystack_length < 65) {
        haystack0_vec = _mm512_maskz_loadu_epi8(_bzhi_u64(0xFFFFFFFFFFFFFFFF, haystack_length), haystack);
        haystack1_vec = _mm512_maskz_loadu_epi8(_bzhi_u64(0xFFFFFFFFFFFFFFFF, haystack_length - 1), haystack + 1);
        haystack_length = 0;
    }
    else {
        haystack0_vec = _mm512_loadu_epi8(haystack);
        haystack1_vec = _mm512_loadu_epi8(haystack + 1);
        haystack_length -= 64;
    }

    // Match all loaded characters.
    __mmask64 matches0 = _mm512_cmp_epu16_mask(haystack0_vec, needle_vec, _MM_CMPINT_EQ);
    __mmask64 matches1 = _mm512_cmp_epu16_mask(haystack1_vec, needle_vec, _MM_CMPINT_EQ);
    if (matches0 | matches1)
        return haystack + sz_ctz64((matches0 & 0x1111111111111111) | (matches1 & 0x2222222222222222));

    // Jump forward, or exit if nothing is left.
    haystack += 64;
    if (haystack_length) goto sz_find_2byte_avx512_cycle;
    return NULL;
}

SZ_EXPORT sz_cptr_t sz_find_3byte_avx512(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {

    // Shifting the bytes across the 512-register is quite expensive.
    // Instead we can simply load twice, and let the CPU do the heavy lifting.
    __m512i needle_vec = _mm512_set1_epi16(*(short const *)(needle));
    __m512i haystack0_vec, haystack1_vec, haystack2_vec;

sz_find_3byte_avx512_cycle:
    if (haystack_length < 66) {
        haystack0_vec = _mm512_maskz_loadu_epi8(_bzhi_u64(0xFFFFFFFFFFFFFFFF, haystack_length), haystack);
        haystack1_vec = _mm512_maskz_loadu_epi8(_bzhi_u64(0xFFFFFFFFFFFFFFFF, haystack_length - 1), haystack + 1);
        haystack2_vec = _mm512_maskz_loadu_epi8(_bzhi_u64(0xFFFFFFFFFFFFFFFF, haystack_length - 2), haystack + 2);
        haystack_length = 0;
    }
    else {
        haystack0_vec = _mm512_loadu_epi8(haystack);
        haystack1_vec = _mm512_loadu_epi8(haystack + 1);
        haystack2_vec = _mm512_loadu_epi8(haystack + 2);
        haystack_length -= 64;
    }

    // Match all loaded characters.
    __mmask64 matches0 = _mm512_cmp_epu16_mask(haystack0_vec, needle_vec, _MM_CMPINT_EQ);
    __mmask64 matches1 = _mm512_cmp_epu16_mask(haystack1_vec, needle_vec, _MM_CMPINT_EQ);
    __mmask64 matches2 = _mm512_cmp_epu16_mask(haystack2_vec, needle_vec, _MM_CMPINT_EQ);
    if (matches0 | matches1 | matches2)
        return haystack + sz_ctz64((matches0 & 0x1111111111111111) | //
                                   (matches1 & 0x2222222222222222) | //
                                   (matches2 & 0x4444444444444444));

    // Jump forward, or exit if nothing is left.
    haystack += 64;
    if (haystack_length) goto sz_find_3byte_avx512_cycle;
    return NULL;
}

SZ_EXPORT sz_cptr_t sz_find_4byte_avx512(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {

    // Shifting the bytes across the 512-register is quite expensive.
    // Instead we can simply load twice, and let the CPU do the heavy lifting.
    __m512i needle_vec = _mm512_set1_epi32(*(unsigned const *)(needle));
    __m512i haystack0_vec, haystack1_vec, haystack2_vec, haystack3_vec;

sz_find_4byte_avx512_cycle:
    if (haystack_length < 67) {
        haystack0_vec = _mm512_maskz_loadu_epi8(_bzhi_u64(0xFFFFFFFFFFFFFFFF, haystack_length), haystack);
        haystack1_vec = _mm512_maskz_loadu_epi8(_bzhi_u64(0xFFFFFFFFFFFFFFFF, haystack_length - 1), haystack + 1);
        haystack2_vec = _mm512_maskz_loadu_epi8(_bzhi_u64(0xFFFFFFFFFFFFFFFF, haystack_length - 2), haystack + 2);
        haystack3_vec = _mm512_maskz_loadu_epi8(_bzhi_u64(0xFFFFFFFFFFFFFFFF, haystack_length - 3), haystack + 3);
        haystack_length = 0;
    }
    else {
        haystack0_vec = _mm512_loadu_epi8(haystack);
        haystack1_vec = _mm512_loadu_epi8(haystack + 1);
        haystack2_vec = _mm512_loadu_epi8(haystack + 2);
        haystack3_vec = _mm512_loadu_epi8(haystack + 3);
        haystack_length -= 64;
    }

    // Match all loaded characters.
    __mmask64 matches0 = _mm512_cmp_epu16_mask(haystack0_vec, needle_vec, _MM_CMPINT_EQ);
    __mmask64 matches1 = _mm512_cmp_epu16_mask(haystack1_vec, needle_vec, _MM_CMPINT_EQ);
    __mmask64 matches2 = _mm512_cmp_epu16_mask(haystack2_vec, needle_vec, _MM_CMPINT_EQ);
    __mmask64 matches3 = _mm512_cmp_epu16_mask(haystack3_vec, needle_vec, _MM_CMPINT_EQ);
    if (matches0 | matches1 | matches2 | matches3)
        return haystack + sz_ctz64((matches0 & 0x1111111111111111) | //
                                   (matches1 & 0x2222222222222222) | //
                                   (matches2 & 0x4444444444444444) | //
                                   (matches3 & 0x8888888888888888));

    // Jump forward, or exit if nothing is left.
    haystack += 64;
    if (haystack_length) goto sz_find_4byte_avx512_cycle;
    return NULL;
}

SZ_EXPORT sz_cptr_t sz_find_avx512(sz_cptr_t const haystack, sz_size_t const haystack_length, sz_cptr_t const needle,
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
