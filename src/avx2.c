#include <stringzilla/stringzilla.h>

#if SZ_USE_X86_AVX2
#include <x86intrin.h>

SZ_PUBLIC sz_cptr_t sz_find_byte_avx2(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    __m256i const n_vec = _mm256_set1_epi8(n[0]);
    sz_cptr_t const h_end = h + h_length;

    while (h + 1 + 32 <= h_end) {
        __m256i h0 = _mm256_loadu_si256((__m256i const *)(h + 0));
        int matches0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(h0, n_vec));
        if (matches0) {
            sz_size_t first_match_offset = sz_u64_ctz(matches0);
            return h + first_match_offset;
        }
        else { h += 32; }
    }
    // Handle the last few characters
    return sz_find_serial(h, h_end - h, n, 1);
}

SZ_PUBLIC sz_cptr_t sz_find_2byte_avx2(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u64_parts_t n_parts;
    n_parts.u64 = 0;
    n_parts.u8s[0] = n[0];
    n_parts.u8s[1] = n[1];

    __m256i const n_vec = _mm256_set1_epi16(n_parts.u16s[0]);
    sz_cptr_t const h_end = h + h_length;

    while (h + 2 + 32 <= h_end) {
        __m256i h0 = _mm256_loadu_si256((__m256i const *)(h + 0));
        int matches0 = _mm256_movemask_epi8(_mm256_cmpeq_epi16(h0, n_vec));
        __m256i h1 = _mm256_loadu_si256((__m256i const *)(h + 1));
        int matches1 = _mm256_movemask_epi8(_mm256_cmpeq_epi16(h1, n_vec));

        if (matches0 | matches1) {
            int combined_matches = (matches0 & 0x55555555) | (matches1 & 0xAAAAAAAA);
            sz_size_t first_match_offset = sz_u64_ctz(combined_matches);
            return h + first_match_offset;
        }
        else { h += 32; }
    }
    // Handle the last few characters
    return sz_find_serial(h, h_end - h, n, 2);
}

SZ_PUBLIC sz_cptr_t sz_find_4byte_avx2(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u64_parts_t n_parts;
    n_parts.u64 = 0;
    n_parts.u8s[0] = n[0];
    n_parts.u8s[1] = n[1];
    n_parts.u8s[2] = n[2];
    n_parts.u8s[3] = n[3];

    __m256i const n_vec = _mm256_set1_epi32(n_parts.u32s[0]);
    sz_cptr_t const h_end = h + h_length;

    while (h + 4 + 32 <= h_end) {
        // Top level for-loop changes dramatically.
        // In sequential computing model for 32 offsets we would do:
        //  + 32 comparions.
        //  + 32 branches.
        // In vectorized computations models:
        //  + 4 vectorized comparisons.
        //  + 4 movemasks.
        //  + 3 bitwise ANDs.
        __m256i h0 = _mm256_loadu_si256((__m256i const *)(h + 0));
        int matches0 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h0, n_vec));
        __m256i h1 = _mm256_loadu_si256((__m256i const *)(h + 1));
        int matches1 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h1, n_vec));
        __m256i h2 = _mm256_loadu_si256((__m256i const *)(h + 2));
        int matches2 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h2, n_vec));
        __m256i h3 = _mm256_loadu_si256((__m256i const *)(h + 3));
        int matches3 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h3, n_vec));

        if (matches0 | matches1 | matches2 | matches3) {
            int matches =                 //
                (matches0 & 0x11111111) | //
                (matches1 & 0x22222222) | //
                (matches2 & 0x44444444) | //
                (matches3 & 0x88888888);
            sz_size_t first_match_offset = sz_u64_ctz(matches);
            return h + first_match_offset;
        }
        else { h += 32; }
    }
    // Handle the last few characters
    return sz_find_serial(h, h_end - h, n, 4);
}

SZ_PUBLIC sz_cptr_t sz_find_3byte_avx2(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u64_parts_t n_parts;
    n_parts.u64 = 0;
    n_parts.u8s[0] = n[0];
    n_parts.u8s[1] = n[1];
    n_parts.u8s[2] = n[2];

    // This implementation is more complex than the `sz_find_4byte_avx2`,
    // as we are going to match only 3 bytes within each 4-byte word.
    sz_u64_parts_t mask_parts;
    mask_parts.u64 = 0;
    mask_parts.u8s[0] = mask_parts.u8s[1] = mask_parts.u8s[2] = 0xFF, mask_parts.u8s[3] = 0;

    __m256i const n_vec = _mm256_set1_epi32(n_parts.u32s[0]);
    __m256i const mask_vec = _mm256_set1_epi32(mask_parts.u32s[0]);
    sz_cptr_t const h_end = h + h_length;

    while (h + 4 + 32 <= h_end) {
        __m256i h0 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(h + 0)), mask_vec);
        int matches0 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h0, n_vec));
        __m256i h1 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(h + 1)), mask_vec);
        int matches1 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h1, n_vec));
        __m256i h2 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(h + 2)), mask_vec);
        int matches2 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h2, n_vec));
        __m256i h3 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(h + 3)), mask_vec);
        int matches3 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h3, n_vec));

        if (matches0 | matches1 | matches2 | matches3) {
            int matches =                 //
                (matches0 & 0x11111111) | //
                (matches1 & 0x22222222) | //
                (matches2 & 0x44444444) | //
                (matches3 & 0x88888888);
            sz_size_t first_match_offset = sz_u64_ctz(matches);
            return h + first_match_offset;
        }
        else { h += 32; }
    }
    // Handle the last few characters
    return sz_find_serial(h, h_end - h, n, 3);
}

SZ_PUBLIC sz_cptr_t sz_find_avx2(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    if (h_length < n_length) return NULL;

    // For very short strings a lookup table for an optimized backend makes a lot of sense
    switch (n_length) {
    case 0: return NULL;
    case 1: return sz_find_byte_avx2(h, h_length, n);
    case 2: return sz_find_2byte_avx2(h, h_length, n);
    case 3: return sz_find_3byte_avx2(h, h_length, n);
    case 4: return sz_find_4byte_avx2(h, h_length, n);
    default:
    }

    // For longer needles, use exact matching for the first 4 bytes and then check the rest
    sz_size_t prefix_length = 4;
    for (sz_size_t i = 0; i <= h_length - n_length; ++i) {
        sz_cptr_t found = sz_find_4byte_avx2(h + i, h_length - i, n);
        if (!found) return NULL;

        // Verify the remaining part of the needle
        if (sz_equal_serial(found + prefix_length, n + prefix_length, n_length - prefix_length)) return found;

        // Adjust the position
        i = found - h + prefix_length - 1;
    }

    return NULL;
}

#endif
