/**
 *  @brief LoongArch LASX (256-bit) backend for compare.
 *  @file include/stringzilla/compare/lasx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_LASX_H_
#define STRINGZILLA_COMPARE_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX

/*  LASX has no single "movemask" instruction. `__lasx_xvmskltz_b` collects the sign bit of every
 *  byte into a per-128-bit-lane 16-bit mask, deposited into word 0 (low lane) and word 4 (high lane).
 *  Recombining them yields the same 32-bit mask AVX2's `_mm256_movemask_epi8` would produce, so the
 *  byte order matches and `ctz`/`clz` index bytes identically to the Haswell backend. */
SZ_INTERNAL sz_u32_t sz_xvmovemask_b_compare_lasx_(__m256i sign_extended) {
    __m256i collected_u8x32 = __lasx_xvmskltz_b(sign_extended);
    unsigned int low = __lasx_xvpickve2gr_wu(collected_u8x32, 0);
    unsigned int high = __lasx_xvpickve2gr_wu(collected_u8x32, 4);
    return (low & 0xFFFFu) | ((high & 0xFFFFu) << 16);
}

/*  The 128-bit LSX analog: `__lsx_vmskltz_b` packs the 16 byte sign bits into the low 16 bits of element 0,
 *  so a single GPR extraction yields the SSE-style 16-bit `_mm_movemask_epi8` value. LSX is the natural fit
 *  for sub-32-byte inputs, where a 256-bit LASX register would be half-empty and a serial byte loop wastes
 *  the wide datapath the Loongson cores expose. */
SZ_INTERNAL sz_u32_t sz_vmovemask_b_compare_lsx_(__m128i sign_extended) {
    return (unsigned int)__lsx_vpickve2gr_wu(__lsx_vmskltz_b(sign_extended), 0) & 0xFFFFu;
}

SZ_PUBLIC sz_ordering_t sz_order_lasx(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Before optimizing this, read the "Operations Not Worth Optimizing" in Contributions Guide:
    //! https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md#general-performance-observations
    return sz_order_serial(a, a_length, b, b_length);
}

SZ_PUBLIC sz_bool_t sz_equal_lasx(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {

    if (length < 8) {
        sz_cptr_t const a_end = a + length;
        while (a != a_end && *a == *b) a++, b++;
        return (sz_bool_t)(a_end == a);
    }
    // Two overlapping 64-bit words cover everything in [8, 16] without a branch.
    else if (length <= 16) {
        sz_u64_t a_first_word = sz_u64_load(a).u64, b_first_word = sz_u64_load(b).u64;
        sz_u64_t a_second_word = sz_u64_load(a + length - 8).u64, b_second_word = sz_u64_load(b + length - 8).u64;
        return (sz_bool_t)((a_first_word == b_first_word) & (a_second_word == b_second_word));
    }
    // Two overlapping 128-bit LSX loads cover [17, 32]; LSX keeps the whole register busy on short inputs
    // instead of leaving half of a 256-bit LASX register idle.
    else if (length <= 32) {
        sz_u128_vec_t a_first_vec, b_first_vec, a_second_vec, b_second_vec;
        a_first_vec.lsx = __lsx_vld(a, 0);
        b_first_vec.lsx = __lsx_vld(b, 0);
        a_second_vec.lsx = __lsx_vld(a + length - 16, 0);
        b_second_vec.lsx = __lsx_vld(b + length - 16, 0);
        __m128i first_matches_u8x16 = __lsx_vseq_b(a_first_vec.lsx, b_first_vec.lsx);
        __m128i second_matches_u8x16 = __lsx_vseq_b(a_second_vec.lsx, b_second_vec.lsx);
        __m128i both_matches_u8x16 = __lsx_vand_v(first_matches_u8x16, second_matches_u8x16);
        return (sz_bool_t)(sz_vmovemask_b_compare_lsx_(both_matches_u8x16) == 0xFFFFu);
    }
    // We can use 2x 256-bit interleaving loads, similar to the AVX2 backend, to handle up to 64 bytes.
    else if (length <= 64) {
        sz_u256_vec_t a_first_vec, b_first_vec, a_second_vec, b_second_vec;
        a_first_vec.lasx = __lasx_xvld(a, 0);
        b_first_vec.lasx = __lasx_xvld(b, 0);
        a_second_vec.lasx = __lasx_xvld(a + length - 32, 0);
        b_second_vec.lasx = __lasx_xvld(b + length - 32, 0);
        __m256i first_matches_u8x32 = __lasx_xvseq_b(a_first_vec.lasx, b_first_vec.lasx);
        __m256i second_matches_u8x32 = __lasx_xvseq_b(a_second_vec.lasx, b_second_vec.lasx);
        __m256i both_matches_u8x32 = __lasx_xvand_v(first_matches_u8x32, second_matches_u8x32);
        return (sz_bool_t)(sz_xvmovemask_b_compare_lasx_(both_matches_u8x32) == 0xFFFFFFFFu);
    }
    else {
        sz_size_t byte_index = 0;
        sz_u256_vec_t a_vec, b_vec;
        do {
            a_vec.lasx = __lasx_xvld(a + byte_index, 0);
            b_vec.lasx = __lasx_xvld(b + byte_index, 0);
            if (sz_xvmovemask_b_compare_lasx_(__lasx_xvseq_b(a_vec.lasx, b_vec.lasx)) != 0xFFFFFFFFu) return sz_false_k;
            byte_index += 32;
        } while (byte_index + 32 <= length);
        a_vec.lasx = __lasx_xvld(a + length - 32, 0);
        b_vec.lasx = __lasx_xvld(b + length - 32, 0);
        return (sz_bool_t)(sz_xvmovemask_b_compare_lasx_(__lasx_xvseq_b(a_vec.lasx, b_vec.lasx)) == 0xFFFFFFFFu);
    }
}

#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_LASX_H_
