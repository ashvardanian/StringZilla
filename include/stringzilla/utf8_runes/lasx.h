/**
 *  @brief LoongArch LASX backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_runes/lasx.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_LASX_H_
#define STRINGZILLA_UTF8_RUNES_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX
/** @brief  Recombine the two per-128-bit-lane 16-bit `__lasx_xvmskltz_b` sign-bit masks into one 32-bit mask,
 *          matching AVX2's `_mm256_movemask_epi8` (word 0 = low lane, word 4 = high lane). */
SZ_HELPER_INLINE sz_u32_t sz_xvmovemask_b_utf8_lasx_(__m256i sign_extended) {
    __m256i comparison_result_u8x32 = __lasx_xvmskltz_b(sign_extended);
    sz_u32_t low = (sz_u32_t)__lasx_xvpickve2gr_wu(comparison_result_u8x32, 0);
    sz_u32_t high = (sz_u32_t)__lasx_xvpickve2gr_wu(comparison_result_u8x32, 4);
    return (low & 0xFFFFu) | ((high & 0xFFFFu) << 16);
}

/** @brief  Per-lane logical right shift by 4 bits, keeping only the low nibble of every byte lane. The single shift
 *          amount the classifier needs is spelled out (LASX `xvsrli.h` requires an immediate; it shifts 16-bit lanes,
 *          so we mask back to byte width afterwards). */
SZ_HELPER_INLINE __m256i sz_utf8_high_nibble_lasx_(__m256i value) {
    return __lasx_xvand_v(__lasx_xvsrli_h(value, 4), __lasx_xvreplgr2vr_b((char)0x0F));
}

/** @brief  Shift the whole 32-byte window left by one byte (lane `i` receives original lane `i + 1`), zero-filling the
 *          top. Built from an in-lane `xvbsrl.v` (per-128-bit down-shift) stitched with the cross-lane high half via
 *          `xvpermi.q` so byte 16 receives original byte 17 across the 128-bit seam. */
SZ_HELPER_INLINE __m256i sz_utf8_next1_lasx_(__m256i window) {
    __m256i const high_half_duplicated_u8x32 = __lasx_xvpermi_q(window, window, 0x31); // both halves := original high
    __m256i const shift_within_lane_u8x32 = __lasx_xvbsrl_v(window, 1); // bytes down by 1 within each 128-bit half
    __m256i const cross_lane_carry_u8x32 = __lasx_xvbsll_v(high_half_duplicated_u8x32,
                                                           15); // high-half byte 0 → lane 15
    return __lasx_xvor_v(shift_within_lane_u8x32, cross_lane_carry_u8x32);
}

#pragma region Shuffle LUT left pack

/** @brief  256-entry shuffle LUT: row `m` left-packs the set-bit positions of the 8-bit mask `m` into the low bytes
 *          (value = bit index 0..7), in `.rodata` (no per-call rebuild). The caller reads only the leading
 *          `popcount(m)` lanes; the 0x80 fill marks the unused tail. 16 bytes per row so one `xvld` brings a row into
 *          a 128-bit `xvshuf.b` source lane. */
SZ_HELPER_INLINE sz_u8_t const *sz_utf8_pack8_lut_lasx_(void) {
    // 256 rows x 16 bytes of `.rodata`: row `m` lists the set-bit positions of the 8-bit mask `m`; only the leading
    // `popcount(m)` lanes are read, the 0x80 fill is an unused-tail marker.
    static sz_u8_t const table[256][16] = {
        {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 4, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 4, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 4, 5, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 4, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {3, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 3, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 3, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 3, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {2, 3, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 2, 3, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {1, 2, 3, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        {0, 1, 2, 3, 4, 5, 6, 7, 128, 128, 128, 128, 128, 128, 128, 128}};
    return (sz_u8_t const *)table;
}

/**
 *  @brief  Left-pack the set lane indices (ascending, in [0,32)) of the 32-bit @p mask into @p out via a `xvshuf.b`
 *          shuffle-LUT permutation, NOT a scalar `ctz` per-bit walk. Each 128-bit window lane carries 16 byte lanes
 *          ([0,16) and [16,32)); the mask is split into four 8-bit halves, each routed through the 256-entry
 *          @ref sz_utf8_pack8_lut_lasx_ so `xvshuf.b` collapses the set positions to the low bytes of the
 *          lane. The four packed groups are stitched in order by their `popcount` offset (the group base + local
 *          index gives the absolute byte offset). @return the popcount of @p mask.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_pack_indices_lasx_(sz_u32_t mask, sz_u8_t *out) {
    sz_u8_t const *lut = sz_utf8_pack8_lut_lasx_();
    // Lane-local byte identity {0..15, 0..15}: each 128-bit lane shuffles its own 0..15 identity by the LUT row, so
    // the packed values are the within-half bit positions; we add the half base to recover absolute offsets.
    static sz_u8_t const identity_bytes[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    __m256i const byte_identity_iota_u8x32 = __lasx_xvld(identity_bytes, 0);
    sz_size_t total = 0;
    for (int half = 0; half < 4; ++half) {
        sz_u32_t const half_mask = (mask >> (half * 8)) & 0xFFu;
        int const half_base = half * 8;
        __m256i const shuffle_lookup_row_u8x32 = __lasx_xvld(lut + (sz_size_t)half_mask * 16, 0);
        // The low 128-bit lane holds the packed locals.
        __m256i const packed_indices_u8x32 = __lasx_xvshuf_b(byte_identity_iota_u8x32, byte_identity_iota_u8x32,
                                                             shuffle_lookup_row_u8x32);
        sz_u256_vec_t packed_bytes_vec;
        packed_bytes_vec.lasx = packed_indices_u8x32;
        sz_size_t const count = (sz_size_t)sz_u32_popcount(half_mask);
        for (sz_size_t k = 0; k < count; ++k) out[total++] = (sz_u8_t)(half_base + packed_bytes_vec.u8s[k]);
    }
    return total;
}

#pragma endregion Shuffle LUT left pack

/**
 *  @brief  Decode the emitted-start lanes @p emit_starts of a classified @p window into sequential UTF-32 runes, the
 *          LASX sibling of `sz_utf8_rune_drain_icelake_`. The start byte-offsets are left-packed by a
 *          single `xvshuf.b` shuffle-LUT (no scalar per-bit walk, no spill compaction); the lead/+1/+2/+3 bytes are
 *          gathered at the packed offsets with `xvshuf.b` over a broadcast window, the value is reconstructed by a
 *          branchless 1/2/3/4-byte width-blend, and every @p ill_formed lane is overwritten with U+FFFD. The resume
 *          cursor delta = last packed start offset + its compacted maximal-subpart length (so an ill-formed trailing
 *          lane never skips bytes owing their own next U+FFFD).
 *  @return Number of runes emitted; sets @p consumed_bytes to the byte span they cover.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_lasx_( //
    __m256i window, sz_u32_t emit_starts, sz_u32_t ill_formed, __m256i consumed_length, int has_three, int has_four,
    sz_size_t emit_count, sz_rune_t *runes, sz_size_t capacity, sz_size_t *consumed_bytes) {

    // The width-blend gathers the lead, 2nd byte, and 2-byte form unconditionally; the 3-byte and 4-byte forms are
    // gathered and assembled only when `has_three`/`has_four` say a 3- or 4-byte start exists in this window, so a
    // 3-byte CJK window pays for the 3rd byte yet skips the 4th-byte gather/assembly (perf-neutral, Ice Lake parity).

    // Left-pack the emitted-start byte-offsets via the shuffle-LUT (one `xvshuf.b` per 8-bit mask half).
    sz_u8_t start_offsets[32];
    sz_utf8_pack_indices_lasx_(emit_starts, start_offsets);

    // Broadcast the window's low/high 128-bit halves into both lanes so a single `xvshuf.b` gathers any byte 0..31:
    // index 0..15 selects the low half (`window_low_u8x32`), 16..31 the high half (`window_high_u8x32`), in EVERY
    // 128-bit lane.
    __m256i const window_low_u8x32 = __lasx_xvpermi_q(window, window, 0x00);
    __m256i const window_high_u8x32 = __lasx_xvpermi_q(window, window, 0x11);
    // The per-lane maximal-subpart length, read at the packed start offsets for the resume cursor.
    sz_u256_vec_t consumed_byte_lengths_vec;
    consumed_byte_lengths_vec.lasx = consumed_length;

    __m256i const mask_word_1f_u32x8 = __lasx_xvreplgr2vr_w(0x1F), mask_word_3f_u32x8 = __lasx_xvreplgr2vr_w(0x3F);
    __m256i const mask_word_0f_u32x8 = __lasx_xvreplgr2vr_w(0x0F), mask_word_07_u32x8 = __lasx_xvreplgr2vr_w(0x07);
    __m256i const mask_word_c0_u32x8 = __lasx_xvreplgr2vr_w(0xC0), mask_word_e0_u32x8 = __lasx_xvreplgr2vr_w(0xE0),
                  mask_word_f0_u32x8 = __lasx_xvreplgr2vr_w(0xF0);
    __m256i const replacement_codepoint_u32x8 = __lasx_xvreplgr2vr_w((int)sz_rune_replacement_k);

    sz_size_t const want = emit_count < capacity ? emit_count : capacity;
    sz_size_t produced = 0;
    // One 4-codepoint block per outer step: `xvsllwil` widens the low 4 bytes of each 128-bit lane, so the four
    // gathered bytes placed in lanes [0,4) land in word lanes [0,4). Decode only the K emitted starts.
    for (sz_size_t block_start = 0; block_start < want; block_start += 4) {
        // Load up to 4 packed start offsets into the low bytes of an index vector. Trailing lanes are filled with the
        // last valid offset (an in-window index): `xvshuf.b` keys off the low bits only and does NOT zero a high-bit
        // index, so a defined in-range fill keeps those (unstored) lanes from gathering across the 128-bit seam.
        sz_size_t const block_lanes = (want - block_start) < 4 ? (want - block_start) : 4;
        sz_u256_vec_t gather_index_vec;
        for (int lane = 0; lane < 32; ++lane) gather_index_vec.u8s[lane] = start_offsets[block_start];
        for (sz_size_t lane = 0; lane < block_lanes; ++lane)
            gather_index_vec.u8s[lane] = start_offsets[block_start + lane];
        __m256i const gather_index_u8x32 = gather_index_vec.lasx;

        // Gather the lead and 2nd byte per codepoint at the packed start offsets (one `xvshuf.b` each over the broadcast
        // window). The 3rd/4th bytes are gathered only inside the `has_three`/`has_four` siblings below.
        __m256i const lead_byte_u8x32 = __lasx_xvshuf_b(window_high_u8x32, window_low_u8x32, gather_index_u8x32);
        __m256i const continuation_byte_1_u8x32 = __lasx_xvshuf_b(
            window_high_u8x32, window_low_u8x32, __lasx_xvadd_b(gather_index_u8x32, __lasx_xvreplgr2vr_b(1)));

        // Widen the low 8 byte lanes (this block's gathered bytes) to 8x 32-bit words in the low 128-bit half via the
        // byte→half→word ladder; only the low half is consumed so no cross-128-bit shuffle is needed.
        __m256i const lead_u32x8 = __lasx_xvsllwil_wu_hu(__lasx_xvsllwil_hu_bu(lead_byte_u8x32, 0), 0);
        __m256i const continuation_byte_1_u32x8 = __lasx_xvsllwil_wu_hu(
            __lasx_xvsllwil_hu_bu(continuation_byte_1_u8x32, 0), 0);

        __m256i decoded_codepoint_u32x8 = lead_u32x8; /* ASCII keeps the lead. */
        // 2-byte: lead in [C0,E0): ((b0 & 0x1F) << 6) | (b1 & 0x3F).
        __m256i const two_byte_codepoint_u32x8 = __lasx_xvor_v(
            __lasx_xvslli_w(__lasx_xvand_v(lead_u32x8, mask_word_1f_u32x8), 6),
            __lasx_xvand_v(continuation_byte_1_u32x8, mask_word_3f_u32x8));
        __m256i const two_byte_mask_u32x8 = __lasx_xvandn_v(__lasx_xvsle_wu(mask_word_e0_u32x8, lead_u32x8),
                                                            __lasx_xvsle_wu(mask_word_c0_u32x8, lead_u32x8));
        decoded_codepoint_u32x8 = __lasx_xvbitsel_v(decoded_codepoint_u32x8, two_byte_codepoint_u32x8,
                                                    two_byte_mask_u32x8);

        // The 3rd byte is hoisted so the 4-byte sibling reuses it; `has_four` implies `has_three`, so it is set there.
        __m256i continuation_byte_2_u32x8 = __lasx_xvreplgr2vr_w(0);
        if (has_three) {
            // 3-byte: lead in [E0,F0): ((b0&0x0F)<<12)|((b1&0x3F)<<6)|(b2&0x3F). Gather the 3rd byte only here.
            __m256i const continuation_byte_2_u8x32 = __lasx_xvshuf_b(
                window_high_u8x32, window_low_u8x32, __lasx_xvadd_b(gather_index_u8x32, __lasx_xvreplgr2vr_b(2)));
            continuation_byte_2_u32x8 = __lasx_xvsllwil_wu_hu(__lasx_xvsllwil_hu_bu(continuation_byte_2_u8x32, 0), 0);
            __m256i const three_byte_codepoint_u32x8 = __lasx_xvor_v(
                __lasx_xvor_v(__lasx_xvslli_w(__lasx_xvand_v(lead_u32x8, mask_word_0f_u32x8), 12),
                              __lasx_xvslli_w(__lasx_xvand_v(continuation_byte_1_u32x8, mask_word_3f_u32x8), 6)),
                __lasx_xvand_v(continuation_byte_2_u32x8, mask_word_3f_u32x8));
            __m256i const three_byte_mask_u32x8 = __lasx_xvandn_v(__lasx_xvsle_wu(mask_word_f0_u32x8, lead_u32x8),
                                                                  __lasx_xvsle_wu(mask_word_e0_u32x8, lead_u32x8));
            decoded_codepoint_u32x8 = __lasx_xvbitsel_v(decoded_codepoint_u32x8, three_byte_codepoint_u32x8,
                                                        three_byte_mask_u32x8);
        }
        if (has_four) { /* SIBLING; `has_four` ⟹ `has_three`, so `continuation_byte_2_u32x8` is set above. */
            // 4-byte: lead >= F0: ((b0&7)<<18)|((b1&0x3F)<<12)|((b2&0x3F)<<6)|(b3&0x3F). Gather the 4th byte only here.
            __m256i const continuation_byte_3_u8x32 = __lasx_xvshuf_b(
                window_high_u8x32, window_low_u8x32, __lasx_xvadd_b(gather_index_u8x32, __lasx_xvreplgr2vr_b(3)));
            __m256i const continuation_byte_3_u32x8 = __lasx_xvsllwil_wu_hu(
                __lasx_xvsllwil_hu_bu(continuation_byte_3_u8x32, 0), 0);
            __m256i const four_byte_codepoint_u32x8 = __lasx_xvor_v(
                __lasx_xvor_v(__lasx_xvslli_w(__lasx_xvand_v(lead_u32x8, mask_word_07_u32x8), 18),
                              __lasx_xvslli_w(__lasx_xvand_v(continuation_byte_1_u32x8, mask_word_3f_u32x8), 12)),
                __lasx_xvor_v(__lasx_xvslli_w(__lasx_xvand_v(continuation_byte_2_u32x8, mask_word_3f_u32x8), 6),
                              __lasx_xvand_v(continuation_byte_3_u32x8, mask_word_3f_u32x8)));
            __m256i const four_byte_mask_u32x8 = __lasx_xvsle_wu(mask_word_f0_u32x8, lead_u32x8);
            decoded_codepoint_u32x8 = __lasx_xvbitsel_v(decoded_codepoint_u32x8, four_byte_codepoint_u32x8,
                                                        four_byte_mask_u32x8);
        }

        // Overwrite every ill-formed lane in this block with U+FFFD. Build the per-word ill-formed selector from the
        // compacted `ill_formed` mask bits for this block (word j set ⟺ the (block_start+j)-th emitted start is ill).
        sz_u256_vec_t ill_formed_mask_vec;
        ill_formed_mask_vec.lasx = __lasx_xvreplgr2vr_w(0);
        for (sz_size_t lane = 0; lane < block_lanes; ++lane) {
            sz_size_t const emitted_index = block_start + lane;
            ill_formed_mask_vec.u32s[lane] = ((ill_formed >> start_offsets[emitted_index]) & 1u) ? ~0u : 0u;
        }
        decoded_codepoint_u32x8 = __lasx_xvbitsel_v(decoded_codepoint_u32x8, replacement_codepoint_u32x8,
                                                    ill_formed_mask_vec.lasx);

        // Store the low `block_lanes` words (the four decoded codepoints live in word lanes [0,4)). Extract each via
        // `xvpickve2gr.wu` to a scalar register and write through `runes` so the result is a plain integer store the
        // caller's same-typed load reliably observes.
        runes[produced + 0] = (sz_rune_t)__lasx_xvpickve2gr_wu(decoded_codepoint_u32x8, 0);
        if (block_lanes >= 2) runes[produced + 1] = (sz_rune_t)__lasx_xvpickve2gr_wu(decoded_codepoint_u32x8, 1);
        if (block_lanes >= 3) runes[produced + 2] = (sz_rune_t)__lasx_xvpickve2gr_wu(decoded_codepoint_u32x8, 2);
        if (block_lanes >= 4) runes[produced + 3] = (sz_rune_t)__lasx_xvpickve2gr_wu(decoded_codepoint_u32x8, 3);
        produced += block_lanes;
    }

    // Resume cursor delta = end of the last emitted start = its byte offset + its maximal-subpart length.
    sz_u8_t const last_off = start_offsets[produced - 1];
    sz_u8_t const last_length = consumed_byte_lengths_vec.u8s[last_off];
    *consumed_bytes = (sz_size_t)last_off + (sz_size_t)last_length;
    return produced;
}

/**
 *  @brief  Decode one <=32-byte window of @p text into dense UTF-32 @p runes by the uniform classify → per-lane
 *          well-formed + orphan promotion → compress emitted starts → gather → width-blend → blend U+FFFD path,
 *          emitting at most @p runes_capacity runes and returning the resume cursor. The decode is TOTAL: clean and
 *          dirty bytes are handled in-vector, one U+FFFD per maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C),
 *          bit-exact with @ref sz_utf8_decode_serial. Mirrors `sz_utf8_decode_once_icelake_`; the step
 *          declines (`*runes_unpacked == 0`, cursor unchanged) ONLY when the first lead's declared sequence crosses
 *          the window edge (a boundary truncation), which the public entry finalizes without a serial re-decode.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_decode_once_lasx_( //
    sz_cptr_t text, sz_size_t length,               //
    sz_rune_t *runes, sz_size_t runes_capacity,     //
    sz_size_t *runes_unpacked) {

    sz_size_t const chunk = length < 32 ? length : 32;
    sz_u32_t const loaded_mask = chunk >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << chunk) - 1u);

    // Load <=32 bytes; tail lanes beyond `chunk` are zeroed (whole buffer pre-zeroed) then byte-filled, so we never
    // read past `length`. The 64-byte union is padded so the per-block widen never runs past the end.
    sz_u512_vec_t staging_vec;
    __lasx_xvst(__lasx_xvreplgr2vr_b(0), staging_vec.u8s, 0);
    __lasx_xvst(__lasx_xvreplgr2vr_b(0), staging_vec.u8s + 32, 0);
    for (sz_size_t i = 0; i < chunk; ++i) staging_vec.u8s[i] = ((sz_u8_t const *)text)[i];
    __m256i const window_u8x32 = __lasx_xvld(staging_vec.u8s, 0);

    // ASCII fast lane: a window with no high bit set widens directly. `xvmskltz_b` flags lanes with the sign bit set.
    sz_u32_t const high_bits = sz_xvmovemask_b_utf8_lasx_(window_u8x32) & loaded_mask;
    if (high_bits == 0) {
        sz_size_t const runes_to_unpack = chunk < runes_capacity ? chunk : runes_capacity;
        sz_size_t group = 0;
        for (; group + 4 <= runes_to_unpack; group += 4) {
            __m256i const widened_ascii_u32x8 = __lasx_xvsllwil_wu_hu(
                __lasx_xvsllwil_hu_bu(__lasx_xvld(staging_vec.u8s + group, 0), 0), 0);
            __lasx_xvstelm_d(widened_ascii_u32x8, runes + group, 0, 0);
            __lasx_xvstelm_d(widened_ascii_u32x8, runes + group, 8, 1);
        }
        for (; group < runes_to_unpack; ++group) runes[group] = (sz_rune_t)staging_vec.u8s[group];
        *runes_unpacked = runes_to_unpack;
        return text + runes_to_unpack;
    }

    // Single-source classification: a high-nibble lookup gives per-lane byte length {1x12,2,2,3,4}; both the validity
    // gate and the width-blend derive from it, so a lead and its length can never disagree.
    sz_u8_t const length_lut_bytes[32] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    __m256i const length_lookup_table_u8x32 = __lasx_xvld(length_lut_bytes, 0);
    __m256i const high_nibble_u8x32 = sz_utf8_high_nibble_lasx_(window_u8x32);
    // 128-bit-lane LUT, table duplicated.
    __m256i const sequence_length_u8x32 = __lasx_xvshuf_b(length_lookup_table_u8x32, length_lookup_table_u8x32,
                                                          high_nibble_u8x32);

    // start = non-continuation lane: a byte whose (b & 0xC0) != 0x80. Continuation bytes are signed -128..-65.
    __m256i const continuation_mask_u8x32 = __lasx_xvseq_b(
        __lasx_xvand_v(window_u8x32, __lasx_xvreplgr2vr_b((char)0xC0)), __lasx_xvreplgr2vr_b((char)0x80));
    sz_u32_t const continuation_bits = sz_xvmovemask_b_utf8_lasx_(continuation_mask_u8x32) & loaded_mask;
    sz_u32_t const starts_bits = (~continuation_bits) & loaded_mask;

    // Defer every start whose declared sequence would reach past the window; the FIRST overrunning start bounds the
    // decodable prefix (well-formed text has only the trailing truncation, but a malformed `E0 C0` overruns earlier).
    sz_u8_t const lane_identity_raw[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    sz_u8_t const lane_high_bias[32] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16};
    __m256i const global_lane_offset_u8x32 = __lasx_xvadd_b(__lasx_xvld(lane_identity_raw, 0),
                                                            __lasx_xvld(lane_high_bias, 0));
    __m256i const sequence_end_offset_u8x32 = __lasx_xvadd_b(global_lane_offset_u8x32, sequence_length_u8x32);
    // sequence_end > chunk, restricted to starts. Compare unsigned bytes.
    sz_u32_t const past_end = sz_xvmovemask_b_utf8_lasx_(
                                  __lasx_xvslt_bu(__lasx_xvreplgr2vr_b((char)chunk), sequence_end_offset_u8x32)) &
                              starts_bits;
    sz_size_t const decodable_end = past_end ? (sz_size_t)sz_u32_ctz(past_end) : chunk;
    sz_u32_t const decodable_mask = decodable_end >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << decodable_end) - 1u);

    // Per-length start masks from the LUT length (NOT the lead bit-pattern): a stray lead such as 0xF8 maps to length 4
    // in the LUT, so it MUST be classed as a length-4 start here for the bad-lead gate to reject it.
    sz_u32_t const len2 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(sequence_length_u8x32, __lasx_xvreplgr2vr_b(2))) &
                          starts_bits;
    sz_u32_t const len3 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(sequence_length_u8x32, __lasx_xvreplgr2vr_b(3))) &
                          starts_bits;
    sz_u32_t const len4 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(sequence_length_u8x32, __lasx_xvreplgr2vr_b(4))) &
                          starts_bits;
    sz_u32_t const len_ge2 = len2 | len3 | len4;
    sz_u32_t const len_ge3 = len3 | len4;
    sz_u32_t const len_ge4 = len4;
    sz_u32_t const len1 = starts_bits & ~len_ge2;

    // Bad lead: 0xC0/0xC1 (overlong 2-byte by the LUT) or 0xF5..0xFF (out of range, length-4 by the LUT).
    sz_u32_t const lead_lt_c2 = sz_xvmovemask_b_utf8_lasx_(
        __lasx_xvslt_bu(window_u8x32, __lasx_xvreplgr2vr_b((char)0xC2)));
    sz_u32_t const lead_gt_f4 = sz_xvmovemask_b_utf8_lasx_(
        __lasx_xvslt_bu(__lasx_xvreplgr2vr_b((char)0xF4), window_u8x32));
    sz_u32_t const bad_lead = ((len2 & lead_lt_c2) | (len4 & lead_gt_f4)) & starts_bits;

    // First-continuation range violations for E0/ED/F0/F4 (overlong 3/4-byte, surrogate, > U+10FFFF), computed only
    // when a 3/4-byte lead is present; for any other lead this mask is empty, so `b1_range_ok` keeps the lane.
    int const has_three = len_ge3 != 0;
    int const has_four = len_ge4 != 0;
    sz_u32_t overlong_or_surrogate_or_range = 0;
    if (has_three || has_four) {
        __m256i const next_byte_u8x32 = sz_utf8_next1_lasx_(window_u8x32);
        sz_u32_t const is_e0 = sz_xvmovemask_b_utf8_lasx_(
            __lasx_xvseq_b(window_u8x32, __lasx_xvreplgr2vr_b((char)0xE0)));
        sz_u32_t const is_ed = sz_xvmovemask_b_utf8_lasx_(
            __lasx_xvseq_b(window_u8x32, __lasx_xvreplgr2vr_b((char)0xED)));
        sz_u32_t const is_f0 = sz_xvmovemask_b_utf8_lasx_(
            __lasx_xvseq_b(window_u8x32, __lasx_xvreplgr2vr_b((char)0xF0)));
        sz_u32_t const is_f4 = sz_xvmovemask_b_utf8_lasx_(
            __lasx_xvseq_b(window_u8x32, __lasx_xvreplgr2vr_b((char)0xF4)));
        sz_u32_t const n1_lt_a0 = sz_xvmovemask_b_utf8_lasx_(
            __lasx_xvslt_bu(next_byte_u8x32, __lasx_xvreplgr2vr_b((char)0xA0)));
        sz_u32_t const n1_ge_a0 = (~n1_lt_a0) & loaded_mask;
        sz_u32_t const n1_lt_90 = sz_xvmovemask_b_utf8_lasx_(
            __lasx_xvslt_bu(next_byte_u8x32, __lasx_xvreplgr2vr_b((char)0x90)));
        sz_u32_t const n1_ge_90 = (~n1_lt_90) & loaded_mask;
        overlong_or_surrogate_or_range =
            ((is_e0 & n1_lt_a0) | (is_ed & n1_ge_a0) | (is_f0 & n1_lt_90) | (is_f4 & n1_ge_90)) & starts_bits;
    }

    // Per-lane continuation availability at the declared trailing slots, evaluated at the lead lane.
    sz_u32_t const cont1 = starts_bits & (continuation_bits >> 1);
    sz_u32_t const cont2 = starts_bits & (continuation_bits >> 2);
    sz_u32_t const cont3 = starts_bits & (continuation_bits >> 3);
    sz_u32_t const b1_range_ok = starts_bits & ~overlong_or_surrogate_or_range;
    sz_u32_t const first_ok = cont1 & b1_range_ok & ~bad_lead;

    // Well-formed leads (vectorized `sz_rune_decode` success), restricted to the decodable span.
    sz_u32_t const wf1 = len1;
    sz_u32_t const wf2 = len2 & ~bad_lead & cont1;
    sz_u32_t const wf3 = len3 & b1_range_ok & cont1 & cont2;
    sz_u32_t const wf4 = len4 & ~bad_lead & b1_range_ok & cont1 & cont2 & cont3;
    sz_u32_t const well_formed = (wf1 | wf2 | wf3 | wf4) & decodable_mask;

    // Per-lane maximal-subpart length: start at 1 and extend across each continuation slot a well-formed sequence
    // would still accept. For well-formed lanes this equals the declared length; for ill-formed lanes it is the 1-3
    // byte subpart that one U+FFFD consumes.
    sz_u32_t const step2 = len_ge2 & first_ok;
    sz_u32_t const step3 = step2 & len_ge3 & cont2;
    sz_u32_t const step4 = step3 & len_ge4 & cont3;

    // Orphan promotion: a continuation byte not covered by ANY lead's maximal-subpart span becomes its own 1-byte
    // U+FFFD. The subpart spans are exactly the continuation slots the `step2/3/4` adds reached.
    sz_u32_t const covered = ((step2 & decodable_mask) << 1) | ((step3 & decodable_mask) << 2) |
                             ((step4 & decodable_mask) << 3);
    sz_u32_t const orphan = continuation_bits & decodable_mask & ~covered;
    sz_u32_t const emit_starts = (starts_bits | orphan) & decodable_mask;
    sz_size_t const emit_count = (sz_size_t)sz_u32_popcount(emit_starts);
    if (emit_count == 0) { return *runes_unpacked = 0, text; } // Nothing decodable → window-edge finalize in driver.
    sz_u32_t const ill_formed = emit_starts & ~well_formed;    // Orphans are continuations → never well-formed.

    // Per-lane consumed length = 1 + step2 + step3 + step4 (each step adds one continuation byte to the subpart).
    __m256i maximal_subpart_length_u8x32 = __lasx_xvreplgr2vr_b(1);
    __m256i const one_byte_u8x32 = __lasx_xvreplgr2vr_b(1);
    {
        // Smear each 32-bit step mask to a 0xFF-per-lane byte selector so the conditional add stays branchless.
        sz_u256_vec_t step_byte_mask_vec;
        for (int lane = 0; lane < 32; ++lane) step_byte_mask_vec.u8s[lane] = (sz_u8_t)((step2 >> lane) & 1u ? 0xFF : 0);
        maximal_subpart_length_u8x32 = __lasx_xvadd_b(maximal_subpart_length_u8x32,
                                                      __lasx_xvand_v(step_byte_mask_vec.lasx, one_byte_u8x32));
        for (int lane = 0; lane < 32; ++lane) step_byte_mask_vec.u8s[lane] = (sz_u8_t)((step3 >> lane) & 1u ? 0xFF : 0);
        maximal_subpart_length_u8x32 = __lasx_xvadd_b(maximal_subpart_length_u8x32,
                                                      __lasx_xvand_v(step_byte_mask_vec.lasx, one_byte_u8x32));
        for (int lane = 0; lane < 32; ++lane) step_byte_mask_vec.u8s[lane] = (sz_u8_t)((step4 >> lane) & 1u ? 0xFF : 0);
        maximal_subpart_length_u8x32 = __lasx_xvadd_b(maximal_subpart_length_u8x32,
                                                      __lasx_xvand_v(step_byte_mask_vec.lasx, one_byte_u8x32));
    }

    sz_size_t consumed = 0;
    sz_size_t const produced = sz_utf8_rune_drain_lasx_(window_u8x32, emit_starts, ill_formed,
                                                        maximal_subpart_length_u8x32, has_three, has_four, emit_count,
                                                        runes, runes_capacity, &consumed);
    *runes_unpacked = produced;
    return text + consumed;
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_lasx(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    sz_cptr_t cursor = text;
    sz_cptr_t const end = text + length;
    sz_size_t runes_written = 0;
    while (runes_written < runes_capacity && cursor < end) {
        sz_size_t step_unpacked = 0;
        sz_cptr_t next = sz_utf8_decode_once_lasx_(cursor, (sz_size_t)(end - cursor), runes + runes_written,
                                                   runes_capacity - runes_written, &step_unpacked);
        if (step_unpacked) {
            runes_written += step_unpacked;
            cursor = next;
            continue;
        }
        // `step_unpacked == 0` only when the very first lead declares a sequence crossing the window edge (a boundary
        // truncation). A resumable truncation breaks and awaits more bytes; a bad/overlong truncated lead at the edge
        // finalizes to one U+FFFD over its maximal ill-formed subpart - a bounded <=3-byte finalize, never a serial
        // window re-decode.
        if (sz_utf8_incomplete_tail_(cursor, end)) break;
        runes[runes_written++] = (sz_rune_t)sz_rune_replacement_k;
        cursor += sz_utf8_maximal_subpart_(cursor, end);
    }
    *runes_unpacked = runes_written;
    return cursor;
}

#pragma region Multistep Newline & Whitespace Iteration

/*  Multistep newline / whitespace iteration (LoongArch LASX).
 *
 *  Each 32-byte tile is classified branchlessly into a `start_bits` mask of every delimiter start plus disjoint
 *  2-byte / 3-byte start masks (built from `sz_xvmovemask_b_utf8_lasx_` and bitmask shifts). The peel left-packs
 *  with the same dword-index table the AVX2 backend uses, in eight 4-lane sub-blocks. Starts are trusted in lanes
 *  [0,29] and we step 30; a `t[pos-1] == '\r'` carry suppresses an LF completing a tile-straddling CRLF. */

/** @brief  Store the low `group` (1..4) of four 64-bit lanes of `values_u64x4` to `destination`. */
SZ_HELPER_INLINE void sz_utf8_iterate_store_group_lasx_(__m256i values_u64x4, sz_size_t group, sz_size_t *destination) {
    if (group == 4) { __lasx_xvst(values_u64x4, destination, 0); }
    else {
        __lasx_xvstelm_d(values_u64x4, destination, 0, 0);
        if (group >= 2) __lasx_xvstelm_d(values_u64x4, destination, 8, 1);
        if (group >= 3) __lasx_xvstelm_d(values_u64x4, destination, 16, 2);
    }
}

SZ_API_COMPTIME sz_size_t sz_utf8_count_lasx(sz_cptr_t text, sz_size_t length) {
    __m256i continuation_mask_u8x32 = __lasx_xvreplgr2vr_b((char)0xC0);
    __m256i continuation_pattern_u8x32 = __lasx_xvreplgr2vr_b((char)0x80);
    __m256i const one_byte_u8x32 = __lasx_xvreplgr2vr_b(1);
    __m256i const zero_u8x32 = __lasx_xvreplgr2vr_b(0);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Keep a per-lane vector accumulator across the loop and reduce ONCE at the end, instead of doing a
    // movemask + scalar `popcount` per block. A start byte is any byte where `(b & 0xC0) != 0x80`; we map
    // that to a 0/1 byte and fold adjacent pairs into a 16x u16 accumulator via `xvhaddw_hu_bu`. Each u16
    // lane grows by at most 2 per block, so 65535/2 = 32767 blocks fit before overflow; we flush the u16
    // lanes into a 4x u64 accumulator every 32000 blocks to stay safe (rare for realistic inputs).
    __m256i sum_u64x4 = zero_u8x32;
    __m256i sum_u16x16 = zero_u8x32;
    sz_size_t blocks_since_flush = 0;
    while (length >= 32) {
        __m256i text_window_u8x32 = __lasx_xvld(text_u8, 0);
        // Extract top 2 bits of each byte; continuation bytes equal 0x80. `start_mask_u8x32` is 0xFF for starts.
        __m256i header_mask_u8x32 = __lasx_xvand_v(text_window_u8x32, continuation_mask_u8x32);
        __m256i is_continuation_u8x32 = __lasx_xvseq_b(header_mask_u8x32,
                                                       continuation_pattern_u8x32); /* 0xFF where cont */
        // `andn` style: 1 where start, 0 where continuation. (~is_continuation) & 1.
        __m256i start_mask_u8x32 = __lasx_xvandn_v(is_continuation_u8x32, one_byte_u8x32);
        sum_u16x16 = __lasx_xvadd_h(sum_u16x16, __lasx_xvhaddw_hu_bu(start_mask_u8x32, start_mask_u8x32));
        if (++blocks_since_flush == 32000) {
            __m256i widened_sum_u32x8 = __lasx_xvhaddw_wu_hu(sum_u16x16, sum_u16x16);
            sum_u64x4 = __lasx_xvadd_d(sum_u64x4, __lasx_xvhaddw_du_wu(widened_sum_u32x8, widened_sum_u32x8));
            sum_u16x16 = zero_u8x32;
            blocks_since_flush = 0;
        }
        text_u8 += 32, length -= 32;
    }
    {
        __m256i widened_sum_u32x8 = __lasx_xvhaddw_wu_hu(sum_u16x16, sum_u16x16);
        sum_u64x4 = __lasx_xvadd_d(sum_u64x4, __lasx_xvhaddw_du_wu(widened_sum_u32x8, widened_sum_u32x8));
    }
    char_count += (sz_size_t)__lasx_xvpickve2gr_du(sum_u64x4, 0) + (sz_size_t)__lasx_xvpickve2gr_du(sum_u64x4, 1) +
                  (sz_size_t)__lasx_xvpickve2gr_du(sum_u64x4, 2) + (sz_size_t)__lasx_xvpickve2gr_du(sum_u64x4, 3);

    char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

/** @brief  Same block logic as `sz_utf8_count_lasx`, but locating the Nth start byte. LASX has no `PDEP`, so the
 *          start-byte bitmask is fed to `sz_u32_nth_set_bit` to land on the Nth set bit. */
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_lasx(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    __m256i continuation_mask_u8x32 = __lasx_xvreplgr2vr_b((char)0xC0);
    __m256i continuation_pattern_u8x32 = __lasx_xvreplgr2vr_b((char)0x80);
    __m256i const one_byte_u8x32 = __lasx_xvreplgr2vr_b(1);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    sz_u256_vec_t text_window_vec, header_mask_vec;
    while (length >= 32) {
        text_window_vec.lasx = __lasx_xvld(text_u8, 0);
        header_mask_vec.lasx = __lasx_xvand_v(text_window_vec.lasx, continuation_mask_u8x32);
        __m256i is_continuation_u8x32 = __lasx_xvseq_b(header_mask_vec.lasx, continuation_pattern_u8x32);
        sz_u32_t start_byte_mask = ~sz_xvmovemask_b_utf8_lasx_(is_continuation_u8x32);

        // Count start bytes in-vector: each start lane carries a 0x01, so `xvpcnt_d` sums one bit per such lane.
        __m256i start_mask_u8x32 = __lasx_xvandn_v(is_continuation_u8x32, one_byte_u8x32);
        __m256i start_count_u32x8 = __lasx_xvpcnt_d(start_mask_u8x32);
        sz_size_t start_byte_count =
            (sz_size_t)(__lasx_xvpickve2gr_du(start_count_u32x8, 0) + __lasx_xvpickve2gr_du(start_count_u32x8, 1) +
                        __lasx_xvpickve2gr_du(start_count_u32x8, 2) + __lasx_xvpickve2gr_du(start_count_u32x8, 3));

        if (n >= start_byte_count) {
            n -= start_byte_count, text_u8 += 32, length -= 32;
            continue;
        }
        return (sz_cptr_t)(text_u8 + sz_u32_nth_set_bit(start_byte_mask, n));
    }

    return sz_utf8_seek_serial((sz_cptr_t)text_u8, length, n);
}

#pragma region Word boundaries windowed substrate

/*  The 64-byte windowed substrate the UAX-29 Word_Break kernel builds on (@ref utf8_wordbreaks/lasx.h). LASX is
 *  256-bit, so a 64-byte window lives as two `__m256i` halves (`*_low_u8x32` = lanes [0,32), `*_high_u8x32` = lanes [32,64)) - the
 *  Haswell shape, NOT the NEON quarters. Every lane mask is `sz_u64_t` (`sz_xvmovemask_b_utf8_lasx_` per half,
 *  OR-combined). Forward neighbours wrap modulo 64 exactly like Ice Lake's `permutexvar`; the 256-bit seam between the
 *  two halves is stitched with `xvpermi.q` since LASX has no 32-byte cross-lane byte shuffle. */

/** @brief  The decoded 64-byte window for the LASX backend; field names/semantics match @ref sz_utf8_rune_window_t so
 *          the portable rule algebra is unchanged. The 64 bytes and the byte-domain codepoint halves live as two
 *          `__m256i` halves each; masks are `sz_u64_t` (`vpmovmskb`-equivalent per half, OR-combined). */
typedef struct sz_utf8_rune_window_lasx_t {
    __m256i window_low_u8x32;     /**< Raw input bytes for lanes [0, 32). */
    __m256i window_high_u8x32;    /**< Raw input bytes for lanes [32, 64). */
    __m256i high_byte_low_u8x32;  /**< Per-lane `codepoint >> 8` for lanes [0, 32). */
    __m256i high_byte_high_u8x32; /**< Per-lane `codepoint >> 8` for lanes [32, 64). */
    __m256i low_byte_low_u8x32;   /**< Per-lane `codepoint & 0xFF` for lanes [0, 32). */
    __m256i low_byte_high_u8x32;  /**< Per-lane `codepoint & 0xFF` for lanes [32, 64). */
    sz_u64_t continuation;        /**< Bit `i` => lane `i` is a continuation byte `10xxxxxx`. */
    sz_u64_t codepoint_starts;    /**< Bit `i` => lane `i` begins a codepoint (loaded, non-continuation). */
    sz_u64_t two_byte_starts;     /**< Bit `i` => lane `i` is a 2-byte lead `110xxxxx`. */
    sz_u64_t three_byte_starts;   /**< Bit `i` => lane `i` is a 3-byte lead `1110xxxx`. */
    sz_u64_t four_byte_starts;    /**< Bit `i` => lane `i` is a 4-byte lead `11110xxx`. */
    sz_size_t loaded;             /**< Number of bytes actually loaded (<= 64). */
} sz_utf8_rune_window_lasx_t;

/** @brief  Per-byte logical right shift by a constant @p shift keeping the low @p keep bits - the LASX twin of `srl8_`.
 *          The 16-bit `xvsrli.h` shifts across byte pairs, so the per-byte @p keep mask clears the neighbour's bleed. A
 *          macro because `xvsrli.h` demands an immediate shift (unlike the AVX2 wrapper), and every caller passes one. */
#define sz_utf8_srl8_lasx_(value, shift, keep) \
    __lasx_xvand_v(__lasx_xvsrli_h((value), (shift)), __lasx_xvreplgr2vr_b((char)(keep)))

/** @brief  Combine two per-half `sz_xvmovemask_b_utf8_lasx_` results into one 64-bit lane mask. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_mask_combine_lasx_(__m256i low_half_u8x32, __m256i high_half_u8x32) {
    sz_u64_t const low_bits = (sz_u32_t)sz_xvmovemask_b_utf8_lasx_(low_half_u8x32);
    sz_u64_t const high_bits = (sz_u32_t)sz_xvmovemask_b_utf8_lasx_(high_half_u8x32);
    return low_bits | (high_bits << 32);
}

/** @brief  Expand a 32-bit lane mask into a 32-byte select vector (byte `i` = 0xFF when bit `i` is set) to drive
 *          `xvbitsel.v` in place of AVX2's `blendv_epi8`. Broadcasts the four mask bytes, routes each output lane's
 *          byte with `xvshuf.b`, isolates the per-lane bit, then compares equal (in-register, no scalar). */
SZ_HELPER_INLINE __m256i sz_utf8_byte_mask_from_bits_lasx_(sz_u32_t bits) {
    static sz_u8_t const byte_router[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    static sz_u8_t const bit_select[32] = {1, 2, 4, 8, 16, 32, 64, (sz_u8_t)128, 1, 2, 4, 8, 16, 32, 64, (sz_u8_t)128,
                                           1, 2, 4, 8, 16, 32, 64, (sz_u8_t)128, 1, 2, 4, 8, 16, 32, 64, (sz_u8_t)128};
    __m256i const mask_broadcast_u32x8 = __lasx_xvreplgr2vr_w((int)bits);
    __m256i const spread_u8x32 = __lasx_xvshuf_b(mask_broadcast_u32x8, mask_broadcast_u32x8,
                                                 __lasx_xvld(byte_router, 0));
    __m256i const bit_position_u8x32 = __lasx_xvld(bit_select, 0);
    return __lasx_xvseq_b(__lasx_xvand_v(spread_u8x32, bit_position_u8x32), bit_position_u8x32);
}

/** @brief  Masked 64-byte load into two halves; bytes [loaded, 64) read as zero (the LASX stand-in for
 *          `_mm512_maskz_loadu_epi8`). A stack staging union covers the partial tail so we never read past
 *          `text + loaded`. */
SZ_HELPER_AUTO void sz_utf8_load_window_lasx_(sz_u8_t const *text, sz_size_t loaded, __m256i *out_low_u8x32,
                                              __m256i *out_high_u8x32) {
    if (loaded >= 64) {
        *out_low_u8x32 = __lasx_xvld(text + 0, 0);
        *out_high_u8x32 = __lasx_xvld(text + 32, 0);
        return;
    }
    sz_u512_vec_t staging;
    __lasx_xvst(__lasx_xvreplgr2vr_b(0), staging.u8s + 0, 0);
    __lasx_xvst(__lasx_xvreplgr2vr_b(0), staging.u8s + 32, 0);
    for (sz_size_t i = 0; i < loaded; ++i) staging.u8s[i] = text[i];
    *out_low_u8x32 = __lasx_xvld(staging.u8s + 0, 0);
    *out_high_u8x32 = __lasx_xvld(staging.u8s + 32, 0);
}

/** @brief  Forward neighbours `nextK[i] = window[i+K]` for K in {1,2,3} over all 64 lanes, with lanes past the window
 *          WRAPPING modulo 64 to match Ice Lake's `permutexvar_epi8` (so `next1[63]==window[0]`, etc.). LASX has no
 *          32-byte byte permute, so each 128-bit lane's successor bytes arrive from the neighbouring 128-bit block via
 *          `xvpermi.q` (`window_high_u8x32`'s successor is `window_low_u8x32`, byte 64 aliasing byte 0), then an in-lane `xvbsrl`/
 *          `xvbsll` byte-shift pair stitches across the lane boundary - the two-halves generalization of
 *          @ref sz_utf8_next1_lasx_ to distances 2 and 3 plus the 256-bit half seam. */
SZ_HELPER_INLINE void sz_utf8_forward_neighbours_lasx_( //
    __m256i window_low_u8x32, __m256i window_high_u8x32, __m256i *next_byte_1_low_u8x32,
    __m256i *next_byte_1_high_u8x32, __m256i *next_byte_2_low_u8x32, __m256i *next_byte_2_high_u8x32,
    __m256i *next_byte_3_low_u8x32, __m256i *next_byte_3_high_u8x32) {
    // The 128-bit block following window_low is [window_low.high, window_high.low]; the one following window_high wraps
    // to [window_high.high, window_low.low] (byte 64 aliases byte 0). `xvpermi_q(vd, vj, imm)` picks per 128-bit lane
    // from {vj.low, vj.high, vd.low, vd.high} = {0, 1, 2, 3}; 0x21 selects {vj.high, vd.low}.
    __m256i const successor_low_u8x32 = __lasx_xvpermi_q(window_high_u8x32, window_low_u8x32, 0x21);
    __m256i const successor_high_u8x32 = __lasx_xvpermi_q(window_low_u8x32, window_high_u8x32, 0x21);
    // `alignr(a, b, k)` per lane = `(b >> k bytes) | (a << (16 - k) bytes)`: the low part comes from the window, the
    // wrapped high part from the successor block.
    *next_byte_1_low_u8x32 = __lasx_xvor_v(__lasx_xvbsrl_v(window_low_u8x32, 1),
                                           __lasx_xvbsll_v(successor_low_u8x32, 15));
    *next_byte_2_low_u8x32 = __lasx_xvor_v(__lasx_xvbsrl_v(window_low_u8x32, 2),
                                           __lasx_xvbsll_v(successor_low_u8x32, 14));
    *next_byte_3_low_u8x32 = __lasx_xvor_v(__lasx_xvbsrl_v(window_low_u8x32, 3),
                                           __lasx_xvbsll_v(successor_low_u8x32, 13));
    *next_byte_1_high_u8x32 = __lasx_xvor_v(__lasx_xvbsrl_v(window_high_u8x32, 1),
                                            __lasx_xvbsll_v(successor_high_u8x32, 15));
    *next_byte_2_high_u8x32 = __lasx_xvor_v(__lasx_xvbsrl_v(window_high_u8x32, 2),
                                            __lasx_xvbsll_v(successor_high_u8x32, 14));
    *next_byte_3_high_u8x32 = __lasx_xvor_v(__lasx_xvbsrl_v(window_high_u8x32, 3),
                                            __lasx_xvbsll_v(successor_high_u8x32, 13));
}

/** @brief  Load up to 64 bytes (masked tail) and decode every lane into byte-domain halves - the LASX twin of
 *          @ref sz_utf8_rune_decode_window_, bit-identical on every lane. */
SZ_HELPER_AUTO sz_utf8_rune_window_lasx_t sz_utf8_rune_decode_window_lasx_(sz_u8_t const *text, sz_size_t available) {
    sz_utf8_rune_window_lasx_t result;
    result.loaded = available < 64 ? available : 64;

    __m256i window_low_u8x32, window_high_u8x32;
    sz_utf8_load_window_lasx_(text, result.loaded, &window_low_u8x32, &window_high_u8x32);
    result.window_low_u8x32 = window_low_u8x32, result.window_high_u8x32 = window_high_u8x32;

    __m256i next1_low_u8x32, next1_high_u8x32, next2_low_u8x32, next2_high_u8x32, next3_low_u8x32, next3_high_u8x32;
    sz_utf8_forward_neighbours_lasx_(window_low_u8x32, window_high_u8x32, &next1_low_u8x32, &next1_high_u8x32,
                                     &next2_low_u8x32, &next2_high_u8x32, &next3_low_u8x32, &next3_high_u8x32);
    (void)next3_low_u8x32, (void)next3_high_u8x32; // Only next1/next2 feed the 2-/3-byte reconstruction.

    sz_u64_t const loaded_mask = result.loaded >= 64 ? ~(sz_u64_t)0 : (((sz_u64_t)1 << result.loaded) - 1);

    __m256i const continuation_mask_u8x32 = __lasx_xvreplgr2vr_b((char)0xC0);
    __m256i const continuation_pattern_u8x32 = __lasx_xvreplgr2vr_b((char)0x80);
    result.continuation =
        sz_utf8_mask_combine_lasx_(
            __lasx_xvseq_b(__lasx_xvand_v(window_low_u8x32, continuation_mask_u8x32), continuation_pattern_u8x32),
            __lasx_xvseq_b(__lasx_xvand_v(window_high_u8x32, continuation_mask_u8x32), continuation_pattern_u8x32)) &
        loaded_mask;
    result.codepoint_starts = loaded_mask & ~result.continuation;

    __m256i const two_byte_lead_mask_u8x32 = __lasx_xvreplgr2vr_b((char)0xE0);
    result.two_byte_starts =
        sz_utf8_mask_combine_lasx_(
            __lasx_xvseq_b(__lasx_xvand_v(window_low_u8x32, two_byte_lead_mask_u8x32), continuation_mask_u8x32),
            __lasx_xvseq_b(__lasx_xvand_v(window_high_u8x32, two_byte_lead_mask_u8x32), continuation_mask_u8x32)) &
        loaded_mask;
    __m256i const three_byte_lead_mask_u8x32 = __lasx_xvreplgr2vr_b((char)0xF0);
    result.three_byte_starts =
        sz_utf8_mask_combine_lasx_(
            __lasx_xvseq_b(__lasx_xvand_v(window_low_u8x32, three_byte_lead_mask_u8x32), two_byte_lead_mask_u8x32),
            __lasx_xvseq_b(__lasx_xvand_v(window_high_u8x32, three_byte_lead_mask_u8x32), two_byte_lead_mask_u8x32)) &
        loaded_mask;
    __m256i const four_byte_lead_mask_u8x32 = __lasx_xvreplgr2vr_b((char)0xF8);
    result.four_byte_starts =
        sz_utf8_mask_combine_lasx_(
            __lasx_xvseq_b(__lasx_xvand_v(window_low_u8x32, four_byte_lead_mask_u8x32), three_byte_lead_mask_u8x32),
            __lasx_xvseq_b(__lasx_xvand_v(window_high_u8x32, four_byte_lead_mask_u8x32), three_byte_lead_mask_u8x32)) &
        loaded_mask;

    __m256i const mask_five_low_bits_u8x32 = __lasx_xvreplgr2vr_b(0x1F);
    __m256i const mask_six_low_bits_u8x32 = __lasx_xvreplgr2vr_b(0x3F);
    __m256i const mask_two_low_bits_u8x32 = __lasx_xvreplgr2vr_b(0x03);
    __m256i const mask_four_low_bits_u8x32 = __lasx_xvreplgr2vr_b(0x0F);

    // 2-byte: codepoint = ((b0 & 0x1F) << 6) | (b1 & 0x3F); high = codepoint >> 8, low = codepoint & 0xFF.
    __m256i const high_two_byte_low_u8x32 = sz_utf8_srl8_lasx_(
        __lasx_xvand_v(window_low_u8x32, mask_five_low_bits_u8x32), 2, 0x07);
    __m256i const high_two_byte_high_u8x32 = sz_utf8_srl8_lasx_(
        __lasx_xvand_v(window_high_u8x32, mask_five_low_bits_u8x32), 2, 0x07);
    __m256i const low_two_byte_low_u8x32 = __lasx_xvor_v(
        __lasx_xvslli_h(__lasx_xvand_v(window_low_u8x32, mask_two_low_bits_u8x32), 6),
        __lasx_xvand_v(next1_low_u8x32, mask_six_low_bits_u8x32));
    __m256i const low_two_byte_high_u8x32 = __lasx_xvor_v(
        __lasx_xvslli_h(__lasx_xvand_v(window_high_u8x32, mask_two_low_bits_u8x32), 6),
        __lasx_xvand_v(next1_high_u8x32, mask_six_low_bits_u8x32));

    // 3-byte: codepoint = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F).
    __m256i const high_three_byte_low_u8x32 = __lasx_xvor_v(
        __lasx_xvslli_h(__lasx_xvand_v(window_low_u8x32, mask_four_low_bits_u8x32), 4),
        sz_utf8_srl8_lasx_(next1_low_u8x32, 2, 0x0F));
    __m256i const high_three_byte_high_u8x32 = __lasx_xvor_v(
        __lasx_xvslli_h(__lasx_xvand_v(window_high_u8x32, mask_four_low_bits_u8x32), 4),
        sz_utf8_srl8_lasx_(next1_high_u8x32, 2, 0x0F));
    __m256i const low_three_byte_low_u8x32 = __lasx_xvor_v(
        __lasx_xvslli_h(__lasx_xvand_v(next1_low_u8x32, mask_two_low_bits_u8x32), 6),
        __lasx_xvand_v(next2_low_u8x32, mask_six_low_bits_u8x32));
    __m256i const low_three_byte_high_u8x32 = __lasx_xvor_v(
        __lasx_xvslli_h(__lasx_xvand_v(next1_high_u8x32, mask_two_low_bits_u8x32), 6),
        __lasx_xvand_v(next2_high_u8x32, mask_six_low_bits_u8x32));

    // Blend 2-byte vs 3-byte per lane on the three-byte-start selector (`xvbitsel_v` selects on full-byte masks).
    __m256i const is_three_byte_low_select_u8x32 = sz_utf8_byte_mask_from_bits_lasx_(
        (sz_u32_t)result.three_byte_starts);
    __m256i const is_three_byte_high_select_u8x32 = sz_utf8_byte_mask_from_bits_lasx_(
        (sz_u32_t)(result.three_byte_starts >> 32));
    result.high_byte_low_u8x32 = __lasx_xvbitsel_v(high_two_byte_low_u8x32, high_three_byte_low_u8x32,
                                                   is_three_byte_low_select_u8x32);
    result.high_byte_high_u8x32 = __lasx_xvbitsel_v(high_two_byte_high_u8x32, high_three_byte_high_u8x32,
                                                    is_three_byte_high_select_u8x32);
    result.low_byte_low_u8x32 = __lasx_xvbitsel_v(low_two_byte_low_u8x32, low_three_byte_low_u8x32,
                                                  is_three_byte_low_select_u8x32);
    result.low_byte_high_u8x32 = __lasx_xvbitsel_v(low_two_byte_high_u8x32, low_three_byte_high_u8x32,
                                                   is_three_byte_high_select_u8x32);
    return result;
}

/** @brief  256-entry byte LUT addressed by a per-lane byte index in `[0,256)`: `result[lane] = table[index[lane]]` via
 *          a bounded scalar L1 walk (LASX has no gather). The LASX stand-in for the substrate `lut256` leaf. */
SZ_HELPER_AUTO __m256i sz_utf8_rune_lut256_scalar_lasx_(sz_u8_t const *table, __m256i index_u8x32) {
    sz_u256_vec_t index_vec, result_vec;
    index_vec.lasx = index_u8x32;
    for (int lane = 0; lane < 32; ++lane) result_vec.u8s[lane] = table[index_vec.u8s[lane]];
    return result_vec.lasx;
}

/** @brief  One nibble-cascade stage with a sub-256 selector: `result[lane] = table[selector[lane]*16 + within[lane]]`
 *          (0 when @p selector reaches @p tile_count). Each 16-byte row is double-broadcast into both 128-bit lanes with
 *          `xvpermi.q(row, row, 0x00)` and shuffled by @p within (a nibble, so `xvshuf.b` never crosses the seam), then
 *          blended in for the lanes whose @p selector picks that row - the LASX twin of the AVX2 cascade stage. */
SZ_HELPER_AUTO __m256i sz_utf8_rune_cascade_stage_lasx_( //
    sz_u8_t const *table, int tile_count, __m256i selector_u8x32, __m256i within_u8x32) {
    __m256i result_u8x32 = __lasx_xvreplgr2vr_b(0);
    for (int tile = 0; tile < tile_count; ++tile) {
        __m256i const row_raw_u8x32 = __lasx_xvld(table + (sz_size_t)tile * 16, 0); // low 128 = the 16-byte row
        __m256i const row_broadcast_u8x32 = __lasx_xvpermi_q(row_raw_u8x32, row_raw_u8x32, 0x00);
        __m256i const row_picked_u8x32 = __lasx_xvshuf_b(row_broadcast_u8x32, row_broadcast_u8x32, within_u8x32);
        __m256i const selector_match_u8x32 = __lasx_xvseq_b(selector_u8x32, __lasx_xvreplgr2vr_b((char)tile));
        result_u8x32 = __lasx_xvbitsel_v(result_u8x32, row_picked_u8x32, selector_match_u8x32);
    }
    return result_u8x32;
}

/** @brief  Class byte per lane from a page-compressed flat table: `page_lut[high]` selects one 256-byte page, then
 *          `flat[page * 256 + low]` per lane. LASX has no gather, so the leaf read is a bounded scalar L1 walk over the
 *          fused 16-bit index `(page << 8) | low` (the NEON strategy). Lanes whose page index reaches @p page_count
 *          return zero. */
SZ_HELPER_AUTO __m256i sz_utf8_rune_flat_lookup_lasx_( //
    sz_u8_t const *page_lut, sz_u8_t const *flat, int page_count, __m256i high_bytes_u8x32, __m256i low_bytes_u8x32) {
    sz_u256_vec_t high_vec, low_vec, result_vec;
    high_vec.lasx = high_bytes_u8x32;
    low_vec.lasx = low_bytes_u8x32;
    for (int lane = 0; lane < 32; ++lane) {
        sz_u32_t const page = page_lut[high_vec.u8s[lane]];
        result_vec.u8s[lane] = page < (sz_u32_t)page_count ? flat[(sz_size_t)page * 256 + low_vec.u8s[lane]] : 0;
    }
    return result_vec.lasx;
}

/** @brief  Popcount of a 64-bit lane mask via `xvpcnt.d` on a seeded lane (no scalar `popcount` builtin). */
SZ_HELPER_INLINE sz_size_t sz_utf8_rune_popcount64_lasx_(sz_u64_t value) {
    __m256i const seeded_u64x4 = __lasx_xvinsgr2vr_d(__lasx_xvreplgr2vr_d(0), (long long)value, 0);
    return (sz_size_t)__lasx_xvpickve2gr_du(__lasx_xvpcnt_d(seeded_u64x4), 0);
}

/** @brief  Popcount of an 8-bit value via a nibble table (no scalar `popcount` builtin). */
SZ_HELPER_INLINE int sz_utf8_rune_byte_popcount_lasx_(sz_u32_t byte) {
    static sz_u8_t const nibble_popcount[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
    return (int)nibble_popcount[byte & 0x0F] + (int)nibble_popcount[(byte >> 4) & 0x0F];
}

/** @brief  Left-pack the set lane indices (in [0, 64), ascending) of a 64-bit @p mask into @p out[0..popcount) via the
 *          existing 256-row @ref sz_utf8_pack8_lut_lasx_ + `xvshuf.b` (eight 8-bit groups), NOT a scalar `ctz` walk.
 *          @return the popcount of @p mask. */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_pack_boundaries_lasx_(sz_u64_t mask, sz_u8_t *out) {
    sz_u8_t const *lut = sz_utf8_pack8_lut_lasx_();
    static sz_u8_t const identity_bytes[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    __m256i const byte_identity_iota_u8x32 = __lasx_xvld(identity_bytes, 0);
    sz_size_t total = 0;
    for (int group = 0; group < 8; ++group) {
        sz_u32_t const group_mask = (sz_u32_t)((mask >> (group * 8)) & 0xFFu);
        int const group_base = group * 8;
        __m256i const shuffle_lookup_row_u8x32 = __lasx_xvld(lut + (sz_size_t)group_mask * 16, 0);
        __m256i const packed_indices_u8x32 = __lasx_xvshuf_b(byte_identity_iota_u8x32, byte_identity_iota_u8x32,
                                                             shuffle_lookup_row_u8x32);
        sz_u256_vec_t packed_bytes_vec;
        packed_bytes_vec.lasx = packed_indices_u8x32;
        int const count = sz_utf8_rune_byte_popcount_lasx_(group_mask);
        for (int k = 0; k < count; ++k) out[total++] = (sz_u8_t)(group_base + packed_bytes_vec.u8s[k]);
    }
    return total;
}

/** @brief  LASX forward drain - the `vpcompressb`-free twin of @ref sz_utf8_rune_drain_forward_. Emits one
 *          (start, length) per set boundary lane (ascending), honoring @p capacity and the carried previous-boundary
 *          via @p previous_io; bit-exact with the Ice Lake leaf. The set lanes are left-packed once (pack8 LUT), then
 *          streamed in waves of four u64 positions (`xvperm.w` shift + lane-0 carry seat via `xvinsgr2vr.d`, lengths via
 *          `xvsub.d`), with a scalar tail for the final partial wave. Count is `xvpcnt.d`, never a scalar `popcount`. */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_forward_lasx_( //
    sz_u64_t boundary, sz_size_t base, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *previous_io) {
    sz_size_t const boundary_count = sz_utf8_rune_popcount64_lasx_(boundary);
    sz_u64_t previous = (sz_u64_t)*previous_io;
    if (boundary_count == 0 || produced >= capacity) {
        *previous_io = (sz_size_t)previous;
        return produced;
    }

    sz_u512_vec_t indices;
    sz_utf8_rune_pack_boundaries_lasx_(boundary, indices.u8s);

    // Shift the four u64 positions up one lane (lane j <- position j-1), seating the carry at lane 0 afterwards: as
    // 32-bit words, {p0,p0, p0,p1, p2,p3, p4,p5} restated to u64 lanes [p0, p0, p1, p2].
    static sz_u32_t const shift_up_words[8] = {0, 0, 0, 1, 2, 3, 4, 5};
    __m256i const shift_up_u32x8 = __lasx_xvld(shift_up_words, 0);

    sz_size_t emitted = 0;
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const wave = sz_min_of_three(boundary_count - emitted, capacity - produced, 4);
        __m256i positions_u64x4 = __lasx_xvreplgr2vr_d(0);
        positions_u64x4 = __lasx_xvinsgr2vr_d(positions_u64x4, (long long)(base + indices.u8s[emitted + 0]), 0);
        if (wave >= 2)
            positions_u64x4 = __lasx_xvinsgr2vr_d(positions_u64x4, (long long)(base + indices.u8s[emitted + 1]), 1);
        if (wave >= 3)
            positions_u64x4 = __lasx_xvinsgr2vr_d(positions_u64x4, (long long)(base + indices.u8s[emitted + 2]), 2);
        if (wave >= 4)
            positions_u64x4 = __lasx_xvinsgr2vr_d(positions_u64x4, (long long)(base + indices.u8s[emitted + 3]), 3);
        __m256i segment_starts_u64x4 = __lasx_xvperm_w(positions_u64x4, shift_up_u32x8);
        segment_starts_u64x4 = __lasx_xvinsgr2vr_d(segment_starts_u64x4, (long long)previous, 0);
        __m256i const segment_lengths_u64x4 = __lasx_xvsub_d(positions_u64x4, segment_starts_u64x4);
        if (produced + 4 <= capacity && wave == 4) {
            __lasx_xvst(segment_starts_u64x4, starts + produced, 0);
            __lasx_xvst(segment_lengths_u64x4, lengths + produced, 0);
        }
        else {
            sz_size_t tail_starts[4], tail_lengths[4];
            __lasx_xvst(segment_starts_u64x4, tail_starts, 0);
            __lasx_xvst(segment_lengths_u64x4, tail_lengths, 0);
            for (sz_size_t lane = 0; lane < wave; ++lane)
                starts[produced + lane] = tail_starts[lane], lengths[produced + lane] = tail_lengths[lane];
        }
        produced += wave, emitted += wave;
        previous = (sz_u64_t)(starts[produced - 1] + lengths[produced - 1]);
    }
    *previous_io = (sz_size_t)previous;
    return produced;
}

#pragma endregion Word boundaries windowed substrate

#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_LASX_H_
