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
SZ_INTERNAL sz_u32_t sz_xvmovemask_b_utf8_lasx_(__m256i sign_extended) {
    __m256i collected = __lasx_xvmskltz_b(sign_extended);
    sz_u32_t low = (sz_u32_t)__lasx_xvpickve2gr_wu(collected, 0);
    sz_u32_t high = (sz_u32_t)__lasx_xvpickve2gr_wu(collected, 4);
    return (low & 0xFFFFu) | ((high & 0xFFFFu) << 16);
}

/** @brief  Per-lane logical right shift by 4 bits, keeping only the low nibble of every byte lane. The single shift
 *          amount the classifier needs is spelled out (LASX `xvsrli.h` requires an immediate; it shifts 16-bit lanes,
 *          so we mask back to byte width afterwards). */
SZ_INTERNAL __m256i sz_utf8_high_nibble_lasx_(__m256i value) {
    return __lasx_xvand_v(__lasx_xvsrli_h(value, 4), __lasx_xvreplgr2vr_b((char)0x0F));
}

/** @brief  Shift the whole 32-byte window left by one byte (lane `i` receives original lane `i + 1`), zero-filling the
 *          top. Built from an in-lane `xvbsrl.v` (per-128-bit down-shift) stitched with the cross-lane high half via
 *          `xvpermi.q` so byte 16 receives original byte 17 across the 128-bit seam. */
SZ_INTERNAL __m256i sz_utf8_next1_lasx_(__m256i window) {
    __m256i const high_to_low = __lasx_xvpermi_q(window, window, 0x31); // both 128-bit halves := original high half
    __m256i const in_lane = __lasx_xvbsrl_v(window, 1);                 // bytes down by 1 within each 128-bit half
    __m256i const carry = __lasx_xvbsll_v(high_to_low, 15);             // original high-half byte 0 -> lane 15
    return __lasx_xvor_v(in_lane, carry);
}

#pragma region Shuffle LUT left pack

/** @brief  256-entry shuffle LUT: row `m` left-packs the set-bit positions of the 8-bit mask `m` into the low bytes
 *          (value = bit index 0..7), in `.rodata` (no per-call rebuild). The caller reads only the leading
 *          `popcount(m)` lanes; the 0x80 fill marks the unused tail. 16 bytes per row so one `xvld` brings a row into
 *          a 128-bit `xvshuf.b` source lane. */
SZ_INTERNAL sz_u8_t const *sz_utf8_pack8_lut_lasx_(void) {
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
SZ_INTERNAL sz_size_t sz_utf8_pack_indices_lasx_(sz_u32_t mask, sz_u8_t *out) {
    sz_u8_t const *lut = sz_utf8_pack8_lut_lasx_();
    // Lane-local byte identity {0..15, 0..15}: each 128-bit lane shuffles its own 0..15 identity by the LUT row, so
    // the packed values are the within-half bit positions; we add the half base to recover absolute offsets.
    static sz_u8_t const identity_bytes[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    __m256i const identity = __lasx_xvld(identity_bytes, 0);
    sz_size_t total = 0;
    for (int half = 0; half < 4; ++half) {
        sz_u32_t const half_mask = (mask >> (half * 8)) & 0xFFu;
        int const half_base = half * 8;
        __m256i const row = __lasx_xvld(lut + (sz_size_t)half_mask * 16, 0);
        __m256i const packed = __lasx_xvshuf_b(identity, identity, row); // low 128-bit lane holds the packed locals
        sz_u8_t packed_bytes[32];
        __lasx_xvst(packed, packed_bytes, 0);
        sz_size_t const count = (sz_size_t)sz_u32_popcount(half_mask);
        for (sz_size_t k = 0; k < count; ++k) out[total++] = (sz_u8_t)(half_base + packed_bytes[k]);
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
SZ_INTERNAL sz_size_t sz_utf8_rune_drain_lasx_( //
    __m256i window, sz_u32_t emit_starts, sz_u32_t ill_formed, __m256i consumed_length, int has_three, int has_four,
    sz_size_t emit_count, sz_rune_t *runes, sz_size_t capacity, sz_size_t *consumed_bytes) {

    // Left-pack the emitted-start byte-offsets via the shuffle-LUT (one `xvshuf.b` per 8-bit mask half).
    sz_u8_t start_offsets[32];
    sz_utf8_pack_indices_lasx_(emit_starts, start_offsets);

    // Broadcast the window's low/high 128-bit halves into both lanes so a single `xvshuf.b` gathers any byte 0..31:
    // index 0..15 selects the low half (`window_lo`), 16..31 the high half (`window_hi`), in EVERY 128-bit lane.
    __m256i const window_lo = __lasx_xvpermi_q(window, window, 0x00);
    __m256i const window_hi = __lasx_xvpermi_q(window, window, 0x11);
    // The per-lane maximal-subpart length, read at the packed start offsets for the resume cursor.
    sz_u8_t consumed_bytes_lanes[32];
    __lasx_xvst(consumed_length, consumed_bytes_lanes, 0);

    __m256i const c1f = __lasx_xvreplgr2vr_w(0x1F), c3f = __lasx_xvreplgr2vr_w(0x3F);
    __m256i const c0f = __lasx_xvreplgr2vr_w(0x0F), c07 = __lasx_xvreplgr2vr_w(0x07);
    __m256i const cc0 = __lasx_xvreplgr2vr_w(0xC0), ce0 = __lasx_xvreplgr2vr_w(0xE0), cf0 = __lasx_xvreplgr2vr_w(0xF0);
    __m256i const replacement = __lasx_xvreplgr2vr_w((int)sz_rune_replacement_k);

    sz_size_t const want = emit_count < capacity ? emit_count : capacity;
    sz_size_t produced = 0;
    // One 4-codepoint block per outer step: `xvsllwil` widens the low 4 bytes of each 128-bit lane, so the four
    // gathered bytes placed in lanes [0,4) land in word lanes [0,4). Decode only the K emitted starts.
    for (sz_size_t block_start = 0; block_start < want; block_start += 4) {
        // Load up to 4 packed start offsets into the low bytes of an index vector. Trailing lanes are filled with the
        // last valid offset (an in-window index): `xvshuf.b` keys off the low bits only and does NOT zero a high-bit
        // index, so a defined in-range fill keeps those (unstored) lanes from gathering across the 128-bit seam.
        sz_size_t const block_lanes = (want - block_start) < 4 ? (want - block_start) : 4;
        sz_u8_t index_lanes[32];
        for (int lane = 0; lane < 32; ++lane) index_lanes[lane] = start_offsets[block_start];
        for (sz_size_t lane = 0; lane < block_lanes; ++lane) index_lanes[lane] = start_offsets[block_start + lane];
        __m256i const index = __lasx_xvld(index_lanes, 0);

        // Gather lead + up to three trailing bytes per codepoint at the packed start offsets (one `xvshuf.b` each over
        // the broadcast window). For a decodable well-formed sequence the trailing slots are in-window by construction.
        __m256i const lead = __lasx_xvshuf_b(window_hi, window_lo, index);
        __m256i const byte1 = __lasx_xvshuf_b(window_hi, window_lo, __lasx_xvadd_b(index, __lasx_xvreplgr2vr_b(1)));
        __m256i const byte2 = has_three ? __lasx_xvshuf_b(window_hi, window_lo,
                                                          __lasx_xvadd_b(index, __lasx_xvreplgr2vr_b(2)))
                                        : __lasx_xvreplgr2vr_b(0);
        __m256i const byte3 = has_four ? __lasx_xvshuf_b(window_hi, window_lo,
                                                         __lasx_xvadd_b(index, __lasx_xvreplgr2vr_b(3)))
                                       : __lasx_xvreplgr2vr_b(0);

        // Widen the low 8 byte lanes (this block's gathered bytes) to 8x 32-bit words in the low 128-bit half via the
        // byte->half->word ladder; only the low half is consumed so no cross-128-bit shuffle is needed.
        __m256i const b0 = __lasx_xvsllwil_wu_hu(__lasx_xvsllwil_hu_bu(lead, 0), 0);
        __m256i const b1 = __lasx_xvsllwil_wu_hu(__lasx_xvsllwil_hu_bu(byte1, 0), 0);
        __m256i const b2 = __lasx_xvsllwil_wu_hu(__lasx_xvsllwil_hu_bu(byte2, 0), 0);
        __m256i const b3 = __lasx_xvsllwil_wu_hu(__lasx_xvsllwil_hu_bu(byte3, 0), 0);

        __m256i codepoints = b0; // ASCII keeps the lead.
        // 2-byte: lead in [C0,E0): ((b0 & 0x1F) << 6) | (b1 & 0x3F).
        __m256i const two = __lasx_xvor_v(__lasx_xvslli_w(__lasx_xvand_v(b0, c1f), 6), __lasx_xvand_v(b1, c3f));
        __m256i const is_two = __lasx_xvandn_v(__lasx_xvsle_wu(ce0, b0), __lasx_xvsle_wu(cc0, b0));
        codepoints = __lasx_xvbitsel_v(codepoints, two, is_two);
        if (has_three) {
            // 3-byte: lead in [E0,F0): ((b0&0x0F)<<12)|((b1&0x3F)<<6)|(b2&0x3F).
            __m256i const three = __lasx_xvor_v(__lasx_xvor_v(__lasx_xvslli_w(__lasx_xvand_v(b0, c0f), 12),
                                                              __lasx_xvslli_w(__lasx_xvand_v(b1, c3f), 6)),
                                                __lasx_xvand_v(b2, c3f));
            __m256i const is_three = __lasx_xvandn_v(__lasx_xvsle_wu(cf0, b0), __lasx_xvsle_wu(ce0, b0));
            codepoints = __lasx_xvbitsel_v(codepoints, three, is_three);
            if (has_four) {
                // 4-byte: lead >= F0: ((b0&7)<<18)|((b1&0x3F)<<12)|((b2&0x3F)<<6)|(b3&0x3F).
                __m256i const four = __lasx_xvor_v(
                    __lasx_xvor_v(__lasx_xvslli_w(__lasx_xvand_v(b0, c07), 18),
                                  __lasx_xvslli_w(__lasx_xvand_v(b1, c3f), 12)),
                    __lasx_xvor_v(__lasx_xvslli_w(__lasx_xvand_v(b2, c3f), 6), __lasx_xvand_v(b3, c3f)));
                __m256i const is_four = __lasx_xvsle_wu(cf0, b0);
                codepoints = __lasx_xvbitsel_v(codepoints, four, is_four);
            }
        }

        // Overwrite every ill-formed lane in this block with U+FFFD. Build the per-word ill-formed selector from the
        // compacted `ill_formed` mask bits for this block (word j set <=> the (block_start+j)-th emitted start is ill).
        sz_u32_t ill_lane_mask[8] = {0};
        for (sz_size_t lane = 0; lane < block_lanes; ++lane) {
            sz_size_t const emitted_index = block_start + lane;
            ill_lane_mask[lane] = ((ill_formed >> start_offsets[emitted_index]) & 1u) ? ~0u : 0u;
        }
        __m256i const ill_block = __lasx_xvld(ill_lane_mask, 0);
        codepoints = __lasx_xvbitsel_v(codepoints, replacement, ill_block);

        // Store the low `block_lanes` words (the four decoded codepoints live in word lanes [0,4)). Extract each via
        // `xvpickve2gr.wu` to a scalar register and write through `runes` so the result is a plain integer store the
        // caller's same-typed load reliably observes.
        runes[produced + 0] = (sz_rune_t)__lasx_xvpickve2gr_wu(codepoints, 0);
        if (block_lanes >= 2) runes[produced + 1] = (sz_rune_t)__lasx_xvpickve2gr_wu(codepoints, 1);
        if (block_lanes >= 3) runes[produced + 2] = (sz_rune_t)__lasx_xvpickve2gr_wu(codepoints, 2);
        if (block_lanes >= 4) runes[produced + 3] = (sz_rune_t)__lasx_xvpickve2gr_wu(codepoints, 3);
        produced += block_lanes;
    }

    // Resume cursor delta = end of the last emitted start = its byte offset + its maximal-subpart length.
    sz_u8_t const last_off = start_offsets[produced - 1];
    sz_u8_t const last_length = consumed_bytes_lanes[last_off];
    *consumed_bytes = (sz_size_t)last_off + (sz_size_t)last_length;
    return produced;
}

/**
 *  @brief  Decode one <=32-byte window of @p text into dense UTF-32 @p runes by the uniform classify -> per-lane
 *          well-formed + orphan promotion -> compress emitted starts -> gather -> width-blend -> blend U+FFFD path,
 *          emitting at most @p runes_capacity runes and returning the resume cursor. The decode is TOTAL: clean and
 *          dirty bytes are handled in-vector, one U+FFFD per maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C),
 *          bit-exact with @ref sz_utf8_decode_serial. Mirrors `sz_utf8_decode_once_icelake_`; the step
 *          declines (`*runes_unpacked == 0`, cursor unchanged) ONLY when the first lead's declared sequence crosses
 *          the window edge (a boundary truncation), which the public entry finalizes without a serial re-decode.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_decode_once_lasx_( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked) {

    sz_size_t const chunk = length < 32 ? length : 32;
    sz_u32_t const loaded_mask = chunk >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << chunk) - 1u);

    // Load <=32 bytes; tail lanes beyond `chunk` are zeroed via a bounded copy so we never read past `length`.
    sz_u8_t staging[64] = {0}; // padded so the per-block widen never runs past the end.
    {
        sz_size_t i = 0;
        for (; i < chunk; ++i) staging[i] = ((sz_u8_t const *)text)[i];
    }
    __m256i const window = __lasx_xvld(staging, 0);

    // ASCII fast lane: a window with no high bit set widens directly. `xvmskltz_b` flags lanes with the sign bit set.
    sz_u32_t const high_bits = sz_xvmovemask_b_utf8_lasx_(window) & loaded_mask;
    if (high_bits == 0) {
        sz_size_t const runes_to_unpack = chunk < runes_capacity ? chunk : runes_capacity;
        sz_size_t group = 0;
        for (; group + 4 <= runes_to_unpack; group += 4) {
            __m256i const widened = __lasx_xvsllwil_wu_hu(__lasx_xvsllwil_hu_bu(__lasx_xvld(staging + group, 0), 0), 0);
            __lasx_xvstelm_d(widened, runes + group, 0, 0);
            __lasx_xvstelm_d(widened, runes + group, 8, 1);
        }
        for (; group < runes_to_unpack; ++group) runes[group] = (sz_rune_t)staging[group];
        *runes_unpacked = runes_to_unpack;
        return text + runes_to_unpack;
    }

    // Single-source classification: a high-nibble lookup gives per-lane byte length {1x12,2,2,3,4}; both the validity
    // gate and the width-blend derive from it, so a lead and its length can never disagree.
    sz_u8_t const length_lut_bytes[32] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    __m256i const length_lut = __lasx_xvld(length_lut_bytes, 0);
    __m256i const high_nibble = sz_utf8_high_nibble_lasx_(window);
    __m256i const lengths = __lasx_xvshuf_b(length_lut, length_lut, high_nibble); // 128-bit-lane LUT, table duplicated

    // start = non-continuation lane: a byte whose (b & 0xC0) != 0x80. Continuation bytes are signed -128..-65.
    __m256i const cont_pred = __lasx_xvseq_b(__lasx_xvand_v(window, __lasx_xvreplgr2vr_b((char)0xC0)),
                                             __lasx_xvreplgr2vr_b((char)0x80));
    sz_u32_t const continuation_bits = sz_xvmovemask_b_utf8_lasx_(cont_pred) & loaded_mask;
    sz_u32_t const starts_bits = (~continuation_bits) & loaded_mask;

    // Defer every start whose declared sequence would reach past the window; the FIRST overrunning start bounds the
    // decodable prefix (well-formed text has only the trailing truncation, but a malformed `E0 C0` overruns earlier).
    sz_u8_t const lane_identity_raw[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    sz_u8_t const lane_high_bias[32] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16};
    __m256i const lane_identity_bytes = __lasx_xvadd_b(__lasx_xvld(lane_identity_raw, 0),
                                                       __lasx_xvld(lane_high_bias, 0));
    __m256i const sequence_end = __lasx_xvadd_b(lane_identity_bytes, lengths);
    // sequence_end > chunk, restricted to starts. Compare unsigned bytes.
    sz_u32_t const past_end =
        sz_xvmovemask_b_utf8_lasx_(__lasx_xvslt_bu(__lasx_xvreplgr2vr_b((char)chunk), sequence_end)) & starts_bits;
    sz_size_t const decodable_end = past_end ? (sz_size_t)sz_u32_ctz(past_end) : chunk;
    sz_u32_t const decodable_mask = decodable_end >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << decodable_end) - 1u);

    // Per-length start masks from the LUT length (NOT the lead bit-pattern): a stray lead such as 0xF8 maps to length 4
    // in the LUT, so it MUST be classed as a length-4 start here for the bad-lead gate to reject it.
    sz_u32_t const len2 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(lengths, __lasx_xvreplgr2vr_b(2))) & starts_bits;
    sz_u32_t const len3 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(lengths, __lasx_xvreplgr2vr_b(3))) & starts_bits;
    sz_u32_t const len4 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(lengths, __lasx_xvreplgr2vr_b(4))) & starts_bits;
    sz_u32_t const len_ge2 = len2 | len3 | len4;
    sz_u32_t const len_ge3 = len3 | len4;
    sz_u32_t const len_ge4 = len4;
    sz_u32_t const len1 = starts_bits & ~len_ge2;

    // Bad lead: 0xC0/0xC1 (overlong 2-byte by the LUT) or 0xF5..0xFF (out of range, length-4 by the LUT).
    sz_u32_t const lead_lt_c2 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvslt_bu(window, __lasx_xvreplgr2vr_b((char)0xC2)));
    sz_u32_t const lead_gt_f4 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvslt_bu(__lasx_xvreplgr2vr_b((char)0xF4), window));
    sz_u32_t const bad_lead = ((len2 & lead_lt_c2) | (len4 & lead_gt_f4)) & starts_bits;

    // First-continuation range violations for E0/ED/F0/F4 (overlong 3/4-byte, surrogate, > U+10FFFF), computed only
    // when a 3/4-byte lead is present; for any other lead this mask is empty, so `b1_range_ok` keeps the lane.
    int const has_three = len_ge3 != 0;
    int const has_four = len_ge4 != 0;
    sz_u32_t overlong_or_surrogate_or_range = 0;
    if (has_three || has_four) {
        __m256i const next1 = sz_utf8_next1_lasx_(window);
        sz_u32_t const is_e0 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, __lasx_xvreplgr2vr_b((char)0xE0)));
        sz_u32_t const is_ed = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, __lasx_xvreplgr2vr_b((char)0xED)));
        sz_u32_t const is_f0 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, __lasx_xvreplgr2vr_b((char)0xF0)));
        sz_u32_t const is_f4 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, __lasx_xvreplgr2vr_b((char)0xF4)));
        sz_u32_t const n1_lt_a0 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvslt_bu(next1, __lasx_xvreplgr2vr_b((char)0xA0)));
        sz_u32_t const n1_ge_a0 = (~n1_lt_a0) & loaded_mask;
        sz_u32_t const n1_lt_90 = sz_xvmovemask_b_utf8_lasx_(__lasx_xvslt_bu(next1, __lasx_xvreplgr2vr_b((char)0x90)));
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
    if (emit_count == 0) { return *runes_unpacked = 0, text; } // Nothing decodable -> window-edge finalize in driver.
    sz_u32_t const ill_formed = emit_starts & ~well_formed;    // Orphans are continuations -> never well-formed.

    // Per-lane consumed length = 1 + step2 + step3 + step4 (each step adds one continuation byte to the subpart).
    __m256i consumed_length = __lasx_xvreplgr2vr_b(1);
    __m256i const one_b = __lasx_xvreplgr2vr_b(1);
    {
        // Smear each 32-bit step mask to a 0xFF-per-lane byte selector so the conditional add stays branchless.
        sz_u8_t step_bytes[32];
        for (int lane = 0; lane < 32; ++lane) step_bytes[lane] = (sz_u8_t)((step2 >> lane) & 1u ? 0xFF : 0);
        consumed_length = __lasx_xvadd_b(consumed_length, __lasx_xvand_v(__lasx_xvld(step_bytes, 0), one_b));
        for (int lane = 0; lane < 32; ++lane) step_bytes[lane] = (sz_u8_t)((step3 >> lane) & 1u ? 0xFF : 0);
        consumed_length = __lasx_xvadd_b(consumed_length, __lasx_xvand_v(__lasx_xvld(step_bytes, 0), one_b));
        for (int lane = 0; lane < 32; ++lane) step_bytes[lane] = (sz_u8_t)((step4 >> lane) & 1u ? 0xFF : 0);
        consumed_length = __lasx_xvadd_b(consumed_length, __lasx_xvand_v(__lasx_xvld(step_bytes, 0), one_b));
    }

    sz_size_t consumed = 0;
    sz_size_t const produced = sz_utf8_rune_drain_lasx_(window, emit_starts, ill_formed, consumed_length, has_three,
                                                        has_four, emit_count, runes, runes_capacity, &consumed);
    *runes_unpacked = produced;
    return text + consumed;
}

SZ_PUBLIC sz_cptr_t sz_utf8_decode_lasx(        //
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
SZ_INTERNAL void sz_utf8_iterate_store_group_lasx_(__m256i values_u64x4, sz_size_t group, sz_size_t *destination) {
    if (group == 4) { __lasx_xvst(values_u64x4, destination, 0); }
    else {
        __lasx_xvstelm_d(values_u64x4, destination, 0, 0);
        if (group >= 2) __lasx_xvstelm_d(values_u64x4, destination, 8, 1);
        if (group >= 3) __lasx_xvstelm_d(values_u64x4, destination, 16, 2);
    }
}

SZ_PUBLIC sz_size_t sz_utf8_count_lasx(sz_cptr_t text, sz_size_t length) {
    __m256i continuation_mask_vec = __lasx_xvreplgr2vr_b((char)0xC0);
    __m256i continuation_pattern_vec = __lasx_xvreplgr2vr_b((char)0x80);
    __m256i const one_vec = __lasx_xvreplgr2vr_b(1);
    __m256i const zero_vec = __lasx_xvreplgr2vr_b(0);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Keep a per-lane vector accumulator across the loop and reduce ONCE at the end, instead of doing a
    // movemask + scalar `popcount` per block. A start byte is any byte where `(b & 0xC0) != 0x80`; we map
    // that to a 0/1 byte and fold adjacent pairs into a 16x u16 accumulator via `xvhaddw_hu_bu`. Each u16
    // lane grows by at most 2 per block, so 65535/2 = 32767 blocks fit before overflow; we flush the u16
    // lanes into a 4x u64 accumulator every 32000 blocks to stay safe (rare for realistic inputs).
    __m256i sums_d = zero_vec;
    __m256i sums_h = zero_vec;
    sz_size_t blocks_since_flush = 0;
    while (length >= 32) {
        __m256i text_vec = __lasx_xvld(text_u8, 0);
        // Extract top 2 bits of each byte; continuation bytes equal 0x80. `is_start` is 0xFF for starts.
        __m256i headers = __lasx_xvand_v(text_vec, continuation_mask_vec);
        __m256i is_cont = __lasx_xvseq_b(headers, continuation_pattern_vec); // 0xFF where continuation
        // `andn` style: 1 where start, 0 where continuation. (~is_cont) & 1.
        __m256i is_start = __lasx_xvandn_v(is_cont, one_vec); // (~is_cont) & one
        sums_h = __lasx_xvadd_h(sums_h, __lasx_xvhaddw_hu_bu(is_start, is_start));
        if (++blocks_since_flush == 32000) {
            __m256i widened_w = __lasx_xvhaddw_wu_hu(sums_h, sums_h);
            sums_d = __lasx_xvadd_d(sums_d, __lasx_xvhaddw_du_wu(widened_w, widened_w));
            sums_h = zero_vec;
            blocks_since_flush = 0;
        }
        text_u8 += 32, length -= 32;
    }
    {
        __m256i widened_w = __lasx_xvhaddw_wu_hu(sums_h, sums_h);
        sums_d = __lasx_xvadd_d(sums_d, __lasx_xvhaddw_du_wu(widened_w, widened_w));
    }
    char_count += (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 0) + (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 1) +
                  (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 2) + (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 3);

    char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

/** @brief  Same block logic as `sz_utf8_count_lasx`, but locating the Nth start byte. LASX has no `PDEP`, so the
 *          start-byte bitmask is fed to `sz_u32_nth_set_bit` to land on the Nth set bit. */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_lasx(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    __m256i continuation_mask_vec = __lasx_xvreplgr2vr_b((char)0xC0);
    __m256i continuation_pattern_vec = __lasx_xvreplgr2vr_b((char)0x80);
    __m256i const one_vec = __lasx_xvreplgr2vr_b(1);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    sz_u256_vec_t text_vec, headers_vec;
    while (length >= 32) {
        text_vec.lasx = __lasx_xvld(text_u8, 0);
        headers_vec.lasx = __lasx_xvand_v(text_vec.lasx, continuation_mask_vec);
        __m256i is_continuation = __lasx_xvseq_b(headers_vec.lasx, continuation_pattern_vec);
        sz_u32_t start_byte_mask = ~sz_xvmovemask_b_utf8_lasx_(is_continuation);

        // Count start bytes in-vector: each start lane carries a 0x01, so `xvpcnt_d` sums one bit per such lane.
        __m256i is_start = __lasx_xvandn_v(is_continuation, one_vec);
        __m256i start_byte_counts = __lasx_xvpcnt_d(is_start);
        sz_size_t start_byte_count =
            (sz_size_t)(__lasx_xvpickve2gr_du(start_byte_counts, 0) + __lasx_xvpickve2gr_du(start_byte_counts, 1) +
                        __lasx_xvpickve2gr_du(start_byte_counts, 2) + __lasx_xvpickve2gr_du(start_byte_counts, 3));

        if (n >= start_byte_count) {
            n -= start_byte_count, text_u8 += 32, length -= 32;
            continue;
        }
        return (sz_cptr_t)(text_u8 + sz_u32_nth_set_bit(start_byte_mask, n));
    }

    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

/*  UAX-29 word boundary detection (forward & reverse).
 *
 *  The stateful look-around sub-rules stay in the serial reference; LASX accelerates the common case of long
 *  ASCII-letter or ASCII-digit runs whose interior positions can never be boundaries (WB5 / WB8). Skipping only
 *  proven non-boundaries keeps the returned pointer and `boundary_width` byte-for-byte identical to `_serial`. */
#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_LASX_H_
