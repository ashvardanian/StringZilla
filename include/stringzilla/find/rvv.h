/**
 *  @brief RISC-V Vector (RVV 1.0) backend for find.
 *  @file include/stringzilla/find/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_RVV_H_
#define STRINGZILLA_FIND_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/find/serial.h"
#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

/*  Find the first byte equal to `n[0]`. Length-agnostic `vsetvl` loop: on each strip we build the
 *  equality mask and use `vfirst` (the clean replacement for movemask+ctz) to get the first match
 *  index. No scalar tail required. */
SZ_PUBLIC sz_cptr_t sz_find_byte_rvv(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u8_t const *h_u8 = (sz_u8_t const *)h;
    sz_u8_t needle = *(sz_u8_t const *)n;
    while (h_length) {
        size_t vl = __riscv_vsetvl_e8m8(h_length);
        vuint8m8_t h_vec = __riscv_vle8_v_u8m8(h_u8, vl);
        vbool1_t eq_mask = __riscv_vmseq_vx_u8m8_b1(h_vec, needle, vl);
        long match_index = __riscv_vfirst_m_b1(eq_mask, vl);
        if (match_index >= 0) return (sz_cptr_t)(h_u8 + match_index);
        h_u8 += vl, h_length -= vl;
    }
    return SZ_NULL_CHAR;
}

/*  Find the last byte equal to `n[0]`. We scan strips from the end. Within a strip we need the
 *  last matching lane; RVV has no "last set bit" reduction, so we reverse the strip with
 *  `vrgather` (index `vl-1-i`) and take `vfirst` of the reversed mask. */
SZ_PUBLIC sz_cptr_t sz_rfind_byte_rvv(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u8_t const *h_u8 = (sz_u8_t const *)h;
    sz_u8_t needle = *(sz_u8_t const *)n;
    while (h_length) {
        // Cap the strip at 256 elements so byte-wide reverse-gather indices stay in [0, 255].
        size_t strip_length = h_length < 256 ? h_length : 256;
        size_t vl = __riscv_vsetvl_e8m8(strip_length);
        sz_u8_t const *strip = h_u8 + h_length - vl;
        vuint8m8_t h_vec = __riscv_vle8_v_u8m8(strip, vl);
        // Reverse the strip: out[i] = in[vl-1-i], so `vfirst` returns the highest match.
        vuint8m8_t iota_vec = __riscv_vid_v_u8m8(vl);
        vuint8m8_t reverse_index_vec = __riscv_vrsub_vx_u8m8(iota_vec, (sz_u8_t)(vl - 1), vl);
        vuint8m8_t reversed_vec = __riscv_vrgather_vv_u8m8(h_vec, reverse_index_vec, vl);
        vbool1_t eq_mask = __riscv_vmseq_vx_u8m8_b1(reversed_vec, needle, vl);
        long match_index = __riscv_vfirst_m_b1(eq_mask, vl);
        if (match_index >= 0) return (sz_cptr_t)(strip + (vl - 1 - match_index));
        h_length -= vl;
    }
    return SZ_NULL_CHAR;
}

/*  Per-strip byteset membership mask. Mirrors the NEON `_u8s[c>>3] & (1<<(c&7))` formulation,
 *  using `vrgather` to index the 32-byte set and a variable shift to build the per-lane bit. */
SZ_INTERNAL vbool1_t sz_find_byteset_rvv_mask_(vuint8m8_t h_vec, vuint8m8_t set_vec, size_t vl) {
    vuint8m8_t byte_index_vec = __riscv_vsrl_vx_u8m8(h_vec, 3, vl);   // c >> 3, in [0, 31]
    vuint8m8_t bit_position_vec = __riscv_vand_vx_u8m8(h_vec, 7, vl); // c & 7
    vuint8m8_t one_vec = __riscv_vmv_v_x_u8m8(1, vl);
    vuint8m8_t bit_mask_vec = __riscv_vsll_vv_u8m8(one_vec, bit_position_vec, vl); // 1 << (c & 7)
    vuint8m8_t set_bytes_vec = __riscv_vrgather_vv_u8m8(set_vec, byte_index_vec, vl);
    vuint8m8_t anded_vec = __riscv_vand_vv_u8m8(set_bytes_vec, bit_mask_vec, vl);
    return __riscv_vmsne_vx_u8m8_b1(anded_vec, 0, vl);
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_rvv(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    sz_u8_t const *h_u8 = (sz_u8_t const *)h;
    // The set spans 32 bytes; load it into a `u8m8` group (needs `VLEN >= 4`, always true).
    size_t set_vl = __riscv_vsetvl_e8m8(32);
    if (set_vl < 32) return sz_find_byteset_serial(h, h_length, set);
    vuint8m8_t set_vec = __riscv_vle8_v_u8m8(&set->_u8s[0], 32);
    while (h_length) {
        size_t vl = __riscv_vsetvl_e8m8(h_length);
        vuint8m8_t h_vec = __riscv_vle8_v_u8m8(h_u8, vl);
        vbool1_t match_mask = sz_find_byteset_rvv_mask_(h_vec, set_vec, vl);
        long match_index = __riscv_vfirst_m_b1(match_mask, vl);
        if (match_index >= 0) return (sz_cptr_t)(h_u8 + match_index);
        h_u8 += vl, h_length -= vl;
    }
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_rvv(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    sz_u8_t const *h_u8 = (sz_u8_t const *)h;
    size_t set_vl = __riscv_vsetvl_e8m8(32);
    if (set_vl < 32) return sz_rfind_byteset_serial(h, h_length, set);
    vuint8m8_t set_vec = __riscv_vle8_v_u8m8(&set->_u8s[0], 32);
    while (h_length) {
        size_t strip_length = h_length < 256 ? h_length : 256;
        size_t vl = __riscv_vsetvl_e8m8(strip_length);
        sz_u8_t const *strip = h_u8 + h_length - vl;
        vuint8m8_t h_vec = __riscv_vle8_v_u8m8(strip, vl);
        vuint8m8_t iota_vec = __riscv_vid_v_u8m8(vl);
        vuint8m8_t reverse_index_vec = __riscv_vrsub_vx_u8m8(iota_vec, (sz_u8_t)(vl - 1), vl);
        vuint8m8_t reversed_vec = __riscv_vrgather_vv_u8m8(h_vec, reverse_index_vec, vl);
        vbool1_t match_mask = sz_find_byteset_rvv_mask_(reversed_vec, set_vec, vl);
        long match_index = __riscv_vfirst_m_b1(match_mask, vl);
        if (match_index >= 0) return (sz_cptr_t)(strip + (vl - 1 - match_index));
        h_length -= vl;
    }
    return SZ_NULL_CHAR;
}

/*  Substring search. We follow the NEON strategy: pick three "anomaly" offsets in the needle,
 *  scan strips of candidate start positions comparing those three bytes vectorized, and verify
 *  full matches with `sz_equal_rvv`. The scan window slides so each strip covers `vl` candidate
 *  starts; the residual is handled by the serial backend. */
SZ_PUBLIC sz_cptr_t sz_find_rvv(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_rvv(h, h_length, n);

    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);
    sz_u8_t n_first = ((sz_u8_t const *)n)[offset_first];
    sz_u8_t n_mid = ((sz_u8_t const *)n)[offset_mid];
    sz_u8_t n_last = ((sz_u8_t const *)n)[offset_last];

    sz_u8_t const *h_u8 = (sz_u8_t const *)h;
    // Number of candidate start positions is `h_length - n_length + 1`.
    sz_size_t candidates = h_length - n_length + 1;
    sz_size_t position = 0;
    while (position < candidates) {
        // Cap so per-lane index values used for bit-clearing stay within u8 range.
        sz_size_t remaining = candidates - position;
        size_t strip_length = remaining < 256 ? remaining : 256;
        size_t vl = __riscv_vsetvl_e8m8(strip_length);
        vuint8m8_t first_vec = __riscv_vle8_v_u8m8(h_u8 + position + offset_first, vl);
        vuint8m8_t mid_vec = __riscv_vle8_v_u8m8(h_u8 + position + offset_mid, vl);
        vuint8m8_t last_vec = __riscv_vle8_v_u8m8(h_u8 + position + offset_last, vl);
        vbool1_t first_mask = __riscv_vmseq_vx_u8m8_b1(first_vec, n_first, vl);
        vbool1_t mid_mask = __riscv_vmseq_vx_u8m8_b1(mid_vec, n_mid, vl);
        vbool1_t last_mask = __riscv_vmseq_vx_u8m8_b1(last_vec, n_last, vl);
        vbool1_t match_mask = __riscv_vmand_mm_b1(__riscv_vmand_mm_b1(first_mask, mid_mask, vl), last_mask, vl);
        // Iterate over set bits within this strip.
        for (long match_index = __riscv_vfirst_m_b1(match_mask, vl); match_index >= 0;) {
            sz_size_t candidate = position + (sz_size_t)match_index;
            if (sz_equal_rvv((sz_cptr_t)(h_u8 + candidate), n, n_length)) return (sz_cptr_t)(h_u8 + candidate);
            // Clear the consumed bit and re-query.
            match_mask = __riscv_vmandn_mm_b1(
                match_mask, __riscv_vmseq_vx_u8m8_b1(__riscv_vid_v_u8m8(vl), (sz_u8_t)match_index, vl), vl);
            match_index = __riscv_vfirst_m_b1(match_mask, vl);
        }
        position += vl;
    }
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_rvv(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_rfind_byte_rvv(h, h_length, n);

    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);
    sz_u8_t n_first = ((sz_u8_t const *)n)[offset_first];
    sz_u8_t n_mid = ((sz_u8_t const *)n)[offset_mid];
    sz_u8_t n_last = ((sz_u8_t const *)n)[offset_last];

    sz_u8_t const *h_u8 = (sz_u8_t const *)h;
    sz_size_t candidates = h_length - n_length + 1; // start positions 0 .. candidates-1
    sz_size_t remaining = candidates;
    while (remaining) {
        size_t strip_length = remaining < 256 ? remaining : 256;
        size_t vl = __riscv_vsetvl_e8m8(strip_length);
        sz_size_t base = remaining - vl; // strip covers candidate starts [base, base+vl)
        vuint8m8_t first_vec = __riscv_vle8_v_u8m8(h_u8 + base + offset_first, vl);
        vuint8m8_t mid_vec = __riscv_vle8_v_u8m8(h_u8 + base + offset_mid, vl);
        vuint8m8_t last_vec = __riscv_vle8_v_u8m8(h_u8 + base + offset_last, vl);
        vbool1_t first_mask = __riscv_vmseq_vx_u8m8_b1(first_vec, n_first, vl);
        vbool1_t mid_mask = __riscv_vmseq_vx_u8m8_b1(mid_vec, n_mid, vl);
        vbool1_t last_mask = __riscv_vmseq_vx_u8m8_b1(last_vec, n_last, vl);
        vbool1_t match_mask = __riscv_vmand_mm_b1(__riscv_vmand_mm_b1(first_mask, mid_mask, vl), last_mask, vl);
        // Reverse the mask so `vfirst` yields the highest candidate first.
        vuint8m8_t iota_vec = __riscv_vid_v_u8m8(vl);
        vuint8m8_t reverse_index_vec = __riscv_vrsub_vx_u8m8(iota_vec, (sz_u8_t)(vl - 1), vl);
        // Convert mask to bytes, reverse, convert back, to walk high-to-low.
        vuint8m8_t mask_bytes_vec = __riscv_vmerge_vxm_u8m8(__riscv_vmv_v_x_u8m8(0, vl), 1, match_mask, vl);
        vuint8m8_t reversed_bytes_vec = __riscv_vrgather_vv_u8m8(mask_bytes_vec, reverse_index_vec, vl);
        vbool1_t reversed_mask = __riscv_vmsne_vx_u8m8_b1(reversed_bytes_vec, 0, vl);
        for (long reversed_index = __riscv_vfirst_m_b1(reversed_mask, vl); reversed_index >= 0;) {
            sz_size_t candidate = base + (vl - 1 - (sz_size_t)reversed_index);
            if (sz_equal_rvv((sz_cptr_t)(h_u8 + candidate), n, n_length)) return (sz_cptr_t)(h_u8 + candidate);
            reversed_mask = __riscv_vmandn_mm_b1(
                reversed_mask, __riscv_vmseq_vx_u8m8_b1(__riscv_vid_v_u8m8(vl), (sz_u8_t)reversed_index, vl), vl);
            reversed_index = __riscv_vfirst_m_b1(reversed_mask, vl);
        }
        remaining = base;
    }
    return SZ_NULL_CHAR;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_RVV_H_
