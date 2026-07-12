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

SZ_API_COMPTIME sz_cptr_t sz_find_byte_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    sz_u8_t needle_byte = *(sz_u8_t const *)needle;
    while (haystack_length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(haystack_length);
        vuint8m8_t haystack_u8m8 = __riscv_vle8_v_u8m8(haystack_u8, vector_length);
        vbool1_t eq_mask = __riscv_vmseq_vx_u8m8_b1(haystack_u8m8, needle_byte, vector_length);
        long match_index = __riscv_vfirst_m_b1(eq_mask, vector_length);
        if (match_index >= 0) return (sz_cptr_t)(haystack_u8 + match_index);
        haystack_u8 += vector_length, haystack_length -= vector_length;
    }
    return SZ_NULL_CHAR;
}

/**
 *  @brief Reverse a `u8m4` strip via 16-bit gather indices, lifting the byte-index ceiling.
 *      Backward search wants the highest match first, so we flip the strip and read it
 *      forward. `vrgatherei16` indexes 8-bit elements with a 16-bit vector, so a strip may
 *      span up to 65535 lanes — beyond the RVV 1.0 maximum `VLEN` of 64 Kib at `e8m4`
 *      (`VLMAX = VLEN/2`), so no software cap is ever needed.
 */
SZ_HELPER_INLINE vuint8m4_t sz_reverse_strip_rvv_(vuint8m4_t strip_u8m4, sz_size_t vector_length) {
    vuint16m8_t iota_u16m8 = __riscv_vid_v_u16m8(vector_length);
    vuint16m8_t reverse_index_u16m8 = __riscv_vrsub_vx_u16m8(iota_u16m8, (sz_u16_t)(vector_length - 1), vector_length);
    return __riscv_vrgatherei16_vv_u8m4(strip_u8m4, reverse_index_u16m8, vector_length);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    sz_u8_t needle_byte = *(sz_u8_t const *)needle;
    while (haystack_length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m4(haystack_length);
        sz_u8_t const *strip = haystack_u8 + haystack_length - vector_length;
        // Reverse the strip: out[i] = in[vector_length-1-i], so `vfirst` returns the highest match.
        vuint8m4_t reversed_u8m4 = sz_reverse_strip_rvv_(__riscv_vle8_v_u8m4(strip, vector_length), vector_length);
        vbool2_t eq_mask = __riscv_vmseq_vx_u8m4_b2(reversed_u8m4, needle_byte, vector_length);
        long match_index = __riscv_vfirst_m_b2(eq_mask, vector_length);
        if (match_index >= 0) return (sz_cptr_t)(strip + (vector_length - 1 - match_index));
        haystack_length -= vector_length;
    }
    return SZ_NULL_CHAR;
}

/**
 *  @brief Build a per-lane byteset membership mask for a vector strip.
 *      Mirrors the NEON `_u8s[c>>3] & (1<<(c&7))` formulation. The set byte is fetched with an indexed
 *      memory load (`vluxei8`) straight from the 32-byte `_u8s` table, so the index `c >> 3` in [0, 31]
 *      is valid at any `VLEN` — no register-group capacity ceiling and no serial fallback.
 *
 *  @param haystack_u8m8 Vector of bytes to test.
 *  @param set_u8s The 32-byte byteset in memory.
 *  @param vector_length Vector length for this strip.
 *  @return Predicate mask where lane `i` is set if `haystack_u8m8[i]` is in the set.
 */
SZ_HELPER_AUTO vbool1_t sz_find_byteset_rvv_mask_m8_(vuint8m8_t haystack_u8m8, sz_u8_t const *set_u8s,
                                                     sz_size_t vector_length) {
    vuint8m8_t byte_index_u8m8 = __riscv_vsrl_vx_u8m8(haystack_u8m8, 3, vector_length);   // c >> 3, in [0, 31]
    vuint8m8_t bit_position_u8m8 = __riscv_vand_vx_u8m8(haystack_u8m8, 7, vector_length); // c & 7
    vuint8m8_t one_u8m8 = __riscv_vmv_v_x_u8m8(1, vector_length);
    vuint8m8_t bit_mask_u8m8 = __riscv_vsll_vv_u8m8(one_u8m8, bit_position_u8m8, vector_length); // 1 << (c & 7)
    vuint8m8_t set_bytes_u8m8 = __riscv_vluxei8_v_u8m8(set_u8s, byte_index_u8m8, vector_length);
    vuint8m8_t anded_u8m8 = __riscv_vand_vv_u8m8(set_bytes_u8m8, bit_mask_u8m8, vector_length);
    return __riscv_vmsne_vx_u8m8_b1(anded_u8m8, 0, vector_length);
}

/**
 *  @brief `m4` sibling of @ref sz_find_byteset_rvv_mask_m8_, used on the reversed backward strip.
 */
SZ_HELPER_AUTO vbool2_t sz_find_byteset_rvv_mask_m4_(vuint8m4_t haystack_u8m4, sz_u8_t const *set_u8s,
                                                     sz_size_t vector_length) {
    vuint8m4_t byte_index_u8m4 = __riscv_vsrl_vx_u8m4(haystack_u8m4, 3, vector_length);   // c >> 3, in [0, 31]
    vuint8m4_t bit_position_u8m4 = __riscv_vand_vx_u8m4(haystack_u8m4, 7, vector_length); // c & 7
    vuint8m4_t one_u8m4 = __riscv_vmv_v_x_u8m4(1, vector_length);
    vuint8m4_t bit_mask_u8m4 = __riscv_vsll_vv_u8m4(one_u8m4, bit_position_u8m4, vector_length); // 1 << (c & 7)
    vuint8m4_t set_bytes_u8m4 = __riscv_vluxei8_v_u8m4(set_u8s, byte_index_u8m4, vector_length);
    vuint8m4_t anded_u8m4 = __riscv_vand_vv_u8m4(set_bytes_u8m4, bit_mask_u8m4, vector_length);
    return __riscv_vmsne_vx_u8m4_b2(anded_u8m4, 0, vector_length);
}

SZ_API_COMPTIME sz_cptr_t sz_find_byteset_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set) {
    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    while (haystack_length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(haystack_length);
        vuint8m8_t haystack_u8m8 = __riscv_vle8_v_u8m8(haystack_u8, vector_length);
        vbool1_t match_mask = sz_find_byteset_rvv_mask_m8_(haystack_u8m8, &set->_u8s[0], vector_length);
        long match_index = __riscv_vfirst_m_b1(match_mask, vector_length);
        if (match_index >= 0) return (sz_cptr_t)(haystack_u8 + match_index);
        haystack_u8 += vector_length, haystack_length -= vector_length;
    }
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set) {
    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    while (haystack_length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m4(haystack_length);
        sz_u8_t const *strip = haystack_u8 + haystack_length - vector_length;
        vuint8m4_t reversed_u8m4 = sz_reverse_strip_rvv_(__riscv_vle8_v_u8m4(strip, vector_length), vector_length);
        vbool2_t match_mask = sz_find_byteset_rvv_mask_m4_(reversed_u8m4, &set->_u8s[0], vector_length);
        long match_index = __riscv_vfirst_m_b2(match_mask, vector_length);
        if (match_index >= 0) return (sz_cptr_t)(strip + (vector_length - 1 - match_index));
        haystack_length -= vector_length;
    }
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_find_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                      sz_size_t needle_length) {
    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_rvv(haystack, haystack_length, needle);

    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    sz_u8_t n_first = ((sz_u8_t const *)needle)[offset_first];
    sz_u8_t n_mid = ((sz_u8_t const *)needle)[offset_mid];
    sz_u8_t n_last = ((sz_u8_t const *)needle)[offset_last];

    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    // Number of candidate start positions is `haystack_length - needle_length + 1`.
    sz_size_t candidates = haystack_length - needle_length + 1;
    sz_size_t position = 0;
    while (position < candidates) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(candidates - position);
        vuint8m8_t first_u8m8 = __riscv_vle8_v_u8m8(haystack_u8 + position + offset_first, vector_length);
        vuint8m8_t mid_u8m8 = __riscv_vle8_v_u8m8(haystack_u8 + position + offset_mid, vector_length);
        vuint8m8_t last_u8m8 = __riscv_vle8_v_u8m8(haystack_u8 + position + offset_last, vector_length);
        vbool1_t first_mask = __riscv_vmseq_vx_u8m8_b1(first_u8m8, n_first, vector_length);
        vbool1_t mid_mask = __riscv_vmseq_vx_u8m8_b1(mid_u8m8, n_mid, vector_length);
        vbool1_t last_mask = __riscv_vmseq_vx_u8m8_b1(last_u8m8, n_last, vector_length);
        vbool1_t match_mask = __riscv_vmand_mm_b1(__riscv_vmand_mm_b1(first_mask, mid_mask, vector_length), last_mask,
                                                  vector_length);
        // Iterate set bits low-to-high without materializing a per-lane index: `vmsif` marks every
        // lane up to and including the first set bit, then `vmandn` clears them, leaving the rest.
        for (long match_index = __riscv_vfirst_m_b1(match_mask, vector_length); match_index >= 0;
             match_index = __riscv_vfirst_m_b1(match_mask, vector_length)) {
            sz_size_t candidate = position + (sz_size_t)match_index;
            if (sz_equal_rvv((sz_cptr_t)(haystack_u8 + candidate), needle, needle_length))
                return (sz_cptr_t)(haystack_u8 + candidate);
            match_mask = __riscv_vmandn_mm_b1(match_mask, __riscv_vmsif_m_b1(match_mask, vector_length), vector_length);
        }
        position += vector_length;
    }
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length) {
    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_rvv(haystack, haystack_length, needle);

    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    sz_u8_t n_first = ((sz_u8_t const *)needle)[offset_first];
    sz_u8_t n_mid = ((sz_u8_t const *)needle)[offset_mid];
    sz_u8_t n_last = ((sz_u8_t const *)needle)[offset_last];

    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    sz_size_t candidates = haystack_length - needle_length + 1; // start positions 0 .. candidates-1
    sz_size_t remaining = candidates;
    while (remaining) {
        sz_size_t vector_length = __riscv_vsetvl_e8m4(remaining);
        sz_size_t base = remaining - vector_length; // strip covers candidate starts [base, base+vector_length)
        // Reverse the three anchor loads so reversed lane 0 is the highest candidate (base+vector_length-1);
        // `vfirst` then yields the highest match, with no mask byte-spill round trip.
        vuint8m4_t first_rev = sz_reverse_strip_rvv_(
            __riscv_vle8_v_u8m4(haystack_u8 + base + offset_first, vector_length), vector_length);
        vuint8m4_t mid_rev = sz_reverse_strip_rvv_(__riscv_vle8_v_u8m4(haystack_u8 + base + offset_mid, vector_length),
                                                   vector_length);
        vuint8m4_t last_rev = sz_reverse_strip_rvv_(
            __riscv_vle8_v_u8m4(haystack_u8 + base + offset_last, vector_length), vector_length);
        vbool2_t first_mask = __riscv_vmseq_vx_u8m4_b2(first_rev, n_first, vector_length);
        vbool2_t mid_mask = __riscv_vmseq_vx_u8m4_b2(mid_rev, n_mid, vector_length);
        vbool2_t last_mask = __riscv_vmseq_vx_u8m4_b2(last_rev, n_last, vector_length);
        vbool2_t match_mask = __riscv_vmand_mm_b2(__riscv_vmand_mm_b2(first_mask, mid_mask, vector_length), last_mask,
                                                  vector_length);
        // Walk reversed lanes low-to-high (= candidates high-to-low) via `vmsif`/`vmandn`.
        for (long reversed_index = __riscv_vfirst_m_b2(match_mask, vector_length); reversed_index >= 0;
             reversed_index = __riscv_vfirst_m_b2(match_mask, vector_length)) {
            sz_size_t candidate = base + (vector_length - 1 - (sz_size_t)reversed_index);
            if (sz_equal_rvv((sz_cptr_t)(haystack_u8 + candidate), needle, needle_length))
                return (sz_cptr_t)(haystack_u8 + candidate);
            match_mask = __riscv_vmandn_mm_b2(match_mask, __riscv_vmsif_m_b2(match_mask, vector_length), vector_length);
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
