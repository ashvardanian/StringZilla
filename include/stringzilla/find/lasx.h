/**
 *  @brief LoongArch LASX (256-bit) backend for find.
 *  @file include/stringzilla/find/lasx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_LASX_H_
#define STRINGZILLA_FIND_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX

/*  See `compare/lasx.h` for the rationale: `__lasx_xvmskltz_b` packs the sign bit of each byte into a
 *  per-128-bit-lane 16-bit mask (word 0 = low lane, word 4 = high lane). Recombining matches AVX2's
 *  `_mm256_movemask_epi8` byte ordering, so `ctz`/`clz` index bytes identically to the Haswell backend. */
SZ_INTERNAL sz_u32_t sz_xvmovemask_b_find_lasx_(__m256i sign_extended) {
    __m256i collected = __lasx_xvmskltz_b(sign_extended);
    sz_u32_t low = (sz_u32_t)__lasx_xvpickve2gr_wu(collected, 0);
    sz_u32_t high = (sz_u32_t)__lasx_xvpickve2gr_wu(collected, 4);
    return (low & 0xFFFFu) | ((high & 0xFFFFu) << 16);
}

/*  The 128-bit LSX movemask (see `compare/lasx.h`): a single GPR extract yields the SSE-style 16-bit mask.
 *  Used to cover the sub-32-byte head/tail that a 256-bit-only loop would otherwise hand to the serial path. */
SZ_INTERNAL sz_u32_t sz_vmovemask_b_find_lsx_(__m128i sign_extended) {
    return (sz_u32_t)__lsx_vpickve2gr_wu(__lsx_vmskltz_b(sign_extended), 0) & 0xFFFFu;
}

SZ_PUBLIC sz_cptr_t sz_find_byte_lasx(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u256_vec_t h_vec, n_vec, matches_vec;
    // `xvldrepl_b` broadcasts the needle byte straight from memory, fusing the load and the splat that
    // `xvreplgr2vr_b(n[0])` would otherwise spend two instructions on.
    n_vec.lasx = __lasx_xvldrepl_b(n, 0);

    while (h_length >= 32) {
        h_vec.lasx = __lasx_xvld(h, 0);
        matches_vec.lasx = __lasx_xvseq_b(h_vec.lasx, n_vec.lasx);
        // `xbnz_v` (a single `xvsetnez.v`) answers "any match?" with a branchable flag — no vector->GPR read
        // on the match-free common case. On a hit, `xvfrstp` returns the first-match index directly in one
        // op, a shorter dependency chain than the movemask (`xvmskltz` + two extracts + combine) plus `ctz`
        // it replaces. It works per 128-bit lane: the low lane's first-match index lands in byte 0, the high
        // lane's in byte 16 (each 0..15, or 16 when that lane has no match).
        if (__lasx_xbnz_v(matches_vec.lasx)) {
            __m256i first_match_indices = __lasx_xvfrstpi_b(matches_vec.lasx, matches_vec.lasx, 0);
            sz_size_t low_lane_index = (sz_size_t)(__lasx_xvpickve2gr_wu(first_match_indices, 0) & 0xFF);
            if (low_lane_index < 16) return h + low_lane_index;
            return h + 16 + (sz_size_t)(__lasx_xvpickve2gr_wu(first_match_indices, 4) & 0xFF);
        }
        h += 32, h_length -= 32;
    }
    // A single 128-bit LSX block mops up the [16, 32) remainder before the serial tail.
    if (h_length >= 16) {
        sz_u128_vec_t h128_vec, n128_vec, matches128_vec;
        n128_vec.lsx = __lsx_vldrepl_b(n, 0);
        h128_vec.lsx = __lsx_vld(h, 0);
        matches128_vec.lsx = __lsx_vseq_b(h128_vec.lsx, n128_vec.lsx);
        // One lane: `vfrstp` writes the first-match index to byte 0 (`bnz_v` already proved a match exists).
        if (__lsx_bnz_v(matches128_vec.lsx)) {
            __m128i first_match_indices = __lsx_vfrstpi_b(matches128_vec.lsx, matches128_vec.lsx, 0);
            return h + (sz_size_t)(__lsx_vpickve2gr_wu(first_match_indices, 0) & 0xFF);
        }
        h += 16, h_length -= 16;
    }

    return sz_find_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_lasx(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u256_vec_t h_vec, n_vec, matches_vec;
    n_vec.lasx = __lasx_xvldrepl_b(n, 0);
    // `xvfrstp` finds the FIRST match; reverse search wants the LAST, which LASX has no single op for, so we
    // keep `clz` over the movemask. `xbnz_v` still skips the movemask entirely on match-free blocks.
    while (h_length >= 32) {
        h_vec.lasx = __lasx_xvld(h + h_length - 32, 0);
        matches_vec.lasx = __lasx_xvseq_b(h_vec.lasx, n_vec.lasx);
        if (__lasx_xbnz_v(matches_vec.lasx))
            return h + h_length - 1 - sz_u32_clz(sz_xvmovemask_b_find_lasx_(matches_vec.lasx));
        h_length -= 32;
    }
    // A single 128-bit LSX block scans the [16, 32) tail; `clz` of a 16-bit mask indexes from the top byte.
    if (h_length >= 16) {
        sz_u128_vec_t h128_vec, n128_vec, matches128_vec;
        n128_vec.lsx = __lsx_vldrepl_b(n, 0);
        h128_vec.lsx = __lsx_vld(h + h_length - 16, 0);
        matches128_vec.lsx = __lsx_vseq_b(h128_vec.lsx, n128_vec.lsx);
        if (__lsx_bnz_v(matches128_vec.lsx))
            return h + h_length - 1 - (sz_u32_clz(sz_vmovemask_b_find_lsx_(matches128_vec.lsx)) - 16);
        h_length -= 16;
    }

    return sz_rfind_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_find_lasx(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_lasx(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into YMM registers, loading-and-splatting straight from the needle.
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.lasx = __lasx_xvldrepl_b(n + offset_first, 0);
    n_mid_vec.lasx = __lasx_xvldrepl_b(n + offset_mid, 0);
    n_last_vec.lasx = __lasx_xvldrepl_b(n + offset_last, 0);

    // Scan through the string.
    for (; h_length >= n_length + 32; h += 32, h_length -= 32) {
        h_first_vec.lasx = __lasx_xvld(h + offset_first, 0);
        h_mid_vec.lasx = __lasx_xvld(h + offset_mid, 0);
        h_last_vec.lasx = __lasx_xvld(h + offset_last, 0);
        // AND the three equality VECTORS first, then take a SINGLE movemask, instead of recombining a
        // 32-bit mask three times per block (movemask is the expensive part on LASX).
        __m256i first_matches_vec = __lasx_xvseq_b(h_first_vec.lasx, n_first_vec.lasx);
        __m256i mid_matches_vec = __lasx_xvseq_b(h_mid_vec.lasx, n_mid_vec.lasx);
        __m256i last_matches_vec = __lasx_xvseq_b(h_last_vec.lasx, n_last_vec.lasx);
        __m256i all_matches_vec = __lasx_xvand_v(__lasx_xvand_v(first_matches_vec, mid_matches_vec), last_matches_vec);
        // `xbnz_v` gates the movemask: blocks with no triple-match (the overwhelming majority) skip it.
        if (__lasx_xbnz_v(all_matches_vec)) {
            matches_vec.u32 = sz_xvmovemask_b_find_lasx_(all_matches_vec);
            while (matches_vec.u32) {
                int potential_offset = sz_u32_ctz(matches_vec.u32);
                if (sz_equal_lasx(h + potential_offset, n, n_length)) return h + potential_offset;
                matches_vec.u32 &= matches_vec.u32 - 1;
            }
        }
    }

    // A 128-bit LSX block extends coverage down to haystacks of `n_length + 16` bytes, which a 256-bit-only
    // loop would have dropped entirely to the serial path.
    {
        sz_u128_vec_t h_first128_vec, h_mid128_vec, h_last128_vec, n_first128_vec, n_mid128_vec, n_last128_vec;
        n_first128_vec.lsx = __lsx_vldrepl_b(n + offset_first, 0);
        n_mid128_vec.lsx = __lsx_vldrepl_b(n + offset_mid, 0);
        n_last128_vec.lsx = __lsx_vldrepl_b(n + offset_last, 0);
        for (; h_length >= n_length + 16; h += 16, h_length -= 16) {
            h_first128_vec.lsx = __lsx_vld(h + offset_first, 0);
            h_mid128_vec.lsx = __lsx_vld(h + offset_mid, 0);
            h_last128_vec.lsx = __lsx_vld(h + offset_last, 0);
            __m128i first_matches_vec = __lsx_vseq_b(h_first128_vec.lsx, n_first128_vec.lsx);
            __m128i mid_matches_vec = __lsx_vseq_b(h_mid128_vec.lsx, n_mid128_vec.lsx);
            __m128i last_matches_vec = __lsx_vseq_b(h_last128_vec.lsx, n_last128_vec.lsx);
            __m128i all_matches_vec = __lsx_vand_v(__lsx_vand_v(first_matches_vec, mid_matches_vec), last_matches_vec);
            if (__lsx_bnz_v(all_matches_vec)) {
                sz_u32_t matches = sz_vmovemask_b_find_lsx_(all_matches_vec);
                while (matches) {
                    int potential_offset = sz_u32_ctz(matches);
                    if (sz_equal_lasx(h + potential_offset, n, n_length)) return h + potential_offset;
                    matches &= matches - 1;
                }
            }
        }
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_lasx(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_rfind_byte_lasx(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into YMM registers, loading-and-splatting straight from the needle.
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.lasx = __lasx_xvldrepl_b(n + offset_first, 0);
    n_mid_vec.lasx = __lasx_xvldrepl_b(n + offset_mid, 0);
    n_last_vec.lasx = __lasx_xvldrepl_b(n + offset_last, 0);

    // Scan through the string.
    sz_cptr_t h_reversed;
    for (; h_length >= n_length + 32; h_length -= 32) {
        h_reversed = h + h_length - n_length - 32 + 1;
        h_first_vec.lasx = __lasx_xvld(h_reversed + offset_first, 0);
        h_mid_vec.lasx = __lasx_xvld(h_reversed + offset_mid, 0);
        h_last_vec.lasx = __lasx_xvld(h_reversed + offset_last, 0);
        // AND the three equality VECTORS first, then take a SINGLE movemask (see `sz_find_lasx`).
        __m256i first_matches_vec = __lasx_xvseq_b(h_first_vec.lasx, n_first_vec.lasx);
        __m256i mid_matches_vec = __lasx_xvseq_b(h_mid_vec.lasx, n_mid_vec.lasx);
        __m256i last_matches_vec = __lasx_xvseq_b(h_last_vec.lasx, n_last_vec.lasx);
        __m256i all_matches_vec = __lasx_xvand_v(__lasx_xvand_v(first_matches_vec, mid_matches_vec), last_matches_vec);
        // `xbnz_v` gates the movemask: blocks with no triple-match (the overwhelming majority) skip it.
        if (__lasx_xbnz_v(all_matches_vec)) {
            matches_vec.u32 = sz_xvmovemask_b_find_lasx_(all_matches_vec);
            while (matches_vec.u32) {
                int potential_offset = sz_u32_clz(matches_vec.u32);
                if (sz_equal_lasx(h + h_length - n_length - potential_offset, n, n_length))
                    return h + h_length - n_length - potential_offset;
                matches_vec.u32 &= ~(1u << (31 - potential_offset));
            }
        }
    }

    // A 128-bit LSX block extends coverage down to haystacks of `n_length + 16` bytes. The 16-bit mask
    // indexes from the top via `clz - 16` (its set bits live in the low 16 lanes of the 32-bit `clz`).
    {
        sz_u128_vec_t h_first128_vec, h_mid128_vec, h_last128_vec, n_first128_vec, n_mid128_vec, n_last128_vec;
        n_first128_vec.lsx = __lsx_vldrepl_b(n + offset_first, 0);
        n_mid128_vec.lsx = __lsx_vldrepl_b(n + offset_mid, 0);
        n_last128_vec.lsx = __lsx_vldrepl_b(n + offset_last, 0);
        for (; h_length >= n_length + 16; h_length -= 16) {
            sz_cptr_t h_reversed128 = h + h_length - n_length - 16 + 1;
            h_first128_vec.lsx = __lsx_vld(h_reversed128 + offset_first, 0);
            h_mid128_vec.lsx = __lsx_vld(h_reversed128 + offset_mid, 0);
            h_last128_vec.lsx = __lsx_vld(h_reversed128 + offset_last, 0);
            __m128i first_matches_vec = __lsx_vseq_b(h_first128_vec.lsx, n_first128_vec.lsx);
            __m128i mid_matches_vec = __lsx_vseq_b(h_mid128_vec.lsx, n_mid128_vec.lsx);
            __m128i last_matches_vec = __lsx_vseq_b(h_last128_vec.lsx, n_last128_vec.lsx);
            __m128i all_matches_vec = __lsx_vand_v(__lsx_vand_v(first_matches_vec, mid_matches_vec), last_matches_vec);
            if (__lsx_bnz_v(all_matches_vec)) {
                sz_u32_t matches = sz_vmovemask_b_find_lsx_(all_matches_vec);
                while (matches) {
                    int potential_offset = (int)sz_u32_clz(matches) - 16;
                    if (sz_equal_lasx(h + h_length - n_length - potential_offset, n, n_length))
                        return h + h_length - n_length - potential_offset;
                    matches &= ~(1u << (15 - potential_offset));
                }
            }
        }
    }

    return sz_rfind_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_lasx(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {

    // We replicate the strided even/odd bytes of the 32-byte filter into both 128-bit lanes so that the
    // same within-lane `__lasx_xvshuf_b` index works for either half, mirroring the Haswell approach.
    sz_u256_vec_t filter_even_vec, filter_odd_vec;
    sz_u256_vec_t text_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec, matches_vec;

    // Build the even/odd filter tables (16 bytes each), broadcast into both lanes.
    sz_u8_t even_pairs[32], odd_pairs[32];
    sz_u8_t const *filter_bytes = &filter->_u8s[0];
    for (sz_size_t j = 0; j < 16; ++j) {
        even_pairs[j] = even_pairs[j + 16] = filter_bytes[2 * j];
        odd_pairs[j] = odd_pairs[j + 16] = filter_bytes[2 * j + 1];
    }
    filter_even_vec.lasx = __lasx_xvld(even_pairs, 0);
    filter_odd_vec.lasx = __lasx_xvld(odd_pairs, 0);

    // The per-nibble single-bit lookup table, replicated into both lanes.
    sz_u8_t bitmask_table[32];
    for (sz_size_t j = 0; j < 16; ++j) bitmask_table[j] = bitmask_table[j + 16] = (sz_u8_t)(1u << (j & 7));
    bitmask_lookup_vec.lasx = __lasx_xvld(bitmask_table, 0);

    __m256i const zero_vec = __lasx_xvreplgr2vr_b(0);
    __m256i const nibble_mask = __lasx_xvreplgr2vr_b(0x0F);
    __m256i const eight_vec = __lasx_xvreplgr2vr_b(8);

    while (length >= 32) {
        // Transposed equivalent of Wojciech Muła's "SIMD-ized check which bytes are in a set".
        // http://0x80.pl/articles/simd-byte-lookup.html#alternative-implementation-new
        text_vec.lasx = __lasx_xvld(text, 0);
        lower_nibbles_vec.lasx = __lasx_xvand_v(text_vec.lasx, nibble_mask);
        bitmask_vec.lasx = __lasx_xvshuf_b(zero_vec, bitmask_lookup_vec.lasx, lower_nibbles_vec.lasx);

        // Shift right every byte by 4 bits, masking off the carried-in high bits.
        higher_nibbles_vec.lasx = __lasx_xvand_v(__lasx_xvsrli_b(text_vec.lasx, 4), nibble_mask);
        bitset_even_vec.lasx = __lasx_xvshuf_b(zero_vec, filter_even_vec.lasx, higher_nibbles_vec.lasx);
        bitset_odd_vec.lasx = __lasx_xvshuf_b(zero_vec, filter_odd_vec.lasx, higher_nibbles_vec.lasx);

        // Pick the even table when the low nibble is < 8, otherwise the odd table.
        // `__lasx_xvsle_bu(8, lower)` yields 0xFF where `lower >= 8` (selector for `xvbitsel_v` second arg).
        __m256i use_odd_table_mask = __lasx_xvsle_bu(eight_vec, lower_nibbles_vec.lasx);
        bitset_even_vec.lasx = __lasx_xvbitsel_v(bitset_even_vec.lasx, bitset_odd_vec.lasx, use_odd_table_mask);

        // Test the selected bits; a match is any byte where (bitset & bitmask) != 0. `xbnz_v` on this raw
        // product detects an in-set byte in one op, so the common match-free block skips the invert+movemask.
        matches_vec.lasx = __lasx_xvand_v(bitset_even_vec.lasx, bitmask_vec.lasx);
        if (__lasx_xbnz_v(matches_vec.lasx)) {
            matches_vec.lasx = __lasx_xvseq_b(matches_vec.lasx, zero_vec); // 0xFF where NOT matched
            sz_u32_t matches_mask = ~sz_xvmovemask_b_find_lasx_(matches_vec.lasx);
            return text + sz_u32_ctz(matches_mask);
        }
        text += 32, length -= 32;
    }

    // A 128-bit LSX block handles the [16, 32) remainder; the low halves of the 32-byte tables built above
    // are exactly the 16-byte LSX lookup tables, so we reload them without rebuilding anything.
    if (length >= 16) {
        sz_u128_vec_t filter_even128_vec, filter_odd128_vec, bitmask_lookup128_vec;
        sz_u128_vec_t text128_vec, lower_nibbles128_vec, higher_nibbles128_vec;
        sz_u128_vec_t bitmask128_vec, bitset_even128_vec, bitset_odd128_vec, matches128_vec;
        filter_even128_vec.lsx = __lsx_vld(even_pairs, 0);
        filter_odd128_vec.lsx = __lsx_vld(odd_pairs, 0);
        bitmask_lookup128_vec.lsx = __lsx_vld(bitmask_table, 0);
        __m128i const zero128_vec = __lsx_vreplgr2vr_b(0);
        __m128i const nibble_mask128 = __lsx_vreplgr2vr_b(0x0F);
        __m128i const eight128_vec = __lsx_vreplgr2vr_b(8);

        text128_vec.lsx = __lsx_vld(text, 0);
        lower_nibbles128_vec.lsx = __lsx_vand_v(text128_vec.lsx, nibble_mask128);
        bitmask128_vec.lsx = __lsx_vshuf_b(zero128_vec, bitmask_lookup128_vec.lsx, lower_nibbles128_vec.lsx);
        higher_nibbles128_vec.lsx = __lsx_vand_v(__lsx_vsrli_b(text128_vec.lsx, 4), nibble_mask128);
        bitset_even128_vec.lsx = __lsx_vshuf_b(zero128_vec, filter_even128_vec.lsx, higher_nibbles128_vec.lsx);
        bitset_odd128_vec.lsx = __lsx_vshuf_b(zero128_vec, filter_odd128_vec.lsx, higher_nibbles128_vec.lsx);
        __m128i use_odd_table_mask = __lsx_vsle_bu(eight128_vec, lower_nibbles128_vec.lsx);
        bitset_even128_vec.lsx = __lsx_vbitsel_v(bitset_even128_vec.lsx, bitset_odd128_vec.lsx, use_odd_table_mask);
        matches128_vec.lsx = __lsx_vand_v(bitset_even128_vec.lsx, bitmask128_vec.lsx);
        if (__lsx_bnz_v(matches128_vec.lsx)) {
            matches128_vec.lsx = __lsx_vseq_b(matches128_vec.lsx, zero128_vec); // 0xFF where NOT matched
            sz_u32_t matches_mask = (~sz_vmovemask_b_find_lsx_(matches128_vec.lsx)) & 0xFFFFu;
            return text + sz_u32_ctz(matches_mask);
        }
        text += 16, length -= 16;
    }

    return sz_find_byteset_serial(text, length, filter);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_lasx(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    return sz_rfind_byteset_serial(text, length, filter);
}

#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_LASX_H_
