/**
 *  @brief LoongArch LASX (256-bit) backend for memory.
 *  @file include/stringzilla/memory/lasx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_LASX_H_
#define STRINGZILLA_MEMORY_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX

/** @brief  Broadcast a 16-byte slice of the 256-entry lookup table into both 128-bit lanes of a YMM. */
SZ_INTERNAL __m256i sz_lookup_load_lut_lasx_(char const lut[sz_at_least_(256)], sz_size_t offset) {
    sz_u8_t lut_pairs[32];
    for (sz_size_t j = 0; j < 16; ++j) lut_pairs[j] = lut_pairs[j + 16] = (sz_u8_t)lut[offset + j];
    return __lasx_xvld(lut_pairs, 0);
}

SZ_PUBLIC void sz_fill_lasx(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    if (length <= 32) { sz_fill_serial(target, length, value); }
    else {
        __m256i value_vec = __lasx_xvreplgr2vr_b((char)value);
        // Store the unaligned head, then walk the aligned body, overlapping the tail at the end.
        __lasx_xvst(value_vec, target, 0);
        sz_ptr_t end = target + length;
        target = (sz_ptr_t)(((sz_size_t)target + 32) & ~(sz_size_t)31); // First 32-aligned address past the head.
        for (; target + 32 <= end; target += 32) __lasx_xvst(value_vec, target, 0);
        // Store the unaligned tail, overlapping the last aligned body store if needed.
        if (target != end) __lasx_xvst(value_vec, end - 32, 0);
    }
}

SZ_PUBLIC void sz_copy_lasx(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (length < 8) {
        while (length--) *(target++) = *(source++);
    }
    else if (length <= 16) {
        sz_u64_t source_first_word = *(sz_u64_t const *)(source);
        sz_u64_t source_second_word = *(sz_u64_t const *)(source + length - 8);
        *(sz_u64_t *)(target) = source_first_word;
        *(sz_u64_t *)(target + length - 8) = source_second_word;
    }
    else if (length <= 32) {
        // Two overlapping 128-bit LSX halves cover everything in [17, 32].
        sz_u128_vec_t source_first_vec, source_second_vec;
        source_first_vec.lsx = __lsx_vld(source, 0);
        source_second_vec.lsx = __lsx_vld(source + length - 16, 0);
        __lsx_vst(source_first_vec.lsx, target, 0);
        __lsx_vst(source_second_vec.lsx, target + length - 16, 0);
    }
    else if (length <= 64) {
        sz_u256_vec_t source_first_vec, source_second_vec;
        source_first_vec.lasx = __lasx_xvld(source, 0);
        source_second_vec.lasx = __lasx_xvld(source + length - 32, 0);
        __lasx_xvst(source_first_vec.lasx, target, 0);
        __lasx_xvst(source_second_vec.lasx, target + length - 32, 0);
    }
    else {
        // Copy the unaligned head, then run an aligned-store body and an overlapping tail.
        __m256i head_vec = __lasx_xvld(source, 0);
        sz_ptr_t target_end = target + length;
        sz_ptr_t target_aligned = (sz_ptr_t)(((sz_size_t)target + 32) & ~(sz_size_t)31);
        sz_size_t head_bytes = (sz_size_t)(target_aligned - target);
        __lasx_xvst(head_vec, target, 0);
        source += head_bytes;
        for (target = target_aligned; target + 32 <= target_end; target += 32, source += 32)
            __lasx_xvst(__lasx_xvld(source, 0), target, 0);
        if (target != target_end) {
            sz_size_t remaining = (sz_size_t)(target_end - target);
            __lasx_xvst(__lasx_xvld(source + remaining - 32, 0), target_end - 32, 0);
        }
    }
}

SZ_PUBLIC void sz_move_lasx(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (length < 8) {
        if (target < source)
            while (length--) *(target++) = *(source++);
        else {
            target += length, source += length;
            while (length--) *(--target) = *(--source);
        }
    }
    else if (length <= 16) {
        sz_u64_t source_first_word = *(sz_u64_t const *)(source);
        sz_u64_t source_second_word = *(sz_u64_t const *)(source + length - 8);
        *(sz_u64_t *)(target) = source_first_word;
        *(sz_u64_t *)(target + length - 8) = source_second_word;
    }
    else if (length <= 32) {
        // Both overlapping 128-bit halves are loaded before either store, so source/target overlap is safe.
        sz_u128_vec_t source_first_vec, source_second_vec;
        source_first_vec.lsx = __lsx_vld(source, 0);
        source_second_vec.lsx = __lsx_vld(source + length - 16, 0);
        __lsx_vst(source_first_vec.lsx, target, 0);
        __lsx_vst(source_second_vec.lsx, target + length - 16, 0);
    }
    else if (length <= 64) {
        sz_u256_vec_t source_first_vec, source_second_vec;
        source_first_vec.lasx = __lasx_xvld(source, 0);
        source_second_vec.lasx = __lasx_xvld(source + length - 32, 0);
        __lasx_xvst(source_first_vec.lasx, target, 0);
        __lasx_xvst(source_second_vec.lasx, target + length - 32, 0);
    }
    else if (target < source || target >= source + length) {
        for (; length >= 32; target += 32, source += 32, length -= 32) __lasx_xvst(__lasx_xvld(source, 0), target, 0);
        while (length--) *(target++) = *(source++);
    }
    else {
        // Overlapping ranges with `target` ahead of `source`: walk backwards.
        for (target += length, source += length; length >= 32; length -= 32)
            __lasx_xvst(__lasx_xvld(source -= 32, 0), target -= 32, 0);
        while (length--) *(--target) = *(--source);
    }
}

SZ_PUBLIC void sz_lookup_lasx(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]) {

    // The setup cost only pays off for larger inputs.
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    // Pull the 256-entry lookup table into 16x YMM registers, each holding a 16-byte slice broadcast
    // into both 128-bit lanes, so `__lasx_xvshuf_b` can index it with the same nibble in either lane.
    // Unlike AVX2's `_mm256_shuffle_epi8`, `__lasx_xvshuf_b(a, b, idx)` picks `b[idx & 15]` within a
    // lane when `idx & 31 < 16`. Because all our indices are nibbles (0..15), this matches pshufb.
    sz_u256_vec_t lut_0_to_15_vec, lut_16_to_31_vec, lut_32_to_47_vec, lut_48_to_63_vec, //
        lut_64_to_79_vec, lut_80_to_95_vec, lut_96_to_111_vec, lut_112_to_127_vec,       //
        lut_128_to_143_vec, lut_144_to_159_vec, lut_160_to_175_vec, lut_176_to_191_vec,  //
        lut_192_to_207_vec, lut_208_to_223_vec, lut_224_to_239_vec, lut_240_to_255_vec;

    lut_0_to_15_vec.lasx = sz_lookup_load_lut_lasx_(lut, 0);
    lut_16_to_31_vec.lasx = sz_lookup_load_lut_lasx_(lut, 16);
    lut_32_to_47_vec.lasx = sz_lookup_load_lut_lasx_(lut, 32);
    lut_48_to_63_vec.lasx = sz_lookup_load_lut_lasx_(lut, 48);
    lut_64_to_79_vec.lasx = sz_lookup_load_lut_lasx_(lut, 64);
    lut_80_to_95_vec.lasx = sz_lookup_load_lut_lasx_(lut, 80);
    lut_96_to_111_vec.lasx = sz_lookup_load_lut_lasx_(lut, 96);
    lut_112_to_127_vec.lasx = sz_lookup_load_lut_lasx_(lut, 112);
    lut_128_to_143_vec.lasx = sz_lookup_load_lut_lasx_(lut, 128);
    lut_144_to_159_vec.lasx = sz_lookup_load_lut_lasx_(lut, 144);
    lut_160_to_175_vec.lasx = sz_lookup_load_lut_lasx_(lut, 160);
    lut_176_to_191_vec.lasx = sz_lookup_load_lut_lasx_(lut, 176);
    lut_192_to_207_vec.lasx = sz_lookup_load_lut_lasx_(lut, 192);
    lut_208_to_223_vec.lasx = sz_lookup_load_lut_lasx_(lut, 208);
    lut_224_to_239_vec.lasx = sz_lookup_load_lut_lasx_(lut, 224);
    lut_240_to_255_vec.lasx = sz_lookup_load_lut_lasx_(lut, 240);

    sz_u256_vec_t source_vec, source_bot_vec;
    sz_u256_vec_t blended_0_to_31_vec, blended_32_to_63_vec, blended_64_to_95_vec, blended_96_to_127_vec,
        blended_128_to_159_vec, blended_160_to_191_vec, blended_192_to_223_vec, blended_224_to_255_vec;

    __m256i const zero_vec = __lasx_xvreplgr2vr_b(0);
    __m256i const nibble_mask = __lasx_xvreplgr2vr_b(0x0F);

    // `__lasx_xvbitsel_v(a, b, c)` selects bit-by-bit: result = (a & ~c) | (b & c).
    // We build full-byte selector masks (0xFF -> pick `b`, 0x00 -> pick `a`) by testing one source bit
    // via `__lasx_xvslt_bu(0, byte & mask)`, which yields 0xFF whenever the masked byte is non-zero.
    while (length >= 32) {
        source_vec.lasx = __lasx_xvld(source, 0);
        source_bot_vec.lasx = __lasx_xvand_v(source_vec.lasx, nibble_mask);

        // Round 1: select within each 32-entry pair using bit 4 (0x10).
        __m256i bit4_set = __lasx_xvslt_bu(zero_vec, __lasx_xvand_v(source_vec.lasx, __lasx_xvreplgr2vr_b(0x10)));
        blended_0_to_31_vec.lasx = __lasx_xvbitsel_v(                              //
            __lasx_xvshuf_b(zero_vec, lut_0_to_15_vec.lasx, source_bot_vec.lasx),  //
            __lasx_xvshuf_b(zero_vec, lut_16_to_31_vec.lasx, source_bot_vec.lasx), //
            bit4_set);
        blended_32_to_63_vec.lasx = __lasx_xvbitsel_v(                             //
            __lasx_xvshuf_b(zero_vec, lut_32_to_47_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_vec, lut_48_to_63_vec.lasx, source_bot_vec.lasx), //
            bit4_set);
        blended_64_to_95_vec.lasx = __lasx_xvbitsel_v(                             //
            __lasx_xvshuf_b(zero_vec, lut_64_to_79_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_vec, lut_80_to_95_vec.lasx, source_bot_vec.lasx), //
            bit4_set);
        blended_96_to_127_vec.lasx = __lasx_xvbitsel_v(                              //
            __lasx_xvshuf_b(zero_vec, lut_96_to_111_vec.lasx, source_bot_vec.lasx),  //
            __lasx_xvshuf_b(zero_vec, lut_112_to_127_vec.lasx, source_bot_vec.lasx), //
            bit4_set);
        blended_128_to_159_vec.lasx = __lasx_xvbitsel_v(                             //
            __lasx_xvshuf_b(zero_vec, lut_128_to_143_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_vec, lut_144_to_159_vec.lasx, source_bot_vec.lasx), //
            bit4_set);
        blended_160_to_191_vec.lasx = __lasx_xvbitsel_v(                             //
            __lasx_xvshuf_b(zero_vec, lut_160_to_175_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_vec, lut_176_to_191_vec.lasx, source_bot_vec.lasx), //
            bit4_set);
        blended_192_to_223_vec.lasx = __lasx_xvbitsel_v(                             //
            __lasx_xvshuf_b(zero_vec, lut_192_to_207_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_vec, lut_208_to_223_vec.lasx, source_bot_vec.lasx), //
            bit4_set);
        blended_224_to_255_vec.lasx = __lasx_xvbitsel_v(                             //
            __lasx_xvshuf_b(zero_vec, lut_224_to_239_vec.lasx, source_bot_vec.lasx), //
            __lasx_xvshuf_b(zero_vec, lut_240_to_255_vec.lasx, source_bot_vec.lasx), //
            bit4_set);

        // Round 2: select using bit 5 (0x20).
        __m256i bit5_set = __lasx_xvslt_bu(zero_vec, __lasx_xvand_v(source_vec.lasx, __lasx_xvreplgr2vr_b(0x20)));
        blended_0_to_31_vec.lasx = __lasx_xvbitsel_v(blended_0_to_31_vec.lasx, blended_32_to_63_vec.lasx, bit5_set);
        blended_64_to_95_vec.lasx = __lasx_xvbitsel_v(blended_64_to_95_vec.lasx, blended_96_to_127_vec.lasx, bit5_set);
        blended_128_to_159_vec.lasx =
            __lasx_xvbitsel_v(blended_128_to_159_vec.lasx, blended_160_to_191_vec.lasx, bit5_set);
        blended_192_to_223_vec.lasx =
            __lasx_xvbitsel_v(blended_192_to_223_vec.lasx, blended_224_to_255_vec.lasx, bit5_set);

        // Round 3: select using bit 6 (0x40).
        __m256i bit6_set = __lasx_xvslt_bu(zero_vec, __lasx_xvand_v(source_vec.lasx, __lasx_xvreplgr2vr_b(0x40)));
        blended_0_to_31_vec.lasx = __lasx_xvbitsel_v(blended_0_to_31_vec.lasx, blended_64_to_95_vec.lasx, bit6_set);
        blended_128_to_159_vec.lasx =
            __lasx_xvbitsel_v(blended_128_to_159_vec.lasx, blended_192_to_223_vec.lasx, bit6_set);

        // Round 4: select using bit 7 (0x80).
        __m256i bit7_set = __lasx_xvslt_bu(zero_vec, __lasx_xvand_v(source_vec.lasx, __lasx_xvreplgr2vr_b((char)0x80)));
        blended_0_to_31_vec.lasx = __lasx_xvbitsel_v(blended_0_to_31_vec.lasx, blended_128_to_159_vec.lasx, bit7_set);

        __lasx_xvst(blended_0_to_31_vec.lasx, target, 0);
        source += 32, target += 32, length -= 32;
    }

    if (length) sz_lookup_serial(target, length, source, lut);
}

#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_MEMORY_LASX_H_
