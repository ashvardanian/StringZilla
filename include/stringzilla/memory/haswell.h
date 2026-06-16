/**
 *  @brief AVX2 backend for hardware-accelerated memory operations on Haswell and newer x86 CPUs.
 *  @file include/stringzilla/memory/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_HASWELL_H_
#define STRINGZILLA_MEMORY_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

SZ_PUBLIC void sz_fill_haswell(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    char value_char = *(char *)&value;
    __m256i value_vec = _mm256_set1_epi8(value_char);
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "stores".
    //
    //    for (; length >= 32; target += 32, length -= 32) _mm256_storeu_si256(target, value_vec);
    //    sz_fill_serial(target, length, value);
    //
    // When the buffer is small, there isn't much to innovate.
    if (length <= 32) sz_fill_serial(target, length, value);
    // When the buffer is aligned, we can avoid any split-stores.
    else {
        sz_size_t head_length = (32 - ((sz_size_t)target % 32)) % 32; // 31 or less.
        sz_size_t tail_length = (sz_size_t)(target + length) % 32;    // 31 or less.
        sz_size_t body_length = length - head_length - tail_length;   // Multiple of 32.
        sz_u16_t value16 = (sz_u16_t)value * 0x0101u;
        sz_u32_t value32 = (sz_u32_t)value16 * 0x00010001u;
        sz_u64_t value64 = (sz_u64_t)value32 * 0x0000000100000001ull;

        // Fill the head of the buffer. This part is much cleaner with AVX-512.
        if (head_length & 1) *(sz_u8_t *)target = value, target++, head_length--;
        if (head_length & 2) *(sz_u16_t *)target = value16, target += 2, head_length -= 2;
        if (head_length & 4) *(sz_u32_t *)target = value32, target += 4, head_length -= 4;
        if (head_length & 8) *(sz_u64_t *)target = value64, target += 8, head_length -= 8;
        if (head_length & 16)
            _mm_store_si128((__m128i *)target, _mm_set1_epi8(value_char)), target += 16, head_length -= 16;
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");

        // Fill the aligned body of the buffer.
        for (; body_length >= 32; target += 32, body_length -= 32) _mm256_store_si256((__m256i *)target, value_vec);

        // Fill the tail of the buffer. This part is much cleaner with AVX-512.
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");
        if (tail_length & 16)
            _mm_store_si128((__m128i *)target, _mm_set1_epi8(value_char)), target += 16, tail_length -= 16;
        if (tail_length & 8) *(sz_u64_t *)target = value64, target += 8, tail_length -= 8;
        if (tail_length & 4) *(sz_u32_t *)target = value32, target += 4, tail_length -= 4;
        if (tail_length & 2) *(sz_u16_t *)target = value16, target += 2, tail_length -= 2;
        if (tail_length & 1) *(sz_u8_t *)target = value, target++, tail_length--;
    }
}

SZ_PUBLIC void sz_copy_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "stores" and "loads".
    //
    //    for (; length >= 32; target += 32, source += 32, length -= 32)
    //        _mm256_storeu_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
    //    sz_copy_serial(target, source, length);
    //
    // A typical AWS Skylake instance can have 32 KB x 2 blocks of L1 data cache per core,
    // 1 MB x 2 blocks of L2 cache per core, and one shared L3 cache buffer.
    // For now, let's avoid the cases beyond the L2 size.
    int is_huge = length > 1ull * 1024ull * 1024ull;
    if (length < 8) {
        while (length--) *(target++) = *(source++);
    }
    // The next few sections are identical here and in the `sz_move_haswell` function.
    // We can use 2x 64-bit interleaving loads for each string, and then compare them for equality.
    // The same approach is used in GLibC and was suggest by Denis Yaroshevskiy.
    // https://codebrowser.dev/glibc/glibc/sysdeps/x86_64/multiarch/memcmp-avx2-movbe.S.html#518
    // It shouldn't improve performance on microbenchmarks, but should be better in practice.
    else if (length <= 16) {
        sz_u64_t source_first_word = sz_u64_load(source).u64;
        sz_u64_t source_second_word = sz_u64_load(source + length - 8).u64;
        sz_u64_store(target, source_first_word);
        sz_u64_store(target + length - 8, source_second_word);
    }
    // We can use 2x 128-bit interleaving loads for each string, and then compare them for equality.
    else if (length <= 32) {
        sz_u128_vec_t source_first_vec, source_second_vec;
        source_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(source));
        source_second_vec.xmm = _mm_lddqu_si128((__m128i const *)(source + length - 16));
        _mm_storeu_si128((__m128i *)(target), source_first_vec.xmm);
        _mm_storeu_si128((__m128i *)(target + length - 16), source_second_vec.xmm);
    }
    // We can use 2x 256-bit interleaving loads for each string, and then compare them for equality.
    else if (length <= 64) {
        sz_u256_vec_t source_first_vec, source_second_vec;
        source_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source));
        source_second_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source + length - 32));
        _mm256_storeu_si256((__m256i *)(target), source_first_vec.ymm);
        _mm256_storeu_si256((__m256i *)(target + length - 32), source_second_vec.ymm);
    }
    // When dealing with larger arrays, the optimization is not as simple as with the `sz_fill_haswell` function,
    // as both buffers may be unaligned. If we are lucky and the requested operation is some huge page transfer,
    // we can use aligned loads and stores, and the performance will be great.
    else if ((sz_size_t)target % 32 == 0 && (sz_size_t)source % 32 == 0 && !is_huge) {
        for (; length >= 32; target += 32, source += 32, length -= 32)
            _mm256_store_si256((__m256i *)target, _mm256_load_si256((__m256i const *)source));
        if (length) sz_copy_serial(target, source, length);
    }
    // The trickiest case is when both `source` and `target` are not aligned.
    // In such and simpler cases we can copy enough bytes into `target` to reach its cacheline boundary,
    // and then combine unaligned loads with aligned stores.
    else {
        sz_size_t head_length = (32 - ((sz_size_t)target % 32)) % 32; // 31 or less.
        sz_size_t tail_length = (sz_size_t)(target + length) % 32;    // 31 or less.
        sz_size_t body_length = length - head_length - tail_length;   // Multiple of 32.

        // Fill the head of the buffer. This part is much cleaner with AVX-512.
        if (head_length & 1) *(sz_u8_t *)target = *(sz_u8_t *)source, target++, source++, head_length--;
        if (head_length & 2) sz_u16_store(target, sz_u16_load(source).u16), target += 2, source += 2, head_length -= 2;
        if (head_length & 4) sz_u32_store(target, sz_u32_load(source).u32), target += 4, source += 4, head_length -= 4;
        if (head_length & 8) sz_u64_store(target, sz_u64_load(source).u64), target += 8, source += 8, head_length -= 8;
        if (head_length & 16)
            _mm_store_si128((__m128i *)target, _mm_lddqu_si128((__m128i const *)source)), target += 16, source += 16,
                head_length -= 16;
        sz_assert_(head_length == 0 && "The head length should be zero after the head copy.");
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");

        // Fill the aligned body of the buffer.
        if (!is_huge) {
            for (; body_length >= 32; target += 32, source += 32, body_length -= 32)
                _mm256_store_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
        }
        // When the buffer is huge, we can traverse it in 2 directions.
        else {
            size_t tails_bytes_skipped = 0;
            for (; body_length >= 64; target += 32, source += 32, body_length -= 64, tails_bytes_skipped += 32) {
                _mm256_store_si256((__m256i *)(target), _mm256_lddqu_si256((__m256i const *)(source)));
                _mm256_store_si256((__m256i *)(target + body_length - 32),
                                   _mm256_lddqu_si256((__m256i const *)(source + body_length - 32)));
            }
            if (body_length) {
                sz_assert_(body_length == 32 && "The only remaining body length should be 32 bytes.");
                _mm256_store_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
                target += 32, source += 32, body_length -= 32;
            }
            target += tails_bytes_skipped;
            source += tails_bytes_skipped;
        }

        // Fill the tail of the buffer. This part is much cleaner with AVX-512.
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");
        if (tail_length & 16)
            _mm_store_si128((__m128i *)target, _mm_lddqu_si128((__m128i const *)source)), target += 16, source += 16,
                tail_length -= 16;
        if (tail_length & 8) sz_u64_store(target, sz_u64_load(source).u64), target += 8, source += 8, tail_length -= 8;
        if (tail_length & 4) sz_u32_store(target, sz_u32_load(source).u32), target += 4, source += 4, tail_length -= 4;
        if (tail_length & 2) sz_u16_store(target, sz_u16_load(source).u16), target += 2, source += 2, tail_length -= 2;
        if (tail_length & 1) *(sz_u8_t *)target = *(sz_u8_t *)source, target++, source++, tail_length--;
        sz_assert_(tail_length == 0 && "The tail length should be zero after the tail copy.");
    }
}

SZ_PUBLIC void sz_move_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {

    if (length < 8) {
        if (target < source)
            while (length--) *(target++) = *(source++);
        else {
            // Jump to the end and walk backwards:
            target += length, source += length;
            while (length--) *(--target) = *(--source);
        }
    }
    // The next few sections are identical here and in the `sz_copy_haswell` function.
    // We can use 2x 64-bit interleaving loads for each string, and then compare them for equality.
    // The same approach is used in GLibC and was suggest by Denis Yaroshevskiy.
    // https://codebrowser.dev/glibc/glibc/sysdeps/x86_64/multiarch/memcmp-avx2-movbe.S.html#518
    // It shouldn't improve performance on microbenchmarks, but should be better in practice.
    else if (length <= 16) {
        sz_u64_t source_first_word = sz_u64_load(source).u64;
        sz_u64_t source_second_word = sz_u64_load(source + length - 8).u64;
        sz_u64_store(target, source_first_word);
        sz_u64_store(target + length - 8, source_second_word);
    }
    // We can use 2x 128-bit interleaving loads for each string, and then compare them for equality.
    else if (length <= 32) {
        sz_u128_vec_t source_first_vec, source_second_vec;
        source_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(source));
        source_second_vec.xmm = _mm_lddqu_si128((__m128i const *)(source + length - 16));
        _mm_storeu_si128((__m128i *)(target), source_first_vec.xmm);
        _mm_storeu_si128((__m128i *)(target + length - 16), source_second_vec.xmm);
    }
    // We can use 2x 256-bit interleaving loads for each string, and then compare them for equality.
    else if (length <= 64) {
        sz_u256_vec_t source_first_vec, source_second_vec;
        source_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source));
        source_second_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source + length - 32));
        _mm256_storeu_si256((__m256i *)(target), source_first_vec.ymm);
        _mm256_storeu_si256((__m256i *)(target + length - 32), source_second_vec.ymm);
    }
    // When dealing with larger arrays, we keep things simple:
    else if (target < source || target >= source + length) {
        for (; length >= 32; target += 32, source += 32, length -= 32)
            _mm256_storeu_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
        while (length--) *(target++) = *(source++);
    }
    else {
        // Jump to the end and walk backwards:
        for (target += length, source += length; length >= 32; length -= 32)
            _mm256_storeu_si256((__m256i *)(target -= 32), _mm256_lddqu_si256((__m256i const *)(source -= 32)));
        while (length--) *(--target) = *(--source);
    }
}

SZ_PUBLIC void sz_lookup_haswell(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                 char const lut[sz_at_least_(256)]) {

    // If the input is tiny (especially smaller than the look-up table itself), we may end up paying
    // more for organizing the SIMD registers and changing the CPU state, than for the actual computation.
    // But if at least 3 cache lines are touched, the AVX-2 implementation should be faster.
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    // We need to pull the lookup table into 8x YMM registers.
    // The biggest issue is reorganizing the data in the lookup table, as AVX2 doesn't have 256-bit shuffle,
    // it only has 128-bit "within-lane" shuffle. Still, it's wiser to use full YMM registers, instead of XMM,
    // so that we can at least compensate high latency with twice larger window and one more level of lookup.
    sz_u256_vec_t lut_0_to_15_vec, lut_16_to_31_vec, lut_32_to_47_vec, lut_48_to_63_vec, //
        lut_64_to_79_vec, lut_80_to_95_vec, lut_96_to_111_vec, lut_112_to_127_vec,       //
        lut_128_to_143_vec, lut_144_to_159_vec, lut_160_to_175_vec, lut_176_to_191_vec,  //
        lut_192_to_207_vec, lut_208_to_223_vec, lut_224_to_239_vec, lut_240_to_255_vec;

    lut_0_to_15_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut)));
    lut_16_to_31_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 16)));
    lut_32_to_47_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 32)));
    lut_48_to_63_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 48)));
    lut_64_to_79_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 64)));
    lut_80_to_95_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 80)));
    lut_96_to_111_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 96)));
    lut_112_to_127_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 112)));
    lut_128_to_143_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 128)));
    lut_144_to_159_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 144)));
    lut_160_to_175_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 160)));
    lut_176_to_191_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 176)));
    lut_192_to_207_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 192)));
    lut_208_to_223_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 208)));
    lut_224_to_239_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 224)));
    lut_240_to_255_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 240)));

    // Assuming each lookup is performed within 16 elements of 256, we need to reduce the scope by 16x = 2^4.
    sz_u256_vec_t not_first_bit_vec, not_second_bit_vec, not_third_bit_vec, not_fourth_bit_vec;

    // Top and bottom nibbles of the source are used separately.
    sz_u256_vec_t source_vec, source_low_nibble_vec;
    sz_u256_vec_t blended_0_to_31_vec, blended_32_to_63_vec, blended_64_to_95_vec, blended_96_to_127_vec,
        blended_128_to_159_vec, blended_160_to_191_vec, blended_192_to_223_vec, blended_224_to_255_vec;

    // Handling the head.
    while (length >= 32) {
        // Load and separate the nibbles of each byte in the source.
        source_vec.ymm = _mm256_lddqu_si256((__m256i const *)source);
        source_low_nibble_vec.ymm = _mm256_and_si256(source_vec.ymm, _mm256_set1_epi8((char)0x0F));

        // In the first round, we select using the 4th bit.
        not_fourth_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x10), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8(                             //
            _mm256_shuffle_epi8(lut_16_to_31_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_0_to_15_vec.ymm, source_low_nibble_vec.ymm),  //
            not_fourth_bit_vec.ymm);
        blended_32_to_63_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_48_to_63_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_32_to_47_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_64_to_95_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_80_to_95_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_64_to_79_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_96_to_127_vec.ymm = _mm256_blendv_epi8(                             //
            _mm256_shuffle_epi8(lut_112_to_127_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_96_to_111_vec.ymm, source_low_nibble_vec.ymm),  //
            not_fourth_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_144_to_159_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_128_to_143_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_160_to_191_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_176_to_191_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_160_to_175_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_192_to_223_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_208_to_223_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_192_to_207_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_224_to_255_vec.ymm = _mm256_blendv_epi8(                            //
            _mm256_shuffle_epi8(lut_240_to_255_vec.ymm, source_low_nibble_vec.ymm), //
            _mm256_shuffle_epi8(lut_224_to_239_vec.ymm, source_low_nibble_vec.ymm), //
            not_fourth_bit_vec.ymm);

        // Perform a tree-like reduction of the 8x "blended" YMM registers, depending on the "source" content.
        // The first round selects using the 3rd bit.
        not_third_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x20), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( //
            blended_32_to_63_vec.ymm,                 //
            blended_0_to_31_vec.ymm,                  //
            not_third_bit_vec.ymm);
        blended_64_to_95_vec.ymm = _mm256_blendv_epi8( //
            blended_96_to_127_vec.ymm,                 //
            blended_64_to_95_vec.ymm,                  //
            not_third_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8( //
            blended_160_to_191_vec.ymm,                  //
            blended_128_to_159_vec.ymm,                  //
            not_third_bit_vec.ymm);
        blended_192_to_223_vec.ymm = _mm256_blendv_epi8( //
            blended_224_to_255_vec.ymm,                  //
            blended_192_to_223_vec.ymm,                  //
            not_third_bit_vec.ymm);

        // The second round selects using the 2nd bit.
        not_second_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x40), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( //
            blended_64_to_95_vec.ymm,                 //
            blended_0_to_31_vec.ymm,                  //
            not_second_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8( //
            blended_192_to_223_vec.ymm,                  //
            blended_128_to_159_vec.ymm,                  //
            not_second_bit_vec.ymm);

        // The third round selects using the 1st bit.
        not_first_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x80), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( //
            blended_128_to_159_vec.ymm,               //
            blended_0_to_31_vec.ymm,                  //
            not_first_bit_vec.ymm);

        // And dump the result into the target.
        _mm256_storeu_si256((__m256i *)target, blended_0_to_31_vec.ymm);
        source += 32, target += 32, length -= 32;
    }

    // Handle the tail.
    if (length) sz_lookup_serial(target, length, source, lut);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_MEMORY_HASWELL_H_
