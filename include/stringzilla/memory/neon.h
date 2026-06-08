/**
 *  @brief NEON backend for hardware-accelerated memory operations on 64-bit Arm CPUs.
 *  @file include/stringzilla/memory/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_NEON_H_
#define STRINGZILLA_MEMORY_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

SZ_PUBLIC void sz_copy_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    // In most cases the `source` and the `target` are not aligned, but we should
    // at least make sure that writes don't touch many cache lines.
    // NEON has an instruction to load and write 64 bytes at once.
    //
    //    sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
    //    sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
    //    for (; head_length; target += 1, source += 1, head_length -= 1) *target = *source;
    //    length -= head_length;
    //    for (; length >= 64; target += 64, source += 64, length -= 64)
    //        vst4q_u8((sz_u8_t *)target, vld1q_u8_x4((sz_u8_t const *)source));
    //    for (; tail_length; target += 1, source += 1, tail_length -= 1) *target = *source;
    //
    // Sadly, those instructions end up being 20% slower than the code processing 16 bytes at a time:
    for (; length >= 16; target += 16, source += 16, length -= 16)
        vst1q_u8((sz_u8_t *)target, vld1q_u8((sz_u8_t const *)source));
    if (length) sz_copy_serial(target, source, length);
}

SZ_PUBLIC void sz_move_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    // When moving small buffers, using a small buffer on stack as a temporary storage is faster.

    if (target < source || target >= source + length) {
        // Non-overlapping, proceed forward
        sz_copy_neon(target, source, length);
    }
    else {
        // Overlapping, proceed backward
        target += length;
        source += length;

        sz_u128_vec_t source_vec;
        while (length >= 16) {
            target -= 16, source -= 16, length -= 16;
            source_vec.u8x16 = vld1q_u8((sz_u8_t const *)source);
            vst1q_u8((sz_u8_t *)target, source_vec.u8x16);
        }
        while (length) {
            target -= 1, source -= 1, length -= 1;
            *target = *source;
        }
    }
}

SZ_PUBLIC void sz_fill_neon(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    uint8x16_t fill_u8x16 = vdupq_n_u8(value); // Broadcast the value across the register

    while (length >= 16) {
        vst1q_u8((sz_u8_t *)target, fill_u8x16);
        target += 16;
        length -= 16;
    }

    // Handle remaining bytes
    if (length) sz_fill_serial(target, length, value);
}

SZ_PUBLIC void sz_lookup_neon(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]) {

    // If the input is tiny (especially smaller than the look-up table itself), we may end up paying
    // more for organizing the SIMD registers and changing the CPU state, than for the actual computation.
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    sz_size_t head_length = (16 - ((sz_size_t)target % 16)) % 16; // 15 or less.
    sz_size_t tail_length = (sz_size_t)(target + length) % 16;    // 15 or less.
    sz_size_t body_length = length - head_length - tail_length;

    // We need to pull the lookup table into 16x NEON registers. We have a total of 32 such registers.
    // According to the Neoverse V2 manual, the 4-table lookup has a latency of 6 cycles, and 4x throughput.
    uint8x16x4_t lut_0_to_63_vec, lut_64_to_127_vec, lut_128_to_191_vec, lut_192_to_255_vec;
    lut_0_to_63_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 0));
    lut_64_to_127_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 64));
    lut_128_to_191_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 128));
    lut_192_to_255_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 192));

    sz_u128_vec_t source_vec;
    // If the top bit is set in each word of `source_vec`, than we use `lookup_128_to_191_vec` or
    // `lookup_192_to_255_vec`. If the second bit is set, we use `lookup_64_to_127_vec` or `lookup_192_to_255_vec`.
    sz_u128_vec_t lookup_0_to_63_vec, lookup_64_to_127_vec, lookup_128_to_191_vec, lookup_192_to_255_vec;
    sz_u128_vec_t blended_0_to_255_vec;

    // Process the head with serial code
    for (; head_length; target += 1, source += 1, head_length -= 1) *target = lut[*(sz_u8_t const *)source];

    // Table lookups on Arm are much simpler to use than on x86, as we can use the `vqtbl4q_u8` instruction
    // to perform a 4-table lookup in a single instruction. The XORs are used to adjust the lookup position
    // within each 64-byte range of the table.
    // Details on the 4-table lookup: https://lemire.me/blog/2019/07/23/arbitrary-byte-to-byte-maps-using-arm-neon/
    for (; body_length >= 16; source += 16, target += 16, body_length -= 16) {
        source_vec.u8x16 = vld1q_u8((sz_u8_t const *)source);
        lookup_0_to_63_vec.u8x16 = vqtbl4q_u8(lut_0_to_63_vec, source_vec.u8x16);
        lookup_64_to_127_vec.u8x16 = vqtbl4q_u8(lut_64_to_127_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0x40)));
        lookup_128_to_191_vec.u8x16 = vqtbl4q_u8(lut_128_to_191_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0x80)));
        lookup_192_to_255_vec.u8x16 = vqtbl4q_u8(lut_192_to_255_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0xc0)));
        blended_0_to_255_vec.u8x16 = vorrq_u8( //
            vorrq_u8(lookup_0_to_63_vec.u8x16, lookup_64_to_127_vec.u8x16),
            vorrq_u8(lookup_128_to_191_vec.u8x16, lookup_192_to_255_vec.u8x16));
        vst1q_u8((sz_u8_t *)target, blended_0_to_255_vec.u8x16);
    }

    // Process the tail with serial code
    for (; tail_length; target += 1, source += 1, tail_length -= 1) *target = lut[*(sz_u8_t const *)source];
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_MEMORY_NEON_H_
