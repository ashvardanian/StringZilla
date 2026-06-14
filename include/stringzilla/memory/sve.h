/**
 *  @brief SVE backend for hardware-accelerated memory operations on Arm v9 CPUs.
 *  @file include/stringzilla/memory/sve.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_SVE_H_
#define STRINGZILLA_MEMORY_SVE_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

SZ_PUBLIC void sz_fill_sve(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    svuint8_t value_u8x = svdup_u8(value);
    sz_size_t vector_length = svcntb(); // Vector length in bytes (scalable)

    if (length <= vector_length) {
        // Small buffer case: use mask to handle small writes
        svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
        svst1_u8(mask, (sz_u8_t *)target, value_u8x);
    }
    else {
        // Calculate head, body, and tail sizes
        sz_size_t head_length = vector_length - ((sz_size_t)target % vector_length);
        sz_size_t tail_length = (sz_size_t)(target + length) % vector_length;
        sz_size_t body_length = length - head_length - tail_length;

        // Handle unaligned head
        svbool_t head_mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)head_length);
        svst1_u8(head_mask, (sz_u8_t *)target, value_u8x);
        target += head_length;

        // Aligned body loop
        for (; body_length >= vector_length; target += vector_length, body_length -= vector_length)
            svst1_u8(svptrue_b8(), (sz_u8_t *)target, value_u8x);

        // Handle unaligned tail
        svbool_t tail_mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)tail_length);
        svst1_u8(tail_mask, (sz_u8_t *)target, value_u8x);
    }
}

SZ_PUBLIC void sz_copy_sve(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_size_t vector_length = svcntb(); // Vector length in bytes

    // When the buffer is small, there isn't much to innovate.
    if (length <= vector_length) {
        // Small buffer case: use mask to handle small writes
        svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
        svuint8_t source_u8x = svld1_u8(mask, (sz_u8_t *)source);
        svst1_u8(mask, (sz_u8_t *)target, source_u8x);
    }
    // Slightly larger buffers
    else if (2 * length <= vector_length) {
        svbool_t mask_first = svptrue_b8();
        svbool_t mask_second = svwhilelt_b8((sz_u64_t)vector_length, (sz_u64_t)length);
        svuint8_t source_first_u8x = svld1_u8(mask_first, (sz_u8_t *)(source));
        svuint8_t source_second_u8x = svld1_u8(mask_second, (sz_u8_t *)(source + vector_length));
        svst1_u8(mask_first, (sz_u8_t *)(target), source_first_u8x);
        svst1_u8(mask_second, (sz_u8_t *)(target + vector_length), source_second_u8x);
    }
    // For medium-sized buffers, use unidirectional traversal without non-temporal operations
    else {
        // Main loop: full vector copies
        for (; length >= vector_length; source += vector_length, target += vector_length, length -= vector_length) {
            svuint8_t source_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)source);
            svst1_u8(svptrue_b8(), (sz_u8_t *)target, source_u8x);
        }

        // Tail: single masked copy for remainder
        if (length) {
            svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
            svuint8_t source_u8x = svld1_u8(mask, (sz_u8_t const *)source);
            svst1_u8(mask, (sz_u8_t *)target, source_u8x);
        }
    }
}

SZ_PUBLIC void sz_move_sve(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_size_t vector_length = svcntb(); // Vector length in bytes

    // When the buffer is small, there isn't much to innovate.
    if (length <= vector_length) {
        // Small buffer case: use mask to handle small writes
        svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
        svuint8_t source_u8x = svld1_u8(mask, (sz_u8_t *)source);
        svst1_u8(mask, (sz_u8_t *)target, source_u8x);
    }
    // Slightly larger buffers
    else if (2 * length <= vector_length) {
        svbool_t mask_first = svptrue_b8();
        svbool_t mask_second = svwhilelt_b8((sz_u64_t)vector_length, (sz_u64_t)length);
        svuint8_t source_first_u8x = svld1_u8(mask_first, (sz_u8_t *)(source));
        svuint8_t source_second_u8x = svld1_u8(mask_second, (sz_u8_t *)(source + vector_length));
        svst1_u8(mask_first, (sz_u8_t *)(target), source_first_u8x);
        svst1_u8(mask_second, (sz_u8_t *)(target + vector_length), source_second_u8x);
    }
    // For medium-sized buffers, check for overlap
    else {
        // Check if regions overlap with target after source
        int const overlapping = (target > source && target < source + length);

        if (overlapping) {
            // Backward traversal to avoid overwriting source data
            source += length;
            target += length;

            // Backward main loop
            for (; length >= vector_length; length -= vector_length) {
                source -= vector_length;
                target -= vector_length;
                svuint8_t source_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)source);
                svst1_u8(svptrue_b8(), (sz_u8_t *)target, source_u8x);
            }

            // Backward tail
            if (length) {
                source -= length;
                target -= length;
                svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
                svuint8_t source_u8x = svld1_u8(mask, (sz_u8_t const *)source);
                svst1_u8(mask, (sz_u8_t *)target, source_u8x);
            }
        }
        else {
            // Forward traversal (safe for non-overlapping or target < source)
            // Main loop: full vector copies
            for (; length >= vector_length; source += vector_length, target += vector_length, length -= vector_length) {
                svuint8_t source_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)source);
                svst1_u8(svptrue_b8(), (sz_u8_t *)target, source_u8x);
            }

            // Tail: single masked copy for remainder
            if (length) {
                svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
                svuint8_t source_u8x = svld1_u8(mask, (sz_u8_t const *)source);
                svst1_u8(mask, (sz_u8_t *)target, source_u8x);
            }
        }
    }
}

SZ_PUBLIC void sz_lookup_sve(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]) {

    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    // SVE vector length in bytes
    sz_size_t vector_length = svcntb();

    // Load the 256-byte lookup table into 4 SVE vectors
    svuint8_t lut_0_to_63_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)(lut + 0));
    svuint8_t lut_64_to_127_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)(lut + 64));
    svuint8_t lut_128_to_191_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)(lut + 128));
    svuint8_t lut_192_to_255_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)(lut + 192));

    svuint8_t low_6bits_mask_u8x = svdup_u8(0x3f);

    sz_size_t byte_index = 0;

    // Main loop: process full vectors
    while (byte_index + vector_length <= length) {
        svuint8_t source_u8x = svld1_u8(svptrue_b8(), (sz_u8_t const *)(source + byte_index));

        // Create predicates based on top 2 bits (which 64-byte range)
        svbool_t pred_0_63 = svcmplt_n_u8(svptrue_b8(), source_u8x, 64);
        svbool_t pred_64_127 = svcmpge_n_u8(svcmplt_n_u8(svptrue_b8(), source_u8x, 128), source_u8x, 64);
        svbool_t pred_128_191 = svcmpge_n_u8(svcmplt_n_u8(svptrue_b8(), source_u8x, 192), source_u8x, 128);
        svbool_t pred_192_255 = svcmpge_n_u8(svptrue_b8(), source_u8x, 192);

        // Mask indices to bottom 6 bits for indexing within each 64-byte table
        svuint8_t index_u8x = svand_u8_x(svptrue_b8(), source_u8x, low_6bits_mask_u8x);

        // Perform lookups and blend results based on predicates
        svuint8_t result_u8x = svsel_u8(pred_0_63, svtbl_u8(lut_0_to_63_u8x, index_u8x), svdup_u8(0));
        result_u8x = svsel_u8(pred_64_127, svtbl_u8(lut_64_to_127_u8x, index_u8x), result_u8x);
        result_u8x = svsel_u8(pred_128_191, svtbl_u8(lut_128_to_191_u8x, index_u8x), result_u8x);
        result_u8x = svsel_u8(pred_192_255, svtbl_u8(lut_192_to_255_u8x, index_u8x), result_u8x);

        svst1_u8(svptrue_b8(), (sz_u8_t *)target + byte_index, result_u8x);
        byte_index += vector_length;
    }

    // Handle tail: process remaining elements with predicated operations
    if (byte_index < length) {
        svbool_t tail_pred = svwhilelt_b8_u64(byte_index, length);
        svuint8_t source_u8x = svld1_u8(tail_pred, (sz_u8_t const *)(source + byte_index));

        // Create predicates for each range (comparison already uses tail_pred as governing predicate)
        svbool_t pred_0_63 = svcmplt_n_u8(tail_pred, source_u8x, 64);
        svbool_t pred_64_127 = svcmpge_n_u8(svcmplt_n_u8(tail_pred, source_u8x, 128), source_u8x, 64);
        svbool_t pred_128_191 = svcmpge_n_u8(svcmplt_n_u8(tail_pred, source_u8x, 192), source_u8x, 128);
        svbool_t pred_192_255 = svcmpge_n_u8(tail_pred, source_u8x, 192);

        // Mask indices to bottom 6 bits
        svuint8_t index_u8x = svand_u8_x(tail_pred, source_u8x, low_6bits_mask_u8x);

        svuint8_t result_u8x = svsel_u8(pred_0_63, svtbl_u8(lut_0_to_63_u8x, index_u8x), svdup_u8(0));
        result_u8x = svsel_u8(pred_64_127, svtbl_u8(lut_64_to_127_u8x, index_u8x), result_u8x);
        result_u8x = svsel_u8(pred_128_191, svtbl_u8(lut_128_to_191_u8x, index_u8x), result_u8x);
        result_u8x = svsel_u8(pred_192_255, svtbl_u8(lut_192_to_255_u8x, index_u8x), result_u8x);

        svst1_u8(tail_pred, (sz_u8_t *)target + byte_index, result_u8x);
    }
}
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_MEMORY_SVE_H_
