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
    // Slightly larger buffers - between one and two vectors, copied as two predicated halves.
    else if (length <= 2 * vector_length) {
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
    // Slightly larger buffers - between one and two vectors, copied as two predicated halves.
    else if (length <= 2 * vector_length) {
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
    sz_size_t const vector_length = svcntb();

    // Hold the 256-entry table as sixteen 16-byte rows: the high nibble of a source byte picks the row, the
    // low nibble indexes within it. `svtbl_u8` only ever sees indices in [0, 16), so each lookup stays inside
    // one 16-lane row and is correct at every vector length.
    svbool_t const row_predicate = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)16);
    svuint8_t const row_0_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0x00));
    svuint8_t const row_1_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0x10));
    svuint8_t const row_2_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0x20));
    svuint8_t const row_3_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0x30));
    svuint8_t const row_4_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0x40));
    svuint8_t const row_5_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0x50));
    svuint8_t const row_6_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0x60));
    svuint8_t const row_7_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0x70));
    svuint8_t const row_8_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0x80));
    svuint8_t const row_9_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0x90));
    svuint8_t const row_10_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0xA0));
    svuint8_t const row_11_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0xB0));
    svuint8_t const row_12_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0xC0));
    svuint8_t const row_13_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0xD0));
    svuint8_t const row_14_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0xE0));
    svuint8_t const row_15_u8x = svld1_u8(row_predicate, (sz_u8_t const *)(lut + 0xF0));

    // Single predicated stream: `svwhilelt` yields all-true for whole vectors and a partial mask for the final
    // stretch, so the tail folds into the loop with no separate peeled epilogue.
    for (sz_size_t byte_index = 0; byte_index < length; byte_index += vector_length) {
        svbool_t const active = svwhilelt_b8_u64(byte_index, length);
        svuint8_t const source_u8x = svld1_u8(active, (sz_u8_t const *)(source + byte_index));
        svuint8_t const column_u8x = svand_n_u8_x(active, source_u8x, 0x0F);
        svuint8_t const high_nibble_u8x = svlsr_n_u8_x(active, source_u8x, 4);

        // Flat 16-way select tree keyed on the high nibble; `row_0_u8x` is the default for high nibble 0.
        svuint8_t result_u8x = svtbl_u8(row_0_u8x, column_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 1), svtbl_u8(row_1_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 2), svtbl_u8(row_2_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 3), svtbl_u8(row_3_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 4), svtbl_u8(row_4_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 5), svtbl_u8(row_5_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 6), svtbl_u8(row_6_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 7), svtbl_u8(row_7_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 8), svtbl_u8(row_8_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 9), svtbl_u8(row_9_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 10), svtbl_u8(row_10_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 11), svtbl_u8(row_11_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 12), svtbl_u8(row_12_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 13), svtbl_u8(row_13_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 14), svtbl_u8(row_14_u8x, column_u8x), result_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(active, high_nibble_u8x, 15), svtbl_u8(row_15_u8x, column_u8x), result_u8x);

        svst1_u8(active, (sz_u8_t *)(target + byte_index), result_u8x);
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
