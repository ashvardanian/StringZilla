/**
 *  @brief AVX-512 (Skylake) backend for sorting string collections.
 *  @file include/stringzilla/sort/skylake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/sort.h
 */
#ifndef STRINGZILLA_SORT_SKYLAKE_H_
#define STRINGZILLA_SORT_SKYLAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_copy`

#include "stringzilla/sort/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 *
 *  We are going to use VBMI2 for `_mm256_maskz_compress_epi8`.
 */
#if SZ_USE_SKYLAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,evex512,popcnt"))), \
                             apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,popcnt"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2", "popcnt")
#endif

/**
 *  @brief The most important part of the QuickSort algorithm partitioning the elements around the pivot.
 *  @note Unlike the serial algorithm, uses compressed stores to filter and move the elements around the pivot.
 *
 *  @param initial_pgrams Source pgram array; updated in place after copy-back.
 *  @param initial_order Corresponding order array; updated in place after copy-back.
 *  @param partitioned_pgrams Temporary output buffer for the three-way partitioned pgrams.
 *  @param partitioned_order Temporary output buffer for the corresponding order entries.
 *  @param start_in_sequence First index (inclusive) of the range to partition.
 *  @param end_in_sequence One-past-the-last index of the range to partition.
 *  @param first_pivot_offset Receives the index of the first element equal to the pivot.
 *  @param last_pivot_offset Receives the index of the last element equal to the pivot.
 */
SZ_HELPER_AUTO void sz_sequence_argsort_skylake_3way_partition_(                    //
    sz_pgram_t *const initial_pgrams, sz_sorted_idx_t *const initial_order,         //
    sz_pgram_t *const partitioned_pgrams, sz_sorted_idx_t *const partitioned_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,             //
    sz_size_t *const first_pivot_offset, sz_size_t *const last_pivot_offset) {

    sz_size_t const count = end_in_sequence - start_in_sequence;
    sz_size_t const pgrams_per_register = sizeof(sz_u512_vec_t) / sizeof(sz_pgram_t);

    // Choose the pivot offset with Sedgewick's method.
    sz_pgram_t const *pivot_pgram_ptr = sz_sequence_partitioning_pivot_(initial_pgrams + start_in_sequence, count);
    sz_pgram_t const pivot_pgram = *pivot_pgram_ptr;
    sz_u512_vec_t pivot_vec;
    pivot_vec.zmm = _mm512_set1_epi64(pivot_pgram);

    // Reading data is always cheaper than writing, so we can further minimize the writes, if
    // we know exactly, how many elements are smaller or greater than the pivot.
    sz_size_t count_smaller = 0, count_greater = 0;
    sz_size_t const tail_count = count & 7u;
    __mmask8 const tail_mask_m8 = sz_u8_mask_until_(tail_count);

    sz_u512_vec_t pgrams_vec, order_vec;
    for (sz_size_t block_index = start_in_sequence; block_index + pgrams_per_register <= end_in_sequence;
         block_index += pgrams_per_register) {
        pgrams_vec.zmm = _mm512_loadu_si512(initial_pgrams + block_index);
        count_smaller += _mm_popcnt_u32(_mm512_cmplt_epu64_mask(pgrams_vec.zmm, pivot_vec.zmm));
        count_greater += _mm_popcnt_u32(_mm512_cmpgt_epu64_mask(pgrams_vec.zmm, pivot_vec.zmm));
    }
    if (tail_count) {
        pgrams_vec.zmm = _mm512_maskz_loadu_epi64(tail_mask_m8, initial_pgrams + end_in_sequence - tail_count);
        count_smaller += _mm_popcnt_u32(_mm512_mask_cmplt_epu64_mask(tail_mask_m8, pgrams_vec.zmm, pivot_vec.zmm));
        count_greater += _mm_popcnt_u32(_mm512_mask_cmpgt_epu64_mask(tail_mask_m8, pgrams_vec.zmm, pivot_vec.zmm));
    }

    // Now all we need to do is to loop through the collection and export them into the temporary buffer
    // in 3 separate segments - smaller, equal, and greater than the pivot.
    sz_size_t const count_equal = count - count_smaller - count_greater;
    sz_assert_(count_equal >= 1 && "The pivot must be present in the collection.");
    sz_assert_(count_smaller + count_equal + count_greater == count && "The partitioning must be exhaustive.");
    sz_size_t smaller_offset = start_in_sequence;
    sz_size_t equal_offset = start_in_sequence + count_smaller;
    sz_size_t greater_offset = start_in_sequence + count_smaller + count_equal;

    // The naive algorithm - unzip the elements into 3 separate buffers.
    for (sz_size_t block_index = start_in_sequence; block_index < end_in_sequence; block_index += pgrams_per_register) {
        __mmask8 const load_mask_m8 = block_index + pgrams_per_register <= end_in_sequence ? 0xFF : tail_mask_m8;
        pgrams_vec.zmm = _mm512_maskz_loadu_epi64(load_mask_m8, initial_pgrams + block_index);
        order_vec.zmm = _mm512_maskz_loadu_epi64(load_mask_m8, initial_order + block_index);

        __mmask8 const smaller_mask_m8 = _mm512_mask_cmplt_epu64_mask(load_mask_m8, pgrams_vec.zmm, pivot_vec.zmm);
        __mmask8 const greater_mask_m8 = _mm512_mask_cmpgt_epu64_mask(load_mask_m8, pgrams_vec.zmm, pivot_vec.zmm);
        // The equal lanes are the active lanes that are neither smaller nor greater, so a NOR within the
        // load mask derives them without a third masked compare.
        __mmask8 const equal_mask_m8 = (__mmask8)(load_mask_m8 & ~(smaller_mask_m8 | greater_mask_m8));

        // Compress the elements into the temporary buffer.
        _mm512_mask_compressstoreu_epi64(partitioned_pgrams + smaller_offset, smaller_mask_m8, pgrams_vec.zmm);
        _mm512_mask_compressstoreu_epi64(partitioned_order + smaller_offset, smaller_mask_m8, order_vec.zmm);
        smaller_offset += _mm_popcnt_u32(smaller_mask_m8);

        _mm512_mask_compressstoreu_epi64(partitioned_pgrams + equal_offset, equal_mask_m8, pgrams_vec.zmm);
        _mm512_mask_compressstoreu_epi64(partitioned_order + equal_offset, equal_mask_m8, order_vec.zmm);
        equal_offset += _mm_popcnt_u32(equal_mask_m8);

        _mm512_mask_compressstoreu_epi64(partitioned_pgrams + greater_offset, greater_mask_m8, pgrams_vec.zmm);
        _mm512_mask_compressstoreu_epi64(partitioned_order + greater_offset, greater_mask_m8, order_vec.zmm);
        greater_offset += _mm_popcnt_u32(greater_mask_m8);
    }

    // Copy back.
    sz_copy_skylake((sz_ptr_t)(initial_pgrams + start_in_sequence),      //
                    (sz_cptr_t)(partitioned_pgrams + start_in_sequence), //
                    count * sizeof(sz_pgram_t));
    sz_copy_skylake((sz_ptr_t)(initial_order + start_in_sequence),      //
                    (sz_cptr_t)(partitioned_order + start_in_sequence), //
                    count * sizeof(sz_sorted_idx_t));

    // Return the offsets of the equal elements.
    *first_pivot_offset = start_in_sequence + count_smaller;
    *last_pivot_offset = start_in_sequence + count_smaller + count_equal - 1;
}

/**
 *  @brief Recursive Quick-Sort implementation backing both the `sz_sequence_argsort_skylake` and
 *      `sz_pgrams_sort_skylake`, and using the `sz_sequence_argsort_skylake_3way_partition_` under the hood.
 *
 *  @param initial_pgrams Pgram array to sort in place.
 *  @param initial_order Corresponding order array, permuted in sync with `initial_pgrams`.
 *  @param temporary_pgrams Scratch buffer of the same size as `initial_pgrams`, used during partitioning.
 *  @param temporary_order Scratch buffer of the same size as `initial_order`, used during partitioning.
 *  @param start_in_sequence First index (inclusive) of the range to sort.
 *  @param end_in_sequence One-past-the-last index of the range to sort.
 */
SZ_API_COMPTIME void sz_sequence_argsort_skylake_quicksort_pgrams_( //
    sz_pgram_t *initial_pgrams, sz_sorted_idx_t *initial_order,     //
    sz_pgram_t *temporary_pgrams, sz_sorted_idx_t *temporary_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence, sz_size_t const top_count) {

    // On very small inputs, when we don't even have enough input for a single ZMM register,
    // use simple insertion sort without any extra memory.
    sz_size_t const count = end_in_sequence - start_in_sequence;
    sz_size_t const pgrams_per_register = sizeof(sz_u512_vec_t) / sizeof(sz_pgram_t);
    if (count <= pgrams_per_register) {
        sz_pgrams_sort_with_insertion( //
            initial_pgrams + start_in_sequence, count, initial_order + start_in_sequence);
        return;
    }

    // Partition the collection around some pivot
    sz_size_t first_pivot_index, last_pivot_index;
    sz_sequence_argsort_skylake_3way_partition_(                          //
        initial_pgrams, initial_order, temporary_pgrams, temporary_order, //
        start_in_sequence, end_in_sequence,                               //
        &first_pivot_index, &last_pivot_index);

    // Recursively sort the left and right partitions, if there are at least 2 elements in each.
    // The right partition is skipped entirely if it lies past the `top_count` cut-off.
    if (start_in_sequence + 1 < first_pivot_index)
        sz_sequence_argsort_skylake_quicksort_pgrams_(                        //
            initial_pgrams, initial_order, temporary_pgrams, temporary_order, //
            start_in_sequence, first_pivot_index, top_count);
    if (last_pivot_index + 2 < end_in_sequence && (top_count == 0 || last_pivot_index + 1 < top_count))
        sz_sequence_argsort_skylake_quicksort_pgrams_(                        //
            initial_pgrams, initial_order, temporary_pgrams, temporary_order, //
            last_pivot_index + 1, end_in_sequence, top_count);
}

SZ_API_COMPTIME sz_status_t sz_pgrams_sort_skylake(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                                   sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t pgram_index = 0; pgram_index != count; ++pgram_index) order[pgram_index] = pgram_index;

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Allocate memory for partitioning the elements around the pivot.
    sz_size_t memory_usage = sizeof(sz_pgram_t) * count + sizeof(sz_sorted_idx_t) * count;
    sz_pgram_t *temporary_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    if (!temporary_pgrams) return sz_bad_alloc_k;

    // Reuse the string sorting algorithm for sorting the "pgrams" - a plain full ascending sort.
    sz_sequence_argsort_skylake_quicksort_pgrams_(pgrams, order, temporary_pgrams, temporary_order, 0, count, 0);

    // Deallocate the temporary memory used for partitioning.
    alloc->free(temporary_pgrams, memory_usage, alloc);
    return sz_success_k;
}

typedef struct sz_argsort_skylake_scratch_t {
    sz_pgram_t *temporary_pgrams;
    sz_sorted_idx_t *temporary_order;
} sz_argsort_skylake_scratch_t;

SZ_HELPER_AUTO void sz_argsort_skylake_sort_range_(sz_pgram_t *pgrams, sz_sorted_idx_t *order, sz_size_t start,
                                                   sz_size_t end, sz_size_t top_count, void *context) {
    sz_argsort_skylake_scratch_t *scratch = (sz_argsort_skylake_scratch_t *)context;
    sz_sequence_argsort_skylake_quicksort_pgrams_(pgrams, order, scratch->temporary_pgrams, scratch->temporary_order,
                                                  start, end, top_count);
}

/**
 *  @brief Quick-Sort adaptation for strings, that processes the strings a few N-grams at a time.
 *      It combines `sz_sequence_argsort_serial_export_byte_window_` and `sz_sequence_argsort_skylake_quicksort_pgrams_`.
 *      Equal pgrams are walked iteratively, so a long shared prefix does not grow the call stack.
 *
 *  @param sequence The collection of strings to sort.
 *  @param global_pgrams Working pgram array, length at least `sequence->count`.
 *  @param global_order Current permutation array, updated in place.
 *  @param temporary_pgrams Scratch buffer of the same size as `global_pgrams`.
 *  @param temporary_order Scratch buffer of the same size as `global_order`.
 *  @param start_in_sequence First index (inclusive) of the range to process.
 *  @param end_in_sequence One-past-the-last index of the range to process.
 *  @param start_character Byte offset into each string for the current pgram window.
 *  @param top_count Global top-K cut-off forwarded to the partitioner; 0 fully sorts the range.
 *  @param reverse Whether to export complemented keys for descending order.
 *  @param frames Scratch stack of at least `end_in_sequence - start_in_sequence` window frames.
 */
SZ_API_COMPTIME void sz_sequence_argsort_skylake_sort_byte_windows_(            //
    sz_sequence_t const *const sequence,                                        //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,       //
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,         //
    sz_size_t const start_character, sz_size_t const top_count, sz_bool_t const reverse,
    sz_argsort_window_frame_t *const frames) {

    sz_argsort_skylake_scratch_t scratch = {temporary_pgrams, temporary_order};
    sz_argsort_walk_byte_windows_(sequence, global_pgrams, global_order, start_in_sequence, end_in_sequence,
                                  start_character, top_count, reverse, sz_argsort_skylake_sort_range_, &scratch, frames);
}

SZ_API_COMPTIME sz_status_t sz_sequence_argsort_skylake(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                        sz_sorted_idx_t *order, sz_size_t top_count,
                                                        sz_bool_t reverse) {

    // First, initialize the `order` with `std::iota`-like behavior.
    sz_size_t count = sequence->count;
    for (sz_size_t sequence_index = 0; sequence_index != count; ++sequence_index)
        order[sequence_index] = sequence_index;
    if (count < 2) return sz_success_k;

    // On very small ascending collections - just use the quadratic-complexity @b stable insertion sort
    // without any smart optimizations or memory allocations. Descending small inputs fall through.
    if (count <= 32 && !reverse) {
        sz_sequence_argsort_with_insertion(sequence, order);
        return sz_success_k;
    }

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Allocate memory for partitioning the elements around the pivot.
    sz_size_t memory_usage = sizeof(sz_pgram_t) * count * 2 + sizeof(sz_sorted_idx_t) * count +
                             sizeof(sz_argsort_window_frame_t) * count;
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!global_pgrams) return sz_bad_alloc_k;
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    sz_argsort_window_frame_t *frames = (sz_argsort_window_frame_t *)(temporary_order + count);

    sz_sequence_argsort_skylake_sort_byte_windows_(sequence, global_pgrams, order, temporary_pgrams, temporary_order, //
                                                   0, count, 0, top_count, reverse, frames);

    // Free temporary storage.
    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

/**
 *  @brief Uncased twin of `sz_sequence_argsort_skylake_sort_byte_windows_`: the folded code-point export
 *      stays scalar (and is shared with the serial backend), but the pgrams it produces are sorted with the
 *      AVX-512 partition - which is where Skylake beats the fully-serial uncased path.
 */
SZ_API_COMPTIME void sz_sequence_argsort_skylake_sort_casefold_windows_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const folded_skip_count, sz_size_t const top_count,
    sz_bool_t const reverse, sz_argsort_window_frame_t *const frames) {

    sz_argsort_skylake_scratch_t scratch = {temporary_pgrams, temporary_order};
    sz_argsort_walk_casefold_windows_(sequence, global_pgrams, global_order, start_in_sequence, end_in_sequence,
                                      folded_skip_count, top_count, reverse, sz_argsort_skylake_sort_range_, &scratch,
                                      frames);
}

SZ_API_COMPTIME sz_status_t sz_sequence_argsort_uncased_skylake( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc, //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse) {

    sz_size_t const count = sequence->count;
    for (sz_size_t sequence_index = 0; sequence_index != count; ++sequence_index)
        order[sequence_index] = sequence_index;
    if (count < 2) return sz_success_k;

    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Same layout as the byte arg-sort - working pgrams (count) + the partition's two scratch regions. The
    // AVX-512 compress-store is exact, so no slack is needed. The folded export is stateless (re-folds the
    // prefix on demand), so unlike the earlier design there is no per-string cursor array.
    sz_size_t const memory_usage = sizeof(sz_pgram_t) * count * 2 + sizeof(sz_sorted_idx_t) * count +
                                   sizeof(sz_argsort_window_frame_t) * count;
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!global_pgrams) return sz_bad_alloc_k;
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    sz_argsort_window_frame_t *frames = (sz_argsort_window_frame_t *)(temporary_order + count);

    sz_sequence_argsort_skylake_sort_casefold_windows_(sequence, global_pgrams, order, temporary_pgrams,
                                                       temporary_order, 0, count, 0, top_count, reverse, frames);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SKYLAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SORT_SKYLAKE_H_
