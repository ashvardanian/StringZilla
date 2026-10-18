/**
 *  @brief RISC-V Vector (RVV 1.0) backend for sorting string collections.
 *  @file include/stringzilla/sort/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/sort.h
 *
 *  Mirrors the SVE backend's length-agnostic, out-of-place 3-way QuickSort partition. RVV has an exact
 *  `__riscv_vcompress_vm_u64m*` (the analogue of SVE's `svcompact_u64`), so each `vsetvl` strip's smaller /
 *  equal / greater lanes are packed to the front and stored contiguously with no left-pack table and no
 *  region slack (unlike NEON's `vqtbl2q` compaction). `__riscv_vcpop_m_b64` counts the surviving lanes for
 *  the cursor advance. Compaction preserves lane order, so the partition is @b stable - matching the
 *  stable-by-default contract - and the recursion / stability / reverse / top-K / uncased
 *  machinery is reused verbatim from `sort/serial.h`.
 *
 *  Vector-length-agnostic: correct at VLEN 128 / 256 / 512 / 1024, because every strip is bounded by
 *  `__riscv_vsetvl_e64m1` and stores are masked to the per-strip surviving count.
 */
#ifndef STRINGZILLA_SORT_RVV_H_
#define STRINGZILLA_SORT_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_copy_rvv`

#include "stringzilla/sort/serial.h"

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

/**
 *  @brief The most important part of the QuickSort algorithm partitioning the elements around the pivot.
 *  @note Unlike the serial algorithm, uses compressed stores to filter and move the elements around the pivot.
 *  @sa Identical in shape to the @b SVE implementation, but uses variable-length RVV `vsetvl` strips and the
 *      `__riscv_vcompress_vm_u64m1` primitive in place of `svcompact_u64`.
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
SZ_HELPER_AUTO void sz_sequence_argsort_rvv_3way_partition_(
    sz_pgram_t *const initial_pgrams, sz_sorted_idx_t *const initial_order, sz_pgram_t *const partitioned_pgrams,
    sz_sorted_idx_t *const partitioned_order, sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,
    sz_size_t *const first_pivot_offset, sz_size_t *const last_pivot_offset) {
    sz_size_t const count = end_in_sequence - start_in_sequence;

    // Choose the pivot with Sedgewick's method.
    sz_pgram_t const *pivot_pgram_ptr = sz_sequence_partitioning_pivot_(initial_pgrams + start_in_sequence, count);
    sz_pgram_t const pivot_pgram = *pivot_pgram_ptr;

    // Count elements smaller and greater than the pivot, one `vsetvl` strip at a time.
    sz_size_t count_smaller = 0, count_greater = 0;
    for (sz_size_t block_index = start_in_sequence; block_index < end_in_sequence;) {
        sz_size_t const vector_length = __riscv_vsetvl_e64m1(end_in_sequence - block_index);
        vuint64m1_t pgrams_u64m1 = __riscv_vle64_v_u64m1((sz_u64_t const *)(initial_pgrams + block_index),
                                                         vector_length);
        vbool64_t smaller_b64 = __riscv_vmsltu_vx_u64m1_b64(pgrams_u64m1, pivot_pgram, vector_length);
        vbool64_t greater_b64 = __riscv_vmsgtu_vx_u64m1_b64(pgrams_u64m1, pivot_pgram, vector_length);
        count_smaller += __riscv_vcpop_m_b64(smaller_b64, vector_length);
        count_greater += __riscv_vcpop_m_b64(greater_b64, vector_length);
        block_index += vector_length;
    }

    sz_size_t const count_equal = count - count_smaller - count_greater;
    sz_assert_(count_equal >= 1 && "The pivot must be present in the collection.");
    sz_assert_(count_smaller + count_equal + count_greater == count && "The partitioning must be exhaustive.");

    // Set offsets for each partition.
    sz_size_t smaller_offset = start_in_sequence;
    sz_size_t equal_offset = start_in_sequence + count_smaller;
    sz_size_t greater_offset = start_in_sequence + count_smaller + count_equal;

    // Partition elements into three segments using exact compressed stores.
    for (sz_size_t block_index = start_in_sequence; block_index < end_in_sequence;) {
        sz_size_t const vector_length = __riscv_vsetvl_e64m1(end_in_sequence - block_index);
        vuint64m1_t pgrams_u64m1 = __riscv_vle64_v_u64m1((sz_u64_t const *)(initial_pgrams + block_index),
                                                         vector_length);
        vuint64m1_t order_u64m1 = __riscv_vle64_v_u64m1((sz_u64_t const *)(initial_order + block_index), vector_length);

        vbool64_t smaller_b64 = __riscv_vmsltu_vx_u64m1_b64(pgrams_u64m1, pivot_pgram, vector_length);
        vbool64_t greater_b64 = __riscv_vmsgtu_vx_u64m1_b64(pgrams_u64m1, pivot_pgram, vector_length);
        // The equal lanes are the active lanes that are neither smaller nor greater, so a NOR derives them
        // without a third compare; lanes past `vector_length` are ignored by the bounded compress / store.
        vbool64_t equal_b64 = __riscv_vmnor_mm_b64(smaller_b64, greater_b64, vector_length);

        // Compress the elements that satisfy the predicate and store them contiguously. A compress of count
        // zero is a no-op, so no per-segment branch guards the stores.
        sz_size_t block_count_smaller = __riscv_vcpop_m_b64(smaller_b64, vector_length);
        sz_size_t block_count_equal = __riscv_vcpop_m_b64(equal_b64, vector_length);
        sz_size_t block_count_greater = __riscv_vcpop_m_b64(greater_b64, vector_length);

        vuint64m1_t comp_smaller_pgrams_u64m1 = __riscv_vcompress_vm_u64m1(pgrams_u64m1, smaller_b64, vector_length);
        vuint64m1_t comp_smaller_order_u64m1 = __riscv_vcompress_vm_u64m1(order_u64m1, smaller_b64, vector_length);
        __riscv_vse64_v_u64m1((sz_u64_t *)(partitioned_pgrams + smaller_offset), comp_smaller_pgrams_u64m1,
                              block_count_smaller);
        __riscv_vse64_v_u64m1((sz_u64_t *)(partitioned_order + smaller_offset), comp_smaller_order_u64m1,
                              block_count_smaller);
        smaller_offset += block_count_smaller;

        vuint64m1_t comp_equal_pgrams_u64m1 = __riscv_vcompress_vm_u64m1(pgrams_u64m1, equal_b64, vector_length);
        vuint64m1_t comp_equal_order_u64m1 = __riscv_vcompress_vm_u64m1(order_u64m1, equal_b64, vector_length);
        __riscv_vse64_v_u64m1((sz_u64_t *)(partitioned_pgrams + equal_offset), comp_equal_pgrams_u64m1,
                              block_count_equal);
        __riscv_vse64_v_u64m1((sz_u64_t *)(partitioned_order + equal_offset), comp_equal_order_u64m1,
                              block_count_equal);
        equal_offset += block_count_equal;

        vuint64m1_t comp_greater_pgrams_u64m1 = __riscv_vcompress_vm_u64m1(pgrams_u64m1, greater_b64, vector_length);
        vuint64m1_t comp_greater_order_u64m1 = __riscv_vcompress_vm_u64m1(order_u64m1, greater_b64, vector_length);
        __riscv_vse64_v_u64m1((sz_u64_t *)(partitioned_pgrams + greater_offset), comp_greater_pgrams_u64m1,
                              block_count_greater);
        __riscv_vse64_v_u64m1((sz_u64_t *)(partitioned_order + greater_offset), comp_greater_order_u64m1,
                              block_count_greater);
        greater_offset += block_count_greater;
        block_index += vector_length;
    }

    // Copy back.
    sz_copy_rvv((sz_ptr_t)(initial_pgrams + start_in_sequence),      //
                (sz_cptr_t)(partitioned_pgrams + start_in_sequence), //
                count * sizeof(sz_pgram_t));
    sz_copy_rvv((sz_ptr_t)(initial_order + start_in_sequence),      //
                (sz_cptr_t)(partitioned_order + start_in_sequence), //
                count * sizeof(sz_sorted_idx_t));

    // Return the offsets of the equal elements.
    *first_pivot_offset = start_in_sequence + count_smaller;
    *last_pivot_offset = start_in_sequence + count_smaller + count_equal - 1;
}

/**
 *  @brief Recursive Quick-Sort implementation backing both the `sz_sequence_argsort_rvv` and
 *      `sz_pgrams_sort_rvv`, and using the `sz_sequence_argsort_rvv_3way_partition_` under the hood.
 *  @sa Identical in shape to the @b SVE implementation, but uses variable-length RVV `vsetvl` strips.
 *
 *  @param initial_pgrams Pgram array to sort in place.
 *  @param initial_order Corresponding order array, permuted in sync with `initial_pgrams`.
 *  @param temporary_pgrams Scratch buffer of the same size as `initial_pgrams`, used during partitioning.
 *  @param temporary_order Scratch buffer of the same size as `initial_order`, used during partitioning.
 *  @param start_in_sequence First index (inclusive) of the range to sort.
 *  @param end_in_sequence One-past-the-last index of the range to sort.
 *  @param top_count Global top-K cut-off forwarded to the partitioner; 0 fully sorts the range.
 */
SZ_API_COMPTIME void sz_sequence_argsort_rvv_quicksort_pgrams_(
    sz_pgram_t *initial_pgrams, sz_sorted_idx_t *initial_order, sz_pgram_t *temporary_pgrams,
    sz_sorted_idx_t *temporary_order, sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,
    sz_size_t const top_count) {
    sz_size_t const count = end_in_sequence - start_in_sequence;
    if (count <= 32) {
        // For very small arrays use a simple stable insertion sort.
        sz_pgrams_sort_with_insertion(initial_pgrams + start_in_sequence, count, initial_order + start_in_sequence);
        return;
    }

    sz_size_t first_pivot_index, last_pivot_index;
    sz_sequence_argsort_rvv_3way_partition_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                            start_in_sequence, end_in_sequence, &first_pivot_index, &last_pivot_index);

    if (start_in_sequence + 1 < first_pivot_index)
        sz_sequence_argsort_rvv_quicksort_pgrams_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                                  start_in_sequence, first_pivot_index, top_count);
    if (last_pivot_index + 2 < end_in_sequence && (top_count == 0 || last_pivot_index + 1 < top_count))
        sz_sequence_argsort_rvv_quicksort_pgrams_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                                  last_pivot_index + 1, end_in_sequence, top_count);
}

SZ_API_COMPTIME sz_status_t sz_pgrams_sort_rvv(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                               sz_sorted_idx_t *order) {
    // Initialize the order with 0,1,2,...
    for (sz_size_t pgram_index = 0; pgram_index != count; ++pgram_index) order[pgram_index] = pgram_index;

    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Allocate temporary memory for partitioning. The RVV compress-store is exact, so no slack is needed.
    sz_size_t memory_usage = sizeof(sz_pgram_t) * count + sizeof(sz_sorted_idx_t) * count;
    sz_pgram_t *temporary_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    if (!temporary_pgrams) return sz_bad_alloc_k;

    sz_sequence_argsort_rvv_quicksort_pgrams_(pgrams, order, temporary_pgrams, temporary_order, 0, count, 0);

    alloc->free(temporary_pgrams, memory_usage, alloc);
    return sz_success_k;
}

typedef struct sz_argsort_rvv_scratch_t {
    sz_pgram_t *temporary_pgrams;
    sz_sorted_idx_t *temporary_order;
} sz_argsort_rvv_scratch_t;

SZ_HELPER_AUTO void sz_argsort_rvv_sort_range_(sz_pgram_t *pgrams, sz_sorted_idx_t *order, sz_size_t start,
                                               sz_size_t end, sz_size_t top_count, void *context) {
    sz_argsort_rvv_scratch_t *scratch = (sz_argsort_rvv_scratch_t *)context;
    sz_sequence_argsort_rvv_quicksort_pgrams_(pgrams, order, scratch->temporary_pgrams, scratch->temporary_order, start,
                                              end, top_count);
}

/**
 *  @brief Quick-Sort adaptation for strings, that processes the strings a few N-grams at a time.
 *      It combines `sz_sequence_argsort_serial_export_byte_window_` and `sz_sequence_argsort_rvv_quicksort_pgrams_`.
 *      Equal pgrams are walked iteratively, so a long shared prefix does not grow the call stack.
 *  @sa Identical in shape to the @b SVE implementation, but uses variable-length RVV `vsetvl` strips.
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
SZ_API_COMPTIME void sz_sequence_argsort_rvv_sort_byte_windows_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const start_character, sz_size_t const top_count,
    sz_bool_t const reverse, sz_argsort_window_frame_t *const frames) {

    sz_argsort_rvv_scratch_t scratch = {temporary_pgrams, temporary_order};
    sz_argsort_walk_byte_windows_(sequence, global_pgrams, global_order, start_in_sequence, end_in_sequence,
                                  start_character, top_count, reverse, sz_argsort_rvv_sort_range_, &scratch, frames);
}

SZ_API_COMPTIME sz_status_t sz_sequence_argsort_rvv(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse) {
    sz_size_t count = sequence->count;
    for (sz_size_t sequence_index = 0; sequence_index != count; ++sequence_index)
        order[sequence_index] = sequence_index;
    if (count < 2) return sz_success_k;

    if (count <= 32 && !reverse) {
        sz_sequence_argsort_with_insertion(sequence, order);
        return sz_success_k;
    }

    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    sz_size_t memory_usage = sizeof(sz_pgram_t) * count * 2 + sizeof(sz_sorted_idx_t) * count +
                             sizeof(sz_argsort_window_frame_t) * count;
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!global_pgrams) return sz_bad_alloc_k;
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    sz_argsort_window_frame_t *frames = (sz_argsort_window_frame_t *)(temporary_order + count);

    sz_sequence_argsort_rvv_sort_byte_windows_(sequence, global_pgrams, order, temporary_pgrams, temporary_order, 0,
                                               count, 0, top_count, reverse, frames);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

/**
 *  @brief Uncased twin of `sz_sequence_argsort_rvv_sort_byte_windows_`: the folded code-point export
 *      stays scalar (and is shared with the serial backend), but the pgrams it produces are sorted with the
 *      RVV partition - which is where RVV beats the fully-serial uncased path.
 */
SZ_API_COMPTIME void sz_sequence_argsort_rvv_sort_casefold_windows_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const folded_skip_count, sz_size_t const top_count,
    sz_bool_t const reverse, sz_argsort_window_frame_t *const frames) {

    sz_argsort_rvv_scratch_t scratch = {temporary_pgrams, temporary_order};
    sz_argsort_walk_casefold_windows_(sequence, global_pgrams, global_order, start_in_sequence, end_in_sequence,
                                      folded_skip_count, top_count, reverse, sz_argsort_rvv_sort_range_, &scratch,
                                      frames);
}

SZ_API_COMPTIME sz_status_t sz_sequence_argsort_uncased_rvv(     //
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
    // RVV compress-store is exact, so no slack is needed. The folded export is stateless (re-folds the prefix
    // on demand), so unlike the earlier design there is no per-string cursor array.
    sz_size_t const memory_usage = sizeof(sz_pgram_t) * count * 2 + sizeof(sz_sorted_idx_t) * count +
                                   sizeof(sz_argsort_window_frame_t) * count;
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!global_pgrams) return sz_bad_alloc_k;
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    sz_argsort_window_frame_t *frames = (sz_argsort_window_frame_t *)(temporary_order + count);

    sz_sequence_argsort_rvv_sort_casefold_windows_(sequence, global_pgrams, order, temporary_pgrams, temporary_order, 0,
                                                   count, 0, top_count, reverse, frames);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
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

#endif // STRINGZILLA_SORT_RVV_H_
