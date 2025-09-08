/**
 *  @brief  Hardware-accelerated string collection intersections for JOIN-like DBMS operations.
 *  @file   intersect.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs for `sz_sequence_t` string collections with hardware-specific backends:
 *
 *  - `sz_sequence_intersect` - to compute the strict intersection of two deduplicated string collections.
 *  - TODO: `sz_sequence_join` - to compute the intersection of two arbitrary string collections.
 */
#ifndef STRINGZILLA_INTERSECT_H_
#define STRINGZILLA_INTERSECT_H_

#include "types.h"

#include "compare.h" // `sz_compare`
#include "memory.h"  // `sz_fill`
#include "hash.h"    // `sz_hash`

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  The @b power-of-two memory-usage budget @b multiple for the hash table.
 *
 *  The behaviour of hashing-based approaches can often be tuned with different "hyper-parameter" values.
 *  For "unordered set intersections" implemented here, the @p budget argument controls the balance between
 *  throughput and memory usage. The higher the budget, the more memory is used, but the fewer collisions
 *  will be observed
 */
#if !defined(SZ_SEQUENCE_INTERSECT_BUDGET)
#define SZ_SEQUENCE_INTERSECT_BUDGET (1)
#endif

#pragma region Core API

/**
 *  @brief  Intersects two @b deduplicated binary @b string sequences, using a hash table.
 *          Outputs the @p first_positions from the @p first_sequence and @p second_positions from
 *          the @p second_sequence, that contain matched strings. Missing matches are represented as `SZ_SIZE_MAX`.
 *
 *  @param[in] first_sequence First immutable sequence of strings to intersection.
 *  @param[in] second_sequence Second immutable sequence of strings to intersection.
 *  @param[in] semantics JOIN semantics for the intersection, including handling of duplicates.
 *  @param[in] alloc Optional memory allocator for temporary storage.
 *  @param[in] seed Optional seed for the hash table to avoid attacks.
 *  @param[out] intersection_size Number of matching strings in both sequences.
 *  @param[out] first_positions Offset positions of the matching strings from the @p first_sequence.
 *  @param[out] second_positions Offset positions of the matching strings from the @p second_sequence.
 *
 *  @retval `sz_success_k` if the operation was successful.
 *  @retval `sz_bad_alloc_k` if the operation failed due to memory allocation failure.
 *  @pre The @p first_positions array must fit at least `min(first_sequence->count, second_sequence->count)` items.
 *  @pre The @p second_positions array must fit at least `min(first_sequence->count, second_sequence->count)` items.
 *  @warning Doesn't check for duplicates and won't return `sz_contains_duplicates_k`. Duplicates result in UB.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/intersect.h>
 *      int main() {
 *          char const *first[] = {"banana", "apple", "cherry"};
 *          char const *second[] = {"cherry", "orange", "pineapple", "banana"};
 *          sz_sequence_t first_sequence, second_sequence;
 *          sz_sequence_from_null_terminated_strings(first, 3, &first_sequence);
 *          sz_sequence_from_null_terminated_strings(second, 4, &second_sequence);
 *          sz_size_t intersection_size;
 *          sz_sorted_idx_t first_positions[3], second_positions[3]; //? 3 is the size of the smaller sequence
 *          sz_status_t status = sz_sequence_intersect(&first_sequence, &second_sequence,
 *              sz_join_inner_strict_k, NULL, 0,
 *              &intersection_size, first_positions, second_positions);
 *          return status == sz_success_k && intersection_size == 2 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note   The algorithm has linear memory complexity and linear time complexity.
 *  @see    https://en.wikipedia.org/wiki/Join_(SQL)
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_sequence_intersect_serial, sz_sequence_intersect_ice, sz_sequence_intersect_sve
 */
SZ_DYNAMIC sz_status_t sz_sequence_intersect(sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence,
                                             sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_size,
                                             sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions);

/**
 *  @brief  Defines various JOIN semantics for string sequences, including handling of duplicates.
 *  @sa     sz_join_inner_strict_k, sz_join_inner_k, sz_join_left_outer_k, sz_join_right_outer_k, sz_join_full_outer_k,
 *          sz_join_cross_k
 */
typedef enum {
    /**
     *  @brief  Strict inner join with uniqueness enforcement.
     *
     *  In this mode, only unique matching strings from both sequences are returned.
     *  If either sequence contains duplicate strings, the operation will fail.
     *
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana", "cherry" }
     *      second_sequence: { "banana", "cherry", "date" }
     *  - Output:
     *      Result: { ("banana", "banana"), ("cherry", "cherry") }
     *
     *  SQL equivalent:
     *  @code{.sql}
     *  -- Returns unique matching rows only.
     *  SELECT DISTINCT a.*
     *  FROM first_sequence a
     *  INNER JOIN second_sequence b ON a.string = b.string;
     *  @endcode
     */
    sz_join_inner_strict_k = 0,

    /**
     *  @brief  Conventional inner join allowing duplicate entries.
     *
     *  This mode returns all pairs of matching strings from both sequences.
     *  Each occurrence in the first sequence is paired with every matching occurrence
     *  in the second sequence. Order stability is not guaranteed.
     *
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana", "banana" }
     *      second_sequence: { "banana", "banana", "cherry" }
     *  - Output:
     *      Result: { ("banana", "banana"), ("banana", "banana"),
     *                ("banana", "banana"), ("banana", "banana") }
     *      (2 occurrences of "banana" in the first sequence × 2 in the second = 4 pairs)
     *
     *  SQL equivalent:
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a
     *  INNER JOIN second_sequence b ON a.string = b.string;
     *  @endcode
     */
    sz_join_inner_k = 1,

    /**
     *  @brief  Left outer join preserving all entries from the first sequence.
     *
     *  This mode returns every string from the first sequence along with matching strings
     *  from the second sequence. If no match is found for an element in the first sequence,
     *  the corresponding output for the second sequence is NULL (or its equivalent).
     *
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana", "cherry" }
     *      second_sequence: { "banana", "cherry", "date" }
     *  - Output:
     *      Result: { ("apple", NULL), ("banana", "banana"), ("cherry", "cherry") }
     *
     *  SQL equivalent:
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a
     *  LEFT OUTER JOIN second_sequence b ON a.string = b.string;
     *  @endcode
     */
    sz_join_left_outer_k = 2,

    /**
     *  @brief  Right outer join preserving all entries from the second sequence.
     *
     *  This mode returns every string from the second sequence along with matching strings
     *  from the first sequence. If no match is found for an element in the second sequence,
     *  the corresponding output for the first sequence is NULL (or its equivalent).
     *
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana" }
     *      second_sequence: { "banana", "cherry", "date" }
     *  - Output:
     *      Result: { ("banana", "banana"), (NULL, "cherry"), (NULL, "date") }
     *
     *  SQL equivalent:
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a
     *  RIGHT OUTER JOIN second_sequence b ON a.string = b.string;
     *  @endcode
     */
    sz_join_right_outer_k = 3,

    /**
     *  @brief  Full outer join combining all entries from both sequences.
     *
     *  This mode returns all matching pairs along with unmatched strings from both sequences.
     *  For unmatched strings, the corresponding result from the other sequence is NULL.
     *
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana" }
     *      second_sequence: { "banana", "cherry" }
     *  - Output:
     *      Result: { ("apple", NULL), ("banana", "banana"), (NULL, "cherry") }
     *
     *  SQL equivalent:
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a
     *  FULL OUTER JOIN second_sequence b ON a.string = b.string;
     *  @endcode
     */
    sz_join_full_outer_k = 4,

    /**
     *  @brief  Cross join (Cartesian product) of two sequences.
     *
     *  This mode returns the Cartesian product of both sequences, pairing every string in the first sequence
     *  with every string in the second sequence regardless of any matching condition.
     *
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana" }
     *      second_sequence: { "cherry", "date" }
     *  - Output:
     *      Result: { ("apple", "cherry"), ("apple", "date"),
     *                ("banana", "cherry"), ("banana", "date") }
     *
     *  SQL equivalent:
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a, second_sequence b;
     *  @endcode
     */
    sz_join_cross_k = 5,
} sz_sequence_join_semantics_t;

#if SZ_USE_ICE

/** @copydoc sz_sequence_intersect */
SZ_PUBLIC sz_status_t sz_sequence_intersect_ice(                               //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_size, //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions);

#endif

#if SZ_USE_SVE

/** @copydoc sz_sequence_intersect */
SZ_PUBLIC sz_status_t sz_sequence_intersect_sve(                               //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_size, //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions);

#endif

#pragma endregion

#pragma region Serial Implementation

SZ_PUBLIC sz_status_t sz_sequence_intersect_serial(                                 //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence,      //
    sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_count_ptr, //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions) {

    // To join to unordered sets of strings, the simplest approach would be to hash them into a dynamically
    // allocated hash table and then iterate over the second set, checking for the presence of each element in the
    // hash table. This would require O(N) memory and O(N) time complexity, where N is the smaller set.
    sz_sequence_t const *small_sequence, *large_sequence;
    sz_sorted_idx_t *small_positions, *large_positions;
    if (first_sequence->count <= second_sequence->count) {
        small_sequence = first_sequence, large_sequence = second_sequence;
        small_positions = first_positions, large_positions = second_positions;
    }
    else {
        small_sequence = second_sequence, large_sequence = first_sequence;
        small_positions = second_positions, large_positions = first_positions;
    }

    // We may very well have nothing to join
    if (small_sequence->count == 0) {
        *intersection_count_ptr = 0;
        return sz_success_k;
    }

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Allocate memory for the hash table and initialize it with 0xFF.
    // The higher is the `hash_table_slots` multiple - the more memory we will use,
    // but the less likely the collisions will be.
    sz_size_t const hash_table_slots = sz_size_bit_ceil(small_sequence->count) * (1 << SZ_SEQUENCE_INTERSECT_BUDGET);
    sz_size_t const bytes_per_entry = sizeof(sz_size_t) + sizeof(sz_u64_t);
    sz_size_t *const table_positions = (sz_size_t *)alloc->allocate(hash_table_slots * bytes_per_entry, alloc);
    if (!table_positions) return sz_bad_alloc_k;
    sz_u64_t *const table_hashes = (sz_u64_t *)(table_positions + hash_table_slots);
    sz_fill((sz_ptr_t)table_positions, hash_table_slots * bytes_per_entry, 0xFF);

    // Hash the smaller set into the hash table using the default available backend.
    for (sz_size_t small_position = 0; small_position < small_sequence->count; ++small_position) {
        sz_cptr_t const str = small_sequence->get_start(small_sequence->handle, small_position);
        sz_size_t const length = small_sequence->get_length(small_sequence->handle, small_position);
        sz_u64_t const hash = sz_hash(str, length, seed);
        sz_size_t hash_slot = hash & (hash_table_slots - 1);
        // Implement linear probing to find the first free slot.
        // If we somehow face 2 different strings with same hash, we will export that hash 2 times!
        while (table_hashes[hash_slot] != SZ_SIZE_MAX) hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
        table_hashes[hash_slot] = hash;
        table_positions[hash_slot] = small_position;
    }

    // Iterate over the larger set and check for the presence of each element in the hash table.
    sz_size_t intersection_count = 0;
    for (sz_size_t large_position = 0; large_position < large_sequence->count; ++large_position) {
        sz_cptr_t const str = large_sequence->get_start(large_sequence->handle, large_position);
        sz_size_t const length = large_sequence->get_length(large_sequence->handle, large_position);
        sz_u64_t const hash = sz_hash(str, length, seed);
        sz_size_t hash_slot = hash & (hash_table_slots - 1);

        // Implement linear probing to resolve collisions.
        for (; table_hashes[hash_slot] != SZ_SIZE_MAX; hash_slot = (hash_slot + 1) & (hash_table_slots - 1)) {
            sz_u64_t small_hash = table_hashes[hash_slot];
            if (small_hash != hash) continue;

            // The hash matches, compare the strings.
            sz_size_t const small_position = table_positions[hash_slot];
            sz_size_t const small_length = small_sequence->get_length(small_sequence->handle, small_position);
            if (length != small_length) continue;

            // Same hash may still imply different strings, so we need to compare them.
            sz_cptr_t const small_str = small_sequence->get_start(small_sequence->handle, small_position);
            sz_bool_t const same = sz_equal(str, small_str, length);
            if (same != sz_true_k) continue;

            // Finally, there is a match, store the positions.
            small_positions[intersection_count] = small_position;
            large_positions[intersection_count] = large_position;
            ++intersection_count;
            break;
        }
    }

    *intersection_count_ptr = intersection_count;
    return sz_success_k;
}

#pragma endregion // Serial Implementation

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 *
 *  We are going to use VBMI2 for `_mm256_maskz_compress_epi8`.
 */
#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#if defined(__clang__)
#pragma clang attribute push(                                                                                  \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vnni,bmi,bmi2,aes,vaes"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vnni", "bmi", "bmi2", \
                   "aes", "vaes")
#endif

SZ_INTERNAL int sz_u64x4_contains_collisions_haswell_(__m256i v) {
    // Assume `v` stores values: [a, b, c, d].
    __m256i cmp1 = _mm256_cmpeq_epi64(v, _mm256_permute4x64_epi64(v, 0xB1)); // 0xB1 produces [b, a, d, c]
    __m256i cmp2 = _mm256_cmpeq_epi64(v, _mm256_permute4x64_epi64(v, 0x4E)); // 0x4E produces [c, d, a, b]
    __m256i cmp3 = _mm256_cmpeq_epi64(v, _mm256_permute4x64_epi64(v, 0x1B)); // 0x1B produces [d, c, b, a]

    // Combine the results from the three comparisons.
    __m256i cmp = _mm256_or_si256(_mm256_or_si256(cmp1, cmp2), cmp3);

    // Each 64-bit lane comparison yields all ones if equal, so the movemask will be nonzero if any pair matched.
    int mask = _mm256_movemask_epi8(cmp);
    return mask;
}

SZ_PUBLIC sz_status_t sz_sequence_intersect_ice(                                    //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence,      //
    sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_count_ptr, //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions) {

    // To join to unordered sets of strings, the simplest approach would be to hash them into a dynamically
    // allocated hash table and then iterate over the second set, checking for the presence of each element in the
    // hash table. This would require O(N) memory and O(N) time complexity, where N is the smaller set.
    sz_sequence_t const *small_sequence, *large_sequence;
    sz_sorted_idx_t *small_positions, *large_positions;
    if (first_sequence->count <= second_sequence->count) {
        small_sequence = first_sequence, large_sequence = second_sequence;
        small_positions = first_positions, large_positions = second_positions;
    }
    else {
        small_sequence = second_sequence, large_sequence = first_sequence;
        small_positions = second_positions, large_positions = first_positions;
    }

    // We may very well have nothing to join
    if (small_sequence->count == 0) {
        *intersection_count_ptr = 0;
        return sz_success_k;
    }

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Allocate memory for the hash table and initialize it with 0xFF.
    // The higher is the `hash_table_slots` multiple - the more memory we will use,
    // but the less likely the collisions will be.
    sz_size_t const hash_table_slots = sz_size_bit_ceil(small_sequence->count) * (1 << SZ_SEQUENCE_INTERSECT_BUDGET);
    sz_size_t const bytes_per_entry = sizeof(sz_size_t) + sizeof(sz_u64_t);
    sz_size_t *table_positions = (sz_size_t *)alloc->allocate(hash_table_slots * bytes_per_entry, alloc);
    if (!table_positions) return sz_bad_alloc_k;
    sz_u64_t *table_hashes = (sz_u64_t *)(table_positions + hash_table_slots);
    sz_fill((sz_ptr_t)table_positions, hash_table_slots * bytes_per_entry, 0xFF);

    // Conceptually the Ice Lake variant is similar to the serial one, except it takes advantage of:
    // - computing 4x individual high-quality hashes with `_mm512_aesenc_epi128`.
    // - gathering values from the hash-table using `_mm256_mmask_i64gather_epi64`.
    //
    // We still start by hashing the smaller set into the hash table, but we will process 4 entries
    // at a time and will separately handle values under 16 bytes fitting into one AES block and the
    // larger values.
    //
    // For larger entries, we will use a separate loop afterwards to decrease the likelihood of collisions
    // on the shorter entries, that can benefit from vectorized processing.
    sz_hash_minimal_x4_t_ batch_hashes_states_initial;
    sz_hash_minimal_x4_init_ice_(&batch_hashes_states_initial, seed);
    sz_size_t count_longer = 0;
    for (sz_size_t small_position = 0; small_position < small_sequence->count;) {
        sz_string_view_t batch[4];
        sz_u256_vec_t batch_positions;
        sz_size_t batch_size;
        for (batch_size = 0; batch_size < 4 && small_position < small_sequence->count; ++small_position) {
            sz_size_t length = small_sequence->get_length(small_sequence->handle, small_position);
            if (length > 16) {
                count_longer++;
                continue;
            }
            sz_cptr_t str = small_sequence->get_start(small_sequence->handle, small_position);
            batch[batch_size].start = str;
            batch[batch_size].length = length;
            batch_positions.u64s[batch_size] = small_position;
            ++batch_size;
        }

        // If we couldn't populate the whole batch, fall back to the serial solution
        if (batch_size != 4) {
            for (sz_size_t i = 0; i < batch_size; ++i) {
                sz_cptr_t const str = batch[i].start;
                sz_size_t const length = batch[i].length;
                sz_u64_t const hash = sz_hash(str, length, seed);
                sz_size_t hash_slot = hash & (hash_table_slots - 1);
                // Implement linear probing to find the first free slot.
                // If we somehow face 2 different strings with same hash, we will export that hash 2 times!
                while (table_hashes[hash_slot] != SZ_SIZE_MAX) hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
                table_hashes[hash_slot] = hash;
                table_positions[hash_slot] = batch_positions.u64s[i];
            }
        }
        // The batch is successfully populated, let's use the vectorized solution
        else {
            // Now let's load the first bytes of each string.
            sz_u256_vec_t batch_hashes;
            sz_u512_vec_t batch_prefixes;
            batch_prefixes.xmms[0] = _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[0].length), batch[0].start);
            batch_prefixes.xmms[1] = _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[1].length), batch[1].start);
            batch_prefixes.xmms[2] = _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[2].length), batch[2].start);
            batch_prefixes.xmms[3] = _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[3].length), batch[3].start);

            // Reuse the already computed state for hashes
            sz_hash_minimal_x4_t_ batch_hashes_states = batch_hashes_states_initial;
            sz_hash_minimal_x4_update_ice_(&batch_hashes_states, batch_prefixes.zmm);
            batch_hashes.ymm = sz_hash_minimal_x4_finalize_ice_(&batch_hashes_states, batch[0].length, batch[1].length,
                                                                batch[2].length, batch[3].length);
            sz_assert_(batch_hashes.u64s[0] == sz_hash(batch[0].start, batch[0].length, seed));
            sz_assert_(batch_hashes.u64s[1] == sz_hash(batch[1].start, batch[1].length, seed));
            sz_assert_(batch_hashes.u64s[2] == sz_hash(batch[2].start, batch[2].length, seed));
            sz_assert_(batch_hashes.u64s[3] == sz_hash(batch[3].start, batch[3].length, seed));

            // Now let's perform an optimistic hash-table lookup using vectorized gathers
            sz_u256_vec_t batch_slots, existing_hashes;
            batch_slots.ymm = _mm256_and_si256(batch_hashes.ymm, _mm256_set1_epi64x(hash_table_slots - 1));

            // In case of very small inputs, it's more likely, that some of the 4x hashes or their slots will collide
            int const has_slot_collisions = sz_u64x4_contains_collisions_haswell_(batch_slots.ymm);

            // Before scattering the new positions - gather the pre-existing ones.
            // In case of `has_slot_collisions`, this will practically be a "prefetch" operation.
            existing_hashes.ymm =
                _mm256_mmask_i64gather_epi64(_mm256_setzero_si256(), 0xFF, batch_slots.ymm, table_hashes, 8);

            // Check that we don't have any collisions - in that case each value will be equal to `SZ_SIZE_MAX`
            int const all_empty = _mm256_testc_si256(existing_hashes.ymm, _mm256_set1_epi64x(-1));
            if (all_empty && !has_slot_collisions) {
                // Scatter the new positions
                _mm256_mask_i64scatter_epi64(table_hashes, 0xFF, batch_slots.ymm, batch_hashes.ymm, 8);
                _mm256_mask_i64scatter_epi64(table_positions, 0xFF, batch_slots.ymm, batch_positions.ymm, 8);
            }
            else {
                // We have a collision, let's resolve it with a serial solution
                for (sz_size_t i = 0; i < 4; ++i) {
                    sz_size_t hash_slot = batch_slots.u64s[i] & (hash_table_slots - 1);
                    // Implement linear probing to find the first free slot.
                    // If we somehow face 2 different strings with same hash, we will export that hash 2 times!
                    while (table_hashes[hash_slot] != SZ_SIZE_MAX) hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
                    table_hashes[hash_slot] = batch_hashes.u64s[i];
                    table_positions[hash_slot] = batch_positions.u64s[i];
                }
            }
        }
    }

    // Now, let's cross-reference all shorter values from the larger collection.
    sz_size_t intersection_count = 0;
    for (sz_size_t large_position = 0; large_position < large_sequence->count;) {
        sz_string_view_t batch[4];
        sz_u256_vec_t batch_positions;
        sz_size_t batch_size;
        for (batch_size = 0; batch_size < 4 && large_position < large_sequence->count; ++large_position) {
            sz_size_t length = large_sequence->get_length(large_sequence->handle, large_position);
            if (length > 16) {
                count_longer++;
                continue;
            }
            sz_cptr_t str = large_sequence->get_start(large_sequence->handle, large_position);
            batch[batch_size].start = str;
            batch[batch_size].length = length;
            batch_positions.u64s[batch_size] = large_position;
            ++batch_size;
        }

        // If we couldn't populate the whole batch, fall back to the serial solution
        if (batch_size != 4) {
            for (sz_size_t i = 0; i < batch_size; ++i) {
                sz_cptr_t const str = batch[i].start;
                sz_size_t const length = batch[i].length;
                sz_u64_t const hash = sz_hash(str, length, seed);
                sz_size_t hash_slot = hash & (hash_table_slots - 1);
                // Implement linear probing to resolve collisions.
                for (; table_hashes[hash_slot] != SZ_SIZE_MAX; hash_slot = (hash_slot + 1) & (hash_table_slots - 1)) {
                    sz_u64_t small_hash = table_hashes[hash_slot];
                    if (small_hash != hash) continue;

                    // The hash matches, compare the strings.
                    sz_size_t const small_position = table_positions[hash_slot];
                    sz_size_t const small_length = small_sequence->get_length(small_sequence->handle, small_position);
                    if (length != small_length) continue;

                    // Same hash may still imply different strings, so we need to compare them.
                    sz_cptr_t const small_str = small_sequence->get_start(small_sequence->handle, small_position);
                    sz_bool_t const same = sz_equal(str, small_str, length);
                    if (same != sz_true_k) continue;

                    // Finally, there is a match, store the positions.
                    small_positions[intersection_count] = small_position;
                    large_positions[intersection_count] = batch_positions.u64s[i];
                    ++intersection_count;
                    break;
                }
            }
        }
        // The batch is successfully populated, let's use the vectorized solution
        else {
            // Now let's load the first bytes of each string.
            sz_u256_vec_t batch_hashes;
            sz_u512_vec_t batch_prefixes;
            batch_prefixes.xmms[0] = _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[0].length), batch[0].start);
            batch_prefixes.xmms[1] = _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[1].length), batch[1].start);
            batch_prefixes.xmms[2] = _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[2].length), batch[2].start);
            batch_prefixes.xmms[3] = _mm_maskz_loadu_epi8(sz_u16_mask_until_(batch[3].length), batch[3].start);

            // Reuse the already computed state for hashes
            sz_hash_minimal_x4_t_ batch_hashes_states = batch_hashes_states_initial;
            sz_hash_minimal_x4_update_ice_(&batch_hashes_states, batch_prefixes.zmm);
            batch_hashes.ymm = sz_hash_minimal_x4_finalize_ice_(&batch_hashes_states, batch[0].length, batch[1].length,
                                                                batch[2].length, batch[3].length);
            sz_assert_(batch_hashes.u64s[0] == sz_hash(batch[0].start, batch[0].length, seed));
            sz_assert_(batch_hashes.u64s[1] == sz_hash(batch[1].start, batch[1].length, seed));
            sz_assert_(batch_hashes.u64s[2] == sz_hash(batch[2].start, batch[2].length, seed));
            sz_assert_(batch_hashes.u64s[3] == sz_hash(batch[3].start, batch[3].length, seed));

            // Now let's perform an optimistic hash-table lookup using vectorized gathers.
            sz_u256_vec_t batch_slots, existing_hashes;
            batch_slots.ymm = _mm256_and_si256(batch_hashes.ymm, _mm256_set1_epi64x(hash_table_slots - 1));

            // Before scattering the new positions - gather the pre-existing ones.
            // This can help us detect values:
            // - that are definitely missing in the hash table, if the slot is just NULL-ed
            // - that may be present in the hash table, and need to be validated in the loop
            existing_hashes.ymm =
                _mm256_mmask_i64gather_epi64(_mm256_setzero_si256(), 0xFF, batch_slots.ymm, table_hashes, 8);

            // Check if we already have all of those slots populated with exactly the same values
            int const same_hashes = _mm256_movemask_epi8(_mm256_cmpeq_epi64(existing_hashes.ymm, batch_hashes.ymm));
            int const nulled_hashes =
                _mm256_movemask_epi8(_mm256_cmpeq_epi64(existing_hashes.ymm, _mm256_set1_epi64x(-1)));

            // Now for every one of the 4 hashed values we can have several outcomes:
            // - it's an "empty" value → no match
            // - it's a different hash → continue probing
            // - it's the same hash for a different string, so we have a rare collision → continue probing
            // - it's the same hash for the same string, so we have a match → export
            //
            // That logic is too complex to be effectively handled by SIMD, so we switch back to serial code.
            for (sz_size_t i = 0; i < 4; ++i) {
                sz_cptr_t const str = batch[i].start;
                sz_size_t const length = batch[i].length;
                sz_u64_t const hash = batch_hashes.u64s[i];
                int const same_hash = (same_hashes >> (8 * i)) & 0xFF;
                int const nulled_hash = (nulled_hashes >> (8 * i)) & 0xFF;
                if (nulled_hash) continue;

                sz_size_t hash_slot = batch_slots.u64s[i];
                // This optimization may look like just one less  memory load,
                // but it will help us produce a different set of branches and will affect
                // the branch prediction quality on the CPU backend.
                if (same_hash) {
                    // The hash matches, compare the strings.
                    sz_size_t const small_position = table_positions[hash_slot];
                    sz_size_t const small_length = small_sequence->get_length(small_sequence->handle, small_position);
                    if (length == small_length) {
                        // Same hash may still imply different strings, so we need to compare them.
                        sz_cptr_t const small_str = small_sequence->get_start(small_sequence->handle, small_position);
                        sz_bool_t const same = sz_equal(str, small_str, length);
                        if (same == sz_true_k) {
                            // Finally, there is a match, store the positions.
                            small_positions[intersection_count] = small_position;
                            large_positions[intersection_count] = batch_positions.u64s[i];
                            ++intersection_count;
                            // Now go to the next value in the batch.
                            continue;
                        }
                    }
                    // If any of the conditions above didn't hold, just continue probing.
                    hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
                }

                // Implement linear probing to resolve collisions.
                for (; table_hashes[hash_slot] != SZ_SIZE_MAX; hash_slot = (hash_slot + 1) & (hash_table_slots - 1)) {
                    sz_u64_t small_hash = table_hashes[hash_slot];
                    if (small_hash != hash) continue;

                    // The hash matches, compare the strings.
                    sz_size_t const small_position = table_positions[hash_slot];
                    sz_size_t const small_length = small_sequence->get_length(small_sequence->handle, small_position);
                    if (length != small_length) continue;

                    // Same hash may still imply different strings, so we need to compare them.
                    sz_cptr_t const small_str = small_sequence->get_start(small_sequence->handle, small_position);
                    sz_bool_t const same = sz_equal(str, small_str, length);
                    if (same != sz_true_k) continue;

                    // Finally, there is a match, store the positions.
                    small_positions[intersection_count] = small_position;
                    large_positions[intersection_count] = batch_positions.u64s[i];
                    ++intersection_count;
                    break;
                }
            }
        }
    }

    // TODO: Consider one more level of partitioning, separating the values into [17:64] and [64:] ranges.
    if (count_longer) {
        // At this point only large values are remaining, let's process them with the code identical to our
        // serial solution, but dispatching the right Ice Lake kernel under the hood.
        sz_fill((sz_ptr_t)table_positions, hash_table_slots * bytes_per_entry, 0xFF);

        // Hash the smaller set into the hash table using the default available backend.
        for (sz_size_t small_position = 0; small_position < small_sequence->count; ++small_position) {
            sz_size_t const length = small_sequence->get_length(small_sequence->handle, small_position);
            if (length <= 16) continue; //! This is the only difference from the serial solution
            sz_cptr_t const str = small_sequence->get_start(small_sequence->handle, small_position);
            sz_u64_t const hash = sz_hash(str, length, seed);
            sz_size_t hash_slot = hash & (hash_table_slots - 1);
            // Implement linear probing to find the first free slot.
            // If we somehow face 2 different strings with same hash, we will export that hash 2 times!
            while (table_hashes[hash_slot] != SZ_SIZE_MAX) hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
            table_hashes[hash_slot] = hash;
            table_positions[hash_slot] = small_position;
        }

        // Iterate over the larger set and check for the presence of each element in the hash table.
        for (sz_size_t large_position = 0; large_position < large_sequence->count; ++large_position) {
            sz_size_t const length = large_sequence->get_length(large_sequence->handle, large_position);
            if (length <= 16) continue; //! This is the only difference from the serial solution
            sz_cptr_t const str = large_sequence->get_start(large_sequence->handle, large_position);
            sz_u64_t const hash = sz_hash(str, length, seed);
            sz_size_t hash_slot = hash & (hash_table_slots - 1);

            // Implement linear probing to resolve collisions.
            for (; table_hashes[hash_slot] != SZ_SIZE_MAX; hash_slot = (hash_slot + 1) & (hash_table_slots - 1)) {
                sz_u64_t small_hash = table_hashes[hash_slot];
                if (small_hash != hash) continue;

                // The hash matches, compare the strings.
                sz_size_t const small_position = table_positions[hash_slot];
                sz_size_t const small_length = small_sequence->get_length(small_sequence->handle, small_position);
                if (length != small_length) continue;

                // Same hash may still imply different strings, so we need to compare them.
                sz_cptr_t const small_str = small_sequence->get_start(small_sequence->handle, small_position);
                sz_bool_t const same = sz_equal(str, small_str, length);
                if (same != sz_true_k) continue;

                // Finally, there is a match, store the positions.
                small_positions[intersection_count] = small_position;
                large_positions[intersection_count] = large_position;
                ++intersection_count;
                break;
            }
        }
    }

    // Finalize
    *intersection_count_ptr = intersection_count;
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

#pragma region SVE Implementation
#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

SZ_PUBLIC sz_status_t sz_sequence_intersect_sve(sz_sequence_t const *first_sequence,
                                                sz_sequence_t const *second_sequence, //
                                                sz_memory_allocator_t *alloc, sz_u64_t seed,
                                                sz_size_t *intersection_size, sz_sorted_idx_t *first_positions,
                                                sz_sorted_idx_t *second_positions) {
    // TODO: Finalize `sz_hash_sve2_upto16x16_` and integrate here
    return sz_sequence_intersect_serial( //
        first_sequence, second_sequence, //
        alloc, seed, intersection_size,  //
        first_positions, second_positions);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SVE
#pragma endregion // SVE Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_status_t sz_sequence_intersect(sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence,
                                             sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_size,
                                             sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions) {
#if SZ_USE_SKYLAKE
    return sz_sequence_intersect_ice(    //
        first_sequence, second_sequence, //
        alloc, seed, intersection_size,  //
        first_positions, second_positions);
#elif SZ_USE_SVE
    return sz_sequence_intersect_sve(    //
        first_sequence, second_sequence, //
        alloc, seed, intersection_size,  //
        first_positions, second_positions);
#else
    return sz_sequence_intersect_serial( //
        first_sequence, second_sequence, //
        alloc, seed, intersection_size,  //
        first_positions, second_positions);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_INTERSECT_H_
