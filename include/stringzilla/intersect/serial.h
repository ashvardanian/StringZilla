/**
 *  @brief Serial backend for set intersection.
 *  @file include/stringzilla/intersect/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/intersect.h
 */
#ifndef STRINGZILLA_INTERSECT_SERIAL_H_
#define STRINGZILLA_INTERSECT_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_fill`
#include "stringzilla/hash.h"    // `sz_hash`

#ifdef __cplusplus
extern "C" {
#endif

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
    sz_size_t const hash_table_slots = sz_size_bit_ceil(small_sequence->count) * (1u << SZ_SEQUENCE_INTERSECT_BUDGET);
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

    alloc->free(table_positions, hash_table_slots * bytes_per_entry, alloc);
    *intersection_count_ptr = intersection_count;
    return sz_success_k;
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_INTERSECT_SERIAL_H_
