/**
 *  @file include/stringzilla/intersect/serial.h
 *  @author Ash Vardanian
 *  @date March 7, 2025
 *  @brief Serial backend for set intersection.
 *
 *  @sa c/stringzilla/intersect.h
 */
#ifndef STRINGZILLA_INTERSECT_SERIAL_H_
#define STRINGZILLA_INTERSECT_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h" // `sz_equal_serial_`
#include "stringzilla/hash/serial.h"    // `sz_hash_serial_`

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief The @b power-of-two memory-usage budget @b multiple for the hash table.
 *
 *  The behaviour of hashing-based approaches can often be tuned with different "hyper-parameter"
 *  values. For "unordered set intersections" implemented here, the budget controls the balance
 *  between throughput and memory usage. The higher the budget, the more memory is used, but the
 *  fewer collisions will be observed.
 */
#if !defined(STRINGZILLA_INTERSECT_BUDGET)
#define STRINGZILLA_INTERSECT_BUDGET (1)
#endif

STRINGZILLA_INLINE sz_status_t sz_intersect_serial_(                                 //
    sz_strings_view_t const *first_strings, sz_strings_view_t const *second_strings, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count_ptr,     //
    sz_size_t *first_positions, sz_size_t *second_positions, sz_u64_t (*hasher)(sz_cptr_t, sz_size_t, sz_u64_t)) {

    // To join to unordered sets of strings, the simplest approach would be to hash them into a dynamically
    // allocated hash table and then iterate over the second set, checking for the presence of each element in the
    // hash table. This would require O(N) memory and O(N) time complexity, where N is the smaller set.
    sz_strings_view_t const *small_strings, *large_strings;
    sz_size_t *small_positions, *large_positions;
    if (first_strings->count <= second_strings->count) {
        small_strings = first_strings, large_strings = second_strings;
        small_positions = first_positions, large_positions = second_positions;
    }
    else {
        small_strings = second_strings, large_strings = first_strings;
        small_positions = second_positions, large_positions = first_positions;
    }

    // We may very well have nothing to join
    if (small_strings->count == 0) {
        *intersection_count_ptr = 0;
        return sz_success_k;
    }

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_allocator_t global_alloc;
    if (!allocator) {
        sz_status_t const status = sz_allocator_init_heap(&global_alloc);
        if (status != sz_success_k) return status;
        allocator = &global_alloc;
    }

    // Allocate memory for the hash table and initialize it with 0xFF.
    // The higher is the `hash_table_slots` multiple - the more memory we will use,
    // but the less likely the collisions will be.
    sz_size_t const hash_table_slots = sz_size_bit_ceil(small_strings->count) * (1u << STRINGZILLA_INTERSECT_BUDGET);
    sz_assert_(hash_table_slots > small_strings->count && "A full table leaves a missing key probing forever");
    sz_size_t const bytes_per_entry = sizeof(sz_size_t) + sizeof(sz_u64_t);
    sz_size_t *const table_positions = (sz_size_t *)allocator->allocate(hash_table_slots * bytes_per_entry,
                                                                        allocator->handle, STRINGZILLA_NULL);
    if (!table_positions) return sz_bad_alloc_k;
    sz_u64_t *const table_hashes = (sz_u64_t *)(table_positions + hash_table_slots);
    memset(table_positions, 0xFF, hash_table_slots * bytes_per_entry);
    // Empty-slot sentinel for the 64-bit `table_hashes`: the `0xFF` fill makes every slot all-ones.
    // It must be a 64-bit constant, not `STRINGZILLA_SIZE_MAX` - on 32-bit targets (e.g. wasm32)
    // `sz_size_t` is 32-bit, so `STRINGZILLA_SIZE_MAX` (0xFFFFFFFF) never equals the 64-bit fill,
    // and the probe loop would never terminate.
    sz_u64_t const empty_slot = ~(sz_u64_t)0;
    // The top bit of a stored position marks a slot whose distinct value already produced a pair,
    // so duplicate keys on either side never emit again, keeping the result a set no larger than
    // `min(counts)` that never overflows the caller's `positions` arrays. Positions are collection
    // indices, always far below 2^(word_bits-1), so the high bit is free to borrow.
    sz_size_t const consumed_flag = (sz_size_t)1 << (sizeof(sz_size_t) * 8 - 1);

    // Hash the smaller set into the hash table, reading the views 64 at a time.
    sz_string_view_t views[64];
    for (sz_size_t block_start = 0; block_start < small_strings->count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(small_strings, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_u64_t const hash = hasher(views[position].start, views[position].length, seed);
            sz_size_t hash_slot = hash & (hash_table_slots - 1);
            // Linear probing finds the first free slot; two strings sharing a hash export it twice.
            while (table_hashes[hash_slot] != empty_slot) hash_slot = (hash_slot + 1) & (hash_table_slots - 1);
            table_hashes[hash_slot] = hash;
            table_positions[hash_slot] = block_start + position;
        }
    }

    // Iterate over the larger set and check for the presence of each element in the hash table.
    sz_size_t intersection_count = 0;
    for (sz_size_t block_start = 0; block_start < large_strings->count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(large_strings, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_size_t const large_position = block_start + position;
            sz_cptr_t const str = views[position].start;
            sz_size_t const length = views[position].length;
            sz_u64_t const hash = hasher(str, length, seed);
            sz_size_t hash_slot = hash & (hash_table_slots - 1);

            // Implement linear probing to resolve collisions.
            for (; table_hashes[hash_slot] != empty_slot; hash_slot = (hash_slot + 1) & (hash_table_slots - 1)) {
                sz_u64_t small_hash = table_hashes[hash_slot];
                if (small_hash != hash) continue;

                // The hash matches, compare the strings.
                sz_size_t const stored = table_positions[hash_slot];
                sz_size_t const small_position = stored & ~consumed_flag;
                sz_string_view_t small_view;
                sz_strings_get_(small_strings, &small_position, 1, &small_view);
                if (length != small_view.length) continue;

                // Same hash may still imply different strings, so we need to compare them.
                sz_bool_t const same = sz_equal_serial_(str, small_view.start, length);
                if (same != sz_true_k) continue;

                // A duplicate key on either side: this value already contributed its pair.
                if (stored & consumed_flag) break;

                // Finally, there is a match: store the positions and mark the slot consumed.
                small_positions[intersection_count] = small_position;
                large_positions[intersection_count] = large_position;
                ++intersection_count;
                table_positions[hash_slot] = small_position | consumed_flag;
                break;
            }
        }
    }

    allocator->free(table_positions, hash_table_slots * bytes_per_entry, allocator->handle, STRINGZILLA_NULL);
    *intersection_count_ptr = intersection_count;
    return sz_success_k;
}

#if STRINGZILLA_TARGET_SERIAL

STRINGZILLA_API sz_status_t sz_intersect_serial(                                     //
    sz_strings_view_t const *first_strings, sz_strings_view_t const *second_strings, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count_ptr,     //
    sz_size_t *first_positions, sz_size_t *second_positions, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_intersect_serial_(first_strings, second_strings, allocator, seed, intersection_count_ptr, first_positions,
                                second_positions, sz_hash_serial_);
}

#endif // STRINGZILLA_TARGET_SERIAL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_INTERSECT_SERIAL_H_
