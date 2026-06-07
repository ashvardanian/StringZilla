/**
 *  @file c/stringzilla/intersect.c
 *  @brief Per-domain dispatch shim for unordered string-set intersection (`sz_sequence_intersect`).
 *  @author Ash Vardanian
 *  @date January 16, 2024
 */
#include "dispatch.h"
#include <stringzilla/intersect.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_intersect_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->sequence_intersect = sz_sequence_intersect_serial;

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->sequence_intersect = sz_sequence_intersect_icelake; }
#endif

#if SZ_USE_SVE
    if (caps & sz_cap_sve_k) { impl->sequence_intersect = sz_sequence_intersect_sve; }
#endif
}

SZ_DYNAMIC sz_status_t sz_sequence_intersect(sz_sequence_t const *first_array, sz_sequence_t const *second_array,
                                             sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_size,
                                             sz_size_t *first_positions, sz_size_t *second_positions) {
    return sz_dispatch_table.sequence_intersect(first_array, second_array, alloc, seed, intersection_size,
                                                first_positions, second_positions);
}
