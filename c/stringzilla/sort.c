/**
 *  @file c/stringzilla/sort.c
 *  @brief Per-domain dispatch shim for single-threaded sorting (`sz_sequence_argsort`, `sz_pgrams_sort`).
 *  @author Ash Vardanian
 *  @date January 16, 2024
 */
#include "dispatch.h"
#include <stringzilla/sort.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_sort_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->sequence_argsort = sz_sequence_argsort_serial;
    impl->pgrams_sort = sz_pgrams_sort_serial;

#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->sequence_argsort = sz_sequence_argsort_skylake;
        impl->pgrams_sort = sz_pgrams_sort_skylake;
    }
#endif

#if SZ_USE_SVE
    if (caps & sz_cap_sve_k) {
        impl->sequence_argsort = sz_sequence_argsort_sve;
        impl->pgrams_sort = sz_pgrams_sort_sve;
    }
#endif
}

SZ_DYNAMIC sz_status_t sz_pgrams_sort(sz_pgram_t *array, sz_size_t count, sz_memory_allocator_t *alloc,
                                      sz_size_t *order) {
    return sz_dispatch_table.pgrams_sort(array, count, alloc, order);
}

SZ_DYNAMIC sz_status_t sz_sequence_argsort(sz_sequence_t const *array, sz_memory_allocator_t *alloc, sz_size_t *order) {
    return sz_dispatch_table.sequence_argsort(array, alloc, order);
}
