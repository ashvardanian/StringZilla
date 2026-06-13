/**
 *  @file c/stringzilla/sort.c
 *  @brief Per-domain dispatch shim for single-threaded sorting (`sz_sequence_argsort` and the
 *         case-insensitive variant). The integer `sz_pgrams_sort_*` core is an internal helper and is
 *         not runtime-dispatched.
 *  @author Ash Vardanian
 *  @date January 16, 2024
 */
#include "dispatch.h"
#include <stringzilla/sort.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_sort_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->sequence_argsort = sz_sequence_argsort_serial;
    impl->sequence_argsort_utf8_case_insensitive = sz_sequence_argsort_utf8_case_insensitive_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->sequence_argsort = sz_sequence_argsort_haswell;
        impl->sequence_argsort_utf8_case_insensitive = sz_sequence_argsort_utf8_case_insensitive_haswell;
    }
#endif

#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->sequence_argsort = sz_sequence_argsort_skylake;
        impl->sequence_argsort_utf8_case_insensitive = sz_sequence_argsort_utf8_case_insensitive_skylake;
    }
#endif

#if SZ_USE_SVE
    if (caps & sz_cap_sve_k) {
        impl->sequence_argsort = sz_sequence_argsort_sve;
        impl->sequence_argsort_utf8_case_insensitive = sz_sequence_argsort_utf8_case_insensitive_sve;
    }
#endif
#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) {
        impl->sequence_argsort = sz_sequence_argsort_neon;
        impl->sequence_argsort_utf8_case_insensitive = sz_sequence_argsort_utf8_case_insensitive_neon;
    }
#endif
}

SZ_DYNAMIC sz_status_t sz_sequence_argsort(sz_sequence_t const *array, sz_memory_allocator_t *alloc, sz_size_t *order,
                                           sz_size_t top_count, sz_bool_t reverse) {
    return sz_dispatch_table.sequence_argsort(array, alloc, order, top_count, reverse);
}

SZ_DYNAMIC sz_status_t sz_sequence_argsort_utf8_case_insensitive(sz_sequence_t const *array,
                                                                 sz_memory_allocator_t *alloc, sz_size_t *order,
                                                                 sz_size_t top_count, sz_bool_t reverse) {
    return sz_dispatch_table.sequence_argsort_utf8_case_insensitive(array, alloc, order, top_count, reverse);
}
