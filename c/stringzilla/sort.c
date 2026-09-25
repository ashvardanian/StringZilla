/**
 *  @file c/stringzilla/sort.c
 *  @author Ash Vardanian
 *  @date February 15, 2025
 *  @brief Per-domain dispatch shim for single-threaded sorting, cased and uncased.
 *
 *  Dispatches @c sz_sequence_argsort and its uncased variant. The integer `sz_pgrams_sort_*` core
 *  is an internal helper and is not runtime-dispatched.
 */
#include <stringzilla/sort.h>

#include "dispatch.h"

STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_sort_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_cpu_table;
    sz_unused_(caps);

    impl->sequence_argsort = sz_sequence_argsort_serial;
    impl->sequence_argsort_uncased = sz_sequence_argsort_uncased_serial;

#if STRINGZILLA_TARGET_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->sequence_argsort = sz_sequence_argsort_haswell;
        impl->sequence_argsort_uncased = sz_sequence_argsort_uncased_haswell;
    }
#endif

#if STRINGZILLA_TARGET_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->sequence_argsort = sz_sequence_argsort_skylake;
        impl->sequence_argsort_uncased = sz_sequence_argsort_uncased_skylake;
    }
#endif

#if STRINGZILLA_TARGET_NEON
    if (caps & sz_cap_neon_k) {
        impl->sequence_argsort = sz_sequence_argsort_neon;
        impl->sequence_argsort_uncased = sz_sequence_argsort_uncased_neon;
    }
#endif
#if STRINGZILLA_TARGET_SVE
    // At the common 128-bit vector length the NEON argsort leads (Graviton 5 words: 53 vs 43 MiB/s SVE,
    // 50 serial); the scalable kernel pays off only with wider registers.
    if ((caps & sz_cap_sve_k) && sz_sve_wider_than_neon_()) {
        impl->sequence_argsort = sz_sequence_argsort_sve;
        impl->sequence_argsort_uncased = sz_sequence_argsort_uncased_sve;
    }
#endif
#if STRINGZILLA_TARGET_RVV
    if (caps & sz_cap_rvv_k) {
        impl->sequence_argsort = sz_sequence_argsort_rvv;
        impl->sequence_argsort_uncased = sz_sequence_argsort_uncased_rvv;
    }
#endif
}

STRINGZILLA_API_RUNTIME sz_status_t sz_sequence_argsort(sz_sequence_t const *array, sz_memory_allocator_t *alloc,
                                                        sz_size_t *order, sz_size_t top_count, sz_bool_t reverse) {
    return sz_dispatch_cpu_table.sequence_argsort(array, alloc, order, top_count, reverse);
}

STRINGZILLA_API_RUNTIME sz_status_t sz_sequence_argsort_uncased(sz_sequence_t const *array,
                                                                sz_memory_allocator_t *alloc, sz_size_t *order,
                                                                sz_size_t top_count, sz_bool_t reverse) {
    return sz_dispatch_cpu_table.sequence_argsort_uncased(array, alloc, order, top_count, reverse);
}
