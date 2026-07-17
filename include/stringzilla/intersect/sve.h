/**
 *  @brief SVE backend for set intersection.
 *  @file include/stringzilla/intersect/sve.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/intersect.h
 */
#ifndef STRINGZILLA_INTERSECT_SVE_H_
#define STRINGZILLA_INTERSECT_SVE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_fill`
#include "stringzilla/hash.h"    // `sz_hash`
#include "stringzilla/intersect/serial.h"

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

SZ_API_COMPTIME sz_status_t sz_sequence_intersect_sve(sz_sequence_t const *first_sequence,
                                                      sz_sequence_t const *second_sequence, //
                                                      sz_memory_allocator_t *alloc, sz_u64_t seed,
                                                      sz_size_t *intersection_count_ptr,
                                                      sz_sorted_idx_t *first_positions,
                                                      sz_sorted_idx_t *second_positions) {
    // Serial passthrough - a batched SVE2+AES kernel was built and measured, then rejected.
    //
    // The design mirrored the Ice Lake donor (`sz_sequence_intersect_icelake`): hash four ≤16-byte
    // keys at once with a 4-lane analog of `sz_hash_sve2_upto16_` (four interleaved SVE2-AES streams
    // in Z registers, predicated `svld1` loads, `svlasta` extraction - bit-identical to `sz_hash`,
    // no stack staging, no gathers), then linear-probe the table scalar-side. On Graviton 5
    // (Neoverse-V3, VL=128) it did NOT clear the owner's 1.5x gate over serial on xlsum `words`:
    //
    //     sz_sequence_intersect_serial : ~10.0-11.0 MOps/s (~152-177 MiB/s)
    //     sz_sequence_intersect_sve2   : ~9.2-10.0 MOps/s  (~173-176 MiB/s)   =>  ~0.9-1.16x, tied
    //
    // Why: the intersection is dominated by the hash-table probe (pointer-chasing table loads plus
    // indirect `get_length`/`get_start` and `sz_equal`), not by hashing, so batching the hash alone
    // can't reach 1.5x (Amdahl). Ice Lake wins on x86 only because it ALSO vectorizes the probe with
    // `i64gather`/`i64scatter`; at VL=128 SVE gathers are ~9-11c latency and 2-4 B/cycle - a trap on
    // this core - so there is no profitable way to vectorize the probe here. Kernel deleted; if the
    // probe is ever restructured (e.g. software-pipelined slot prefetch), the batched hash can return.
    return sz_sequence_intersect_serial(     //
        first_sequence, second_sequence,     //
        alloc, seed, intersection_count_ptr, //
        first_positions, second_positions);
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

#endif // STRINGZILLA_INTERSECT_SVE_H_
