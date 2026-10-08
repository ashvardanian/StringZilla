/**
 *  @file include/stringzilla/hash/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief ROCm host side of multi-state SHA-256: the residency checks, the launches and the
 *      @c _rocm exports, over the kernels of `hash/simt.cuh`.
 *
 *  @sa include/stringzilla/hash/simt.cuh
 */
#ifndef STRINGZILLA_HASH_ROCM_CUH_
#define STRINGZILLA_HASH_ROCM_CUH_

#include "stringzilla/rocm.cuh"
#include "stringzilla/hash/simt.cuh"

#if STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_ROCM

STRINGZILLA_INLINE sz_status_t sz_sha256_multistate_update_rocm_(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                                 sz_stream_t stream) {
    sz_size_t count = texts->count;
    void const *tape = texts->handle;
    if (!count) return sz_success_k;
    if (texts->get_start != sz_sequence_tape_start || texts->get_length != sz_sequence_tape_length ||
        !sz_memory_accessible_rocm_(tape) || !sz_memory_accessible_rocm_(states))
        return sz_device_memory_mismatch_k;
    void *arguments[] = {&states, &tape, &count};
    dim3 const grid((unsigned)sz_size_divide_round_up(count, sz_sha256_threads_per_block_simt_k));
    return sz_launch_rocm_((void const *)&sz_sha256_multistate_update_simt_kernel_, grid,
                           dim3(sz_sha256_threads_per_block_simt_k), arguments, 0, stream);
}

STRINGZILLA_INLINE sz_status_t sz_sha256_multistate_digest_rocm_(sz_sha256_state_t const *states,
                                                                 sz_size_t states_count, sz_u8_t *digests,
                                                                 sz_stream_t stream) {
    if (!states_count) return sz_success_k;
    if (!sz_memory_accessible_rocm_(states) || !sz_memory_accessible_rocm_(digests)) return sz_device_memory_mismatch_k;
    void *arguments[] = {&states, &states_count, &digests};
    dim3 const grid((unsigned)sz_size_divide_round_up(states_count, sz_sha256_threads_per_block_simt_k));
    return sz_launch_rocm_((void const *)&sz_sha256_multistate_digest_simt_kernel_, grid,
                           dim3(sz_sha256_threads_per_block_simt_k), arguments, 0, stream);
}

STRINGZILLA_API sz_status_t sz_sha256_multistate_update_rocm(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                             sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_sha256_multistate_update_rocm_(states, texts, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_sha256_multistate_digest_rocm(sz_sha256_state_t const *states, sz_size_t states_count,
                                                             sz_u8_t *digests, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_sha256_multistate_digest_rocm_(states, states_count, digests, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

#endif // STRINGZILLA_TARGET_ROCM

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_HASH_ROCM_CUH_
