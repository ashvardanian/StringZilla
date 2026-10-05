/**
 *  @file c/stringzilla.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief The library's cross-family exports: versions, statuses, capability queries, the kernel
 *      finder, and the device queries of the GPU vendors it lacks.
 *
 *  Each family's dispatch points live in its own unit under `c/dispatch/`, each capability's
 *  kernels in its own unit under `c/target/`, and each GPU vendor's kernels and device exports in
 *  `c/target/cuda.cu`, `c/target/rocm.hip` and `c/target/metal.c`.
 */

/** glibc feature selection must precede the libc headers pulled in by <stdatomic.h>. */
#if defined(__linux__) && !defined(_DEFAULT_SOURCE) && !defined(_GNU_SOURCE)
#define _DEFAULT_SOURCE 1
#endif

#include <stdatomic.h> // `atomic_load`, `atomic_store`

#include "stringzilla/stringzilla.h" // `sz_capabilities_detected_cpu_`, the family finders

#include "dispatch.h" // `sz_capabilities_runnable_`, `sz_capability_group_cpu_k`

STRINGZILLA_API sz_status_t sz_find_kernel_punned(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    switch (kind) {
    case sz_kernel_equal_k:
    case sz_kernel_order_k: return sz_compare_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_copy_k:
    case sz_kernel_move_k:
    case sz_kernel_fill_k:
    case sz_kernel_lookup_k: return sz_memory_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_find_k:
    case sz_kernel_rfind_k:
    case sz_kernel_find_byte_k:
    case sz_kernel_rfind_byte_k:
    case sz_kernel_find_byteset_k:
    case sz_kernel_rfind_byteset_k: return sz_find_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_bytesum_k:
    case sz_kernel_hash_k:
    case sz_kernel_hash_multiseed_k:
    case sz_kernel_fill_random_k:
    case sz_kernel_hash_state_init_k:
    case sz_kernel_hash_state_update_k:
    case sz_kernel_hash_state_digest_k:
    case sz_kernel_sha256_state_init_k:
    case sz_kernel_sha256_state_update_k:
    case sz_kernel_sha256_state_digest_k:
    case sz_kernel_sha256_multistate_update_k:
    case sz_kernel_sha256_multistate_digest_k: return sz_hash_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_aes256_key_init_k:
    case sz_kernel_aes256_gcm_key_init_k:
    case sz_kernel_aes256_ctr_xor_k:
    case sz_kernel_aes256_gcm_encrypt_k:
    case sz_kernel_aes256_gcm_decrypt_k:
    case sz_kernel_aes256_gcm_encryptor_init_k:
    case sz_kernel_aes256_gcm_encryptor_associate_k:
    case sz_kernel_aes256_gcm_encryptor_update_k:
    case sz_kernel_aes256_gcm_encryptor_digest_k:
    case sz_kernel_aes256_gcm_decryptor_init_k:
    case sz_kernel_aes256_gcm_decryptor_associate_k:
    case sz_kernel_aes256_gcm_decryptor_update_unverified_k:
    case sz_kernel_aes256_gcm_decryptor_verify_k: return sz_cipher_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_sequence_argsort_k:
    case sz_kernel_sequence_argsort_uncased_k: return sz_sort_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_sequence_intersect_k: return sz_intersect_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_utf8_count_k:
    case sz_kernel_utf8_seek_k:
    case sz_kernel_utf8_decode_k: return sz_utf8_runes_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_utf8_newlines_k:
    case sz_kernel_utf8_whitespaces_k:
    case sz_kernel_utf8_delimiters_k: return sz_utf8_tokens_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_utf8_wordbreaks_k: return sz_utf8_wordbreaks_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_utf8_graphemes_k: return sz_utf8_graphemes_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_utf8_sentences_k: return sz_utf8_sentences_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_utf8_linebreaks_k: return sz_utf8_linebreaks_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_utf8_norm_k:
    case sz_kernel_utf8_find_denormalized_k: return sz_utf8_norm_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_utf8_uncased_fold_k: return sz_utf8_uncased_fold_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_utf8_uncased_needle_init_k:
    case sz_kernel_utf8_uncased_search_k:
    case sz_kernel_utf8_uncased_order_k:
    case sz_kernel_utf8_find_cased_k: return sz_utf8_uncased_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_levenshtein_engine_init_k:
    case sz_kernel_levenshtein_distances_k:
        return sz_levenshtein_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_overlap_engine_init_k:
    case sz_kernel_overlap_scores_k: return sz_overlap_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_substrings_engine_init_k:
    case sz_kernel_substrings_counts_k:
    case sz_kernel_substrings_find_k:
    case sz_kernel_substrings_replace_k:
    case sz_kernel_substrings_bm25_scores_k: return sz_substrings_find_kernel(kind, capabilities, kernel, capability);
    default: *kernel = STRINGZILLA_NULL, *capability = 0; return sz_missing_kernel_k;
    }
}

STRINGZILLA_API int sz_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
STRINGZILLA_API int sz_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
STRINGZILLA_API int sz_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }
STRINGZILLA_API char const *sz_status_name(sz_status_t status) { return sz_status_name_(status); }

STRINGZILLA_API sz_cptr_t sz_sequence_tape_start(void const *handle, sz_size_t i) {
    sz_u64_t const *offsets = (sz_u64_t const *)handle;
    return (sz_cptr_t)handle + offsets[i];
}

STRINGZILLA_API sz_size_t sz_sequence_tape_length(void const *handle, sz_size_t i) {
    sz_u64_t const *offsets = (sz_u64_t const *)handle;
    return (sz_size_t)(offsets[i + 1] - offsets[i]);
}

/*  The CPU's capabilities, probed on the first query: a query can be a system call, and the
 *  bindings ask on every call. Racing threads probe the same CPU and store the same word, never
 *  zero once filled, as it always holds @c sz_cap_serial_k. The GPU queries keep nothing, as their
 *  runtimes answer from their own state. */
static _Atomic sz_capability_t sz_cpu_detected_;

STRINGZILLA_API sz_status_t sz_capabilities_detected_cpu(sz_capability_t *capabilities) {
    sz_capability_t detected = atomic_load(&sz_cpu_detected_);
    if (!detected) atomic_store(&sz_cpu_detected_, detected = sz_capabilities_detected_cpu_());
    *capabilities = detected;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_compiled_cpu(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_compiled_cpu_();
    return sz_success_k;
}

/*  Every lookup clamps a mask to what its group runs, so a caller may pass `sz_cap_cpus_k` itself.
 *  A call before the load-time widening, from another constructor, gets the serial kernels. */
sz_capability_t sz_capabilities_runnable_[sz_capability_groups_k] = {sz_cap_serial_k, ~0ull, ~0ull, ~0ull};

/** Widens the CPU's runnable capabilities to what it detects and this library compiled. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void sz_capabilities_runnable_widen_(void) {
    sz_capability_t detected = 0;
    sz_capabilities_detected_cpu(&detected);
    sz_capabilities_runnable_[sz_capability_group_cpu_k] = detected & sz_capabilities_compiled_cpu_();
}

/*  MSVC has no constructor attribute, but runs every pointer in this section as the library loads. */
#if defined(_MSC_VER) && !defined(__clang__)
#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU")) void (*sz_capabilities_runnable_widener_)(void) = &sz_capabilities_runnable_widen_;
#endif

STRINGZILLA_API sz_status_t sz_capabilities_enabled_cpu(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_runnable_[sz_capability_group_cpu_k];
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_thread_configure_cpu(sz_capability_t capabilities) {
    return sz_thread_configure_cpu_(capabilities);
}

STRINGZILLA_API sz_size_t sz_capabilities_name(sz_capability_t capabilities, char *buffer, sz_size_t capacity) {
    return sz_capabilities_name_(capabilities, buffer, capacity);
}

#pragma region Devices

/*  Only the producers are stubbed: every consumer is a @c _best dispatch point, which answers
 *  @c sz_missing_gpu_k for a vendor the library lacks. A library with CUDA kernels counts, probes
 *  and opens streams on its devices in `c/target/cuda.cu`. */
#if !STRINGZILLA_ARCH_CUDA_

STRINGZILLA_API sz_status_t sz_device_count_cuda(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_detected_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_init_cuda(sz_size_t ordinal, sz_stream_t *stream) {
    sz_unused_(ordinal);
    *stream = STRINGZILLA_NULL;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_free_cuda(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_gpu_k;
}

#endif // !STRINGZILLA_ARCH_CUDA_

STRINGZILLA_API sz_status_t sz_capabilities_compiled_cuda(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_compiled_cuda_();
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_enabled_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_status_t const status = sz_capabilities_detected_cuda(ordinal, capabilities);
    *capabilities &= sz_capabilities_compiled_cuda_();
    return status;
}

/*  A library with ROCm kernels counts, probes and opens streams on its devices in
 *  `c/target/rocm.hip`. */
#if !STRINGZILLA_ARCH_ROCM_

STRINGZILLA_API sz_status_t sz_device_count_rocm(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_detected_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_init_rocm(sz_size_t ordinal, sz_stream_t *stream) {
    sz_unused_(ordinal);
    *stream = STRINGZILLA_NULL;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_free_rocm(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_gpu_k;
}

#endif // !STRINGZILLA_ARCH_ROCM_

STRINGZILLA_API sz_status_t sz_capabilities_compiled_rocm(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_compiled_rocm_();
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_enabled_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_status_t const status = sz_capabilities_detected_rocm(ordinal, capabilities);
    *capabilities &= sz_capabilities_compiled_rocm_();
    return status;
}

/*  With Metal kernels in the library, `c/target/metal.c` counts, probes and opens streams on the
 *  devices instead. */
#if !STRINGZILLA_WITH_METAL

STRINGZILLA_API sz_status_t sz_device_count_metal(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_detected_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    return sz_capabilities_detected_metal_(ordinal, capabilities);
}

STRINGZILLA_API sz_status_t sz_stream_init_metal(sz_size_t ordinal, sz_stream_t *stream) {
    return sz_stream_init_metal_(ordinal, stream);
}

STRINGZILLA_API sz_status_t sz_stream_free_metal(sz_stream_t stream) { return sz_stream_free_metal_(stream); }

#endif // !STRINGZILLA_WITH_METAL

STRINGZILLA_API sz_status_t sz_capabilities_compiled_metal(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_compiled_metal_();
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_enabled_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_status_t const status = sz_capabilities_detected_metal(ordinal, capabilities);
    *capabilities &= sz_capabilities_compiled_metal_();
    return status;
}

#pragma endregion Devices

STRINGZILLA_API sz_status_t sz_stream_synchronize_best(sz_capability_t capabilities, sz_stream_t stream) {
    switch (sz_capability_group_of_(capabilities)) {
    case sz_capability_group_cpu_k: return sz_stream_synchronize_serial(stream);
#if STRINGZILLA_TARGET_CUDA
    case sz_capability_group_cuda_k: return sz_stream_synchronize_cuda(stream);
#endif
#if STRINGZILLA_TARGET_ROCM
    case sz_capability_group_rocm_k: return sz_stream_synchronize_rocm(stream);
#endif
#if STRINGZILLA_TARGET_METAL
    case sz_capability_group_metal_k: return sz_stream_synchronize_metal(stream);
#endif
    default: return sz_missing_gpu_k;
    }
}
