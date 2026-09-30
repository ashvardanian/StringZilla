/**
 *  @file c/stringzilla.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief The library's cross-family exports: versions, statuses, capability queries, the kernel
 *      finder, and the device exports of the GPU vendors it lacks.
 *
 *  Each family's dispatch points live in its own unit under `c/dispatch/`, each capability's
 *  kernels in its own unit under `c/cpu/`, and each GPU vendor's kernels and device exports in
 *  `c/nvidia/cuda.cu`, `c/amd/rocm.hip` and `c/apple/metal.c`.
 */
#include <stringzilla/stringzilla.h> // `sz_cpu_capabilities_detected_` and its twins, the family finders
#include <stringzilla/types.cuh>     // The device exports of the GPU vendors this library lacks

#include <stdatomic.h> // `atomic_load`, `atomic_store`, after StringZilla's headers pick the LibC macros

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
    case sz_kernel_levenshtein_distance_tiled_k:
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

/*  The CPU's capabilities, probed on the first query: a query can be a system call, and the
 *  bindings ask on every call. Racing threads probe the same CPU and store the same word, never
 *  zero once filled, as it always holds @c sz_cap_serial_k. The GPU queries keep nothing, as their
 *  runtimes answer from their own state. */
static _Atomic sz_capability_t sz_cpu_detected_;

STRINGZILLA_API sz_status_t sz_cpu_capabilities_detected(sz_capability_t *capabilities) {
    sz_capability_t detected = atomic_load(&sz_cpu_detected_);
    if (!detected) atomic_store(&sz_cpu_detected_, detected = sz_cpu_capabilities_detected_());
    *capabilities = detected;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_cpu_capabilities_compiled(sz_capability_t *capabilities) {
    *capabilities = sz_cpu_capabilities_compiled_();
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_cpu_capabilities_enabled(sz_capability_t *capabilities) {
    sz_cpu_capabilities_detected(capabilities);
    *capabilities &= sz_cpu_capabilities_compiled_();
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_cpu_configure_thread(sz_capability_t capabilities) {
    return sz_cpu_configure_thread_(capabilities);
}

STRINGZILLA_API sz_size_t sz_capabilities_name(sz_capability_t capabilities, char *buffer, sz_size_t capacity) {
    return sz_capabilities_name_(capabilities, buffer, capacity);
}

#pragma region Devices

/*  A library with CUDA kernels counts, probes and binds its devices in `c/nvidia/cuda.cu`. */
#if !STRINGZILLA_ARCH_CUDA_

STRINGZILLA_API sz_status_t sz_cuda_count_devices(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_cuda_capabilities_detected(sz_size_t device, sz_capability_t *capabilities) {
    return sz_cuda_capabilities_detected_(device, capabilities);
}

STRINGZILLA_API sz_status_t sz_cuda_memory_reaches_device(void const *pointer, sz_bool_t *reaches) {
    sz_unused_(pointer), sz_unused_(reaches);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_cuda_memory_allocator_init_unified(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_cuda_memory_allocator_init_device(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_cuda_memory_allocator_init_pinned(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_cuda_sequence_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                               sz_sequence_t *sequence) {
    sz_unused_(views), sz_unused_(count), sz_unused_(sequence);
    return sz_missing_gpu_k;
}

#endif // !STRINGZILLA_ARCH_CUDA_

STRINGZILLA_API sz_status_t sz_cuda_capabilities_compiled(sz_capability_t *capabilities) {
    *capabilities = sz_cuda_capabilities_compiled_();
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_cuda_capabilities_enabled(sz_size_t device, sz_capability_t *capabilities) {
    sz_status_t const status = sz_cuda_capabilities_detected(device, capabilities);
    *capabilities &= sz_cuda_capabilities_compiled_();
    return status;
}

/*  A library with ROCm kernels counts, probes and binds its devices in `c/amd/rocm.hip`. */
#if !STRINGZILLA_ARCH_ROCM_

STRINGZILLA_API sz_status_t sz_rocm_count_devices(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_rocm_capabilities_detected(sz_size_t device, sz_capability_t *capabilities) {
    return sz_rocm_capabilities_detected_(device, capabilities);
}

STRINGZILLA_API sz_status_t sz_rocm_memory_reaches_device(void const *pointer, sz_bool_t *reaches) {
    sz_unused_(pointer), sz_unused_(reaches);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_unified(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_device(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_pinned(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_rocm_sequence_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                               sz_sequence_t *sequence) {
    sz_unused_(views), sz_unused_(count), sz_unused_(sequence);
    return sz_missing_gpu_k;
}

#endif // !STRINGZILLA_ARCH_ROCM_

STRINGZILLA_API sz_status_t sz_rocm_capabilities_compiled(sz_capability_t *capabilities) {
    *capabilities = sz_rocm_capabilities_compiled_();
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rocm_capabilities_enabled(sz_size_t device, sz_capability_t *capabilities) {
    sz_status_t const status = sz_rocm_capabilities_detected(device, capabilities);
    *capabilities &= sz_rocm_capabilities_compiled_();
    return status;
}

/*  With Metal kernels in the library, `c/apple/metal.c` counts and probes the devices instead. */
#if !STRINGZILLA_WITH_METAL

STRINGZILLA_API sz_status_t sz_metal_count_devices(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_metal_capabilities_detected(sz_size_t device, sz_capability_t *capabilities) {
    return sz_metal_capabilities_detected_(device, capabilities);
}

#endif // !STRINGZILLA_WITH_METAL

STRINGZILLA_API sz_status_t sz_metal_capabilities_compiled(sz_capability_t *capabilities) {
    *capabilities = sz_metal_capabilities_compiled_();
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_metal_capabilities_enabled(sz_size_t device, sz_capability_t *capabilities) {
    sz_status_t const status = sz_metal_capabilities_detected(device, capabilities);
    *capabilities &= sz_metal_capabilities_compiled_();
    return status;
}

#pragma endregion Devices
