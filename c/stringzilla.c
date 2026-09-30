/**
 *  @file c/stringzilla.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief The library's cross-family exports: versions, statuses, capability queries, the kernel
 *      finder, the device exports of the GPU vendors it lacks, and the LibC @c mem* overrides.
 *
 *  Each family's dispatch points live in its own unit under `c/dispatch/`, each capability's
 *  kernels in its own unit under `c/cpu/`, and each GPU vendor's kernels and device exports in
 *  `c/nvidia/cuda.cu`, `c/amd/rocm.hip` and `c/apple/metal.c`.
 */
#if !defined(STRINGZILLA_OVERRIDE_LIBC)
#define STRINGZILLA_OVERRIDE_LIBC (!STRINGZILLA_WITH_LIBC)
#endif
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

/*  Overrides for the LibC `mem*` functions, each resolving its kernel for the enabled capabilities
 *  on first use. Racing threads resolve and store the same kernel, and resolving reaches no `mem*`:
 *  detection reads CPUID or the ID registers, and the finder only walks its lists.
 *
 *  MSVC rejects @c STRINGZILLA_API here with C2375, a linkage conflict with the CRT's @c dllimport
 *  declarations, so these are exported with linker flags instead. A 32-bit build prefixes the
 *  exported name with an underscore, because that is how MSVC decorates @c __cdecl functions:
 *  https://stackoverflow.com/questions/62753691 */
#pragma region LibC Overrides
#if STRINGZILLA_OVERRIDE_LIBC && !defined(__CYGWIN__)

#if !STRINGZILLA_WITH_LIBC
#ifdef _MSC_VER
typedef sz_size_t size_t; // Reuse the type definition we've inferred from `stringzilla.h`
#else
typedef __SIZE_TYPE__ size_t; // For GCC/Clang
#endif
#endif

static sz_kernel_punned_t sz_libc_kernel_(sz_kernel_kind_t kind, _Atomic(sz_kernel_punned_t) *cache) {
    sz_kernel_punned_t kernel = atomic_load_explicit(cache, memory_order_relaxed);
    if (kernel) return kernel;
    sz_capability_t capabilities, capability;
    sz_cpu_capabilities_enabled(&capabilities);
    sz_find_kernel_punned(kind, capabilities, &kernel, &capability);
    atomic_store_explicit(cache, kernel, memory_order_relaxed);
    return kernel;
}

#if defined(_MSC_VER)
#if defined(_WIN64)
#pragma comment(linker, "/export:memcpy")
#else
#pragma comment(linker, "/export:_memcpy")
#endif
void *__cdecl memcpy(void *target, void const *source, size_t length) {
#else
STRINGZILLA_API void *memcpy(void *target, void const *source, size_t length) {
#endif
    static _Atomic(sz_kernel_punned_t) cache;
    ((sz_kernel_copy_t)sz_libc_kernel_(sz_kernel_copy_k, &cache))(target, source, length, STRINGZILLA_NULL);
    return target;
}

#if defined(_MSC_VER)
#if defined(_WIN64)
#pragma comment(linker, "/export:memmove")
#else
#pragma comment(linker, "/export:_memmove")
#endif
void *__cdecl memmove(void *target, void const *source, size_t length) {
#else
STRINGZILLA_API void *memmove(void *target, void const *source, size_t length) {
#endif
    static _Atomic(sz_kernel_punned_t) cache;
    ((sz_kernel_move_t)sz_libc_kernel_(sz_kernel_move_k, &cache))(target, source, length, STRINGZILLA_NULL);
    return target;
}

#if defined(_MSC_VER)
#if defined(_WIN64)
#pragma comment(linker, "/export:memset")
#else
#pragma comment(linker, "/export:_memset")
#endif
void *__cdecl memset(void *target, int value, size_t length) {
#else
STRINGZILLA_API void *memset(void *target, int value, size_t length) {
#endif
    static _Atomic(sz_kernel_punned_t) cache;
    ((sz_kernel_fill_t)sz_libc_kernel_(sz_kernel_fill_k, &cache))(target, length, (sz_u8_t)value, STRINGZILLA_NULL);
    return target;
}

#if defined(_MSC_VER)
#if defined(_WIN64)
#pragma comment(linker, "/export:memchr")
#else
#pragma comment(linker, "/export:_memchr")
#endif
void *__cdecl memchr(void const *haystack, int character_wide, size_t length) {
#else
STRINGZILLA_API void *memchr(void const *haystack, int character_wide, size_t length) {
#endif
    static _Atomic(sz_kernel_punned_t) cache;
    sz_u8_t const character = (sz_u8_t)character_wide;
    sz_cptr_t match;
    ((sz_kernel_find_byte_t)sz_libc_kernel_(sz_kernel_find_byte_k, &cache))(
        (sz_cptr_t)haystack, length, (sz_cptr_t)&character, &match, STRINGZILLA_NULL);
    return (void *)match;
}

#if !defined(_MSC_VER)

STRINGZILLA_API void *memmem(void const *haystack, size_t haystack_length, void const *needle, size_t needle_length) {
    static _Atomic(sz_kernel_punned_t) cache;
    sz_cptr_t match;
    ((sz_kernel_find_t)sz_libc_kernel_(sz_kernel_find_k, &cache))(
        (sz_cptr_t)haystack, haystack_length, (sz_cptr_t)needle, needle_length, &match, STRINGZILLA_NULL);
    return (void *)match;
}

STRINGZILLA_API void *memrchr(void const *haystack, int character_wide, size_t length) {
    static _Atomic(sz_kernel_punned_t) cache;
    sz_u8_t const character = (sz_u8_t)character_wide;
    sz_cptr_t match;
    ((sz_kernel_find_byte_t)sz_libc_kernel_(sz_kernel_rfind_byte_k, &cache))(
        (sz_cptr_t)haystack, length, (sz_cptr_t)&character, &match, STRINGZILLA_NULL);
    return (void *)match;
}

STRINGZILLA_API void memfrob(void *target, size_t length) {
    static _Atomic(sz_kernel_punned_t) cache;
    static sz_u64_t nonce = 42;
    ((sz_kernel_fill_random_t)sz_libc_kernel_(sz_kernel_fill_random_k, &cache))((sz_ptr_t)target, length, nonce++,
                                                                                STRINGZILLA_NULL);
}

#endif // !defined(_MSC_VER)
#endif // STRINGZILLA_OVERRIDE_LIBC && !defined(__CYGWIN__)
#pragma endregion LibC Overrides
