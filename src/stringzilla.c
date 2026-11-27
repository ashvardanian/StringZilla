/**
 *  @file src/stringzilla.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief The library's cross-family exports: versions, statuses, capability queries, and the
 *      kernel finder over every family.
 *
 *  Each family's dispatch points live in its own unit under <tt>src/dispatch/</tt>, each capability's
 *  kernels in its own unit under <tt>src/capabilities/</tt>, and each GPU vendor's kernels and device
 *  exports in <tt>src/capabilities/cuda.cu</tt>, <tt>src/capabilities/rocm.hip</tt> and
 *  <tt>src/capabilities/metal.c</tt>, which CMake swaps for <tt>src/missing/cuda.c</tt>,
 *  <tt>src/missing/rocm.c</tt> and <tt>src/missing/metal.c</tt> in a build without that vendor.
 */

#include "stringzilla/stringzilla.h" // the family finders
#include "stringzilla/cpu.h"         // `sz_capabilities_detected_cpu_`, `sz_configure_thread_cpu_`

#include <stdatomic.h> // `atomic_load`, `atomic_store`

#include "dispatch.h" // `sz_capabilities_enabled_cpu_`, `sz_capability_group_of_`

STRINGZILLA_API sz_status_t sz_find_kernel_punned(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    switch (kind) {
    case sz_kernel_equal_k:
    case sz_kernel_order_k: return sz_compare_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_strings_to_tape_k:
    case sz_kernel_strings_to_rows_k:
    case sz_kernel_strings_escape_k:
    case sz_kernel_strings_unescape_k:
    case sz_kernel_strings_lookup_k:
    case sz_kernel_strings_normalize_k:
    case sz_kernel_strings_split_k:
    case sz_kernel_strings_split_lines_k:
    case sz_kernel_strings_join_k: return sz_strings_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_find_k:
    case sz_kernel_rfind_k:
    case sz_kernel_find_byte_k:
    case sz_kernel_rfind_byte_k:
    case sz_kernel_find_byteset_k:
    case sz_kernel_rfind_byteset_k:
    case sz_kernel_find_strings_k: return sz_find_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_bytesum_k:
    case sz_kernel_hash_k:
    case sz_kernel_hash_multiseed_k:
    case sz_kernel_hash_strings_k:
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
    case sz_kernel_sort_order_k:
    case sz_kernel_sort_uncased_order_k: return sz_sort_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_intersect_k: return sz_intersect_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_utf8_count_k:
    case sz_kernel_utf8_count_strings_k:
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
    case sz_kernel_levenshtein_distances_k: return sz_levenshtein_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_ngrams_engine_init_k:
    case sz_kernel_ngrams_overlaps_k:
    case sz_kernel_ngrams_sketches_k: return sz_ngrams_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_substrings_engine_init_k:
    case sz_kernel_substrings_contains_k:
    case sz_kernel_substrings_counts_k:
    case sz_kernel_substrings_find_k:
    case sz_kernel_substrings_replace_k:
    case sz_kernel_substrings_bm25_scores_k: return sz_substrings_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_regex_engine_init_k:
    case sz_kernel_regex_contains_k:
    case sz_kernel_regex_counts_k:
    case sz_kernel_regex_find_k:
    case sz_kernel_regex_replace_k:
    case sz_kernel_regex_extract_k: return sz_regex_find_kernel(kind, capabilities, kernel, capability);
    case sz_kernel_subdocuments_engine_init_k:
    case sz_kernel_subdocuments_contains_k:
    case sz_kernel_subdocuments_counts_k:
    case sz_kernel_subdocuments_find_k: return sz_subdocuments_find_kernel(kind, capabilities, kernel, capability);
    default: *kernel = STRINGZILLA_NULL, *capability = 0; return sz_missing_kernel_k;
    }
}

STRINGZILLA_API int sz_version_major(void) { return STRINGZILLA_VERSION_MAJOR; }
STRINGZILLA_API int sz_version_minor(void) { return STRINGZILLA_VERSION_MINOR; }
STRINGZILLA_API int sz_version_patch(void) { return STRINGZILLA_VERSION_PATCH; }
STRINGZILLA_API char const *sz_status_name(sz_status_t status) { return sz_status_name_(status); }

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
    *capabilities = sz_capabilities_compiled_cpu_mask();
    return sz_success_k;
}

/*  Every lookup clamps a CPU mask to it, so a caller may pass `sz_cap_cpus_k` itself. A call before
 *  the load-time fill, from another constructor, gets the serial kernels. */
sz_capability_t sz_capabilities_enabled_cpu_ = sz_cap_serial_k;

/** Fills the CPU's enabled capabilities with what it detects and this library compiled, before
 *  the constructors of default priority in a statically linked image. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor(101)))
#endif
static void sz_capabilities_enable_cpu_(void) {
    sz_capability_t detected = 0;
    sz_capabilities_detected_cpu(&detected);
    sz_capabilities_enabled_cpu_ = detected & sz_capabilities_compiled_cpu_mask();
}

/*  MSVC has no constructor attribute, but runs every pointer in this section as the library loads,
 *  and `/include` keeps `/OPT:REF` from dropping the unreferenced pointer. */
#if defined(_MSC_VER) && !defined(__clang__)
#pragma section(".CRT$XCU", read)
#pragma comment(linker, "/include:sz_capabilities_enable_cpu_pointer_")
__declspec(allocate(".CRT$XCU")) void (*sz_capabilities_enable_cpu_pointer_)(void) = &sz_capabilities_enable_cpu_;
#endif

STRINGZILLA_API sz_status_t sz_capabilities_enabled_cpu(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_enabled_cpu_;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_configure_thread_cpu(sz_capability_t capabilities) {
    return sz_configure_thread_cpu_(capabilities);
}

STRINGZILLA_API sz_size_t sz_capabilities_name(sz_capability_t capabilities, char *buffer, sz_size_t capacity) {
    return sz_capabilities_name_(capabilities, buffer, capacity);
}

/*  Each vendor's unit, or its `src/missing/` stand-in, answers the other queries, so enabling is
 *  written once: what the device runs and this binary holds code for. */
#pragma region Devices

STRINGZILLA_API sz_status_t sz_capabilities_enabled_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_capability_t compiled = 0;
    sz_status_t const status = sz_capabilities_detected_cuda(ordinal, capabilities);
    sz_capabilities_compiled_cuda(ordinal, &compiled);
    *capabilities &= compiled;
    return status;
}

STRINGZILLA_API sz_status_t sz_capabilities_enabled_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_capability_t compiled = 0;
    sz_status_t const status = sz_capabilities_detected_rocm(ordinal, capabilities);
    sz_capabilities_compiled_rocm(ordinal, &compiled);
    *capabilities &= compiled;
    return status;
}

STRINGZILLA_API sz_status_t sz_capabilities_enabled_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_capability_t compiled = 0;
    sz_status_t const status = sz_capabilities_detected_metal(ordinal, capabilities);
    sz_capabilities_compiled_metal(ordinal, &compiled);
    *capabilities &= compiled;
    return status;
}

#pragma endregion Devices

STRINGZILLA_API sz_status_t sz_stream_synchronize_best(sz_capability_t capabilities, sz_stream_t stream) {
    switch (sz_capability_group_of_(capabilities)) {
    case sz_capability_group_cpu_k: return sz_stream_synchronize_serial(stream);
    case sz_capability_group_cuda_k: return sz_stream_synchronize_cuda(stream);
    case sz_capability_group_rocm_k: return sz_stream_synchronize_rocm(stream);
    case sz_capability_group_metal_k: return sz_stream_synchronize_metal(stream);
    default: return sz_missing_gpu_k;
    }
}
