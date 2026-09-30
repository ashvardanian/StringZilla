/**
 *  @file bench/cross_arm64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel benchmarks - Arm64 family: NEON, NEON AES, NEON SHA, SVE, SVE2, SVE2 AES.
 */
#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

namespace {

#if STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

/** The pgram sort is always inlined under SVE, so it reaches the portable driver as a plain call,
 *  which a driver without SVE can make. */
sz_status_t pgrams_sort_sve_(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *allocator,
                             sz_sorted_idx_t *order) {
    return sz_pgrams_sort_sve_(pgrams, count, allocator, order);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_SVE

} // namespace

void bench_cross_arm64([[maybe_unused]] corpora_t &corpora) {
#if STRINGZILLA_TARGET_NEON
    if (cross_section("Cross NEON", sz_cap_neon_k)) {
        bench_find_kernels<sz_find_neon, sz_rfind_neon>(corpora, "neon");
        bench_find_byte_kernels<sz_find_byte_neon, sz_rfind_byte_neon>(corpora, "neon");
        bench_find_byteset_kernels<sz_find_byteset_neon, sz_rfind_byteset_neon>(corpora, "neon");
        bench_utf8_count_kernels<sz_utf8_count_neon>(corpora, "neon");
        bench_utf8_seek_kernels<sz_utf8_seek_neon>(corpora, "neon");
        bench_utf8_decode_kernels<sz_utf8_decode_neon>(corpora, "neon");
        bench_utf8_newlines_kernels<sz_utf8_newlines_neon>(corpora, "neon");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_neon>(corpora, "neon");
        bench_utf8_delimiters_kernels<sz_utf8_delimiters_neon>(corpora, "neon");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_neon>(corpora, "neon");
        bench_utf8_sentences_kernels<sz_utf8_sentences_neon>(corpora, "neon");
        bench_utf8_linebreaks_kernels<sz_utf8_linebreaks_neon>(corpora, "neon");
        bench_utf8_norm_kernels<sz_utf8_norm_neon>(corpora, "neon");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_neon>(corpora, "neon");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_neon>(corpora, "neon");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_neon>(corpora, "neon");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_neon>(corpora, "neon");
        bench_bytesum_kernels<sz_bytesum_neon>(corpora, "neon");
        bench_equal_kernels<sz_equal_neon>(corpora, "neon");
        bench_order_kernels<sz_order_neon>(corpora, "neon");
        bench_copy_kernels<sz_copy_neon>(corpora, "neon");
        bench_move_kernels<sz_move_neon>(corpora, "neon");
        bench_fill_kernels<sz_fill_neon>(corpora, "neon");
        bench_lookup_kernels<sz_lookup_neon>(corpora, "neon");
        bench_map_kernels<sz_order_neon>(corpora, "neon");
        bench_sequence_argsort_kernels<sz_sequence_argsort_neon, sz_sequence_argsort_uncased_neon>(corpora, "neon");
#if !STRINGZILLA_HEADER_ONLY
        bench_substrings_kernels<sz_substrings_counts_neon, sz_substrings_find_neon, sz_substrings_replace_neon,
                                 sz_substrings_bm25_scores_neon>(corpora, "neon");
#else
        bench_pgrams_sort_kernels<sz_pgrams_sort_neon_>(corpora, "neon");
#endif
    }
#endif // STRINGZILLA_TARGET_NEON
#if STRINGZILLA_TARGET_NEONAES
    if (cross_section("Cross NEON AES", sz_cap_neonaes_k)) {
        bench_hash_kernels<sz_hash_neonaes>(corpora, "neonaes");
        bench_hash_multiseed_kernels<sz_hash_multiseed_neonaes>(corpora, "neonaes");
        bench_hash_stream_kernels<sz_hash_state_init_neonaes, sz_hash_state_update_neonaes,
                                  sz_hash_state_digest_neonaes>(corpora, "neonaes");
        bench_fill_random_kernels<sz_fill_random_neonaes>(corpora, "neonaes");
        bench_aes256_ctr_kernels<sz_aes256_key_init_neonaes, sz_aes256_ctr_xor_neonaes>(corpora, "neonaes");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_neonaes, sz_aes256_gcm_encrypt_neonaes>(corpora, "neonaes");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_neonaes, sz_aes256_gcm_encryptor_init_neonaes,
                                        sz_aes256_gcm_encryptor_update_neonaes, sz_aes256_gcm_encryptor_digest_neonaes>(
            corpora, "neonaes");
        bench_unordered_map_kernels<sz_hash_neonaes, sz_equal_neon>(corpora, "neonaes", "neon");
        bench_sequence_intersect_kernels<sz_sequence_intersect_neonaes>(corpora, "neonaes");
    }
#endif // STRINGZILLA_TARGET_NEONAES
#if STRINGZILLA_TARGET_NEONSHA
    if (cross_section("Cross NEON SHA", sz_cap_neonsha_k)) {
        bench_sha256_kernels<sz_sha256_state_init_neonsha, sz_sha256_state_update_neonsha,
                             sz_sha256_state_digest_neonsha>(corpora, "neonsha");
    }
#endif // STRINGZILLA_TARGET_NEONSHA
#if STRINGZILLA_TARGET_SVE
    if (cross_section("Cross SVE", sz_cap_sve_k)) {
        bench_find_kernels<sz_find_sve, sz_rfind_sve>(corpora, "sve");
        bench_find_byte_kernels<sz_find_byte_sve, sz_rfind_byte_sve>(corpora, "sve");
        bench_utf8_norm_kernels<sz_utf8_norm_sve>(corpora, "sve");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_sve>(corpora, "sve");
        bench_bytesum_kernels<sz_bytesum_sve>(corpora, "sve");
        bench_equal_kernels<sz_equal_sve>(corpora, "sve");
        bench_order_kernels<sz_order_sve>(corpora, "sve");
        bench_copy_kernels<sz_copy_sve>(corpora, "sve");
        bench_move_kernels<sz_move_sve>(corpora, "sve");
        bench_fill_kernels<sz_fill_sve>(corpora, "sve");
        bench_lookup_kernels<sz_lookup_sve>(corpora, "sve");
        bench_sequence_argsort_kernels<sz_sequence_argsort_sve, sz_sequence_argsort_uncased_sve>(corpora, "sve");
#if STRINGZILLA_HEADER_ONLY
        bench_pgrams_sort_kernels<pgrams_sort_sve_>(corpora, "sve");
#endif
    }
#endif // STRINGZILLA_TARGET_SVE
#if STRINGZILLA_TARGET_SVE2
    if (cross_section("Cross SVE2", sz_cap_sve2_k)) {
        bench_find_byteset_kernels<sz_find_byteset_sve2, sz_rfind_byteset_sve2>(corpora, "sve2");
        bench_utf8_count_kernels<sz_utf8_count_sve2>(corpora, "sve2");
        bench_utf8_seek_kernels<sz_utf8_seek_sve2>(corpora, "sve2");
        bench_utf8_decode_kernels<sz_utf8_decode_sve2>(corpora, "sve2");
        bench_utf8_newlines_kernels<sz_utf8_newlines_sve2>(corpora, "sve2");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_sve2>(corpora, "sve2");
        bench_utf8_delimiters_kernels<sz_utf8_delimiters_sve2>(corpora, "sve2");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_sve2>(corpora, "sve2");
        bench_utf8_graphemes_kernels<sz_utf8_graphemes_sve2>(corpora, "sve2");
        bench_utf8_sentences_kernels<sz_utf8_sentences_sve2>(corpora, "sve2");
        bench_utf8_linebreaks_kernels<sz_utf8_linebreaks_sve2>(corpora, "sve2");
        bench_utf8_norm_kernels<sz_utf8_norm_sve2>(corpora, "sve2");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_sve2>(corpora, "sve2");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_sve2>(corpora, "sve2");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_sve2>(corpora, "sve2");
        bench_bytesum_kernels<sz_bytesum_sve2>(corpora, "sve2");
    }
#endif // STRINGZILLA_TARGET_SVE2
#if STRINGZILLA_TARGET_SVE2AES
    if (cross_section("Cross SVE2 AES", sz_cap_sve2aes_k)) {
        bench_hash_kernels<sz_hash_sve2aes>(corpora, "sve2aes");
        bench_hash_stream_kernels<sz_hash_state_init_sve2aes, sz_hash_state_update_sve2aes,
                                  sz_hash_state_digest_sve2aes>(corpora, "sve2aes");
        bench_fill_random_kernels<sz_fill_random_sve2aes>(corpora, "sve2aes");
        bench_aes256_ctr_kernels<sz_aes256_key_init_sve2aes, sz_aes256_ctr_xor_sve2aes>(corpora, "sve2aes");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_sve2aes, sz_aes256_gcm_encrypt_sve2aes>(corpora, "sve2aes");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_sve2aes, sz_aes256_gcm_encryptor_init_sve2aes,
                                        sz_aes256_gcm_encryptor_update_sve2aes, sz_aes256_gcm_encryptor_digest_sve2aes>(
            corpora, "sve2aes");
    }
#endif // STRINGZILLA_TARGET_SVE2AES
}
