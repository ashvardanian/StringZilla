/**
 *  @file bench/cross_arm64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel benchmarks - Arm64 family: NEON, NEON AES, NEON SHA, SVE, SVE2, SVE2 AES.
 */
#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

void bench_cross_arm64([[maybe_unused]] environment_t &env) {
#if STRINGZILLA_TARGET_NEON
    if (section(env, "Cross NEON", sz_cap_neon_k)) {
        bench_find_kernels<sz_find_neon, sz_rfind_neon>(env, "neon");
        bench_find_byte_kernels<sz_find_byte_neon, sz_rfind_byte_neon>(env, "neon");
        bench_find_byteset_kernels<sz_find_byteset_neon, sz_rfind_byteset_neon>(env, "neon");
        bench_utf8_count_kernels<sz_utf8_count_neon>(env, "neon");
        bench_utf8_seek_kernels<sz_utf8_seek_neon>(env, "neon");
        bench_utf8_decode_kernels<sz_utf8_decode_neon>(env, "neon");
        bench_utf8_newlines_kernels<sz_utf8_newlines_neon>(env, "neon");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_neon>(env, "neon");
        bench_utf8_delimiters_kernels<sz_utf8_delimiters_neon>(env, "neon");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_neon>(env, "neon");
        bench_utf8_graphemes_kernels<sz_utf8_graphemes_neon>(env, "neon");
        bench_utf8_sentences_kernels<sz_utf8_sentences_neon>(env, "neon");
        bench_utf8_linebreaks_kernels<sz_utf8_linebreaks_neon>(env, "neon");
        bench_utf8_norm_kernels<sz_utf8_norm_neon>(env, "neon");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_neon>(env, "neon");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_neon>(env, "neon");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_neon>(env, "neon");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_neon>(env, "neon");
        bench_bytesum_kernels<sz_bytesum_neon>(env, "neon");
        bench_equal_kernels<sz_equal_neon>(env, "neon");
        bench_order_kernels<sz_order_neon>(env, "neon");
        bench_copy_kernels<sz_copy_neon>(env, "neon");
        bench_move_kernels<sz_move_neon>(env, "neon");
        bench_fill_kernels<sz_fill_neon>(env, "neon");
        bench_lookup_kernels<sz_lookup_neon>(env, "neon");
        bench_map_kernels<sz_order_neon>(env, "neon");
        bench_sequence_argsort_kernels<sz_sequence_argsort_neon, sz_sequence_argsort_uncased_neon>(env, "neon");
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_neon, sz_levenshtein_distances_neon>(
            env, "neon", sz_levenshtein_bytes_k);
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_neon, sz_levenshtein_distances_neon>(
            env, "neon", sz_levenshtein_runes_k);
        bench_overlap_scores_kernels<sz_overlap_engine_init_neon, sz_overlap_scores_neon>(env, "neon");
        bench_substrings_kernels<sz_substrings_engine_init_neon, sz_substrings_counts_neon, sz_substrings_find_neon,
                                 sz_substrings_replace_neon, sz_substrings_bm25_scores_neon>(env, "neon");
    }
#endif // STRINGZILLA_TARGET_NEON
#if STRINGZILLA_TARGET_NEONAES
    if (section(env, "Cross NEON AES", sz_cap_neonaes_k)) {
        bench_hash_kernels<sz_hash_neonaes>(env, "neonaes");
        bench_hash_multiseed_kernels<sz_hash_multiseed_neonaes>(env, "neonaes");
        bench_hash_stream_kernels<sz_hash_state_init_neonaes, sz_hash_state_update_neonaes,
                                  sz_hash_state_digest_neonaes>(env, "neonaes");
        bench_fill_random_kernels<sz_fill_random_neonaes>(env, "neonaes");
        bench_aes256_ctr_kernels<sz_aes256_key_init_neonaes, sz_aes256_ctr_xor_neonaes>(env, "neonaes");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_neonaes, sz_aes256_gcm_encrypt_neonaes>(env, "neonaes");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_neonaes, sz_aes256_gcm_encryptor_init_neonaes,
                                        sz_aes256_gcm_encryptor_update_neonaes, sz_aes256_gcm_encryptor_digest_neonaes>(
            env, "neonaes");
        bench_unordered_map_kernels<sz_hash_neonaes, sz_equal_neon>(env, "neonaes", "neon");
        bench_sequence_intersect_kernels<sz_sequence_intersect_neonaes>(env, "neonaes");
    }
#endif // STRINGZILLA_TARGET_NEONAES
#if STRINGZILLA_TARGET_NEONSHA
    if (section(env, "Cross NEON SHA", sz_cap_neonsha_k)) {
        bench_sha256_kernels<sz_sha256_state_init_neonsha, sz_sha256_state_update_neonsha,
                             sz_sha256_state_digest_neonsha>(env, "neonsha");
        bench_sha256_multistate_kernels<sz_sha256_multistate_update_neonsha, sz_sha256_multistate_digest_neonsha>(
            env, "neonsha");
    }
#endif // STRINGZILLA_TARGET_NEONSHA
#if STRINGZILLA_TARGET_SVE
    if (section(env, "Cross SVE", sz_cap_sve_k)) {
        bench_find_kernels<sz_find_sve, sz_rfind_sve>(env, "sve");
        bench_find_byte_kernels<sz_find_byte_sve, sz_rfind_byte_sve>(env, "sve");
        bench_utf8_norm_kernels<sz_utf8_norm_sve>(env, "sve");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_sve>(env, "sve");
        bench_bytesum_kernels<sz_bytesum_sve>(env, "sve");
        bench_equal_kernels<sz_equal_sve>(env, "sve");
        bench_order_kernels<sz_order_sve>(env, "sve");
        bench_copy_kernels<sz_copy_sve>(env, "sve");
        bench_move_kernels<sz_move_sve>(env, "sve");
        bench_fill_kernels<sz_fill_sve>(env, "sve");
        bench_lookup_kernels<sz_lookup_sve>(env, "sve");
        bench_sequence_argsort_kernels<sz_sequence_argsort_sve, sz_sequence_argsort_uncased_sve>(env, "sve");
    }
#endif // STRINGZILLA_TARGET_SVE
#if STRINGZILLA_TARGET_SVE2
    if (section(env, "Cross SVE2", sz_cap_sve2_k)) {
        bench_find_byteset_kernels<sz_find_byteset_sve2, sz_rfind_byteset_sve2>(env, "sve2");
        bench_utf8_count_kernels<sz_utf8_count_sve2>(env, "sve2");
        bench_utf8_seek_kernels<sz_utf8_seek_sve2>(env, "sve2");
        bench_utf8_decode_kernels<sz_utf8_decode_sve2>(env, "sve2");
        bench_utf8_newlines_kernels<sz_utf8_newlines_sve2>(env, "sve2");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_sve2>(env, "sve2");
        bench_utf8_delimiters_kernels<sz_utf8_delimiters_sve2>(env, "sve2");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_sve2>(env, "sve2");
        bench_utf8_graphemes_kernels<sz_utf8_graphemes_sve2>(env, "sve2");
        bench_utf8_sentences_kernels<sz_utf8_sentences_sve2>(env, "sve2");
        bench_utf8_linebreaks_kernels<sz_utf8_linebreaks_sve2>(env, "sve2");
        bench_utf8_norm_kernels<sz_utf8_norm_sve2>(env, "sve2");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_sve2>(env, "sve2");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_sve2>(env, "sve2");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_sve2>(env, "sve2");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_sve2>(env, "sve2");
        bench_bytesum_kernels<sz_bytesum_sve2>(env, "sve2");
    }
#endif // STRINGZILLA_TARGET_SVE2
#if STRINGZILLA_TARGET_SVE2AES
    if (section(env, "Cross SVE2 AES", sz_cap_sve2aes_k)) {
        bench_hash_kernels<sz_hash_sve2aes>(env, "sve2aes");
        bench_hash_stream_kernels<sz_hash_state_init_sve2aes, sz_hash_state_update_sve2aes,
                                  sz_hash_state_digest_sve2aes>(env, "sve2aes");
        bench_fill_random_kernels<sz_fill_random_sve2aes>(env, "sve2aes");
        bench_aes256_ctr_kernels<sz_aes256_key_init_sve2aes, sz_aes256_ctr_xor_sve2aes>(env, "sve2aes");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_sve2aes, sz_aes256_gcm_encrypt_sve2aes>(env, "sve2aes");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_sve2aes, sz_aes256_gcm_encryptor_init_sve2aes,
                                        sz_aes256_gcm_encryptor_update_sve2aes, sz_aes256_gcm_encryptor_digest_sve2aes>(
            env, "sve2aes");
    }
#endif // STRINGZILLA_TARGET_SVE2AES
}

} // namespace ashvardanian::stringzilla::bench
