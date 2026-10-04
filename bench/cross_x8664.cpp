/**
 *  @file bench/cross_x8664.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel benchmarks - x86-64 family: Westmere, Goldmont, Haswell, Skylake, Ice Lake.
 */
#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

void bench_cross_x8664([[maybe_unused]] environment_t &env) {
#if STRINGZILLA_TARGET_WESTMERE
    if (section(env, "Cross Westmere", sz_cap_westmere_k)) {
        bench_find_kernels<sz_find_westmere, sz_rfind_westmere>(env, "westmere");
        bench_find_byte_kernels<sz_find_byte_westmere, sz_rfind_byte_westmere>(env, "westmere");
        bench_hash_kernels<sz_hash_westmere>(env, "westmere");
        bench_hash_multiseed_kernels<sz_hash_multiseed_westmere>(env, "westmere");
        bench_hash_stream_kernels<sz_hash_state_init_westmere, sz_hash_state_update_westmere,
                                  sz_hash_state_digest_westmere>(env, "westmere");
        bench_equal_kernels<sz_equal_westmere>(env, "westmere");
        bench_order_kernels<sz_order_westmere>(env, "westmere");
        bench_fill_random_kernels<sz_fill_random_westmere>(env, "westmere");
        bench_aes256_ctr_kernels<sz_aes256_key_init_westmere, sz_aes256_ctr_xor_westmere>(env, "westmere");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_westmere, sz_aes256_gcm_encrypt_westmere>(env, "westmere");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_westmere, sz_aes256_gcm_encryptor_init_westmere,
                                        sz_aes256_gcm_encryptor_update_westmere,
                                        sz_aes256_gcm_encryptor_digest_westmere>(env, "westmere");
        bench_sequence_intersect_kernels<sz_sequence_intersect_westmere>(env, "westmere");
    }
#endif // STRINGZILLA_TARGET_WESTMERE
#if STRINGZILLA_TARGET_GOLDMONT
    if (section(env, "Cross Goldmont", sz_cap_goldmont_k)) {
        bench_sha256_kernels<sz_sha256_state_init_goldmont, sz_sha256_state_update_goldmont,
                             sz_sha256_state_digest_goldmont>(env, "goldmont");
        bench_sha256_multistate_kernels<sz_sha256_multistate_update_goldmont, sz_sha256_multistate_digest_goldmont>(
            env, "goldmont");
    }
#endif // STRINGZILLA_TARGET_GOLDMONT
#if STRINGZILLA_TARGET_HASWELL
    if (section(env, "Cross Haswell", sz_cap_haswell_k)) {
        bench_find_kernels<sz_find_haswell, sz_rfind_haswell>(env, "haswell");
        bench_find_byte_kernels<sz_find_byte_haswell, sz_rfind_byte_haswell>(env, "haswell");
        bench_find_byteset_kernels<sz_find_byteset_haswell, sz_rfind_byteset_haswell>(env, "haswell");
        bench_utf8_count_kernels<sz_utf8_count_haswell>(env, "haswell");
        bench_utf8_seek_kernels<sz_utf8_seek_haswell>(env, "haswell");
        bench_utf8_decode_kernels<sz_utf8_decode_haswell>(env, "haswell");
        bench_utf8_newlines_kernels<sz_utf8_newlines_haswell>(env, "haswell");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_haswell>(env, "haswell");
        bench_utf8_delimiters_kernels<sz_utf8_delimiters_haswell>(env, "haswell");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_haswell>(env, "haswell");
        bench_utf8_graphemes_kernels<sz_utf8_graphemes_haswell>(env, "haswell");
        bench_utf8_sentences_kernels<sz_utf8_sentences_haswell>(env, "haswell");
        bench_utf8_linebreaks_kernels<sz_utf8_linebreaks_haswell>(env, "haswell");
        bench_utf8_norm_kernels<sz_utf8_norm_haswell>(env, "haswell");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_haswell>(env, "haswell");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_haswell>(env, "haswell");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_haswell>(env, "haswell");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_haswell>(env, "haswell");
        bench_bytesum_kernels<sz_bytesum_haswell>(env, "haswell");
        bench_sha256_multistate_kernels<sz_sha256_multistate_update_haswell, sz_sha256_multistate_digest_haswell>(
            env, "haswell");
        bench_equal_kernels<sz_equal_haswell>(env, "haswell");
        bench_order_kernels<sz_order_haswell>(env, "haswell");
        bench_copy_kernels<sz_copy_haswell>(env, "haswell");
        bench_move_kernels<sz_move_haswell>(env, "haswell");
        bench_fill_kernels<sz_fill_haswell>(env, "haswell");
        bench_lookup_kernels<sz_lookup_haswell>(env, "haswell");
        bench_map_kernels<sz_order_haswell>(env, "haswell");
#if STRINGZILLA_TARGET_WESTMERE
        // Haswell equality pairs with Westmere hashing, which also requires AES-NI.
        if (env.machine.detected & sz_cap_westmere_k)
            bench_unordered_map_kernels<sz_hash_westmere, sz_equal_haswell>(env, "westmere", "haswell");
#endif
        bench_sequence_argsort_kernels<sz_sequence_argsort_haswell, sz_sequence_argsort_uncased_haswell>(env,
                                                                                                         "haswell");
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_haswell, sz_levenshtein_distances_haswell>(
            env, "haswell", sz_levenshtein_bytes_k);
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_haswell, sz_levenshtein_distances_haswell>(
            env, "haswell", sz_levenshtein_runes_k);
        bench_overlap_scores_kernels<sz_overlap_engine_init_haswell, sz_overlap_scores_haswell>(env, "haswell");
        bench_substrings_kernels<sz_substrings_engine_init_haswell, sz_substrings_counts_haswell,
                                 sz_substrings_find_haswell, sz_substrings_replace_haswell,
                                 sz_substrings_bm25_scores_haswell>(env, "haswell");
    }
#endif // STRINGZILLA_TARGET_HASWELL
#if STRINGZILLA_TARGET_SKYLAKE
    if (section(env, "Cross Skylake", sz_cap_skylake_k)) {
        bench_find_kernels<sz_find_skylake, sz_rfind_skylake>(env, "skylake");
        bench_find_byte_kernels<sz_find_byte_skylake, sz_rfind_byte_skylake>(env, "skylake");
        bench_utf8_norm_kernels<sz_utf8_norm_skylake>(env, "skylake");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_skylake>(env, "skylake");
        bench_bytesum_kernels<sz_bytesum_skylake>(env, "skylake");
        bench_hash_kernels<sz_hash_skylake>(env, "skylake");
        bench_hash_stream_kernels<sz_hash_state_init_skylake, sz_hash_state_update_skylake,
                                  sz_hash_state_digest_skylake>(env, "skylake");
        bench_sha256_multistate_kernels<sz_sha256_multistate_update_skylake, sz_sha256_multistate_digest_skylake>(
            env, "skylake");
        bench_equal_kernels<sz_equal_skylake>(env, "skylake");
        bench_order_kernels<sz_order_skylake>(env, "skylake");
        bench_copy_kernels<sz_copy_skylake>(env, "skylake");
        bench_move_kernels<sz_move_skylake>(env, "skylake");
        bench_fill_kernels<sz_fill_skylake>(env, "skylake");
        bench_fill_random_kernels<sz_fill_random_skylake>(env, "skylake");
        bench_map_kernels<sz_order_skylake>(env, "skylake");
        bench_unordered_map_kernels<sz_hash_skylake, sz_equal_skylake>(env, "skylake", "skylake");
        bench_sequence_argsort_kernels<sz_sequence_argsort_skylake, sz_sequence_argsort_uncased_skylake>(env,
                                                                                                         "skylake");
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_skylake, sz_levenshtein_distances_skylake>(
            env, "skylake", sz_levenshtein_bytes_k);
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_skylake, sz_levenshtein_distances_skylake>(
            env, "skylake", sz_levenshtein_runes_k);
        bench_overlap_scores_kernels<sz_overlap_engine_init_skylake, sz_overlap_scores_skylake>(env, "skylake");
    }
#endif // STRINGZILLA_TARGET_SKYLAKE
#if STRINGZILLA_TARGET_ICELAKE
    if (section(env, "Cross Ice Lake", sz_cap_icelake_k)) {
        bench_find_byteset_kernels<sz_find_byteset_icelake, sz_rfind_byteset_icelake>(env, "icelake");
        bench_utf8_count_kernels<sz_utf8_count_icelake>(env, "icelake");
        bench_utf8_seek_kernels<sz_utf8_seek_icelake>(env, "icelake");
        bench_utf8_decode_kernels<sz_utf8_decode_icelake>(env, "icelake");
        bench_utf8_newlines_kernels<sz_utf8_newlines_icelake>(env, "icelake");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_icelake>(env, "icelake");
        bench_utf8_delimiters_kernels<sz_utf8_delimiters_icelake>(env, "icelake");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_icelake>(env, "icelake");
        bench_utf8_graphemes_kernels<sz_utf8_graphemes_icelake>(env, "icelake");
        bench_utf8_sentences_kernels<sz_utf8_sentences_icelake>(env, "icelake");
        bench_utf8_linebreaks_kernels<sz_utf8_linebreaks_icelake>(env, "icelake");
        bench_utf8_norm_kernels<sz_utf8_norm_icelake>(env, "icelake");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_icelake>(env, "icelake");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_icelake>(env, "icelake");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_icelake>(env, "icelake");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_icelake>(env, "icelake");
        bench_bytesum_kernels<sz_bytesum_icelake>(env, "icelake");
        bench_hash_kernels<sz_hash_icelake>(env, "icelake");
        bench_hash_multiseed_kernels<sz_hash_multiseed_icelake>(env, "icelake");
        bench_hash_stream_kernels<sz_hash_state_init_icelake, sz_hash_state_update_icelake,
                                  sz_hash_state_digest_icelake>(env, "icelake");
        bench_fill_random_kernels<sz_fill_random_icelake>(env, "icelake");
        bench_lookup_kernels<sz_lookup_icelake>(env, "icelake");
        bench_aes256_ctr_kernels<sz_aes256_key_init_icelake, sz_aes256_ctr_xor_icelake>(env, "icelake");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_icelake, sz_aes256_gcm_encrypt_icelake>(env, "icelake");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_icelake, sz_aes256_gcm_encryptor_init_icelake,
                                        sz_aes256_gcm_encryptor_update_icelake, sz_aes256_gcm_encryptor_digest_icelake>(
            env, "icelake");
        bench_sequence_intersect_kernels<sz_sequence_intersect_icelake>(env, "icelake");
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_icelake, sz_levenshtein_distances_icelake>(
            env, "icelake", sz_levenshtein_bytes_k);
        bench_substrings_kernels<sz_substrings_engine_init_icelake, sz_substrings_counts_icelake,
                                 sz_substrings_find_icelake, sz_substrings_replace_icelake,
                                 sz_substrings_bm25_scores_icelake>(env, "icelake");
    }
#endif // STRINGZILLA_TARGET_ICELAKE
}

} // namespace ashvardanian::stringzilla::bench
