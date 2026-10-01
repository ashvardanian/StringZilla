/**
 *  @file bench/cross_serial.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel benchmarks - the serial kernels, which every other capability is logged against.
 */
#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

void bench_cross_serial(environment_t &env) {
    section(env, "Cross Serial", sz_cap_serial_k);
    bench_find_kernels<sz_find_serial, sz_rfind_serial>(env, "serial");
    bench_find_byte_kernels<sz_find_byte_serial, sz_rfind_byte_serial>(env, "serial");
    bench_find_byteset_kernels<sz_find_byteset_serial, sz_rfind_byteset_serial>(env, "serial");
    bench_utf8_count_kernels<sz_utf8_count_serial>(env, "serial");
    bench_utf8_seek_kernels<sz_utf8_seek_serial>(env, "serial");
    bench_utf8_decode_kernels<sz_utf8_decode_serial>(env, "serial");
    bench_utf8_newlines_kernels<sz_utf8_newlines_serial>(env, "serial");
    bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_serial>(env, "serial");
    bench_utf8_delimiters_kernels<sz_utf8_delimiters_serial>(env, "serial");
    bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_serial>(env, "serial");
    bench_utf8_graphemes_kernels<sz_utf8_graphemes_serial>(env, "serial");
    bench_utf8_sentences_kernels<sz_utf8_sentences_serial>(env, "serial");
    bench_utf8_linebreaks_kernels<sz_utf8_linebreaks_serial>(env, "serial");
    bench_utf8_norm_kernels<sz_utf8_norm_serial>(env, "serial");
    bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_serial>(env, "serial");
    bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_serial>(env, "serial");
    bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_serial>(env, "serial");
    bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_serial>(env, "serial");
    bench_bytesum_kernels<sz_bytesum_serial>(env, "serial");
    bench_hash_kernels<sz_hash_serial>(env, "serial");
    bench_hash_multiseed_kernels<sz_hash_multiseed_serial>(env, "serial");
    bench_hash_stream_kernels<sz_hash_state_init_serial, sz_hash_state_update_serial, sz_hash_state_digest_serial>(
        env, "serial");
    bench_sha256_kernels<sz_sha256_state_init_serial, sz_sha256_state_update_serial, sz_sha256_state_digest_serial>(
        env, "serial");
    bench_sha256_multistate_kernels<sz_sha256_multistate_update_serial, sz_sha256_multistate_digest_serial>(env,
                                                                                                            "serial");
    bench_equal_kernels<sz_equal_serial>(env, "serial");
    bench_order_kernels<sz_order_serial>(env, "serial");
    bench_copy_kernels<sz_copy_serial>(env, "serial");
    bench_move_kernels<sz_move_serial>(env, "serial");
    bench_fill_kernels<sz_fill_serial>(env, "serial");
    bench_fill_random_kernels<sz_fill_random_serial>(env, "serial");
    bench_lookup_kernels<sz_lookup_serial>(env, "serial");
    bench_aes256_ctr_kernels<sz_aes256_key_init_serial, sz_aes256_ctr_xor_serial>(env, "serial");
    bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_serial, sz_aes256_gcm_encrypt_serial>(env, "serial");
    bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_serial, sz_aes256_gcm_encryptor_init_serial,
                                    sz_aes256_gcm_encryptor_update_serial, sz_aes256_gcm_encryptor_digest_serial>(
        env, "serial");
    bench_map_kernels<sz_order_serial>(env, "serial");
    bench_unordered_map_kernels<sz_hash_serial, sz_equal_serial>(env, "serial", "serial");
    bench_pgrams_sort_kernels<sz_pgrams_sort_serial_>(env, "serial");
    bench_sequence_argsort_kernels<sz_sequence_argsort_serial, sz_sequence_argsort_uncased_serial>(env, "serial");
    bench_sequence_intersect_kernels<sz_sequence_intersect_serial>(env, "serial");
    bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_serial, sz_levenshtein_distances_serial>(
        env, "serial", sz_levenshtein_bytes_k);
    bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_serial, sz_levenshtein_distances_serial>(
        env, "serial", sz_levenshtein_runes_k);
    bench_levenshtein_query_prepare(env);
    bench_levenshtein_step_kernels<levenshtein_step_from_serial>(env, "sz_levenshtein_u64x1_step_serial");
    bench_overlap_step_kernels<sz_overlap_serial_f64x1_positions_per_step_k, sz_overlap_f64x1_prefix_hash_step_serial,
                               sz_overlap_f64x1_prefix_hash_step_tail_serial, sz_overlap_f64x1_window_hash_step_serial,
                               sz_overlap_f64x1_window_hash_step_tail_serial, sz_overlap_u32x1_btree_sort_serial,
                               sz_overlap_u32x1_btree_probe_serial>(env, "serial");
    bench_overlap_scores_kernels<sz_overlap_engine_init_serial, sz_overlap_scores_serial>(env, "serial");
#if !STRINGZILLA_HEADER_ONLY
    bench_substrings_kernels<sz_substrings_counts_serial, sz_substrings_find_serial, sz_substrings_replace_serial,
                             sz_substrings_bm25_scores_serial>(env, "serial");
#endif
}

} // namespace ashvardanian::stringzilla::bench
