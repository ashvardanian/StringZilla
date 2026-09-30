/**
 *  @file bench/cross_serial.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel benchmarks - the serial kernels, which every other capability is logged against.
 */
#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

void bench_cross_serial(corpora_t &corpora) {
    cross_section("Cross Serial", sz_cap_serial_k);
    bench_find_kernels<sz_find_serial, sz_rfind_serial>(corpora, "serial");
    bench_find_byte_kernels<sz_find_byte_serial, sz_rfind_byte_serial>(corpora, "serial");
    bench_find_byteset_kernels<sz_find_byteset_serial, sz_rfind_byteset_serial>(corpora, "serial");
    bench_utf8_count_kernels<sz_utf8_count_serial>(corpora, "serial");
    bench_utf8_seek_kernels<sz_utf8_seek_serial>(corpora, "serial");
    bench_utf8_decode_kernels<sz_utf8_decode_serial>(corpora, "serial");
    bench_utf8_newlines_kernels<sz_utf8_newlines_serial>(corpora, "serial");
    bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_serial>(corpora, "serial");
    bench_utf8_delimiters_kernels<sz_utf8_delimiters_serial>(corpora, "serial");
    bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_serial>(corpora, "serial");
    bench_utf8_graphemes_kernels<sz_utf8_graphemes_serial>(corpora, "serial");
    bench_utf8_sentences_kernels<sz_utf8_sentences_serial>(corpora, "serial");
    bench_utf8_linebreaks_kernels<sz_utf8_linebreaks_serial>(corpora, "serial");
    bench_utf8_norm_kernels<sz_utf8_norm_serial>(corpora, "serial");
    bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_serial>(corpora, "serial");
    bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_serial>(corpora, "serial");
    bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_serial>(corpora, "serial");
    bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_serial>(corpora, "serial");
    bench_bytesum_kernels<sz_bytesum_serial>(corpora, "serial");
    bench_hash_kernels<sz_hash_serial>(corpora, "serial");
    bench_hash_multiseed_kernels<sz_hash_multiseed_serial>(corpora, "serial");
    bench_hash_stream_kernels<sz_hash_state_init_serial, sz_hash_state_update_serial, sz_hash_state_digest_serial>(
        corpora, "serial");
    bench_sha256_kernels<sz_sha256_state_init_serial, sz_sha256_state_update_serial, sz_sha256_state_digest_serial>(
        corpora, "serial");
    bench_sha256_multistate_kernels<sz_sha256_multistate_update_serial, sz_sha256_multistate_digest_serial>(corpora,
                                                                                                            "serial");
    bench_equal_kernels<sz_equal_serial>(corpora, "serial");
    bench_order_kernels<sz_order_serial>(corpora, "serial");
    bench_copy_kernels<sz_copy_serial>(corpora, "serial");
    bench_move_kernels<sz_move_serial>(corpora, "serial");
    bench_fill_kernels<sz_fill_serial>(corpora, "serial");
    bench_fill_random_kernels<sz_fill_random_serial>(corpora, "serial");
    bench_lookup_kernels<sz_lookup_serial>(corpora, "serial");
    bench_aes256_ctr_kernels<sz_aes256_key_init_serial, sz_aes256_ctr_xor_serial>(corpora, "serial");
    bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_serial, sz_aes256_gcm_encrypt_serial>(corpora, "serial");
    bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_serial, sz_aes256_gcm_encryptor_init_serial,
                                    sz_aes256_gcm_encryptor_update_serial, sz_aes256_gcm_encryptor_digest_serial>(
        corpora, "serial");
    bench_map_kernels<sz_order_serial>(corpora, "serial");
    bench_unordered_map_kernels<sz_hash_serial, sz_equal_serial>(corpora, "serial", "serial");
    bench_pgrams_sort_kernels<sz_pgrams_sort_serial_>(corpora, "serial");
    bench_sequence_argsort_kernels<sz_sequence_argsort_serial, sz_sequence_argsort_uncased_serial>(corpora, "serial");
    bench_sequence_intersect_kernels<sz_sequence_intersect_serial>(corpora, "serial");
    bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_serial, sz_levenshtein_distances_serial>(
        corpora, "serial", sz_levenshtein_bytes_k);
    bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_serial, sz_levenshtein_distances_serial>(
        corpora, "serial", sz_levenshtein_runes_k);
    bench_levenshtein_query_prepare(corpora);
    bench_levenshtein_step_kernels<levenshtein_step_from_serial>(corpora, "sz_levenshtein_u64x1_step_serial");
    bench_overlap_step_kernels<sz_overlap_serial_f64x1_positions_per_step_k, sz_overlap_f64x1_prefix_hash_step_serial,
                               sz_overlap_f64x1_prefix_hash_step_tail_serial, sz_overlap_f64x1_window_hash_step_serial,
                               sz_overlap_f64x1_window_hash_step_tail_serial, sz_overlap_u32x1_btree_sort_serial,
                               sz_overlap_u32x1_btree_probe_serial>(corpora, "serial");
    bench_overlap_scores_kernels<sz_overlap_engine_init_serial, sz_overlap_scores_serial>(corpora, "serial");
#if !STRINGZILLA_HEADER_ONLY
    bench_substrings_kernels<sz_substrings_counts_serial, sz_substrings_find_serial, sz_substrings_replace_serial,
                             sz_substrings_bm25_scores_serial>(corpora, "serial");
#endif
}
