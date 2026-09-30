/**
 *  @file bench/cross_wasm.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel benchmarks - WebAssembly family: V128, V128 Relaxed.
 */
#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

void bench_cross_wasm([[maybe_unused]] corpora_t &corpora) {
#if STRINGZILLA_TARGET_V128
    if (cross_section("Cross V128", sz_cap_v128_k)) {
        bench_find_kernels<sz_find_v128, sz_rfind_v128>(corpora, "v128");
        bench_find_byte_kernels<sz_find_byte_v128, sz_rfind_byte_v128>(corpora, "v128");
        bench_find_byteset_kernels<sz_find_byteset_v128, sz_rfind_byteset_v128>(corpora, "v128");
        bench_utf8_count_kernels<sz_utf8_count_v128>(corpora, "v128");
        bench_utf8_seek_kernels<sz_utf8_seek_v128>(corpora, "v128");
        bench_utf8_decode_kernels<sz_utf8_decode_v128>(corpora, "v128");
        bench_utf8_newlines_kernels<sz_utf8_newlines_v128>(corpora, "v128");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_v128>(corpora, "v128");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_v128>(corpora, "v128");
        bench_utf8_norm_kernels<sz_utf8_norm_v128>(corpora, "v128");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_v128>(corpora, "v128");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_v128>(corpora, "v128");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_v128>(corpora, "v128");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_v128>(corpora, "v128");
        bench_bytesum_kernels<sz_bytesum_v128>(corpora, "v128");
        bench_hash_kernels<sz_hash_v128>(corpora, "v128");
        bench_hash_multiseed_kernels<sz_hash_multiseed_v128>(corpora, "v128");
        bench_hash_stream_kernels<sz_hash_state_init_v128, sz_hash_state_update_v128, sz_hash_state_digest_v128>(
            corpora, "v128");
        bench_sha256_kernels<sz_sha256_state_init_v128, sz_sha256_state_update_v128, sz_sha256_state_digest_v128>(
            corpora, "v128");
        bench_equal_kernels<sz_equal_v128>(corpora, "v128");
        bench_order_kernels<sz_order_v128>(corpora, "v128");
        bench_copy_kernels<sz_copy_v128>(corpora, "v128");
        bench_move_kernels<sz_move_v128>(corpora, "v128");
        bench_fill_kernels<sz_fill_v128>(corpora, "v128");
        bench_fill_random_kernels<sz_fill_random_v128>(corpora, "v128");
        bench_lookup_kernels<sz_lookup_v128>(corpora, "v128");
        bench_aes256_ctr_kernels<sz_aes256_key_init_v128, sz_aes256_ctr_xor_v128>(corpora, "v128");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_v128, sz_aes256_gcm_encrypt_v128>(corpora, "v128");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_v128, sz_aes256_gcm_encryptor_init_v128,
                                        sz_aes256_gcm_encryptor_update_v128, sz_aes256_gcm_encryptor_digest_v128>(
            corpora, "v128");
    }
#endif // STRINGZILLA_TARGET_V128
#if STRINGZILLA_TARGET_V128RELAXED
    if (cross_section("Cross V128 Relaxed", sz_cap_v128relaxed_k)) {
        bench_find_kernels<sz_find_v128relaxed, sz_rfind_v128relaxed>(corpora, "v128relaxed");
        bench_find_byte_kernels<sz_find_byte_v128relaxed, sz_rfind_byte_v128relaxed>(corpora, "v128relaxed");
        bench_find_byteset_kernels<sz_find_byteset_v128relaxed, sz_rfind_byteset_v128relaxed>(corpora, "v128relaxed");
        bench_utf8_count_kernels<sz_utf8_count_v128relaxed>(corpora, "v128relaxed");
        bench_utf8_seek_kernels<sz_utf8_seek_v128relaxed>(corpora, "v128relaxed");
        bench_utf8_norm_kernels<sz_utf8_norm_v128relaxed>(corpora, "v128relaxed");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_v128relaxed>(corpora, "v128relaxed");
        bench_bytesum_kernels<sz_bytesum_v128relaxed>(corpora, "v128relaxed");
        bench_hash_kernels<sz_hash_v128relaxed>(corpora, "v128relaxed");
        bench_hash_multiseed_kernels<sz_hash_multiseed_v128relaxed>(corpora, "v128relaxed");
        bench_hash_stream_kernels<sz_hash_state_init_v128relaxed, sz_hash_state_update_v128relaxed,
                                  sz_hash_state_digest_v128relaxed>(corpora, "v128relaxed");
        bench_equal_kernels<sz_equal_v128relaxed>(corpora, "v128relaxed");
        bench_order_kernels<sz_order_v128relaxed>(corpora, "v128relaxed");
        bench_copy_kernels<sz_copy_v128relaxed>(corpora, "v128relaxed");
        bench_move_kernels<sz_move_v128relaxed>(corpora, "v128relaxed");
        bench_fill_kernels<sz_fill_v128relaxed>(corpora, "v128relaxed");
        bench_fill_random_kernels<sz_fill_random_v128relaxed>(corpora, "v128relaxed");
        bench_lookup_kernels<sz_lookup_v128relaxed>(corpora, "v128relaxed");
        bench_aes256_ctr_kernels<sz_aes256_key_init_v128relaxed, sz_aes256_ctr_xor_v128relaxed>(corpora, "v128relaxed");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_v128relaxed, sz_aes256_gcm_encrypt_v128relaxed>(corpora,
                                                                                                        "v128relaxed");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_v128relaxed, sz_aes256_gcm_encryptor_init_v128relaxed,
                                        sz_aes256_gcm_encryptor_update_v128relaxed,
                                        sz_aes256_gcm_encryptor_digest_v128relaxed>(corpora, "v128relaxed");
    }
#endif // STRINGZILLA_TARGET_V128RELAXED
}
