/**
 *  @file bench/cross_ppc64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel benchmarks - Power family: VSX.
 */
#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

void bench_cross_ppc64([[maybe_unused]] corpora_t &corpora) {
#if STRINGZILLA_TARGET_POWERVSX
    if (cross_section("Cross Power VSX", sz_cap_powervsx_k)) {
        bench_find_kernels<sz_find_powervsx, sz_rfind_powervsx>(corpora, "powervsx");
        bench_find_byte_kernels<sz_find_byte_powervsx, sz_rfind_byte_powervsx>(corpora, "powervsx");
        bench_find_byteset_kernels<sz_find_byteset_powervsx, sz_rfind_byteset_powervsx>(corpora, "powervsx");
        bench_utf8_count_kernels<sz_utf8_count_powervsx>(corpora, "powervsx");
        bench_utf8_seek_kernels<sz_utf8_seek_powervsx>(corpora, "powervsx");
        bench_utf8_decode_kernels<sz_utf8_decode_powervsx>(corpora, "powervsx");
        bench_utf8_newlines_kernels<sz_utf8_newlines_powervsx>(corpora, "powervsx");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_powervsx>(corpora, "powervsx");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_powervsx>(corpora, "powervsx");
        bench_utf8_norm_kernels<sz_utf8_norm_powervsx>(corpora, "powervsx");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_powervsx>(corpora, "powervsx");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_powervsx>(corpora, "powervsx");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_powervsx>(corpora, "powervsx");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_powervsx>(corpora, "powervsx");
        bench_bytesum_kernels<sz_bytesum_powervsx>(corpora, "powervsx");
        bench_hash_kernels<sz_hash_powervsx>(corpora, "powervsx");
        bench_hash_stream_kernels<sz_hash_state_init_powervsx, sz_hash_state_update_powervsx,
                                  sz_hash_state_digest_powervsx>(corpora, "powervsx");
        bench_sha256_kernels<sz_sha256_state_init_powervsx, sz_sha256_state_update_powervsx,
                             sz_sha256_state_digest_powervsx>(corpora, "powervsx");
        bench_equal_kernels<sz_equal_powervsx>(corpora, "powervsx");
        bench_order_kernels<sz_order_powervsx>(corpora, "powervsx");
        bench_copy_kernels<sz_copy_powervsx>(corpora, "powervsx");
        bench_move_kernels<sz_move_powervsx>(corpora, "powervsx");
        bench_fill_kernels<sz_fill_powervsx>(corpora, "powervsx");
        bench_fill_random_kernels<sz_fill_random_powervsx>(corpora, "powervsx");
        bench_lookup_kernels<sz_lookup_powervsx>(corpora, "powervsx");
        bench_aes256_ctr_kernels<sz_aes256_key_init_powervsx, sz_aes256_ctr_xor_powervsx>(corpora, "powervsx");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_powervsx, sz_aes256_gcm_encrypt_powervsx>(corpora, "powervsx");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_powervsx, sz_aes256_gcm_encryptor_init_powervsx,
                                        sz_aes256_gcm_encryptor_update_powervsx,
                                        sz_aes256_gcm_encryptor_digest_powervsx>(corpora, "powervsx");
    }
#endif // STRINGZILLA_TARGET_POWERVSX
}
