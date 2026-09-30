/**
 *  @file bench/cross_riscv64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel benchmarks - RISC-V family: RVV, RVV Crypto.
 */
#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

namespace {

#if STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

/** The pgram sort is always inlined under RVV, so it reaches the portable driver as a plain call,
 *  which a driver without RVV can make. */
sz_status_t pgrams_sort_rvv_(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *allocator,
                             sz_sorted_idx_t *order) {
    return sz_pgrams_sort_rvv_(pgrams, count, allocator, order);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_RVV

} // namespace

void bench_cross_riscv64([[maybe_unused]] corpora_t &corpora) {
#if STRINGZILLA_TARGET_RVV
    if (cross_section("Cross RVV", sz_cap_rvv_k)) {
        bench_find_kernels<sz_find_rvv, sz_rfind_rvv>(corpora, "rvv");
        bench_find_byte_kernels<sz_find_byte_rvv, sz_rfind_byte_rvv>(corpora, "rvv");
        bench_find_byteset_kernels<sz_find_byteset_rvv, sz_rfind_byteset_rvv>(corpora, "rvv");
        bench_utf8_count_kernels<sz_utf8_count_rvv>(corpora, "rvv");
        bench_utf8_seek_kernels<sz_utf8_seek_rvv>(corpora, "rvv");
        bench_utf8_decode_kernels<sz_utf8_decode_rvv>(corpora, "rvv");
        bench_utf8_newlines_kernels<sz_utf8_newlines_rvv>(corpora, "rvv");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_rvv>(corpora, "rvv");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_rvv>(corpora, "rvv");
        bench_utf8_norm_kernels<sz_utf8_norm_rvv>(corpora, "rvv");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_rvv>(corpora, "rvv");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_rvv>(corpora, "rvv");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_rvv>(corpora, "rvv");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_rvv>(corpora, "rvv");
        bench_bytesum_kernels<sz_bytesum_rvv>(corpora, "rvv");
        bench_hash_kernels<sz_hash_rvv>(corpora, "rvv");
        bench_hash_stream_kernels<sz_hash_state_init_rvv, sz_hash_state_update_rvv, sz_hash_state_digest_rvv>(corpora,
                                                                                                              "rvv");
        bench_sha256_kernels<sz_sha256_state_init_rvv, sz_sha256_state_update_rvv, sz_sha256_state_digest_rvv>(corpora,
                                                                                                               "rvv");
        bench_equal_kernels<sz_equal_rvv>(corpora, "rvv");
        bench_order_kernels<sz_order_rvv>(corpora, "rvv");
        bench_copy_kernels<sz_copy_rvv>(corpora, "rvv");
        bench_move_kernels<sz_move_rvv>(corpora, "rvv");
        bench_fill_kernels<sz_fill_rvv>(corpora, "rvv");
        bench_fill_random_kernels<sz_fill_random_rvv>(corpora, "rvv");
        bench_lookup_kernels<sz_lookup_rvv>(corpora, "rvv");
        bench_sequence_argsort_kernels<sz_sequence_argsort_rvv, sz_sequence_argsort_uncased_rvv>(corpora, "rvv");
#if STRINGZILLA_HEADER_ONLY
        bench_pgrams_sort_kernels<pgrams_sort_rvv_>(corpora, "rvv");
#endif
    }
#endif // STRINGZILLA_TARGET_RVV
#if STRINGZILLA_TARGET_RVVCRYPTO
    if (cross_section("Cross RVV Crypto", sz_cap_rvvcrypto_k)) {
        bench_aes256_ctr_kernels<sz_aes256_key_init_rvvcrypto, sz_aes256_ctr_xor_rvvcrypto>(corpora, "rvvcrypto");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_rvvcrypto, sz_aes256_gcm_encrypt_rvvcrypto>(corpora,
                                                                                                    "rvvcrypto");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_rvvcrypto, sz_aes256_gcm_encryptor_init_rvvcrypto,
                                        sz_aes256_gcm_encryptor_update_rvvcrypto,
                                        sz_aes256_gcm_encryptor_digest_rvvcrypto>(corpora, "rvvcrypto");
    }
#endif // STRINGZILLA_TARGET_RVVCRYPTO
}
