/**
 *  @file bench/cross_loongarch64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel benchmarks - LoongArch family: LASX.
 */
#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

void bench_cross_loongarch64([[maybe_unused]] corpora_t &corpora) {
#if STRINGZILLA_TARGET_LOONGSONASX
    if (cross_section("Cross Loongson ASX", sz_cap_loongsonasx_k)) {
        bench_find_kernels<sz_find_loongsonasx, sz_rfind_loongsonasx>(corpora, "loongsonasx");
        bench_find_byte_kernels<sz_find_byte_loongsonasx, sz_rfind_byte_loongsonasx>(corpora, "loongsonasx");
        bench_find_byteset_kernels<sz_find_byteset_loongsonasx, sz_rfind_byteset_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_count_kernels<sz_utf8_count_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_seek_kernels<sz_utf8_seek_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_decode_kernels<sz_utf8_decode_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_newlines_kernels<sz_utf8_newlines_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_norm_kernels<sz_utf8_norm_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_loongsonasx>(corpora, "loongsonasx");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_loongsonasx>(corpora, "loongsonasx");
        bench_bytesum_kernels<sz_bytesum_loongsonasx>(corpora, "loongsonasx");
        bench_hash_kernels<sz_hash_loongsonasx>(corpora, "loongsonasx");
        bench_hash_stream_kernels<sz_hash_state_init_loongsonasx, sz_hash_state_update_loongsonasx,
                                  sz_hash_state_digest_loongsonasx>(corpora, "loongsonasx");
        bench_sha256_kernels<sz_sha256_state_init_loongsonasx, sz_sha256_state_update_loongsonasx,
                             sz_sha256_state_digest_loongsonasx>(corpora, "loongsonasx");
        bench_equal_kernels<sz_equal_loongsonasx>(corpora, "loongsonasx");
        bench_order_kernels<sz_order_loongsonasx>(corpora, "loongsonasx");
        bench_copy_kernels<sz_copy_loongsonasx>(corpora, "loongsonasx");
        bench_move_kernels<sz_move_loongsonasx>(corpora, "loongsonasx");
        bench_fill_kernels<sz_fill_loongsonasx>(corpora, "loongsonasx");
        bench_fill_random_kernels<sz_fill_random_loongsonasx>(corpora, "loongsonasx");
        bench_lookup_kernels<sz_lookup_loongsonasx>(corpora, "loongsonasx");
    }
#endif // STRINGZILLA_TARGET_LOONGSONASX
}
