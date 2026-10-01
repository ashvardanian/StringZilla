/**
 *  @file test/cross_riscv64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - RISC-V family: RVV, RVV Crypto.
 */
#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

std::size_t test_cross_riscv64(environment_t const &env) {
    [[maybe_unused]] cross_section_t check(env);

#if STRINGZILLA_TARGET_RVV
    check.section("Cross RVV", sz_cap_rvv_k);

    check("test_compare_unit_rvv", [] { check_compare_unit_({"rvv", sz_equal_rvv, sz_order_rvv}); });

    constexpr memory_backend_t memory_rvv {"rvv", sz_copy_rvv, sz_move_rvv, sz_fill_rvv};
    check("test_memory_unit_rvv", [&] { check_memory_unit_(memory_rvv); });
    check("test_memory_equivalence_rvv",
          [&](test_context_t &context) { check_memory_equivalence_(context, memory_rvv); });
    check("test_memory_safety_rvv", [&] { check_memory_safety_(memory_rvv); });

    constexpr lookup_backend_t lookup_rvv {"rvv", sz_lookup_rvv};
    check("test_lookup_unit_rvv", [&] { check_lookup_unit_(lookup_rvv); });
    check("test_lookup_equivalence_rvv",
          [&](test_context_t &context) { check_lookup_equivalence_(context, lookup_rvv); });
    check("test_lookup_safety_rvv", [&] { check_lookup_safety_(lookup_rvv); });

    constexpr find_backend_t find_rvv {
        .name = "rvv",
        .find = sz_find_rvv,
        .rfind = sz_rfind_rvv,
        .find_byte = sz_find_byte_rvv,
        .rfind_byte = sz_rfind_byte_rvv,
        .find_byteset = sz_find_byteset_rvv,
        .rfind_byteset = sz_rfind_byteset_rvv,
    };
    check("test_find_unit_rvv", [&] { check_find_unit_(find_rvv); });
    check("test_find_equivalence_rvv", [&](test_context_t &context) { check_find_equivalence_(context, find_rvv); });
    check("test_find_safety_rvv", [&] { check_find_safety_(find_rvv); });

    constexpr hash_backend_t hash_rvv {
        .hash_kernel = sz_hash_rvv,
        .init_kernel = sz_hash_state_init_rvv,
        .update_kernel = sz_hash_state_update_rvv,
        .digest_kernel = sz_hash_state_digest_rvv,
    };
    constexpr sha256_backend_t sha256_rvv {
        .init_kernel = sz_sha256_state_init_rvv,
        .update_kernel = sz_sha256_state_update_rvv,
        .digest_kernel = sz_sha256_state_digest_rvv,
    };
    check("test_hash_equivalence_rvv", [&](test_context_t &context) {
        check_bytesum_equivalence_(context, sz_bytesum_rvv);
        check_hash_equivalence_(context, hash_rvv);
        check_fill_random_equivalence_(context, sz_fill_random_rvv);
        check_sha256_equivalence_(context, sha256_rvv);
    });

    constexpr sort_backend_t sort_rvv {"rvv", sz_sequence_argsort_rvv, sz_sequence_argsort_uncased_rvv};
    check("test_sort_unit_rvv", [&] { check_sort_unit_(sort_rvv); });
    check("test_sort_equivalence_rvv", [&](test_context_t &context) { check_sort_equivalence_(context, sort_rvv); });
    check("test_sort_safety_rvv", [&] { check_sort_safety_(sort_rvv); });

    constexpr utf8_runes_backend_t utf8_runes_rvv {"rvv", sz_utf8_count_rvv, sz_utf8_seek_rvv, sz_utf8_decode_rvv};
    check("test_utf8_runes_unit_rvv", [&] { check_utf8_runes_unit_(utf8_runes_rvv); });
    check("test_utf8_runes_safety_rvv",
          [&](test_context_t &context) { check_utf8_runes_safety_(context, utf8_runes_rvv); });
    check("test_utf8_runes_equivalence_rvv",
          [&](test_context_t &context) { check_utf8_runes_equivalence_(context, utf8_runes_rvv); });

    constexpr utf8_tokens_backend_t utf8_tokens_rvv {"rvv", sz_utf8_count_rvv, sz_utf8_newlines_rvv,
                                                     sz_utf8_whitespaces_rvv};
    check("test_utf8_tokens_unit_rvv", [&] { check_utf8_tokens_unit_(utf8_tokens_rvv); });
    check("test_utf8_tokens_safety_rvv",
          [&](test_context_t &context) { check_utf8_tokens_safety_(context, utf8_tokens_rvv); });
    check("test_utf8_tokens_equivalence_rvv",
          [&](test_context_t &context) { check_utf8_tokens_equivalence_(context, utf8_tokens_rvv); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_rvv {"rvv", sz_utf8_wordbreaks_rvv};
    check("test_utf8_wordbreaks_unit_rvv", [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_rvv); });
    check("test_utf8_wordbreaks_rules_rvv", [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_rvv); });
    check("test_utf8_wordbreaks_safety_rvv",
          [&](test_context_t &context) { check_utf8_wordbreaks_safety_(context, utf8_wordbreaks_rvv); });
    check("test_utf8_wordbreaks_equivalence_rvv",
          [&](test_context_t &context) { check_utf8_wordbreaks_equivalence_(context, utf8_wordbreaks_rvv); });

    constexpr utf8_norm_kernels_t utf8_norm_rvv {sz_utf8_norm_rvv, sz_utf8_find_denormalized_rvv};
    check("test_utf8_norm_unit_rvv", [&] { check_utf8_norm_unit_(utf8_norm_rvv); });
    check("test_utf8_norm_equivalence_rvv",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_rvv); });
    check("test_utf8_norm_safety_rvv",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_rvv); });

    constexpr utf8_uncased_kernels_t utf8_uncased_rvv {sz_utf8_uncased_fold_rvv, sz_utf8_uncased_search_rvv,
                                                       sz_utf8_uncased_order_rvv, sz_utf8_find_cased_rvv};
    check("test_utf8_uncased_unit_rvv", [&] { check_utf8_uncased_unit_(utf8_uncased_rvv); });
    check("test_utf8_uncased_equivalence_rvv",
          [&](test_context_t &context) { check_utf8_uncased_equivalence_(context, utf8_uncased_rvv); });
    check("test_utf8_uncased_safety_rvv",
          [&](test_context_t &context) { check_utf8_uncased_safety_(context, utf8_uncased_rvv); });
#endif // STRINGZILLA_TARGET_RVV

#if STRINGZILLA_TARGET_RVVCRYPTO
    check.section("Cross RVV Crypto", sz_cap_rvvcrypto_k);

    constexpr hash_backend_t hash_rvvcrypto {
        .hash_kernel = sz_hash_rvvcrypto,
        .init_kernel = sz_hash_state_init_rvvcrypto,
        .update_kernel = sz_hash_state_update_rvvcrypto,
        .digest_kernel = sz_hash_state_digest_rvvcrypto,
    };
    constexpr sha256_backend_t sha256_rvvcrypto {
        .init_kernel = sz_sha256_state_init_rvvcrypto,
        .update_kernel = sz_sha256_state_update_rvvcrypto,
        .digest_kernel = sz_sha256_state_digest_rvvcrypto,
    };
    check("test_hash_equivalence_rvvcrypto", [&](test_context_t &context) {
        check_hash_equivalence_(context, hash_rvvcrypto);
        check_fill_random_equivalence_(context, sz_fill_random_rvvcrypto);
        check_sha256_equivalence_(context, sha256_rvvcrypto);
    });

    constexpr ctr_backend_t ctr_rvvcrypto {"rvvcrypto", sz_aes256_key_init_rvvcrypto, sz_aes256_ctr_xor_rvvcrypto};
    constexpr gcm_backend_t gcm_rvvcrypto {
        .name = "rvvcrypto",
        .key_init = sz_aes256_gcm_key_init_rvvcrypto,
        .encrypt = sz_aes256_gcm_encrypt_rvvcrypto,
        .decrypt = sz_aes256_gcm_decrypt_rvvcrypto,
        .sealer_init = sz_aes256_gcm_encryptor_init_rvvcrypto,
        .sealer_associate = sz_aes256_gcm_encryptor_associate_rvvcrypto,
        .sealer_update = sz_aes256_gcm_encryptor_update_rvvcrypto,
        .sealer_digest = sz_aes256_gcm_encryptor_digest_rvvcrypto,
        .opener_init = sz_aes256_gcm_decryptor_init_rvvcrypto,
        .opener_associate = sz_aes256_gcm_decryptor_associate_rvvcrypto,
        .opener_update = sz_aes256_gcm_decryptor_update_unverified_rvvcrypto,
        .opener_verify = sz_aes256_gcm_decryptor_verify_rvvcrypto,
    };
    check("test_cipher_unit_rvvcrypto", [&] { check_cipher_unit_(ctr_rvvcrypto, gcm_rvvcrypto); });
    check("test_cipher_equivalence_rvvcrypto",
          [&](test_context_t &context) { check_cipher_equivalence_(context, ctr_rvvcrypto, gcm_rvvcrypto); });
#endif // STRINGZILLA_TARGET_RVVCRYPTO

    return check.failures;
}

} // namespace ashvardanian::stringzilla::test
