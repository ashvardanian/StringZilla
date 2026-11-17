/**
 *  @file test/cross_riscv64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - RISC-V family: RVV, RVV Crypto.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

void test_cross_riscv64() {
#if STRINGZILLA_TARGET_RVV
    capability_gate_t const rvv(sz_cap_rvv_k);

    run_test_case("check_compare_unit_({sz_equal_rvv, sz_order_rvv})", rvv,
                  [] { check_compare_unit_({"rvv", sz_equal_rvv, sz_order_rvv}); });

    constexpr strings_backend_t strings_rvv {"rvv",
                                             sz_strings_to_tape_rvv,
                                             sz_strings_to_rows_rvv,
                                             sz_strings_escape_rvv,
                                             sz_strings_unescape_rvv,
                                             sz_strings_lookup_rvv,
                                             sz_strings_normalize_rvv};
    run_test_case("check_strings_unit_(strings_rvv)", rvv, [&] { check_strings_unit_(strings_rvv); });
    run_test_case("check_strings_equivalence_(strings_rvv)", rvv,
                  [&](split_mix_generator_t &generator) { check_strings_equivalence_(generator, strings_rvv); });
    run_test_case("check_strings_safety_(strings_rvv)", rvv, [&] { check_strings_safety_(strings_rvv); });

    constexpr find_backend_t find_rvv {
        .name = "rvv",
        .find = sz_find_rvv,
        .rfind = sz_rfind_rvv,
        .find_byte = sz_find_byte_rvv,
        .rfind_byte = sz_rfind_byte_rvv,
        .find_byteset = sz_find_byteset_rvv,
        .rfind_byteset = sz_rfind_byteset_rvv,
    };
    run_test_case("check_find_unit_(find_rvv)", rvv, [&] { check_find_unit_(find_rvv); });
    run_test_case("check_find_equivalence_(find_rvv)", rvv,
                  [&](split_mix_generator_t &generator) { check_find_equivalence_(generator, find_rvv); });
    run_test_case("check_find_safety_(find_rvv)", rvv, [&] { check_find_safety_(find_rvv); });

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
    run_test_case("check_bytesum_equivalence_(sz_bytesum_rvv)", rvv, [&](split_mix_generator_t &generator) {
        check_bytesum_equivalence_(generator, sz_bytesum_rvv);
        check_hash_equivalence_(generator, hash_rvv);
        check_fill_random_equivalence_(sz_fill_random_rvv);
        check_sha256_equivalence_(generator, sha256_rvv);
    });

    constexpr sort_backend_t sort_rvv {"rvv", sz_sort_order_rvv, sz_sort_uncased_order_rvv};
    run_test_case("check_sort_unit_(sort_rvv)", rvv, [&] { check_sort_unit_(sort_rvv); });
    run_test_case("check_sort_equivalence_(sort_rvv)", rvv,
                  [&](split_mix_generator_t &generator) { check_sort_equivalence_(generator, sort_rvv); });
    run_test_case("check_sort_safety_(sort_rvv)", rvv, [&] { check_sort_safety_(sort_rvv); });

    constexpr levenshtein_backend_t levenshtein_rvv {"rvv", sz_levenshtein_engine_init_rvv,
                                                     sz_levenshtein_distances_rvv};
    run_test_case("check_levenshtein_unit_(levenshtein_rvv)", rvv, [&] { check_levenshtein_unit_(levenshtein_rvv); });
    run_test_case("check_levenshtein_equivalence_(levenshtein_rvv)", rvv, [&](split_mix_generator_t &generator) {
        check_levenshtein_equivalence_(generator, levenshtein_rvv);
    });
    run_test_case("check_levenshtein_safety_(levenshtein_rvv)", rvv,
                  [&] { check_levenshtein_safety_(levenshtein_rvv); });

    constexpr substrings_tier_t substrings_rvv {
        .init = sz_substrings_engine_init_rvv,
        .counts = sz_substrings_counts_rvv,
        .find = sz_substrings_find_rvv,
        .replace = sz_substrings_replace_rvv,
        .bm25_scores = sz_substrings_bm25_scores_rvv,
    };
    run_test_case("check_substrings_unit_(substrings_rvv)", rvv, [&] { check_substrings_unit_(substrings_rvv); });
    run_test_case("check_substrings_equivalence_(substrings_rvv)", rvv,
                  [&](split_mix_generator_t &generator) { check_substrings_equivalence_(generator, substrings_rvv); });

    constexpr subdocuments_tier_t subdocuments_rvv {
        .init = sz_subdocuments_engine_init_rvv,
        .counts = sz_subdocuments_counts_rvv,
        .find = sz_subdocuments_find_rvv,
    };
    run_test_case("check_subdocuments_unit_(subdocuments_rvv)", rvv,
                  [&] { check_subdocuments_unit_(subdocuments_rvv); });
    run_test_case("check_subdocuments_safety_(subdocuments_rvv)", rvv,
                  [&] { check_subdocuments_safety_(subdocuments_rvv); });
    run_test_case("check_subdocuments_equivalence_(subdocuments_rvv)", rvv, [&](split_mix_generator_t &generator) {
        check_subdocuments_equivalence_(generator, subdocuments_rvv);
    });

    constexpr utf8_runes_backend_t utf8_runes_rvv {"rvv", sz_utf8_count_rvv, sz_utf8_seek_rvv, sz_utf8_decode_rvv};
    run_test_case("check_utf8_runes_unit_(utf8_runes_rvv)", rvv, [&] { check_utf8_runes_unit_(utf8_runes_rvv); });
    run_test_case("check_utf8_runes_safety_(utf8_runes_rvv)", rvv,
                  [&](split_mix_generator_t &generator) { check_utf8_runes_safety_(generator, utf8_runes_rvv); });
    run_test_case("check_utf8_runes_equivalence_(utf8_runes_rvv)", rvv,
                  [&](split_mix_generator_t &generator) { check_utf8_runes_equivalence_(generator, utf8_runes_rvv); });

    constexpr utf8_tokens_backend_t utf8_tokens_rvv {"rvv", sz_utf8_count_rvv, sz_utf8_newlines_rvv,
                                                     sz_utf8_whitespaces_rvv};
    run_test_case("check_utf8_tokens_unit_(utf8_tokens_rvv)", rvv, [&] { check_utf8_tokens_unit_(utf8_tokens_rvv); });
    run_test_case("check_utf8_tokens_safety_(utf8_tokens_rvv)", rvv,
                  [&](split_mix_generator_t &generator) { check_utf8_tokens_safety_(generator, utf8_tokens_rvv); });
    run_test_case("check_utf8_tokens_equivalence_(utf8_tokens_rvv)", rvv, [&](split_mix_generator_t &generator) {
        check_utf8_tokens_equivalence_(generator, utf8_tokens_rvv);
    });

    constexpr utf8_delimiters_backend_t utf8_delimiters_rvv {"rvv", sz_utf8_delimiters_rvv};
    run_test_case("check_utf8_delimiters_unit_(utf8_delimiters_rvv)", rvv,
                  [&] { check_utf8_delimiters_unit_(utf8_delimiters_rvv); });
    run_test_case("check_utf8_delimiters_safety_(utf8_delimiters_rvv)", rvv, [&](split_mix_generator_t &generator) {
        check_utf8_delimiters_safety_(generator, utf8_delimiters_rvv);
    });
    run_test_case(
        "check_utf8_delimiters_equivalence_(utf8_delimiters_rvv)", rvv,
        [&](split_mix_generator_t &generator) { check_utf8_delimiters_equivalence_(generator, utf8_delimiters_rvv); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_rvv {"rvv", sz_utf8_wordbreaks_rvv};
    run_test_case("check_utf8_wordbreaks_unit_(utf8_wordbreaks_rvv)", rvv,
                  [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_rvv); });
    run_test_case("check_utf8_wordbreaks_rules_(utf8_wordbreaks_rvv)", rvv,
                  [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_rvv); });
    run_test_case("check_utf8_wordbreaks_safety_(utf8_wordbreaks_rvv)", rvv, [&](split_mix_generator_t &generator) {
        check_utf8_wordbreaks_safety_(generator, utf8_wordbreaks_rvv);
    });
    run_test_case(
        "check_utf8_wordbreaks_equivalence_(utf8_wordbreaks_rvv)", rvv,
        [&](split_mix_generator_t &generator) { check_utf8_wordbreaks_equivalence_(generator, utf8_wordbreaks_rvv); });

    constexpr utf8_norm_kernels_t utf8_norm_rvv {sz_utf8_norm_rvv, sz_utf8_find_denormalized_rvv};
    run_test_case("check_utf8_norm_unit_(utf8_norm_rvv)", rvv, [&] { check_utf8_norm_unit_(utf8_norm_rvv); });
    run_test_case("check_utf8_norm_equivalence_(utf8_norm_rvv)", rvv,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_equivalence_(generator, utf8_norm_rvv); });
    run_test_case("check_utf8_norm_safety_(utf8_norm_rvv)", rvv,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_safety_(generator, utf8_norm_rvv); });

    constexpr utf8_uncased_kernels_t utf8_uncased_rvv {sz_utf8_uncased_fold_rvv, sz_utf8_uncased_search_rvv,
                                                       sz_utf8_uncased_order_rvv, sz_utf8_find_cased_rvv};
    run_test_case("check_utf8_uncased_unit_(utf8_uncased_rvv)", rvv,
                  [&] { check_utf8_uncased_unit_(utf8_uncased_rvv); });
    run_test_case("check_utf8_uncased_equivalence_(utf8_uncased_rvv)", rvv, [&](split_mix_generator_t &generator) {
        check_utf8_uncased_equivalence_(generator, utf8_uncased_rvv);
    });
    run_test_case("check_utf8_uncased_safety_(utf8_uncased_rvv)", rvv,
                  [&](split_mix_generator_t &generator) { check_utf8_uncased_safety_(generator, utf8_uncased_rvv); });
#endif // STRINGZILLA_TARGET_RVV

#if STRINGZILLA_TARGET_RVVCRYPTO
    capability_gate_t const rvvcrypto(sz_cap_rvvcrypto_k);

    constexpr hash_backend_t hash_rvvcrypto {
        .hash_kernel = sz_hash_rvvcrypto,
        .init_kernel = sz_hash_state_init_rvvcrypto,
        .update_kernel = sz_hash_state_update_rvvcrypto,
        .digest_kernel = sz_hash_state_digest_rvvcrypto,
        .strings_kernel = sz_hash_strings_rvvcrypto,
    };
    constexpr sha256_backend_t sha256_rvvcrypto {
        .init_kernel = sz_sha256_state_init_rvvcrypto,
        .update_kernel = sz_sha256_state_update_rvvcrypto,
        .digest_kernel = sz_sha256_state_digest_rvvcrypto,
    };
    run_test_case("check_hash_multiseed_equivalence_({sz_hash_multiseed_rvvcrypto, sz_hash_rvvcrypto})", rvvcrypto,
                  [](split_mix_generator_t &generator) {
                      check_hash_multiseed_equivalence_(generator, {sz_hash_multiseed_rvvcrypto, sz_hash_rvvcrypto});
                  });
    constexpr sha256_multistate_backend_t sha256_multistate_rvvcrypto {
        .update_kernel = sz_sha256_multistate_update_rvvcrypto,
        .digest_kernel = sz_sha256_multistate_digest_rvvcrypto,
    };
    run_test_case("check_hash_equivalence_(hash_rvvcrypto)", rvvcrypto, [&](split_mix_generator_t &generator) {
        check_hash_equivalence_(generator, hash_rvvcrypto);
        check_fill_random_equivalence_(sz_fill_random_rvvcrypto);
        check_sha256_equivalence_(generator, sha256_rvvcrypto);
        check_sha256_multistate_equivalence_(generator, sha256_multistate_rvvcrypto);
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
    run_test_case("check_cipher_unit_(ctr_rvvcrypto, gcm_rvvcrypto)", rvvcrypto,
                  [&] { check_cipher_unit_(ctr_rvvcrypto, gcm_rvvcrypto); });
    run_test_case(
        "check_cipher_equivalence_(ctr_rvvcrypto, gcm_rvvcrypto)", rvvcrypto,
        [&](split_mix_generator_t &generator) { check_cipher_equivalence_(generator, ctr_rvvcrypto, gcm_rvvcrypto); });
#endif // STRINGZILLA_TARGET_RVVCRYPTO
}

} // namespace ashvardanian::stringzilla::test
