/**
 *  @file test/cross_ppc64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - Power ISA family, VSX.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

void test_cross_ppc64() {
#if STRINGZILLA_TARGET_POWERVSX
    capability_gate_t const powervsx(sz_cap_powervsx_k);

    run_test_case("check_compare_unit_({sz_equal_powervsx, sz_order_powervsx})", powervsx,
                  [] { check_compare_unit_({"powervsx", sz_equal_powervsx, sz_order_powervsx}); });

    constexpr strings_backend_t strings_powervsx {"powervsx",
                                                  sz_strings_to_tape_powervsx,
                                                  sz_strings_to_rows_powervsx,
                                                  sz_strings_escape_powervsx,
                                                  sz_strings_unescape_powervsx,
                                                  sz_strings_lookup_powervsx,
                                                  sz_strings_normalize_powervsx};
    run_test_case("check_strings_unit_(strings_powervsx)", powervsx, [&] { check_strings_unit_(strings_powervsx); });
    run_test_case("check_strings_equivalence_(strings_powervsx)", powervsx,
                  [&](split_mix_generator_t &generator) { check_strings_equivalence_(generator, strings_powervsx); });
    run_test_case("check_strings_safety_(strings_powervsx)", powervsx,
                  [&] { check_strings_safety_(strings_powervsx); });

    constexpr find_backend_t find_powervsx {
        .name = "powervsx",
        .find = sz_find_powervsx,
        .rfind = sz_rfind_powervsx,
        .find_byte = sz_find_byte_powervsx,
        .rfind_byte = sz_rfind_byte_powervsx,
        .find_byteset = sz_find_byteset_powervsx,
        .rfind_byteset = sz_rfind_byteset_powervsx,
    };
    run_test_case("check_find_unit_(find_powervsx)", powervsx, [&] { check_find_unit_(find_powervsx); });
    run_test_case("check_find_equivalence_(find_powervsx)", powervsx,
                  [&](split_mix_generator_t &generator) { check_find_equivalence_(generator, find_powervsx); });
    run_test_case("check_find_safety_(find_powervsx)", powervsx, [&] { check_find_safety_(find_powervsx); });

    constexpr hash_backend_t hash_powervsx {
        .hash_kernel = sz_hash_powervsx,
        .init_kernel = sz_hash_state_init_powervsx,
        .update_kernel = sz_hash_state_update_powervsx,
        .digest_kernel = sz_hash_state_digest_powervsx,
        .strings_kernel = sz_hash_strings_powervsx,
    };
    constexpr sha256_backend_t sha256_powervsx {
        .init_kernel = sz_sha256_state_init_powervsx,
        .update_kernel = sz_sha256_state_update_powervsx,
        .digest_kernel = sz_sha256_state_digest_powervsx,
    };
    constexpr sha256_multistate_backend_t sha256_multistate_powervsx {
        .update_kernel = sz_sha256_multistate_update_powervsx,
        .digest_kernel = sz_sha256_multistate_digest_powervsx,
    };
    run_test_case("check_sha256_unit_(sha256_powervsx)", powervsx, [&] { check_sha256_unit_(sha256_powervsx); });
    run_test_case("check_bytesum_equivalence_(sz_bytesum_powervsx)", powervsx, [&](split_mix_generator_t &generator) {
        check_bytesum_equivalence_(generator, sz_bytesum_powervsx);
        check_hash_equivalence_(generator, hash_powervsx);
        check_fill_random_equivalence_(sz_fill_random_powervsx);
        check_sha256_equivalence_(generator, sha256_powervsx);
        check_sha256_multistate_equivalence_(generator, sha256_multistate_powervsx);
    });

    run_test_case("check_hash_multiseed_equivalence_({sz_hash_multiseed_powervsx, sz_hash_powervsx})", powervsx,
                  [](split_mix_generator_t &generator) {
                      check_hash_multiseed_equivalence_(generator, {sz_hash_multiseed_powervsx, sz_hash_powervsx});
                  });

    constexpr ctr_backend_t ctr_powervsx {"powervsx", sz_aes256_key_init_powervsx, sz_aes256_ctr_xor_powervsx};
    constexpr gcm_backend_t gcm_powervsx {
        .name = "powervsx",
        .key_init = sz_aes256_gcm_key_init_powervsx,
        .encrypt = sz_aes256_gcm_encrypt_powervsx,
        .decrypt = sz_aes256_gcm_decrypt_powervsx,
        .sealer_init = sz_aes256_gcm_encryptor_init_powervsx,
        .sealer_associate = sz_aes256_gcm_encryptor_associate_powervsx,
        .sealer_update = sz_aes256_gcm_encryptor_update_powervsx,
        .sealer_digest = sz_aes256_gcm_encryptor_digest_powervsx,
        .opener_init = sz_aes256_gcm_decryptor_init_powervsx,
        .opener_associate = sz_aes256_gcm_decryptor_associate_powervsx,
        .opener_update = sz_aes256_gcm_decryptor_update_unverified_powervsx,
        .opener_verify = sz_aes256_gcm_decryptor_verify_powervsx,
    };
    run_test_case("check_cipher_unit_(ctr_powervsx, gcm_powervsx)", powervsx,
                  [&] { check_cipher_unit_(ctr_powervsx, gcm_powervsx); });
    run_test_case(
        "check_cipher_equivalence_(ctr_powervsx, gcm_powervsx)", powervsx,
        [&](split_mix_generator_t &generator) { check_cipher_equivalence_(generator, ctr_powervsx, gcm_powervsx); });

    constexpr levenshtein_backend_t levenshtein_powervsx {"powervsx", sz_levenshtein_engine_init_powervsx,
                                                          sz_levenshtein_distances_powervsx};
    run_test_case("check_levenshtein_unit_(levenshtein_powervsx)", powervsx,
                  [&] { check_levenshtein_unit_(levenshtein_powervsx); });
    run_test_case(
        "check_levenshtein_equivalence_(levenshtein_powervsx)", powervsx,
        [&](split_mix_generator_t &generator) { check_levenshtein_equivalence_(generator, levenshtein_powervsx); });
    run_test_case("check_levenshtein_safety_(levenshtein_powervsx)", powervsx,
                  [&] { check_levenshtein_safety_(levenshtein_powervsx); });

    constexpr overlap_backend_t overlap_powervsx {"powervsx", sz_overlap_engine_init_powervsx,
                                                  sz_overlap_scores_powervsx};
    run_test_case("check_overlap_equivalence_(overlap_powervsx)", powervsx,
                  [&](split_mix_generator_t &generator) { check_overlap_equivalence_(generator, overlap_powervsx); });
    run_test_case("check_overlap_safety_(overlap_powervsx)", powervsx,
                  [&] { check_overlap_safety_(overlap_powervsx); });

    constexpr substrings_tier_t substrings_powervsx {
        .init = sz_substrings_engine_init_powervsx,
        .counts = sz_substrings_counts_powervsx,
        .find = sz_substrings_find_powervsx,
        .replace = sz_substrings_replace_powervsx,
        .bm25_scores = sz_substrings_bm25_scores_powervsx,
    };
    run_test_case("check_substrings_unit_(substrings_powervsx)", powervsx,
                  [&] { check_substrings_unit_(substrings_powervsx); });
    run_test_case(
        "check_substrings_equivalence_(substrings_powervsx)", powervsx,
        [&](split_mix_generator_t &generator) { check_substrings_equivalence_(generator, substrings_powervsx); });

    constexpr regex_tier_t regex_powervsx {
        .init = sz_regex_engine_init_powervsx,
        .counts = sz_regex_counts_powervsx,
        .find = sz_regex_find_powervsx,
    };
    constexpr regex_rewriting_t regex_rewriting_powervsx {sz_regex_replace_powervsx, sz_regex_extract_powervsx};
    run_test_case("check_regex_unit_(regex_powervsx, regex_rewriting_powervsx)", powervsx,
                  [&] { check_regex_unit_(regex_powervsx, regex_rewriting_powervsx); });
    run_test_case("check_regex_equivalence_(regex_powervsx, regex_rewriting_powervsx)", powervsx,
                  [&](split_mix_generator_t &generator) {
                      check_regex_equivalence_(generator, regex_powervsx, regex_rewriting_powervsx);
                  });

    constexpr subdocuments_tier_t subdocuments_powervsx {
        .init = sz_subdocuments_engine_init_powervsx,
        .counts = sz_subdocuments_counts_powervsx,
        .find = sz_subdocuments_find_powervsx,
    };
    run_test_case("check_subdocuments_unit_(subdocuments_powervsx)", powervsx,
                  [&] { check_subdocuments_unit_(subdocuments_powervsx); });
    run_test_case("check_subdocuments_safety_(subdocuments_powervsx)", powervsx,
                  [&] { check_subdocuments_safety_(subdocuments_powervsx); });
    run_test_case(
        "check_subdocuments_equivalence_(subdocuments_powervsx)", powervsx,
        [&](split_mix_generator_t &generator) { check_subdocuments_equivalence_(generator, subdocuments_powervsx); });

    constexpr utf8_runes_backend_t utf8_runes_powervsx {"powervsx", sz_utf8_count_powervsx, sz_utf8_seek_powervsx,
                                                        sz_utf8_decode_powervsx};
    run_test_case("check_utf8_runes_unit_(utf8_runes_powervsx)", powervsx,
                  [&] { check_utf8_runes_unit_(utf8_runes_powervsx); });
    run_test_case("check_utf8_runes_safety_(utf8_runes_powervsx)", powervsx,
                  [&](split_mix_generator_t &generator) { check_utf8_runes_safety_(generator, utf8_runes_powervsx); });
    run_test_case(
        "check_utf8_runes_equivalence_(utf8_runes_powervsx)", powervsx,
        [&](split_mix_generator_t &generator) { check_utf8_runes_equivalence_(generator, utf8_runes_powervsx); });

    constexpr utf8_tokens_backend_t utf8_tokens_powervsx {"powervsx", sz_utf8_count_powervsx, sz_utf8_newlines_powervsx,
                                                          sz_utf8_whitespaces_powervsx};
    run_test_case("check_utf8_tokens_unit_(utf8_tokens_powervsx)", powervsx,
                  [&] { check_utf8_tokens_unit_(utf8_tokens_powervsx); });
    run_test_case("check_utf8_tokens_safety_(utf8_tokens_powervsx)", powervsx, [&](split_mix_generator_t &generator) {
        check_utf8_tokens_safety_(generator, utf8_tokens_powervsx);
    });
    run_test_case(
        "check_utf8_tokens_equivalence_(utf8_tokens_powervsx)", powervsx,
        [&](split_mix_generator_t &generator) { check_utf8_tokens_equivalence_(generator, utf8_tokens_powervsx); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_powervsx {"powervsx", sz_utf8_delimiters_powervsx};
    run_test_case("check_utf8_delimiters_unit_(utf8_delimiters_powervsx)", powervsx,
                  [&] { check_utf8_delimiters_unit_(utf8_delimiters_powervsx); });
    run_test_case(
        "check_utf8_delimiters_safety_(utf8_delimiters_powervsx)", powervsx,
        [&](split_mix_generator_t &generator) { check_utf8_delimiters_safety_(generator, utf8_delimiters_powervsx); });
    run_test_case("check_utf8_delimiters_equivalence_(utf8_delimiters_powervsx)", powervsx,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_delimiters_equivalence_(generator, utf8_delimiters_powervsx);
                  });

    constexpr utf8_segment_backend_t utf8_wordbreaks_powervsx {"powervsx", sz_utf8_wordbreaks_powervsx};
    run_test_case("check_utf8_wordbreaks_unit_(utf8_wordbreaks_powervsx)", powervsx,
                  [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_powervsx); });
    run_test_case("check_utf8_wordbreaks_rules_(utf8_wordbreaks_powervsx)", powervsx,
                  [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_powervsx); });
    run_test_case(
        "check_utf8_wordbreaks_safety_(utf8_wordbreaks_powervsx)", powervsx,
        [&](split_mix_generator_t &generator) { check_utf8_wordbreaks_safety_(generator, utf8_wordbreaks_powervsx); });
    run_test_case("check_utf8_wordbreaks_equivalence_(utf8_wordbreaks_powervsx)", powervsx,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_wordbreaks_equivalence_(generator, utf8_wordbreaks_powervsx);
                  });

    constexpr utf8_norm_kernels_t utf8_norm_powervsx {sz_utf8_norm_powervsx, sz_utf8_find_denormalized_powervsx};
    run_test_case("check_utf8_norm_unit_(utf8_norm_powervsx)", powervsx,
                  [&] { check_utf8_norm_unit_(utf8_norm_powervsx); });
    run_test_case("check_utf8_norm_equivalence_(utf8_norm_powervsx)", powervsx, [&](split_mix_generator_t &generator) {
        check_utf8_norm_equivalence_(generator, utf8_norm_powervsx);
    });
    run_test_case("check_utf8_norm_safety_(utf8_norm_powervsx)", powervsx,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_safety_(generator, utf8_norm_powervsx); });

    constexpr utf8_uncased_kernels_t utf8_uncased_powervsx {
        sz_utf8_uncased_fold_powervsx, sz_utf8_uncased_search_powervsx, sz_utf8_uncased_order_powervsx,
        sz_utf8_find_cased_powervsx};
    run_test_case("check_utf8_uncased_unit_(utf8_uncased_powervsx)", powervsx,
                  [&] { check_utf8_uncased_unit_(utf8_uncased_powervsx); });
    run_test_case(
        "check_utf8_uncased_equivalence_(utf8_uncased_powervsx)", powervsx,
        [&](split_mix_generator_t &generator) { check_utf8_uncased_equivalence_(generator, utf8_uncased_powervsx); });
    run_test_case("check_utf8_uncased_safety_(utf8_uncased_powervsx)", powervsx, [&](split_mix_generator_t &generator) {
        check_utf8_uncased_safety_(generator, utf8_uncased_powervsx);
    });
#endif // STRINGZILLA_TARGET_POWERVSX
}

} // namespace ashvardanian::stringzilla::test
