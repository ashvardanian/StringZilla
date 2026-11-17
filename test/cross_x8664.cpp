/**
 *  @file test/cross_x8664.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - x86-64 family: Westmere, Goldmont, Haswell, Skylake, Ice Lake.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

void test_cross_x8664() {
#if STRINGZILLA_TARGET_WESTMERE
    capability_gate_t const westmere(sz_cap_westmere_k);

    run_test_case("check_compare_unit_({sz_equal_westmere, sz_order_westmere})", westmere,
                  [] { check_compare_unit_({"westmere", sz_equal_westmere, sz_order_westmere}); });

    constexpr find_backend_t find_westmere {
        .name = "westmere",
        .find = sz_find_westmere,
        .rfind = sz_rfind_westmere,
        .find_byte = sz_find_byte_westmere,
        .rfind_byte = sz_rfind_byte_westmere,
    };
    run_test_case("check_find_unit_(find_westmere)", westmere, [&] { check_find_unit_(find_westmere); });
    run_test_case("check_find_equivalence_(find_westmere)", westmere,
                  [&](split_mix_generator_t &generator) { check_find_equivalence_(generator, find_westmere); });
    run_test_case("check_find_safety_(find_westmere)", westmere, [&] { check_find_safety_(find_westmere); });

    constexpr hash_backend_t hash_westmere {
        .hash_kernel = sz_hash_westmere,
        .init_kernel = sz_hash_state_init_westmere,
        .update_kernel = sz_hash_state_update_westmere,
        .digest_kernel = sz_hash_state_digest_westmere,
        .strings_kernel = sz_hash_strings_westmere,
    };
    run_test_case("check_hash_equivalence_(hash_westmere)", westmere, [&](split_mix_generator_t &generator) {
        check_hash_equivalence_(generator, hash_westmere);
        check_fill_random_equivalence_(sz_fill_random_westmere);
    });
    run_test_case("check_hash_multiseed_equivalence_({sz_hash_multiseed_westmere, sz_hash_westmere})", westmere,
                  [](split_mix_generator_t &generator) {
                      check_hash_multiseed_equivalence_(generator, {sz_hash_multiseed_westmere, sz_hash_westmere});
                  });

    constexpr ctr_backend_t ctr_westmere {"westmere", sz_aes256_key_init_westmere, sz_aes256_ctr_xor_westmere};
    constexpr gcm_backend_t gcm_westmere {
        .name = "westmere",
        .key_init = sz_aes256_gcm_key_init_westmere,
        .encrypt = sz_aes256_gcm_encrypt_westmere,
        .decrypt = sz_aes256_gcm_decrypt_westmere,
        .sealer_init = sz_aes256_gcm_encryptor_init_westmere,
        .sealer_associate = sz_aes256_gcm_encryptor_associate_westmere,
        .sealer_update = sz_aes256_gcm_encryptor_update_westmere,
        .sealer_digest = sz_aes256_gcm_encryptor_digest_westmere,
        .opener_init = sz_aes256_gcm_decryptor_init_westmere,
        .opener_associate = sz_aes256_gcm_decryptor_associate_westmere,
        .opener_update = sz_aes256_gcm_decryptor_update_unverified_westmere,
        .opener_verify = sz_aes256_gcm_decryptor_verify_westmere,
    };
    run_test_case("check_cipher_unit_(ctr_westmere, gcm_westmere)", westmere,
                  [&] { check_cipher_unit_(ctr_westmere, gcm_westmere); });
    run_test_case(
        "check_cipher_equivalence_(ctr_westmere, gcm_westmere)", westmere,
        [&](split_mix_generator_t &generator) { check_cipher_equivalence_(generator, ctr_westmere, gcm_westmere); });

    run_test_case("check_intersect_unit_(sz_intersect_westmere)", westmere,
                  [] { check_intersect_unit_(sz_intersect_westmere); });
#endif // STRINGZILLA_TARGET_WESTMERE

#if STRINGZILLA_TARGET_GOLDMONT
    capability_gate_t const goldmont(sz_cap_goldmont_k);

    constexpr sha256_backend_t sha256_goldmont {
        .init_kernel = sz_sha256_state_init_goldmont,
        .update_kernel = sz_sha256_state_update_goldmont,
        .digest_kernel = sz_sha256_state_digest_goldmont,
    };
    constexpr sha256_multistate_backend_t sha256_multistate_goldmont {
        .update_kernel = sz_sha256_multistate_update_goldmont,
        .digest_kernel = sz_sha256_multistate_digest_goldmont,
    };
    run_test_case("check_sha256_unit_(sha256_goldmont)", goldmont, [&] { check_sha256_unit_(sha256_goldmont); });
    run_test_case("check_sha256_equivalence_(sha256_goldmont)", goldmont, [&](split_mix_generator_t &generator) {
        check_sha256_equivalence_(generator, sha256_goldmont);
        check_sha256_multistate_equivalence_(generator, sha256_multistate_goldmont);
    });
#endif // STRINGZILLA_TARGET_GOLDMONT

#if STRINGZILLA_TARGET_HASWELL
    capability_gate_t const haswell(sz_cap_haswell_k);

    run_test_case("check_compare_unit_({sz_equal_haswell, sz_order_haswell})", haswell,
                  [] { check_compare_unit_({"haswell", sz_equal_haswell, sz_order_haswell}); });

    constexpr strings_backend_t strings_haswell {"haswell",
                                                 sz_strings_to_tape_haswell,
                                                 sz_strings_to_rows_haswell,
                                                 sz_strings_escape_haswell,
                                                 sz_strings_unescape_haswell,
                                                 sz_strings_lookup_haswell,
                                                 sz_strings_normalize_haswell};
    run_test_case("check_strings_unit_(strings_haswell)", haswell, [&] { check_strings_unit_(strings_haswell); });
    run_test_case("check_strings_equivalence_(strings_haswell)", haswell,
                  [&](split_mix_generator_t &generator) { check_strings_equivalence_(generator, strings_haswell); });
    run_test_case("check_strings_safety_(strings_haswell)", haswell, [&] { check_strings_safety_(strings_haswell); });

    constexpr find_backend_t find_haswell {
        .name = "haswell",
        .find = sz_find_haswell,
        .rfind = sz_rfind_haswell,
        .find_byte = sz_find_byte_haswell,
        .rfind_byte = sz_rfind_byte_haswell,
        .find_byteset = sz_find_byteset_haswell,
        .rfind_byteset = sz_rfind_byteset_haswell,
        .find_strings = sz_find_strings_haswell,
    };
    run_test_case("check_find_unit_(find_haswell)", haswell, [&] { check_find_unit_(find_haswell); });
    run_test_case("check_find_equivalence_(find_haswell)", haswell,
                  [&](split_mix_generator_t &generator) { check_find_equivalence_(generator, find_haswell); });
    run_test_case("check_find_safety_(find_haswell)", haswell, [&] { check_find_safety_(find_haswell); });

    constexpr sha256_multistate_backend_t sha256_multistate_haswell {
        .update_kernel = sz_sha256_multistate_update_haswell,
        .digest_kernel = sz_sha256_multistate_digest_haswell,
    };
    run_test_case("check_bytesum_equivalence_(sz_bytesum_haswell)", haswell, [&](split_mix_generator_t &generator) {
        check_bytesum_equivalence_(generator, sz_bytesum_haswell);
        check_sha256_multistate_equivalence_(generator, sha256_multistate_haswell);
    });

    constexpr sort_backend_t sort_haswell {"haswell", sz_sort_order_haswell, sz_sort_uncased_order_haswell};
    run_test_case("check_sort_unit_(sort_haswell)", haswell, [&] { check_sort_unit_(sort_haswell); });
    run_test_case("check_sort_equivalence_(sort_haswell)", haswell,
                  [&](split_mix_generator_t &generator) { check_sort_equivalence_(generator, sort_haswell); });
    run_test_case("check_sort_safety_(sort_haswell)", haswell, [&] { check_sort_safety_(sort_haswell); });

    constexpr levenshtein_backend_t levenshtein_haswell {"haswell", sz_levenshtein_engine_init_haswell,
                                                         sz_levenshtein_distances_haswell};
    run_test_case("check_levenshtein_unit_(levenshtein_haswell)", haswell,
                  [&] { check_levenshtein_unit_(levenshtein_haswell); });
    run_test_case(
        "check_levenshtein_equivalence_(levenshtein_haswell)", haswell,
        [&](split_mix_generator_t &generator) { check_levenshtein_equivalence_(generator, levenshtein_haswell); });
    run_test_case("check_levenshtein_safety_(levenshtein_haswell)", haswell,
                  [&] { check_levenshtein_safety_(levenshtein_haswell); });

    constexpr overlap_backend_t overlap_haswell {"haswell", sz_overlap_engine_init_haswell, sz_overlap_scores_haswell};
    run_test_case("check_overlap_equivalence_(overlap_haswell)", haswell,
                  [&](split_mix_generator_t &generator) { check_overlap_equivalence_(generator, overlap_haswell); });
    run_test_case("check_overlap_safety_(overlap_haswell)", haswell, [&] { check_overlap_safety_(overlap_haswell); });

    constexpr substrings_tier_t substrings_haswell {
        .init = sz_substrings_engine_init_haswell,
        .counts = sz_substrings_counts_haswell,
        .find = sz_substrings_find_haswell,
        .replace = sz_substrings_replace_haswell,
        .bm25_scores = sz_substrings_bm25_scores_haswell,
    };
    run_test_case("check_substrings_unit_(substrings_haswell)", haswell,
                  [&] { check_substrings_unit_(substrings_haswell); });
    run_test_case("check_substrings_equivalence_(substrings_haswell)", haswell, [&](split_mix_generator_t &generator) {
        check_substrings_equivalence_(generator, substrings_haswell);
    });

    constexpr regex_tier_t regex_haswell {
        .init = sz_regex_engine_init_haswell,
        .counts = sz_regex_counts_haswell,
        .find = sz_regex_find_haswell,
    };
    run_test_case("check_regex_unit_(regex_haswell)", haswell, [&] { check_regex_unit_(regex_haswell); });
    run_test_case("check_regex_equivalence_(regex_haswell)", haswell,
                  [&](split_mix_generator_t &generator) { check_regex_equivalence_(generator, regex_haswell); });

    constexpr subdocuments_tier_t subdocuments_haswell {
        .init = sz_subdocuments_engine_init_haswell,
        .counts = sz_subdocuments_counts_haswell,
        .find = sz_subdocuments_find_haswell,
    };
    run_test_case("check_subdocuments_unit_(subdocuments_haswell)", haswell,
                  [&] { check_subdocuments_unit_(subdocuments_haswell); });
    run_test_case("check_subdocuments_safety_(subdocuments_haswell)", haswell,
                  [&] { check_subdocuments_safety_(subdocuments_haswell); });
    run_test_case(
        "check_subdocuments_equivalence_(subdocuments_haswell)", haswell,
        [&](split_mix_generator_t &generator) { check_subdocuments_equivalence_(generator, subdocuments_haswell); });

    constexpr utf8_runes_backend_t utf8_runes_haswell {"haswell", sz_utf8_count_haswell, sz_utf8_seek_haswell,
                                                       sz_utf8_decode_haswell, sz_utf8_count_strings_haswell};
    run_test_case("check_utf8_runes_unit_(utf8_runes_haswell)", haswell,
                  [&] { check_utf8_runes_unit_(utf8_runes_haswell); });
    run_test_case("check_utf8_runes_safety_(utf8_runes_haswell)", haswell,
                  [&](split_mix_generator_t &generator) { check_utf8_runes_safety_(generator, utf8_runes_haswell); });
    run_test_case("check_utf8_runes_equivalence_(utf8_runes_haswell)", haswell, [&](split_mix_generator_t &generator) {
        check_utf8_runes_equivalence_(generator, utf8_runes_haswell);
    });

    constexpr utf8_tokens_backend_t utf8_tokens_haswell {"haswell", sz_utf8_count_haswell, sz_utf8_newlines_haswell,
                                                         sz_utf8_whitespaces_haswell};
    run_test_case("check_utf8_tokens_unit_(utf8_tokens_haswell)", haswell,
                  [&] { check_utf8_tokens_unit_(utf8_tokens_haswell); });
    run_test_case("check_utf8_tokens_safety_(utf8_tokens_haswell)", haswell,
                  [&](split_mix_generator_t &generator) { check_utf8_tokens_safety_(generator, utf8_tokens_haswell); });
    run_test_case(
        "check_utf8_tokens_equivalence_(utf8_tokens_haswell)", haswell,
        [&](split_mix_generator_t &generator) { check_utf8_tokens_equivalence_(generator, utf8_tokens_haswell); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_haswell {"haswell", sz_utf8_delimiters_haswell};
    run_test_case("check_utf8_delimiters_unit_(utf8_delimiters_haswell)", haswell,
                  [&] { check_utf8_delimiters_unit_(utf8_delimiters_haswell); });
    run_test_case(
        "check_utf8_delimiters_safety_(utf8_delimiters_haswell)", haswell,
        [&](split_mix_generator_t &generator) { check_utf8_delimiters_safety_(generator, utf8_delimiters_haswell); });
    run_test_case("check_utf8_delimiters_equivalence_(utf8_delimiters_haswell)", haswell,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_delimiters_equivalence_(generator, utf8_delimiters_haswell);
                  });

    constexpr utf8_segment_backend_t utf8_wordbreaks_haswell {"haswell", sz_utf8_wordbreaks_haswell};
    run_test_case("check_utf8_wordbreaks_unit_(utf8_wordbreaks_haswell)", haswell,
                  [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_haswell); });
    run_test_case("check_utf8_wordbreaks_rules_(utf8_wordbreaks_haswell)", haswell,
                  [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_haswell); });
    run_test_case(
        "check_utf8_wordbreaks_safety_(utf8_wordbreaks_haswell)", haswell,
        [&](split_mix_generator_t &generator) { check_utf8_wordbreaks_safety_(generator, utf8_wordbreaks_haswell); });
    run_test_case("check_utf8_wordbreaks_equivalence_(utf8_wordbreaks_haswell)", haswell,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_wordbreaks_equivalence_(generator, utf8_wordbreaks_haswell);
                  });

    constexpr utf8_segment_backend_t utf8_graphemes_haswell {"haswell", sz_utf8_graphemes_haswell};
    run_test_case("check_utf8_graphemes_unit_(utf8_graphemes_haswell)", haswell,
                  [&] { check_utf8_graphemes_unit_(utf8_graphemes_haswell); });
    run_test_case("check_utf8_graphemes_rules_(utf8_graphemes_haswell)", haswell,
                  [&] { check_utf8_graphemes_rules_(utf8_graphemes_haswell); });
    run_test_case(
        "check_utf8_graphemes_safety_(utf8_graphemes_haswell)", haswell,
        [&](split_mix_generator_t &generator) { check_utf8_graphemes_safety_(generator, utf8_graphemes_haswell); });
    run_test_case("check_utf8_graphemes_equivalence_(utf8_graphemes_haswell)", haswell,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_graphemes_equivalence_(generator, utf8_graphemes_haswell);
                  });

    constexpr utf8_segment_backend_t utf8_sentences_haswell {"haswell", sz_utf8_sentences_haswell};
    run_test_case("check_utf8_sentences_unit_(utf8_sentences_haswell)", haswell,
                  [&] { check_utf8_sentences_unit_(utf8_sentences_haswell); });
    run_test_case("check_utf8_sentences_rules_(utf8_sentences_haswell)", haswell,
                  [&] { check_utf8_sentences_rules_(utf8_sentences_haswell); });
    run_test_case(
        "check_utf8_sentences_safety_(utf8_sentences_haswell)", haswell,
        [&](split_mix_generator_t &generator) { check_utf8_sentences_safety_(generator, utf8_sentences_haswell); });
    run_test_case("check_utf8_sentences_equivalence_(utf8_sentences_haswell)", haswell,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_sentences_equivalence_(generator, utf8_sentences_haswell);
                  });

    constexpr utf8_segment_backend_t utf8_linebreaks_haswell {"haswell", sz_utf8_linebreaks_haswell};
    run_test_case("check_utf8_linebreaks_unit_(utf8_linebreaks_haswell)", haswell,
                  [&] { check_utf8_linebreaks_unit_(utf8_linebreaks_haswell); });
    run_test_case("check_utf8_linebreaks_rules_(utf8_linebreaks_haswell)", haswell,
                  [&] { check_utf8_linebreaks_rules_(utf8_linebreaks_haswell); });
    run_test_case(
        "check_utf8_linebreaks_safety_(utf8_linebreaks_haswell)", haswell,
        [&](split_mix_generator_t &generator) { check_utf8_linebreaks_safety_(generator, utf8_linebreaks_haswell); });
    run_test_case("check_utf8_linebreaks_equivalence_(utf8_linebreaks_haswell)", haswell,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_linebreaks_equivalence_(generator, utf8_linebreaks_haswell);
                  });

    constexpr utf8_norm_kernels_t utf8_norm_haswell {sz_utf8_norm_haswell, sz_utf8_find_denormalized_haswell};
    run_test_case("check_utf8_norm_unit_(utf8_norm_haswell)", haswell,
                  [&] { check_utf8_norm_unit_(utf8_norm_haswell); });
    run_test_case("check_utf8_norm_equivalence_(utf8_norm_haswell)", haswell, [&](split_mix_generator_t &generator) {
        check_utf8_norm_equivalence_(generator, utf8_norm_haswell);
    });
    run_test_case("check_utf8_norm_safety_(utf8_norm_haswell)", haswell,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_safety_(generator, utf8_norm_haswell); });

    constexpr utf8_uncased_kernels_t utf8_uncased_haswell {sz_utf8_uncased_fold_haswell, sz_utf8_uncased_search_haswell,
                                                           sz_utf8_uncased_order_haswell, sz_utf8_find_cased_haswell};
    run_test_case("check_utf8_uncased_unit_(utf8_uncased_haswell)", haswell,
                  [&] { check_utf8_uncased_unit_(utf8_uncased_haswell); });
    run_test_case(
        "check_utf8_uncased_equivalence_(utf8_uncased_haswell)", haswell,
        [&](split_mix_generator_t &generator) { check_utf8_uncased_equivalence_(generator, utf8_uncased_haswell); });
    run_test_case("check_utf8_uncased_safety_(utf8_uncased_haswell)", haswell, [&](split_mix_generator_t &generator) {
        check_utf8_uncased_safety_(generator, utf8_uncased_haswell);
    });
#endif // STRINGZILLA_TARGET_HASWELL

#if STRINGZILLA_TARGET_SKYLAKE
    capability_gate_t const skylake(sz_cap_skylake_k);

    run_test_case("check_compare_unit_({sz_equal_skylake, sz_order_skylake})", skylake,
                  [] { check_compare_unit_({"skylake", sz_equal_skylake, sz_order_skylake}); });

    constexpr find_backend_t find_skylake {
        .name = "skylake",
        .find = sz_find_skylake,
        .rfind = sz_rfind_skylake,
        .find_byte = sz_find_byte_skylake,
        .rfind_byte = sz_rfind_byte_skylake,
        .find_strings = sz_find_strings_skylake,
    };
    run_test_case("check_find_unit_(find_skylake)", skylake, [&] { check_find_unit_(find_skylake); });
    run_test_case("check_find_equivalence_(find_skylake)", skylake,
                  [&](split_mix_generator_t &generator) { check_find_equivalence_(generator, find_skylake); });
    run_test_case("check_find_safety_(find_skylake)", skylake, [&] { check_find_safety_(find_skylake); });

    constexpr hash_backend_t hash_skylake {
        .hash_kernel = sz_hash_skylake,
        .init_kernel = sz_hash_state_init_skylake,
        .update_kernel = sz_hash_state_update_skylake,
        .digest_kernel = sz_hash_state_digest_skylake,
        .strings_kernel = sz_hash_strings_skylake,
    };
    constexpr sha256_multistate_backend_t sha256_multistate_skylake {
        .update_kernel = sz_sha256_multistate_update_skylake,
        .digest_kernel = sz_sha256_multistate_digest_skylake,
    };
    run_test_case("check_bytesum_equivalence_(sz_bytesum_skylake)", skylake, [&](split_mix_generator_t &generator) {
        check_bytesum_equivalence_(generator, sz_bytesum_skylake);
        check_hash_equivalence_(generator, hash_skylake);
        check_fill_random_equivalence_(sz_fill_random_skylake);
        check_sha256_multistate_equivalence_(generator, sha256_multistate_skylake);
    });

    constexpr sort_backend_t sort_skylake {"skylake", sz_sort_order_skylake, sz_sort_uncased_order_skylake};
    run_test_case("check_sort_unit_(sort_skylake)", skylake, [&] { check_sort_unit_(sort_skylake); });
    run_test_case("check_sort_equivalence_(sort_skylake)", skylake,
                  [&](split_mix_generator_t &generator) { check_sort_equivalence_(generator, sort_skylake); });
    run_test_case("check_sort_safety_(sort_skylake)", skylake, [&] { check_sort_safety_(sort_skylake); });

    constexpr levenshtein_backend_t levenshtein_skylake {"skylake", sz_levenshtein_engine_init_skylake,
                                                         sz_levenshtein_distances_skylake};
    run_test_case("check_levenshtein_unit_(levenshtein_skylake)", skylake,
                  [&] { check_levenshtein_unit_(levenshtein_skylake); });
    run_test_case(
        "check_levenshtein_equivalence_(levenshtein_skylake)", skylake,
        [&](split_mix_generator_t &generator) { check_levenshtein_equivalence_(generator, levenshtein_skylake); });
    run_test_case("check_levenshtein_safety_(levenshtein_skylake)", skylake,
                  [&] { check_levenshtein_safety_(levenshtein_skylake); });

    constexpr overlap_backend_t overlap_skylake {"skylake", sz_overlap_engine_init_skylake, sz_overlap_scores_skylake};
    run_test_case("check_overlap_equivalence_(overlap_skylake)", skylake,
                  [&](split_mix_generator_t &generator) { check_overlap_equivalence_(generator, overlap_skylake); });
    run_test_case("check_overlap_safety_(overlap_skylake)", skylake, [&] { check_overlap_safety_(overlap_skylake); });

    constexpr utf8_norm_kernels_t utf8_norm_skylake {sz_utf8_norm_skylake, sz_utf8_find_denormalized_skylake};
    run_test_case("check_utf8_norm_unit_(utf8_norm_skylake)", skylake,
                  [&] { check_utf8_norm_unit_(utf8_norm_skylake); });
    run_test_case("check_utf8_norm_equivalence_(utf8_norm_skylake)", skylake, [&](split_mix_generator_t &generator) {
        check_utf8_norm_equivalence_(generator, utf8_norm_skylake);
    });
    run_test_case("check_utf8_norm_safety_(utf8_norm_skylake)", skylake,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_safety_(generator, utf8_norm_skylake); });
#endif // STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_TARGET_ICELAKE
    capability_gate_t const icelake(sz_cap_icelake_k);

    constexpr strings_backend_t strings_icelake {"icelake",
                                                 sz_strings_to_tape_icelake,
                                                 sz_strings_to_rows_icelake,
                                                 sz_strings_escape_icelake,
                                                 sz_strings_unescape_icelake,
                                                 sz_strings_lookup_icelake,
                                                 sz_strings_normalize_icelake,
                                                 sz_strings_split_icelake,
                                                 sz_strings_split_lines_icelake,
                                                 sz_strings_join_icelake};
    run_test_case("check_strings_unit_(strings_icelake)", icelake, [&] { check_strings_unit_(strings_icelake); });
    run_test_case("check_strings_equivalence_(strings_icelake)", icelake,
                  [&](split_mix_generator_t &generator) { check_strings_equivalence_(generator, strings_icelake); });
    run_test_case("check_strings_safety_(strings_icelake)", icelake, [&] { check_strings_safety_(strings_icelake); });

    constexpr find_backend_t find_icelake {
        .name = "icelake",
        .find_byteset = sz_find_byteset_icelake,
        .rfind_byteset = sz_rfind_byteset_icelake,
    };
    run_test_case("check_find_unit_(find_icelake)", icelake, [&] { check_find_unit_(find_icelake); });
    run_test_case("check_find_equivalence_(find_icelake)", icelake,
                  [&](split_mix_generator_t &generator) { check_find_equivalence_(generator, find_icelake); });
    run_test_case("check_find_safety_(find_icelake)", icelake, [&] { check_find_safety_(find_icelake); });

    constexpr hash_backend_t hash_icelake {
        .hash_kernel = sz_hash_icelake,
        .init_kernel = sz_hash_state_init_icelake,
        .update_kernel = sz_hash_state_update_icelake,
        .digest_kernel = sz_hash_state_digest_icelake,
        .strings_kernel = sz_hash_strings_icelake,
    };
    run_test_case("check_bytesum_unit_(sz_bytesum_icelake)", icelake, [] { check_bytesum_unit_(sz_bytesum_icelake); });
    run_test_case("check_bytesum_equivalence_(sz_bytesum_icelake)", icelake, [&](split_mix_generator_t &generator) {
        check_bytesum_equivalence_(generator, sz_bytesum_icelake);
        check_hash_equivalence_(generator, hash_icelake);
        check_fill_random_equivalence_(sz_fill_random_icelake);
    });
    run_test_case("check_hash_multiseed_equivalence_({sz_hash_multiseed_icelake, sz_hash_icelake})", icelake,
                  [](split_mix_generator_t &generator) {
                      check_hash_multiseed_equivalence_(generator, {sz_hash_multiseed_icelake, sz_hash_icelake});
                  });

    constexpr ctr_backend_t ctr_icelake {"icelake", sz_aes256_key_init_icelake, sz_aes256_ctr_xor_icelake};
    constexpr gcm_backend_t gcm_icelake {
        .name = "icelake",
        .key_init = sz_aes256_gcm_key_init_icelake,
        .encrypt = sz_aes256_gcm_encrypt_icelake,
        .decrypt = sz_aes256_gcm_decrypt_icelake,
        .sealer_init = sz_aes256_gcm_encryptor_init_icelake,
        .sealer_associate = sz_aes256_gcm_encryptor_associate_icelake,
        .sealer_update = sz_aes256_gcm_encryptor_update_icelake,
        .sealer_digest = sz_aes256_gcm_encryptor_digest_icelake,
        .opener_init = sz_aes256_gcm_decryptor_init_icelake,
        .opener_associate = sz_aes256_gcm_decryptor_associate_icelake,
        .opener_update = sz_aes256_gcm_decryptor_update_unverified_icelake,
        .opener_verify = sz_aes256_gcm_decryptor_verify_icelake,
    };
    run_test_case("check_cipher_unit_(ctr_icelake, gcm_icelake)", icelake,
                  [&] { check_cipher_unit_(ctr_icelake, gcm_icelake); });
    run_test_case(
        "check_cipher_equivalence_(ctr_icelake, gcm_icelake)", icelake,
        [&](split_mix_generator_t &generator) { check_cipher_equivalence_(generator, ctr_icelake, gcm_icelake); });

    run_test_case("check_intersect_unit_(sz_intersect_icelake)", icelake,
                  [] { check_intersect_unit_(sz_intersect_icelake); });

    constexpr levenshtein_backend_t levenshtein_icelake {"icelake", sz_levenshtein_engine_init_icelake,
                                                         sz_levenshtein_distances_icelake};
    run_test_case("check_levenshtein_unit_(levenshtein_icelake)", icelake,
                  [&] { check_levenshtein_unit_(levenshtein_icelake); });
    run_test_case(
        "check_levenshtein_equivalence_(levenshtein_icelake)", icelake,
        [&](split_mix_generator_t &generator) { check_levenshtein_equivalence_(generator, levenshtein_icelake); });
    run_test_case("check_levenshtein_safety_(levenshtein_icelake)", icelake,
                  [&] { check_levenshtein_safety_(levenshtein_icelake); });

    constexpr substrings_tier_t substrings_icelake {
        .init = sz_substrings_engine_init_icelake,
        .counts = sz_substrings_counts_icelake,
        .find = sz_substrings_find_icelake,
        .replace = sz_substrings_replace_icelake,
        .bm25_scores = sz_substrings_bm25_scores_icelake,
    };
    run_test_case("check_substrings_unit_(substrings_icelake)", icelake,
                  [&] { check_substrings_unit_(substrings_icelake); });
    run_test_case("check_substrings_equivalence_(substrings_icelake)", icelake, [&](split_mix_generator_t &generator) {
        check_substrings_equivalence_(generator, substrings_icelake);
    });

    constexpr regex_tier_t regex_icelake {
        .init = sz_regex_engine_init_icelake,
        .counts = sz_regex_counts_icelake,
        .find = sz_regex_find_icelake,
    };
    run_test_case("check_regex_unit_(regex_icelake)", icelake, [&] { check_regex_unit_(regex_icelake); });
    run_test_case("check_regex_equivalence_(regex_icelake)", icelake,
                  [&](split_mix_generator_t &generator) { check_regex_equivalence_(generator, regex_icelake); });

    constexpr subdocuments_tier_t subdocuments_icelake {
        .init = sz_subdocuments_engine_init_icelake,
        .counts = sz_subdocuments_counts_icelake,
        .find = sz_subdocuments_find_icelake,
    };
    run_test_case("check_subdocuments_unit_(subdocuments_icelake)", icelake,
                  [&] { check_subdocuments_unit_(subdocuments_icelake); });
    run_test_case("check_subdocuments_safety_(subdocuments_icelake)", icelake,
                  [&] { check_subdocuments_safety_(subdocuments_icelake); });
    run_test_case(
        "check_subdocuments_equivalence_(subdocuments_icelake)", icelake,
        [&](split_mix_generator_t &generator) { check_subdocuments_equivalence_(generator, subdocuments_icelake); });

    constexpr utf8_runes_backend_t utf8_runes_icelake {"icelake", sz_utf8_count_icelake, sz_utf8_seek_icelake,
                                                       sz_utf8_decode_icelake, sz_utf8_count_strings_icelake};
    run_test_case("check_utf8_runes_unit_(utf8_runes_icelake)", icelake,
                  [&] { check_utf8_runes_unit_(utf8_runes_icelake); });
    run_test_case("check_utf8_runes_safety_(utf8_runes_icelake)", icelake,
                  [&](split_mix_generator_t &generator) { check_utf8_runes_safety_(generator, utf8_runes_icelake); });
    run_test_case("check_utf8_runes_equivalence_(utf8_runes_icelake)", icelake, [&](split_mix_generator_t &generator) {
        check_utf8_runes_equivalence_(generator, utf8_runes_icelake);
    });

    constexpr utf8_tokens_backend_t utf8_tokens_icelake {"icelake", sz_utf8_count_icelake, sz_utf8_newlines_icelake,
                                                         sz_utf8_whitespaces_icelake};
    run_test_case("check_utf8_tokens_unit_(utf8_tokens_icelake)", icelake,
                  [&] { check_utf8_tokens_unit_(utf8_tokens_icelake); });
    run_test_case("check_utf8_tokens_safety_(utf8_tokens_icelake)", icelake,
                  [&](split_mix_generator_t &generator) { check_utf8_tokens_safety_(generator, utf8_tokens_icelake); });
    run_test_case(
        "check_utf8_tokens_equivalence_(utf8_tokens_icelake)", icelake,
        [&](split_mix_generator_t &generator) { check_utf8_tokens_equivalence_(generator, utf8_tokens_icelake); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_icelake {"icelake", sz_utf8_delimiters_icelake};
    run_test_case("check_utf8_delimiters_unit_(utf8_delimiters_icelake)", icelake,
                  [&] { check_utf8_delimiters_unit_(utf8_delimiters_icelake); });
    run_test_case(
        "check_utf8_delimiters_safety_(utf8_delimiters_icelake)", icelake,
        [&](split_mix_generator_t &generator) { check_utf8_delimiters_safety_(generator, utf8_delimiters_icelake); });
    run_test_case("check_utf8_delimiters_equivalence_(utf8_delimiters_icelake)", icelake,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_delimiters_equivalence_(generator, utf8_delimiters_icelake);
                  });

    constexpr utf8_segment_backend_t utf8_wordbreaks_icelake {"icelake", sz_utf8_wordbreaks_icelake};
    run_test_case("check_utf8_wordbreaks_unit_(utf8_wordbreaks_icelake)", icelake,
                  [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_icelake); });
    run_test_case("check_utf8_wordbreaks_rules_(utf8_wordbreaks_icelake)", icelake,
                  [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_icelake); });
    run_test_case(
        "check_utf8_wordbreaks_safety_(utf8_wordbreaks_icelake)", icelake,
        [&](split_mix_generator_t &generator) { check_utf8_wordbreaks_safety_(generator, utf8_wordbreaks_icelake); });
    run_test_case("check_utf8_wordbreaks_equivalence_(utf8_wordbreaks_icelake)", icelake,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_wordbreaks_equivalence_(generator, utf8_wordbreaks_icelake);
                  });

    constexpr utf8_segment_backend_t utf8_graphemes_icelake {"icelake", sz_utf8_graphemes_icelake};
    run_test_case("check_utf8_graphemes_unit_(utf8_graphemes_icelake)", icelake,
                  [&] { check_utf8_graphemes_unit_(utf8_graphemes_icelake); });
    run_test_case("check_utf8_graphemes_rules_(utf8_graphemes_icelake)", icelake,
                  [&] { check_utf8_graphemes_rules_(utf8_graphemes_icelake); });
    run_test_case(
        "check_utf8_graphemes_safety_(utf8_graphemes_icelake)", icelake,
        [&](split_mix_generator_t &generator) { check_utf8_graphemes_safety_(generator, utf8_graphemes_icelake); });
    run_test_case("check_utf8_graphemes_equivalence_(utf8_graphemes_icelake)", icelake,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_graphemes_equivalence_(generator, utf8_graphemes_icelake);
                  });

    constexpr utf8_segment_backend_t utf8_sentences_icelake {"icelake", sz_utf8_sentences_icelake};
    run_test_case("check_utf8_sentences_unit_(utf8_sentences_icelake)", icelake,
                  [&] { check_utf8_sentences_unit_(utf8_sentences_icelake); });
    run_test_case("check_utf8_sentences_rules_(utf8_sentences_icelake)", icelake,
                  [&] { check_utf8_sentences_rules_(utf8_sentences_icelake); });
    run_test_case(
        "check_utf8_sentences_safety_(utf8_sentences_icelake)", icelake,
        [&](split_mix_generator_t &generator) { check_utf8_sentences_safety_(generator, utf8_sentences_icelake); });
    run_test_case("check_utf8_sentences_equivalence_(utf8_sentences_icelake)", icelake,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_sentences_equivalence_(generator, utf8_sentences_icelake);
                  });

    constexpr utf8_segment_backend_t utf8_linebreaks_icelake {"icelake", sz_utf8_linebreaks_icelake};
    run_test_case("check_utf8_linebreaks_unit_(utf8_linebreaks_icelake)", icelake,
                  [&] { check_utf8_linebreaks_unit_(utf8_linebreaks_icelake); });
    run_test_case("check_utf8_linebreaks_rules_(utf8_linebreaks_icelake)", icelake,
                  [&] { check_utf8_linebreaks_rules_(utf8_linebreaks_icelake); });
    run_test_case(
        "check_utf8_linebreaks_safety_(utf8_linebreaks_icelake)", icelake,
        [&](split_mix_generator_t &generator) { check_utf8_linebreaks_safety_(generator, utf8_linebreaks_icelake); });
    run_test_case("check_utf8_linebreaks_equivalence_(utf8_linebreaks_icelake)", icelake,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_linebreaks_equivalence_(generator, utf8_linebreaks_icelake);
                  });

    constexpr utf8_norm_kernels_t utf8_norm_icelake {sz_utf8_norm_icelake, sz_utf8_find_denormalized_icelake};
    run_test_case("check_utf8_norm_unit_(utf8_norm_icelake)", icelake,
                  [&] { check_utf8_norm_unit_(utf8_norm_icelake); });
    run_test_case("check_utf8_norm_equivalence_(utf8_norm_icelake)", icelake, [&](split_mix_generator_t &generator) {
        check_utf8_norm_equivalence_(generator, utf8_norm_icelake);
    });
    run_test_case("check_utf8_norm_safety_(utf8_norm_icelake)", icelake,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_safety_(generator, utf8_norm_icelake); });

    constexpr utf8_uncased_kernels_t utf8_uncased_icelake {sz_utf8_uncased_fold_icelake, sz_utf8_uncased_search_icelake,
                                                           sz_utf8_uncased_order_icelake, sz_utf8_find_cased_icelake};
    run_test_case("check_utf8_uncased_unit_(utf8_uncased_icelake)", icelake,
                  [&] { check_utf8_uncased_unit_(utf8_uncased_icelake); });
    run_test_case(
        "check_utf8_uncased_equivalence_(utf8_uncased_icelake)", icelake,
        [&](split_mix_generator_t &generator) { check_utf8_uncased_equivalence_(generator, utf8_uncased_icelake); });
    run_test_case("check_utf8_uncased_safety_(utf8_uncased_icelake)", icelake, [&](split_mix_generator_t &generator) {
        check_utf8_uncased_safety_(generator, utf8_uncased_icelake);
    });
#endif // STRINGZILLA_TARGET_ICELAKE
}

} // namespace ashvardanian::stringzilla::test
