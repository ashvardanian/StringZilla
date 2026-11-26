/**
 *  @file test/cross_arm64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - Arm64 family: NEON, NEON AES, NEON SHA, SVE, SVE2, SVE2 AES.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

void test_cross_arm64() {
#if STRINGZILLA_TARGET_NEON
    capability_gate_t const neon(sz_cap_neon_k);

    run_test_case("check_compare_unit_({sz_equal_neon, sz_order_neon})", neon,
                  [] { check_compare_unit_({"neon", sz_equal_neon, sz_order_neon}); });

    constexpr strings_backend_t strings_neon {"neon",
                                              sz_strings_to_tape_neon,
                                              sz_strings_to_rows_neon,
                                              sz_strings_escape_neon,
                                              sz_strings_unescape_neon,
                                              sz_strings_lookup_neon,
                                              sz_strings_normalize_neon,
                                              sz_strings_split_neon,
                                              sz_strings_split_lines_neon,
                                              sz_strings_join_neon};
    run_test_case("check_strings_unit_(strings_neon)", neon, [&] { check_strings_unit_(strings_neon); });
    run_test_case("check_strings_equivalence_(strings_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_strings_equivalence_(generator, strings_neon); });
    run_test_case("check_strings_safety_(strings_neon)", neon, [&] { check_strings_safety_(strings_neon); });

    constexpr find_backend_t find_neon {
        .name = "neon",
        .find = sz_find_neon,
        .rfind = sz_rfind_neon,
        .find_byte = sz_find_byte_neon,
        .rfind_byte = sz_rfind_byte_neon,
        .find_byteset = sz_find_byteset_neon,
        .rfind_byteset = sz_rfind_byteset_neon,
        .find_strings = sz_find_strings_neon,
    };
    run_test_case("check_find_unit_(find_neon)", neon, [&] { check_find_unit_(find_neon); });
    run_test_case("check_find_equivalence_(find_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_find_equivalence_(generator, find_neon); });
    run_test_case("check_find_safety_(find_neon)", neon, [&] { check_find_safety_(find_neon); });

    run_test_case("check_bytesum_equivalence_(sz_bytesum_neon)", neon,
                  [](split_mix_generator_t &generator) { check_bytesum_equivalence_(generator, sz_bytesum_neon); });

    constexpr sort_backend_t sort_neon {"neon", sz_sort_order_neon, sz_sort_uncased_order_neon};
    run_test_case("check_sort_unit_(sort_neon)", neon, [&] { check_sort_unit_(sort_neon); });
    run_test_case("check_sort_equivalence_(sort_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_sort_equivalence_(generator, sort_neon); });
    run_test_case("check_sort_safety_(sort_neon)", neon, [&] { check_sort_safety_(sort_neon); });

    constexpr levenshtein_backend_t levenshtein_neon {"neon", sz_levenshtein_engine_init_neon,
                                                      sz_levenshtein_distances_neon};
    run_test_case("check_levenshtein_unit_(levenshtein_neon)", neon,
                  [&] { check_levenshtein_unit_(levenshtein_neon); });
    run_test_case("check_levenshtein_equivalence_(levenshtein_neon)", neon, [&](split_mix_generator_t &generator) {
        check_levenshtein_equivalence_(generator, levenshtein_neon);
    });
    run_test_case("check_levenshtein_safety_(levenshtein_neon)", neon,
                  [&] { check_levenshtein_safety_(levenshtein_neon); });

    constexpr ngrams_backend_t ngrams_neon {"neon",
                                            sz_ngrams_engine_init_neon,
                                            sz_ngrams_overlaps_neon,
                                            sz_ngrams_sketches_neon,
                                            sz_ngrams_residues_neon,
                                            sz_ngrams_sketch_neon};
    run_test_case("check_ngrams_overlaps_(ngrams_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_ngrams_overlaps_(generator, ngrams_neon); });
    run_test_case("check_ngrams_residues_(ngrams_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_ngrams_residues_(generator, ngrams_neon); });
    run_test_case("check_ngrams_sketches_(ngrams_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_ngrams_sketches_(generator, ngrams_neon); });
    run_test_case("check_ngrams_jaccard_(ngrams_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_ngrams_jaccard_(generator, ngrams_neon); });
    run_test_case("check_ngrams_safety_(ngrams_neon)", neon, [&] { check_ngrams_safety_(ngrams_neon); });

    constexpr substrings_tier_t substrings_neon {
        .init = sz_substrings_engine_init_neon,
        .counts = sz_substrings_counts_neon,
        .find = sz_substrings_find_neon,
        .replace = sz_substrings_replace_neon,
        .bm25_scores = sz_substrings_bm25_scores_neon,
    };
    run_test_case("check_substrings_unit_(substrings_neon)", neon, [&] { check_substrings_unit_(substrings_neon); });
    run_test_case("check_substrings_equivalence_(substrings_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_substrings_equivalence_(generator, substrings_neon); });

    constexpr regex_tier_t regex_neon {
        .init = sz_regex_engine_init_neon,
        .counts = sz_regex_counts_neon,
        .find = sz_regex_find_neon,
    };
    run_test_case("check_regex_unit_(regex_neon)", neon, [&] { check_regex_unit_(regex_neon); });
    run_test_case("check_regex_equivalence_(regex_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_regex_equivalence_(generator, regex_neon); });

    constexpr subdocuments_tier_t subdocuments_neon {
        .init = sz_subdocuments_engine_init_neon,
        .counts = sz_subdocuments_counts_neon,
        .find = sz_subdocuments_find_neon,
    };
    run_test_case("check_subdocuments_unit_(subdocuments_neon)", neon,
                  [&] { check_subdocuments_unit_(subdocuments_neon); });
    run_test_case("check_subdocuments_safety_(subdocuments_neon)", neon,
                  [&] { check_subdocuments_safety_(subdocuments_neon); });
    run_test_case("check_subdocuments_equivalence_(subdocuments_neon)", neon, [&](split_mix_generator_t &generator) {
        check_subdocuments_equivalence_(generator, subdocuments_neon);
    });

    constexpr utf8_runes_backend_t utf8_runes_neon {"neon", sz_utf8_count_neon, sz_utf8_seek_neon, sz_utf8_decode_neon,
                                                    sz_utf8_count_strings_neon};
    run_test_case("check_utf8_runes_unit_(utf8_runes_neon)", neon, [&] { check_utf8_runes_unit_(utf8_runes_neon); });
    run_test_case("check_utf8_runes_safety_(utf8_runes_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_utf8_runes_safety_(generator, utf8_runes_neon); });
    run_test_case("check_utf8_runes_equivalence_(utf8_runes_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_utf8_runes_equivalence_(generator, utf8_runes_neon); });

    constexpr utf8_tokens_backend_t utf8_tokens_neon {"neon", sz_utf8_count_neon, sz_utf8_newlines_neon,
                                                      sz_utf8_whitespaces_neon};
    run_test_case("check_utf8_tokens_unit_(utf8_tokens_neon)", neon,
                  [&] { check_utf8_tokens_unit_(utf8_tokens_neon); });
    run_test_case("check_utf8_tokens_safety_(utf8_tokens_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_utf8_tokens_safety_(generator, utf8_tokens_neon); });
    run_test_case("check_utf8_tokens_equivalence_(utf8_tokens_neon)", neon, [&](split_mix_generator_t &generator) {
        check_utf8_tokens_equivalence_(generator, utf8_tokens_neon);
    });

    constexpr utf8_delimiters_backend_t utf8_delimiters_neon {"neon", sz_utf8_delimiters_neon};
    run_test_case("check_utf8_delimiters_unit_(utf8_delimiters_neon)", neon,
                  [&] { check_utf8_delimiters_unit_(utf8_delimiters_neon); });
    run_test_case("check_utf8_delimiters_safety_(utf8_delimiters_neon)", neon, [&](split_mix_generator_t &generator) {
        check_utf8_delimiters_safety_(generator, utf8_delimiters_neon);
    });
    run_test_case(
        "check_utf8_delimiters_equivalence_(utf8_delimiters_neon)", neon,
        [&](split_mix_generator_t &generator) { check_utf8_delimiters_equivalence_(generator, utf8_delimiters_neon); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_neon {"neon", sz_utf8_wordbreaks_neon};
    run_test_case("check_utf8_wordbreaks_unit_(utf8_wordbreaks_neon)", neon,
                  [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_neon); });
    run_test_case("check_utf8_wordbreaks_rules_(utf8_wordbreaks_neon)", neon,
                  [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_neon); });
    run_test_case("check_utf8_wordbreaks_safety_(utf8_wordbreaks_neon)", neon, [&](split_mix_generator_t &generator) {
        check_utf8_wordbreaks_safety_(generator, utf8_wordbreaks_neon);
    });
    run_test_case(
        "check_utf8_wordbreaks_equivalence_(utf8_wordbreaks_neon)", neon,
        [&](split_mix_generator_t &generator) { check_utf8_wordbreaks_equivalence_(generator, utf8_wordbreaks_neon); });

    constexpr utf8_segment_backend_t utf8_graphemes_neon {"neon", sz_utf8_graphemes_neon};
    run_test_case("check_utf8_graphemes_unit_(utf8_graphemes_neon)", neon,
                  [&] { check_utf8_graphemes_unit_(utf8_graphemes_neon); });
    run_test_case("check_utf8_graphemes_rules_(utf8_graphemes_neon)", neon,
                  [&] { check_utf8_graphemes_rules_(utf8_graphemes_neon); });
    run_test_case("check_utf8_graphemes_safety_(utf8_graphemes_neon)", neon, [&](split_mix_generator_t &generator) {
        check_utf8_graphemes_safety_(generator, utf8_graphemes_neon);
    });
    run_test_case(
        "check_utf8_graphemes_equivalence_(utf8_graphemes_neon)", neon,
        [&](split_mix_generator_t &generator) { check_utf8_graphemes_equivalence_(generator, utf8_graphemes_neon); });

    constexpr utf8_segment_backend_t utf8_sentences_neon {"neon", sz_utf8_sentences_neon};
    run_test_case("check_utf8_sentences_unit_(utf8_sentences_neon)", neon,
                  [&] { check_utf8_sentences_unit_(utf8_sentences_neon); });
    run_test_case("check_utf8_sentences_rules_(utf8_sentences_neon)", neon,
                  [&] { check_utf8_sentences_rules_(utf8_sentences_neon); });
    run_test_case("check_utf8_sentences_safety_(utf8_sentences_neon)", neon, [&](split_mix_generator_t &generator) {
        check_utf8_sentences_safety_(generator, utf8_sentences_neon);
    });
    run_test_case(
        "check_utf8_sentences_equivalence_(utf8_sentences_neon)", neon,
        [&](split_mix_generator_t &generator) { check_utf8_sentences_equivalence_(generator, utf8_sentences_neon); });

    constexpr utf8_segment_backend_t utf8_linebreaks_neon {"neon", sz_utf8_linebreaks_neon};
    run_test_case("check_utf8_linebreaks_unit_(utf8_linebreaks_neon)", neon,
                  [&] { check_utf8_linebreaks_unit_(utf8_linebreaks_neon); });
    run_test_case("check_utf8_linebreaks_rules_(utf8_linebreaks_neon)", neon,
                  [&] { check_utf8_linebreaks_rules_(utf8_linebreaks_neon); });
    run_test_case("check_utf8_linebreaks_safety_(utf8_linebreaks_neon)", neon, [&](split_mix_generator_t &generator) {
        check_utf8_linebreaks_safety_(generator, utf8_linebreaks_neon);
    });
    run_test_case(
        "check_utf8_linebreaks_equivalence_(utf8_linebreaks_neon)", neon,
        [&](split_mix_generator_t &generator) { check_utf8_linebreaks_equivalence_(generator, utf8_linebreaks_neon); });

    constexpr utf8_norm_kernels_t utf8_norm_neon {sz_utf8_norm_neon, sz_utf8_find_denormalized_neon};
    run_test_case("check_utf8_norm_unit_(utf8_norm_neon)", neon, [&] { check_utf8_norm_unit_(utf8_norm_neon); });
    run_test_case("check_utf8_norm_equivalence_(utf8_norm_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_equivalence_(generator, utf8_norm_neon); });
    run_test_case("check_utf8_norm_safety_(utf8_norm_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_safety_(generator, utf8_norm_neon); });

    constexpr utf8_uncased_kernels_t utf8_uncased_neon {sz_utf8_uncased_fold_neon, sz_utf8_uncased_search_neon,
                                                        sz_utf8_uncased_order_neon, sz_utf8_find_cased_neon};
    run_test_case("check_utf8_uncased_unit_(utf8_uncased_neon)", neon,
                  [&] { check_utf8_uncased_unit_(utf8_uncased_neon); });
    run_test_case("check_utf8_uncased_equivalence_(utf8_uncased_neon)", neon, [&](split_mix_generator_t &generator) {
        check_utf8_uncased_equivalence_(generator, utf8_uncased_neon);
    });
    run_test_case("check_utf8_uncased_safety_(utf8_uncased_neon)", neon,
                  [&](split_mix_generator_t &generator) { check_utf8_uncased_safety_(generator, utf8_uncased_neon); });
#endif // STRINGZILLA_TARGET_NEON

#if STRINGZILLA_TARGET_NEONAES
    capability_gate_t const neonaes(sz_cap_neonaes_k);

    constexpr hash_backend_t hash_neonaes {
        .hash_kernel = sz_hash_neonaes,
        .init_kernel = sz_hash_state_init_neonaes,
        .update_kernel = sz_hash_state_update_neonaes,
        .digest_kernel = sz_hash_state_digest_neonaes,
        .strings_kernel = sz_hash_strings_neonaes,
    };
    run_test_case("check_hash_equivalence_(hash_neonaes)", neonaes, [&](split_mix_generator_t &generator) {
        check_hash_equivalence_(generator, hash_neonaes);
        check_fill_random_equivalence_(sz_fill_random_neonaes);
    });
    run_test_case("check_hash_multiseed_equivalence_({sz_hash_multiseed_neonaes, sz_hash_neonaes})", neonaes,
                  [](split_mix_generator_t &generator) {
                      check_hash_multiseed_equivalence_(generator, {sz_hash_multiseed_neonaes, sz_hash_neonaes});
                  });

    constexpr ctr_backend_t ctr_neonaes {"neonaes", sz_aes256_key_init_neonaes, sz_aes256_ctr_xor_neonaes};
    constexpr gcm_backend_t gcm_neonaes {
        .name = "neonaes",
        .key_init = sz_aes256_gcm_key_init_neonaes,
        .encrypt = sz_aes256_gcm_encrypt_neonaes,
        .decrypt = sz_aes256_gcm_decrypt_neonaes,
        .sealer_init = sz_aes256_gcm_encryptor_init_neonaes,
        .sealer_associate = sz_aes256_gcm_encryptor_associate_neonaes,
        .sealer_update = sz_aes256_gcm_encryptor_update_neonaes,
        .sealer_digest = sz_aes256_gcm_encryptor_digest_neonaes,
        .opener_init = sz_aes256_gcm_decryptor_init_neonaes,
        .opener_associate = sz_aes256_gcm_decryptor_associate_neonaes,
        .opener_update = sz_aes256_gcm_decryptor_update_unverified_neonaes,
        .opener_verify = sz_aes256_gcm_decryptor_verify_neonaes,
    };
    run_test_case("check_cipher_unit_(ctr_neonaes, gcm_neonaes)", neonaes,
                  [&] { check_cipher_unit_(ctr_neonaes, gcm_neonaes); });
    run_test_case(
        "check_cipher_equivalence_(ctr_neonaes, gcm_neonaes)", neonaes,
        [&](split_mix_generator_t &generator) { check_cipher_equivalence_(generator, ctr_neonaes, gcm_neonaes); });

    run_test_case("check_intersect_unit_(sz_intersect_neonaes)", neonaes,
                  [] { check_intersect_unit_(sz_intersect_neonaes); });
#endif // STRINGZILLA_TARGET_NEONAES

#if STRINGZILLA_TARGET_NEONSHA
    capability_gate_t const neonsha(sz_cap_neonsha_k);

    constexpr sha256_backend_t sha256_neonsha {
        .init_kernel = sz_sha256_state_init_neonsha,
        .update_kernel = sz_sha256_state_update_neonsha,
        .digest_kernel = sz_sha256_state_digest_neonsha,
    };
    constexpr sha256_multistate_backend_t sha256_multistate_neonsha {
        .update_kernel = sz_sha256_multistate_update_neonsha,
        .digest_kernel = sz_sha256_multistate_digest_neonsha,
    };
    run_test_case("check_sha256_equivalence_(sha256_neonsha)", neonsha, [&](split_mix_generator_t &generator) {
        check_sha256_equivalence_(generator, sha256_neonsha);
        check_sha256_multistate_equivalence_(generator, sha256_multistate_neonsha);
    });
#endif // STRINGZILLA_TARGET_NEONSHA

#if STRINGZILLA_TARGET_SVE
    capability_gate_t const sve(sz_cap_sve_k);

    run_test_case("check_compare_unit_({sz_equal_sve, sz_order_sve})", sve,
                  [] { check_compare_unit_({"sve", sz_equal_sve, sz_order_sve}); });

    constexpr strings_backend_t strings_sve {"sve",
                                             sz_strings_to_tape_sve,
                                             sz_strings_to_rows_sve,
                                             sz_strings_escape_sve,
                                             sz_strings_unescape_sve,
                                             sz_strings_lookup_sve,
                                             sz_strings_normalize_sve};
    run_test_case("check_strings_unit_(strings_sve)", sve, [&] { check_strings_unit_(strings_sve); });
    run_test_case("check_strings_equivalence_(strings_sve)", sve,
                  [&](split_mix_generator_t &generator) { check_strings_equivalence_(generator, strings_sve); });
    run_test_case("check_strings_safety_(strings_sve)", sve, [&] { check_strings_safety_(strings_sve); });

    constexpr find_backend_t find_sve {
        .name = "sve",
        .find = sz_find_sve,
        .rfind = sz_rfind_sve,
        .find_byte = sz_find_byte_sve,
        .rfind_byte = sz_rfind_byte_sve,
    };
    run_test_case("check_find_unit_(find_sve)", sve, [&] { check_find_unit_(find_sve); });
    run_test_case("check_find_equivalence_(find_sve)", sve,
                  [&](split_mix_generator_t &generator) { check_find_equivalence_(generator, find_sve); });
    run_test_case("check_find_safety_(find_sve)", sve, [&] { check_find_safety_(find_sve); });

    run_test_case("check_bytesum_equivalence_(sz_bytesum_sve)", sve,
                  [](split_mix_generator_t &generator) { check_bytesum_equivalence_(generator, sz_bytesum_sve); });

    constexpr sort_backend_t sort_sve {"sve", sz_sort_order_sve, sz_sort_uncased_order_sve};
    run_test_case("check_sort_unit_(sort_sve)", sve, [&] { check_sort_unit_(sort_sve); });
    run_test_case("check_sort_equivalence_(sort_sve)", sve,
                  [&](split_mix_generator_t &generator) { check_sort_equivalence_(generator, sort_sve); });
    run_test_case("check_sort_safety_(sort_sve)", sve, [&] { check_sort_safety_(sort_sve); });

    constexpr utf8_norm_kernels_t utf8_norm_sve {sz_utf8_norm_sve, sz_utf8_find_denormalized_sve};
    run_test_case("check_utf8_norm_unit_(utf8_norm_sve)", sve, [&] { check_utf8_norm_unit_(utf8_norm_sve); });
    run_test_case("check_utf8_norm_equivalence_(utf8_norm_sve)", sve,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_equivalence_(generator, utf8_norm_sve); });
    run_test_case("check_utf8_norm_safety_(utf8_norm_sve)", sve,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_safety_(generator, utf8_norm_sve); });
#endif // STRINGZILLA_TARGET_SVE

#if STRINGZILLA_TARGET_SVE2
    capability_gate_t const sve2(sz_cap_sve2_k);

    constexpr find_backend_t find_sve2 {
        .name = "sve2",
        .find_byteset = sz_find_byteset_sve2,
        .rfind_byteset = sz_rfind_byteset_sve2,
    };
    run_test_case("check_find_unit_(find_sve2)", sve2, [&] { check_find_unit_(find_sve2); });
    run_test_case("check_find_equivalence_(find_sve2)", sve2,
                  [&](split_mix_generator_t &generator) { check_find_equivalence_(generator, find_sve2); });
    run_test_case("check_find_safety_(find_sve2)", sve2, [&] { check_find_safety_(find_sve2); });

    run_test_case("check_bytesum_equivalence_(sz_bytesum_sve2)", sve2,
                  [](split_mix_generator_t &generator) { check_bytesum_equivalence_(generator, sz_bytesum_sve2); });

    constexpr substrings_tier_t substrings_sve2 {
        .init = sz_substrings_engine_init_sve2,
        .counts = sz_substrings_counts_sve2,
        .find = sz_substrings_find_sve2,
        .replace = sz_substrings_replace_sve2,
        .bm25_scores = sz_substrings_bm25_scores_sve2,
    };
    run_test_case("check_substrings_unit_(substrings_sve2)", sve2, [&] { check_substrings_unit_(substrings_sve2); });
    run_test_case("check_substrings_equivalence_(substrings_sve2)", sve2,
                  [&](split_mix_generator_t &generator) { check_substrings_equivalence_(generator, substrings_sve2); });

    constexpr subdocuments_tier_t subdocuments_sve2 {
        .init = sz_subdocuments_engine_init_sve2,
        .counts = sz_subdocuments_counts_sve2,
        .find = sz_subdocuments_find_sve2,
    };
    run_test_case("check_subdocuments_unit_(subdocuments_sve2)", sve2,
                  [&] { check_subdocuments_unit_(subdocuments_sve2); });
    run_test_case("check_subdocuments_safety_(subdocuments_sve2)", sve2,
                  [&] { check_subdocuments_safety_(subdocuments_sve2); });
    run_test_case("check_subdocuments_equivalence_(subdocuments_sve2)", sve2, [&](split_mix_generator_t &generator) {
        check_subdocuments_equivalence_(generator, subdocuments_sve2);
    });

    constexpr utf8_runes_backend_t utf8_runes_sve2 {"sve2", sz_utf8_count_sve2, sz_utf8_seek_sve2, sz_utf8_decode_sve2};
    run_test_case("check_utf8_runes_unit_(utf8_runes_sve2)", sve2, [&] { check_utf8_runes_unit_(utf8_runes_sve2); });
    run_test_case("check_utf8_runes_safety_(utf8_runes_sve2)", sve2,
                  [&](split_mix_generator_t &generator) { check_utf8_runes_safety_(generator, utf8_runes_sve2); });
    run_test_case("check_utf8_runes_equivalence_(utf8_runes_sve2)", sve2,
                  [&](split_mix_generator_t &generator) { check_utf8_runes_equivalence_(generator, utf8_runes_sve2); });

    constexpr utf8_tokens_backend_t utf8_tokens_sve2 {"sve2", sz_utf8_count_sve2, sz_utf8_newlines_sve2,
                                                      sz_utf8_whitespaces_sve2};
    run_test_case("check_utf8_tokens_unit_(utf8_tokens_sve2)", sve2,
                  [&] { check_utf8_tokens_unit_(utf8_tokens_sve2); });
    run_test_case("check_utf8_tokens_safety_(utf8_tokens_sve2)", sve2,
                  [&](split_mix_generator_t &generator) { check_utf8_tokens_safety_(generator, utf8_tokens_sve2); });
    run_test_case("check_utf8_tokens_equivalence_(utf8_tokens_sve2)", sve2, [&](split_mix_generator_t &generator) {
        check_utf8_tokens_equivalence_(generator, utf8_tokens_sve2);
    });

    constexpr utf8_delimiters_backend_t utf8_delimiters_sve2 {"sve2", sz_utf8_delimiters_sve2};
    run_test_case("check_utf8_delimiters_unit_(utf8_delimiters_sve2)", sve2,
                  [&] { check_utf8_delimiters_unit_(utf8_delimiters_sve2); });
    run_test_case("check_utf8_delimiters_safety_(utf8_delimiters_sve2)", sve2, [&](split_mix_generator_t &generator) {
        check_utf8_delimiters_safety_(generator, utf8_delimiters_sve2);
    });
    run_test_case(
        "check_utf8_delimiters_equivalence_(utf8_delimiters_sve2)", sve2,
        [&](split_mix_generator_t &generator) { check_utf8_delimiters_equivalence_(generator, utf8_delimiters_sve2); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_sve2 {"sve2", sz_utf8_wordbreaks_sve2};
    run_test_case("check_utf8_wordbreaks_unit_(utf8_wordbreaks_sve2)", sve2,
                  [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_sve2); });
    run_test_case("check_utf8_wordbreaks_rules_(utf8_wordbreaks_sve2)", sve2,
                  [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_sve2); });
    run_test_case("check_utf8_wordbreaks_safety_(utf8_wordbreaks_sve2)", sve2, [&](split_mix_generator_t &generator) {
        check_utf8_wordbreaks_safety_(generator, utf8_wordbreaks_sve2);
    });
    run_test_case(
        "check_utf8_wordbreaks_equivalence_(utf8_wordbreaks_sve2)", sve2,
        [&](split_mix_generator_t &generator) { check_utf8_wordbreaks_equivalence_(generator, utf8_wordbreaks_sve2); });

    constexpr utf8_segment_backend_t utf8_graphemes_sve2 {"sve2", sz_utf8_graphemes_sve2};
    run_test_case("check_utf8_graphemes_unit_(utf8_graphemes_sve2)", sve2,
                  [&] { check_utf8_graphemes_unit_(utf8_graphemes_sve2); });
    run_test_case("check_utf8_graphemes_rules_(utf8_graphemes_sve2)", sve2,
                  [&] { check_utf8_graphemes_rules_(utf8_graphemes_sve2); });
    run_test_case("check_utf8_graphemes_safety_(utf8_graphemes_sve2)", sve2, [&](split_mix_generator_t &generator) {
        check_utf8_graphemes_safety_(generator, utf8_graphemes_sve2);
    });
    run_test_case(
        "check_utf8_graphemes_equivalence_(utf8_graphemes_sve2)", sve2,
        [&](split_mix_generator_t &generator) { check_utf8_graphemes_equivalence_(generator, utf8_graphemes_sve2); });

    constexpr utf8_segment_backend_t utf8_sentences_sve2 {"sve2", sz_utf8_sentences_sve2};
    run_test_case("check_utf8_sentences_unit_(utf8_sentences_sve2)", sve2,
                  [&] { check_utf8_sentences_unit_(utf8_sentences_sve2); });
    run_test_case("check_utf8_sentences_rules_(utf8_sentences_sve2)", sve2,
                  [&] { check_utf8_sentences_rules_(utf8_sentences_sve2); });
    run_test_case("check_utf8_sentences_safety_(utf8_sentences_sve2)", sve2, [&](split_mix_generator_t &generator) {
        check_utf8_sentences_safety_(generator, utf8_sentences_sve2);
    });
    run_test_case(
        "check_utf8_sentences_equivalence_(utf8_sentences_sve2)", sve2,
        [&](split_mix_generator_t &generator) { check_utf8_sentences_equivalence_(generator, utf8_sentences_sve2); });

    constexpr utf8_segment_backend_t utf8_linebreaks_sve2 {"sve2", sz_utf8_linebreaks_sve2};
    run_test_case("check_utf8_linebreaks_unit_(utf8_linebreaks_sve2)", sve2,
                  [&] { check_utf8_linebreaks_unit_(utf8_linebreaks_sve2); });
    run_test_case("check_utf8_linebreaks_rules_(utf8_linebreaks_sve2)", sve2,
                  [&] { check_utf8_linebreaks_rules_(utf8_linebreaks_sve2); });
    run_test_case("check_utf8_linebreaks_safety_(utf8_linebreaks_sve2)", sve2, [&](split_mix_generator_t &generator) {
        check_utf8_linebreaks_safety_(generator, utf8_linebreaks_sve2);
    });
    run_test_case(
        "check_utf8_linebreaks_equivalence_(utf8_linebreaks_sve2)", sve2,
        [&](split_mix_generator_t &generator) { check_utf8_linebreaks_equivalence_(generator, utf8_linebreaks_sve2); });

    constexpr utf8_norm_kernels_t utf8_norm_sve2 {sz_utf8_norm_sve2, sz_utf8_find_denormalized_sve2};
    run_test_case("check_utf8_norm_unit_(utf8_norm_sve2)", sve2, [&] { check_utf8_norm_unit_(utf8_norm_sve2); });
    run_test_case("check_utf8_norm_equivalence_(utf8_norm_sve2)", sve2,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_equivalence_(generator, utf8_norm_sve2); });
    run_test_case("check_utf8_norm_safety_(utf8_norm_sve2)", sve2,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_safety_(generator, utf8_norm_sve2); });

    constexpr utf8_uncased_kernels_t utf8_uncased_sve2 {sz_utf8_uncased_fold_sve2, sz_utf8_uncased_search_sve2,
                                                        sz_utf8_uncased_order_sve2, sz_utf8_find_cased_sve2};
    run_test_case("check_utf8_uncased_unit_(utf8_uncased_sve2)", sve2,
                  [&] { check_utf8_uncased_unit_(utf8_uncased_sve2); });
    run_test_case("check_utf8_uncased_equivalence_(utf8_uncased_sve2)", sve2, [&](split_mix_generator_t &generator) {
        check_utf8_uncased_equivalence_(generator, utf8_uncased_sve2);
    });
    run_test_case("check_utf8_uncased_safety_(utf8_uncased_sve2)", sve2,
                  [&](split_mix_generator_t &generator) { check_utf8_uncased_safety_(generator, utf8_uncased_sve2); });
#endif // STRINGZILLA_TARGET_SVE2

#if STRINGZILLA_TARGET_SVE2AES
    capability_gate_t const sve2aes(sz_cap_sve2aes_k);

    constexpr hash_backend_t hash_sve2aes {
        .hash_kernel = sz_hash_sve2aes,
        .init_kernel = sz_hash_state_init_sve2aes,
        .update_kernel = sz_hash_state_update_sve2aes,
        .digest_kernel = sz_hash_state_digest_sve2aes,
    };
    run_test_case("check_hash_equivalence_(hash_sve2aes)", sve2aes, [&](split_mix_generator_t &generator) {
        check_hash_equivalence_(generator, hash_sve2aes);
        check_fill_random_equivalence_(sz_fill_random_sve2aes);
    });

    constexpr ctr_backend_t ctr_sve2aes {"sve2aes", sz_aes256_key_init_sve2aes, sz_aes256_ctr_xor_sve2aes};
    constexpr gcm_backend_t gcm_sve2aes {
        .name = "sve2aes",
        .key_init = sz_aes256_gcm_key_init_sve2aes,
        .encrypt = sz_aes256_gcm_encrypt_sve2aes,
        .decrypt = sz_aes256_gcm_decrypt_sve2aes,
        .sealer_init = sz_aes256_gcm_encryptor_init_sve2aes,
        .sealer_associate = sz_aes256_gcm_encryptor_associate_sve2aes,
        .sealer_update = sz_aes256_gcm_encryptor_update_sve2aes,
        .sealer_digest = sz_aes256_gcm_encryptor_digest_sve2aes,
        .opener_init = sz_aes256_gcm_decryptor_init_sve2aes,
        .opener_associate = sz_aes256_gcm_decryptor_associate_sve2aes,
        .opener_update = sz_aes256_gcm_decryptor_update_unverified_sve2aes,
        .opener_verify = sz_aes256_gcm_decryptor_verify_sve2aes,
    };
    run_test_case("check_cipher_unit_(ctr_sve2aes, gcm_sve2aes)", sve2aes,
                  [&] { check_cipher_unit_(ctr_sve2aes, gcm_sve2aes); });
    run_test_case(
        "check_cipher_equivalence_(ctr_sve2aes, gcm_sve2aes)", sve2aes,
        [&](split_mix_generator_t &generator) { check_cipher_equivalence_(generator, ctr_sve2aes, gcm_sve2aes); });
#endif // STRINGZILLA_TARGET_SVE2AES
}

} // namespace ashvardanian::stringzilla::test
