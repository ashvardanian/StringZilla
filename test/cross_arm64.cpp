/**
 *  @file test/cross_arm64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - Arm64 family: NEON, NEON AES, NEON SHA, SVE, SVE2, SVE2 AES.
 */
#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

std::size_t test_cross_arm64(environment_t const &env) {
    [[maybe_unused]] cross_section_t check(env);

#if STRINGZILLA_TARGET_NEON
    check.section("Cross NEON", sz_cap_neon_k);

    check("test_compare_unit_neon", [] { check_compare_unit_({"neon", sz_equal_neon, sz_order_neon}); });

    constexpr memory_backend_t memory_neon {"neon", sz_copy_neon, sz_move_neon, sz_fill_neon};
    check("test_memory_unit_neon", [&] { check_memory_unit_(memory_neon); });
    check("test_memory_equivalence_neon",
          [&](test_context_t &context) { check_memory_equivalence_(context, memory_neon); });
    check("test_memory_safety_neon", [&] { check_memory_safety_(memory_neon); });

    constexpr lookup_backend_t lookup_neon {"neon", sz_lookup_neon};
    check("test_lookup_unit_neon", [&] { check_lookup_unit_(lookup_neon); });
    check("test_lookup_equivalence_neon",
          [&](test_context_t &context) { check_lookup_equivalence_(context, lookup_neon); });
    check("test_lookup_safety_neon", [&] { check_lookup_safety_(lookup_neon); });

    constexpr find_backend_t find_neon {
        .name = "neon",
        .find = sz_find_neon,
        .rfind = sz_rfind_neon,
        .find_byte = sz_find_byte_neon,
        .rfind_byte = sz_rfind_byte_neon,
        .find_byteset = sz_find_byteset_neon,
        .rfind_byteset = sz_rfind_byteset_neon,
    };
    check("test_find_unit_neon", [&] { check_find_unit_(find_neon); });
    check("test_find_equivalence_neon", [&](test_context_t &context) { check_find_equivalence_(context, find_neon); });
    check("test_find_safety_neon", [&] { check_find_safety_(find_neon); });

    check("test_hash_equivalence_neon",
          [](test_context_t &context) { check_bytesum_equivalence_(context, sz_bytesum_neon); });

    constexpr sort_backend_t sort_neon {"neon", sz_sequence_argsort_neon, sz_sequence_argsort_uncased_neon};
    check("test_sort_unit_neon", [&] { check_sort_unit_(sort_neon); });
    check("test_sort_equivalence_neon", [&](test_context_t &context) { check_sort_equivalence_(context, sort_neon); });
    check("test_sort_safety_neon", [&] { check_sort_safety_(sort_neon); });

    constexpr levenshtein_backend_t levenshtein_neon {"neon", sz_levenshtein_engine_init_neon,
                                                      sz_levenshtein_distances_neon};
    check("test_levenshtein_unit_neon", [&] { check_levenshtein_unit_(levenshtein_neon); });
    check("test_levenshtein_equivalence_neon",
          [&](test_context_t &context) { check_levenshtein_equivalence_(context, levenshtein_neon); });
    check("test_levenshtein_safety_neon", [&] { check_levenshtein_safety_(levenshtein_neon); });

    constexpr overlap_backend_t overlap_neon {"neon", sz_overlap_engine_init_neon, sz_overlap_scores_neon};
    check("test_overlap_equivalence_neon",
          [&](test_context_t &context) { check_overlap_equivalence_(context, overlap_neon); });
    check("test_overlap_safety_neon", [&] { check_overlap_safety_(overlap_neon); });

    constexpr substrings_tier_t substrings_neon {
        .init = sz_substrings_engine_init_neon,
        .counts = sz_substrings_counts_neon,
        .find = sz_substrings_find_neon,
        .replace = sz_substrings_replace_neon,
        .bm25_scores = sz_substrings_bm25_scores_neon,
    };
    check("test_substrings_unit_neon", [&] { check_substrings_unit_(substrings_neon); });
    check("test_substrings_equivalence_neon",
          [&](test_context_t &context) { check_substrings_equivalence_(context, substrings_neon); });

    constexpr utf8_runes_backend_t utf8_runes_neon {"neon", sz_utf8_count_neon, sz_utf8_seek_neon, sz_utf8_decode_neon};
    check("test_utf8_runes_unit_neon", [&] { check_utf8_runes_unit_(utf8_runes_neon); });
    check("test_utf8_runes_safety_neon",
          [&](test_context_t &context) { check_utf8_runes_safety_(context, utf8_runes_neon); });
    check("test_utf8_runes_equivalence_neon",
          [&](test_context_t &context) { check_utf8_runes_equivalence_(context, utf8_runes_neon); });

    constexpr utf8_tokens_backend_t utf8_tokens_neon {"neon", sz_utf8_count_neon, sz_utf8_newlines_neon,
                                                      sz_utf8_whitespaces_neon};
    check("test_utf8_tokens_unit_neon", [&] { check_utf8_tokens_unit_(utf8_tokens_neon); });
    check("test_utf8_tokens_safety_neon",
          [&](test_context_t &context) { check_utf8_tokens_safety_(context, utf8_tokens_neon); });
    check("test_utf8_tokens_equivalence_neon",
          [&](test_context_t &context) { check_utf8_tokens_equivalence_(context, utf8_tokens_neon); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_neon {"neon", sz_utf8_delimiters_neon};
    check("test_utf8_delimiters_unit_neon", [&] { check_utf8_delimiters_unit_(utf8_delimiters_neon); });
    check("test_utf8_delimiters_safety_neon",
          [&](test_context_t &context) { check_utf8_delimiters_safety_(context, utf8_delimiters_neon); });
    check("test_utf8_delimiters_equivalence_neon",
          [&](test_context_t &context) { check_utf8_delimiters_equivalence_(context, utf8_delimiters_neon); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_neon {"neon", sz_utf8_wordbreaks_neon};
    check("test_utf8_wordbreaks_unit_neon", [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_neon); });
    check("test_utf8_wordbreaks_rules_neon", [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_neon); });
    check("test_utf8_wordbreaks_safety_neon",
          [&](test_context_t &context) { check_utf8_wordbreaks_safety_(context, utf8_wordbreaks_neon); });
    check("test_utf8_wordbreaks_equivalence_neon",
          [&](test_context_t &context) { check_utf8_wordbreaks_equivalence_(context, utf8_wordbreaks_neon); });

    constexpr utf8_segment_backend_t utf8_graphemes_neon {"neon", sz_utf8_graphemes_neon};
    check("test_utf8_graphemes_unit_neon", [&] { check_utf8_graphemes_unit_(utf8_graphemes_neon); });
    check("test_utf8_graphemes_rules_neon", [&] { check_utf8_graphemes_rules_(utf8_graphemes_neon); });
    check("test_utf8_graphemes_safety_neon",
          [&](test_context_t &context) { check_utf8_graphemes_safety_(context, utf8_graphemes_neon); });
    check("test_utf8_graphemes_equivalence_neon",
          [&](test_context_t &context) { check_utf8_graphemes_equivalence_(context, utf8_graphemes_neon); });

    constexpr utf8_segment_backend_t utf8_sentences_neon {"neon", sz_utf8_sentences_neon};
    check("test_utf8_sentences_unit_neon", [&] { check_utf8_sentences_unit_(utf8_sentences_neon); });
    check("test_utf8_sentences_rules_neon", [&] { check_utf8_sentences_rules_(utf8_sentences_neon); });
    check("test_utf8_sentences_safety_neon",
          [&](test_context_t &context) { check_utf8_sentences_safety_(context, utf8_sentences_neon); });
    check("test_utf8_sentences_equivalence_neon",
          [&](test_context_t &context) { check_utf8_sentences_equivalence_(context, utf8_sentences_neon); });

    constexpr utf8_segment_backend_t utf8_linebreaks_neon {"neon", sz_utf8_linebreaks_neon};
    check("test_utf8_linebreaks_unit_neon", [&] { check_utf8_linebreaks_unit_(utf8_linebreaks_neon); });
    check("test_utf8_linebreaks_rules_neon", [&] { check_utf8_linebreaks_rules_(utf8_linebreaks_neon); });
    check("test_utf8_linebreaks_safety_neon",
          [&](test_context_t &context) { check_utf8_linebreaks_safety_(context, utf8_linebreaks_neon); });
    check("test_utf8_linebreaks_equivalence_neon",
          [&](test_context_t &context) { check_utf8_linebreaks_equivalence_(context, utf8_linebreaks_neon); });

    constexpr utf8_norm_kernels_t utf8_norm_neon {sz_utf8_norm_neon, sz_utf8_find_denormalized_neon};
    check("test_utf8_norm_unit_neon", [&] { check_utf8_norm_unit_(utf8_norm_neon); });
    check("test_utf8_norm_equivalence_neon",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_neon); });
    check("test_utf8_norm_safety_neon",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_neon); });

    constexpr utf8_uncased_kernels_t utf8_uncased_neon {sz_utf8_uncased_fold_neon, sz_utf8_uncased_search_neon,
                                                        sz_utf8_uncased_order_neon, sz_utf8_find_cased_neon};
    check("test_utf8_uncased_unit_neon", [&] { check_utf8_uncased_unit_(utf8_uncased_neon); });
    check("test_utf8_uncased_equivalence_neon",
          [&](test_context_t &context) { check_utf8_uncased_equivalence_(context, utf8_uncased_neon); });
    check("test_utf8_uncased_safety_neon",
          [&](test_context_t &context) { check_utf8_uncased_safety_(context, utf8_uncased_neon); });
#endif // STRINGZILLA_TARGET_NEON

#if STRINGZILLA_TARGET_NEONAES
    check.section("Cross NEON AES", sz_cap_neonaes_k);

    constexpr hash_backend_t hash_neonaes {
        .hash_kernel = sz_hash_neonaes,
        .init_kernel = sz_hash_state_init_neonaes,
        .update_kernel = sz_hash_state_update_neonaes,
        .digest_kernel = sz_hash_state_digest_neonaes,
    };
    check("test_hash_equivalence_neonaes", [&](test_context_t &context) {
        check_hash_equivalence_(context, hash_neonaes);
        check_fill_random_equivalence_(context, sz_fill_random_neonaes);
    });
    check("test_hash_multiseed_equivalence_neonaes", [](test_context_t &context) {
        check_hash_multiseed_equivalence_(context, {sz_hash_multiseed_neonaes, sz_hash_neonaes});
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
    check("test_cipher_unit_neonaes", [&] { check_cipher_unit_(ctr_neonaes, gcm_neonaes); });
    check("test_cipher_equivalence_neonaes",
          [&](test_context_t &context) { check_cipher_equivalence_(context, ctr_neonaes, gcm_neonaes); });

    check("test_intersect_unit_neonaes", [] { check_intersect_unit_(sz_sequence_intersect_neonaes); });
#endif // STRINGZILLA_TARGET_NEONAES

#if STRINGZILLA_TARGET_NEONSHA
    check.section("Cross NEON SHA", sz_cap_neonsha_k);

    constexpr sha256_backend_t sha256_neonsha {
        .init_kernel = sz_sha256_state_init_neonsha,
        .update_kernel = sz_sha256_state_update_neonsha,
        .digest_kernel = sz_sha256_state_digest_neonsha,
    };
    check("test_hash_equivalence_neonsha",
          [&](test_context_t &context) { check_sha256_equivalence_(context, sha256_neonsha); });
#endif // STRINGZILLA_TARGET_NEONSHA

#if STRINGZILLA_TARGET_SVE
    check.section("Cross SVE", sz_cap_sve_k);

    check("test_compare_unit_sve", [] { check_compare_unit_({"sve", sz_equal_sve, sz_order_sve}); });

    constexpr memory_backend_t memory_sve {"sve", sz_copy_sve, sz_move_sve, sz_fill_sve};
    check("test_memory_unit_sve", [&] { check_memory_unit_(memory_sve); });
    check("test_memory_equivalence_sve",
          [&](test_context_t &context) { check_memory_equivalence_(context, memory_sve); });
    check("test_memory_safety_sve", [&] { check_memory_safety_(memory_sve); });

    constexpr lookup_backend_t lookup_sve {"sve", sz_lookup_sve};
    check("test_lookup_unit_sve", [&] { check_lookup_unit_(lookup_sve); });
    check("test_lookup_equivalence_sve",
          [&](test_context_t &context) { check_lookup_equivalence_(context, lookup_sve); });
    check("test_lookup_safety_sve", [&] { check_lookup_safety_(lookup_sve); });

    constexpr find_backend_t find_sve {
        .name = "sve",
        .find = sz_find_sve,
        .rfind = sz_rfind_sve,
        .find_byte = sz_find_byte_sve,
        .rfind_byte = sz_rfind_byte_sve,
    };
    check("test_find_unit_sve", [&] { check_find_unit_(find_sve); });
    check("test_find_equivalence_sve", [&](test_context_t &context) { check_find_equivalence_(context, find_sve); });
    check("test_find_safety_sve", [&] { check_find_safety_(find_sve); });

    check("test_hash_equivalence_sve",
          [](test_context_t &context) { check_bytesum_equivalence_(context, sz_bytesum_sve); });

    constexpr sort_backend_t sort_sve {"sve", sz_sequence_argsort_sve, sz_sequence_argsort_uncased_sve};
    check("test_sort_unit_sve", [&] { check_sort_unit_(sort_sve); });
    check("test_sort_equivalence_sve", [&](test_context_t &context) { check_sort_equivalence_(context, sort_sve); });
    check("test_sort_safety_sve", [&] { check_sort_safety_(sort_sve); });

    constexpr utf8_norm_kernels_t utf8_norm_sve {sz_utf8_norm_sve, sz_utf8_find_denormalized_sve};
    check("test_utf8_norm_unit_sve", [&] { check_utf8_norm_unit_(utf8_norm_sve); });
    check("test_utf8_norm_equivalence_sve",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_sve); });
    check("test_utf8_norm_safety_sve",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_sve); });
#endif // STRINGZILLA_TARGET_SVE

#if STRINGZILLA_TARGET_SVE2
    check.section("Cross SVE2", sz_cap_sve2_k);

    constexpr find_backend_t find_sve2 {
        .name = "sve2",
        .find_byteset = sz_find_byteset_sve2,
        .rfind_byteset = sz_rfind_byteset_sve2,
    };
    check("test_find_unit_sve2", [&] { check_find_unit_(find_sve2); });
    check("test_find_equivalence_sve2", [&](test_context_t &context) { check_find_equivalence_(context, find_sve2); });
    check("test_find_safety_sve2", [&] { check_find_safety_(find_sve2); });

    check("test_hash_equivalence_sve2",
          [](test_context_t &context) { check_bytesum_equivalence_(context, sz_bytesum_sve2); });

    constexpr utf8_runes_backend_t utf8_runes_sve2 {"sve2", sz_utf8_count_sve2, sz_utf8_seek_sve2, sz_utf8_decode_sve2};
    check("test_utf8_runes_unit_sve2", [&] { check_utf8_runes_unit_(utf8_runes_sve2); });
    check("test_utf8_runes_safety_sve2",
          [&](test_context_t &context) { check_utf8_runes_safety_(context, utf8_runes_sve2); });
    check("test_utf8_runes_equivalence_sve2",
          [&](test_context_t &context) { check_utf8_runes_equivalence_(context, utf8_runes_sve2); });

    constexpr utf8_tokens_backend_t utf8_tokens_sve2 {"sve2", sz_utf8_count_sve2, sz_utf8_newlines_sve2,
                                                      sz_utf8_whitespaces_sve2};
    check("test_utf8_tokens_unit_sve2", [&] { check_utf8_tokens_unit_(utf8_tokens_sve2); });
    check("test_utf8_tokens_safety_sve2",
          [&](test_context_t &context) { check_utf8_tokens_safety_(context, utf8_tokens_sve2); });
    check("test_utf8_tokens_equivalence_sve2",
          [&](test_context_t &context) { check_utf8_tokens_equivalence_(context, utf8_tokens_sve2); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_sve2 {"sve2", sz_utf8_delimiters_sve2};
    check("test_utf8_delimiters_unit_sve2", [&] { check_utf8_delimiters_unit_(utf8_delimiters_sve2); });
    check("test_utf8_delimiters_safety_sve2",
          [&](test_context_t &context) { check_utf8_delimiters_safety_(context, utf8_delimiters_sve2); });
    check("test_utf8_delimiters_equivalence_sve2",
          [&](test_context_t &context) { check_utf8_delimiters_equivalence_(context, utf8_delimiters_sve2); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_sve2 {"sve2", sz_utf8_wordbreaks_sve2};
    check("test_utf8_wordbreaks_unit_sve2", [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_sve2); });
    check("test_utf8_wordbreaks_rules_sve2", [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_sve2); });
    check("test_utf8_wordbreaks_safety_sve2",
          [&](test_context_t &context) { check_utf8_wordbreaks_safety_(context, utf8_wordbreaks_sve2); });
    check("test_utf8_wordbreaks_equivalence_sve2",
          [&](test_context_t &context) { check_utf8_wordbreaks_equivalence_(context, utf8_wordbreaks_sve2); });

    constexpr utf8_segment_backend_t utf8_graphemes_sve2 {"sve2", sz_utf8_graphemes_sve2};
    check("test_utf8_graphemes_unit_sve2", [&] { check_utf8_graphemes_unit_(utf8_graphemes_sve2); });
    check("test_utf8_graphemes_rules_sve2", [&] { check_utf8_graphemes_rules_(utf8_graphemes_sve2); });
    check("test_utf8_graphemes_safety_sve2",
          [&](test_context_t &context) { check_utf8_graphemes_safety_(context, utf8_graphemes_sve2); });
    check("test_utf8_graphemes_equivalence_sve2",
          [&](test_context_t &context) { check_utf8_graphemes_equivalence_(context, utf8_graphemes_sve2); });

    constexpr utf8_segment_backend_t utf8_sentences_sve2 {"sve2", sz_utf8_sentences_sve2};
    check("test_utf8_sentences_unit_sve2", [&] { check_utf8_sentences_unit_(utf8_sentences_sve2); });
    check("test_utf8_sentences_rules_sve2", [&] { check_utf8_sentences_rules_(utf8_sentences_sve2); });
    check("test_utf8_sentences_safety_sve2",
          [&](test_context_t &context) { check_utf8_sentences_safety_(context, utf8_sentences_sve2); });
    check("test_utf8_sentences_equivalence_sve2",
          [&](test_context_t &context) { check_utf8_sentences_equivalence_(context, utf8_sentences_sve2); });

    constexpr utf8_segment_backend_t utf8_linebreaks_sve2 {"sve2", sz_utf8_linebreaks_sve2};
    check("test_utf8_linebreaks_unit_sve2", [&] { check_utf8_linebreaks_unit_(utf8_linebreaks_sve2); });
    check("test_utf8_linebreaks_rules_sve2", [&] { check_utf8_linebreaks_rules_(utf8_linebreaks_sve2); });
    check("test_utf8_linebreaks_safety_sve2",
          [&](test_context_t &context) { check_utf8_linebreaks_safety_(context, utf8_linebreaks_sve2); });
    check("test_utf8_linebreaks_equivalence_sve2",
          [&](test_context_t &context) { check_utf8_linebreaks_equivalence_(context, utf8_linebreaks_sve2); });

    constexpr utf8_norm_kernels_t utf8_norm_sve2 {sz_utf8_norm_sve2, sz_utf8_find_denormalized_sve2};
    check("test_utf8_norm_unit_sve2", [&] { check_utf8_norm_unit_(utf8_norm_sve2); });
    check("test_utf8_norm_equivalence_sve2",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_sve2); });
    check("test_utf8_norm_safety_sve2",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_sve2); });

    constexpr utf8_uncased_kernels_t utf8_uncased_sve2 {sz_utf8_uncased_fold_sve2, sz_utf8_uncased_search_sve2,
                                                        sz_utf8_uncased_order_sve2, sz_utf8_find_cased_sve2};
    check("test_utf8_uncased_unit_sve2", [&] { check_utf8_uncased_unit_(utf8_uncased_sve2); });
    check("test_utf8_uncased_equivalence_sve2",
          [&](test_context_t &context) { check_utf8_uncased_equivalence_(context, utf8_uncased_sve2); });
    check("test_utf8_uncased_safety_sve2",
          [&](test_context_t &context) { check_utf8_uncased_safety_(context, utf8_uncased_sve2); });
#endif // STRINGZILLA_TARGET_SVE2

#if STRINGZILLA_TARGET_SVE2AES
    check.section("Cross SVE2 AES", sz_cap_sve2aes_k);

    constexpr hash_backend_t hash_sve2aes {
        .hash_kernel = sz_hash_sve2aes,
        .init_kernel = sz_hash_state_init_sve2aes,
        .update_kernel = sz_hash_state_update_sve2aes,
        .digest_kernel = sz_hash_state_digest_sve2aes,
    };
    check("test_hash_equivalence_sve2aes", [&](test_context_t &context) {
        check_hash_equivalence_(context, hash_sve2aes);
        check_fill_random_equivalence_(context, sz_fill_random_sve2aes);
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
    check("test_cipher_unit_sve2aes", [&] { check_cipher_unit_(ctr_sve2aes, gcm_sve2aes); });
    check("test_cipher_equivalence_sve2aes",
          [&](test_context_t &context) { check_cipher_equivalence_(context, ctr_sve2aes, gcm_sve2aes); });
#endif // STRINGZILLA_TARGET_SVE2AES

    return check.failures;
}

} // namespace ashvardanian::stringzilla::test
