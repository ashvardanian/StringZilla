/**
 *  @file test/cross_x8664.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - x86-64 family: Westmere, Goldmont, Haswell, Skylake, Ice Lake.
 */
#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

std::size_t test_cross_x8664(environment_t const &env) {
    [[maybe_unused]] cross_section_t check(env);

#if STRINGZILLA_TARGET_WESTMERE
    check.section("Cross Westmere", sz_cap_westmere_k);

    check("test_compare_unit_westmere",
          [] { check_compare_unit_({"westmere", sz_equal_westmere, sz_order_westmere}); });

    constexpr find_backend_t find_westmere {
        .name = "westmere",
        .find = sz_find_westmere,
        .rfind = sz_rfind_westmere,
        .find_byte = sz_find_byte_westmere,
        .rfind_byte = sz_rfind_byte_westmere,
    };
    check("test_find_unit_westmere", [&] { check_find_unit_(find_westmere); });
    check("test_find_equivalence_westmere",
          [&](test_context_t &context) { check_find_equivalence_(context, find_westmere); });
    check("test_find_safety_westmere", [&] { check_find_safety_(find_westmere); });

    constexpr hash_backend_t hash_westmere {
        .hash_kernel = sz_hash_westmere,
        .init_kernel = sz_hash_state_init_westmere,
        .update_kernel = sz_hash_state_update_westmere,
        .digest_kernel = sz_hash_state_digest_westmere,
    };
    check("test_hash_equivalence_westmere", [&](test_context_t &context) {
        check_hash_equivalence_(context, hash_westmere);
        check_fill_random_equivalence_(context, sz_fill_random_westmere);
    });
    check("test_hash_multiseed_equivalence_westmere", [](test_context_t &context) {
        check_hash_multiseed_equivalence_(context, {sz_hash_multiseed_westmere, sz_hash_westmere});
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
    check("test_cipher_unit_westmere", [&] { check_cipher_unit_(ctr_westmere, gcm_westmere); });
    check("test_cipher_equivalence_westmere",
          [&](test_context_t &context) { check_cipher_equivalence_(context, ctr_westmere, gcm_westmere); });

    check("test_intersect_unit_westmere", [] { check_intersect_unit_(sz_sequence_intersect_westmere); });
#endif // STRINGZILLA_TARGET_WESTMERE

#if STRINGZILLA_TARGET_GOLDMONT
    check.section("Cross Goldmont", sz_cap_goldmont_k);

    constexpr sha256_backend_t sha256_goldmont {
        .init_kernel = sz_sha256_state_init_goldmont,
        .update_kernel = sz_sha256_state_update_goldmont,
        .digest_kernel = sz_sha256_state_digest_goldmont,
    };
    constexpr sha256_multistate_backend_t sha256_multistate_goldmont {
        .update_kernel = sz_sha256_multistate_update_goldmont,
        .digest_kernel = sz_sha256_multistate_digest_goldmont,
    };
    check("test_hash_unit_goldmont", [&] { check_sha256_unit_(sha256_goldmont); });
    check("test_hash_equivalence_goldmont", [&](test_context_t &context) {
        check_sha256_equivalence_(context, sha256_goldmont);
        check_sha256_multistate_equivalence_(context, sha256_multistate_goldmont);
    });
#endif // STRINGZILLA_TARGET_GOLDMONT

#if STRINGZILLA_TARGET_HASWELL
    check.section("Cross Haswell", sz_cap_haswell_k);

    check("test_compare_unit_haswell", [] { check_compare_unit_({"haswell", sz_equal_haswell, sz_order_haswell}); });

    constexpr memory_backend_t memory_haswell {"haswell", sz_copy_haswell, sz_move_haswell, sz_fill_haswell};
    check("test_memory_unit_haswell", [&] { check_memory_unit_(memory_haswell); });
    check("test_memory_equivalence_haswell",
          [&](test_context_t &context) { check_memory_equivalence_(context, memory_haswell); });
    check("test_memory_safety_haswell", [&] { check_memory_safety_(memory_haswell); });

    constexpr lookup_backend_t lookup_haswell {"haswell", sz_lookup_haswell};
    check("test_lookup_unit_haswell", [&] { check_lookup_unit_(lookup_haswell); });
    check("test_lookup_equivalence_haswell",
          [&](test_context_t &context) { check_lookup_equivalence_(context, lookup_haswell); });
    check("test_lookup_safety_haswell", [&] { check_lookup_safety_(lookup_haswell); });

    constexpr find_backend_t find_haswell {
        .name = "haswell",
        .find = sz_find_haswell,
        .rfind = sz_rfind_haswell,
        .find_byte = sz_find_byte_haswell,
        .rfind_byte = sz_rfind_byte_haswell,
        .find_byteset = sz_find_byteset_haswell,
        .rfind_byteset = sz_rfind_byteset_haswell,
    };
    check("test_find_unit_haswell", [&] { check_find_unit_(find_haswell); });
    check("test_find_equivalence_haswell",
          [&](test_context_t &context) { check_find_equivalence_(context, find_haswell); });
    check("test_find_safety_haswell", [&] { check_find_safety_(find_haswell); });

    constexpr sha256_multistate_backend_t sha256_multistate_haswell {
        .update_kernel = sz_sha256_multistate_update_haswell,
        .digest_kernel = sz_sha256_multistate_digest_haswell,
    };
    check("test_hash_equivalence_haswell", [&](test_context_t &context) {
        check_bytesum_equivalence_(context, sz_bytesum_haswell);
        check_sha256_multistate_equivalence_(context, sha256_multistate_haswell);
    });

    constexpr sort_backend_t sort_haswell {"haswell", sz_sequence_argsort_haswell, sz_sequence_argsort_uncased_haswell};
    check("test_sort_unit_haswell", [&] { check_sort_unit_(sort_haswell); });
    check("test_sort_equivalence_haswell",
          [&](test_context_t &context) { check_sort_equivalence_(context, sort_haswell); });
    check("test_sort_safety_haswell", [&] { check_sort_safety_(sort_haswell); });

    constexpr levenshtein_backend_t levenshtein_haswell {"haswell", sz_levenshtein_engine_init_haswell,
                                                         sz_levenshtein_distances_haswell};
    check("test_levenshtein_unit_haswell", [&] { check_levenshtein_unit_(levenshtein_haswell); });
    check("test_levenshtein_equivalence_haswell",
          [&](test_context_t &context) { check_levenshtein_equivalence_(context, levenshtein_haswell); });
    check("test_levenshtein_safety_haswell", [&] { check_levenshtein_safety_(levenshtein_haswell); });

#if STRINGZILLA_HEADER_ONLY // The step verbs are inline helpers, compiled only with the capability headers
    constexpr overlap_step_backend_t overlap_steps_haswell {
        .name = "haswell",
        .positions_per_step = sz_overlap_f64x4_positions_per_step_haswell_k,
        .prefix_hash_step = sz_overlap_f64x4_prefix_hash_step_haswell,
        .prefix_hash_step_tail = sz_overlap_f64x4_prefix_hash_step_tail_haswell,
        .window_hash_step = sz_overlap_f64x4_window_hash_step_haswell,
        .window_hash_step_tail = sz_overlap_f64x4_window_hash_step_tail_haswell,
        .btree_sort = sz_overlap_u32x8_btree_sort_haswell,
        .btree_probe = sz_overlap_u32x8_btree_probe_haswell,
    };
    check("test_overlap_steps_equivalence_haswell",
          [&](test_context_t &context) { check_overlap_steps_equivalence_(context, overlap_steps_haswell); });
#endif // STRINGZILLA_HEADER_ONLY
    constexpr overlap_backend_t overlap_haswell {"haswell", sz_overlap_engine_init_haswell, sz_overlap_scores_haswell};
    check("test_overlap_equivalence_haswell",
          [&](test_context_t &context) { check_overlap_equivalence_(context, overlap_haswell); });
    check("test_overlap_safety_haswell", [&] { check_overlap_safety_(overlap_haswell); });

    constexpr substrings_tier_t substrings_haswell {
        .init = sz_substrings_engine_init_haswell,
        .counts = sz_substrings_counts_haswell,
        .find = sz_substrings_find_haswell,
        .replace = sz_substrings_replace_haswell,
        .bm25_scores = sz_substrings_bm25_scores_haswell,
    };
    check("test_substrings_unit_haswell", [&] { check_substrings_unit_(substrings_haswell); });
    check("test_substrings_equivalence_haswell",
          [&](test_context_t &context) { check_substrings_equivalence_(context, substrings_haswell); });

    constexpr utf8_runes_backend_t utf8_runes_haswell {"haswell", sz_utf8_count_haswell, sz_utf8_seek_haswell,
                                                       sz_utf8_decode_haswell};
    check("test_utf8_runes_unit_haswell", [&] { check_utf8_runes_unit_(utf8_runes_haswell); });
    check("test_utf8_runes_safety_haswell",
          [&](test_context_t &context) { check_utf8_runes_safety_(context, utf8_runes_haswell); });
    check("test_utf8_runes_equivalence_haswell",
          [&](test_context_t &context) { check_utf8_runes_equivalence_(context, utf8_runes_haswell); });

    constexpr utf8_tokens_backend_t utf8_tokens_haswell {"haswell", sz_utf8_count_haswell, sz_utf8_newlines_haswell,
                                                         sz_utf8_whitespaces_haswell};
    check("test_utf8_tokens_unit_haswell", [&] { check_utf8_tokens_unit_(utf8_tokens_haswell); });
    check("test_utf8_tokens_safety_haswell",
          [&](test_context_t &context) { check_utf8_tokens_safety_(context, utf8_tokens_haswell); });
    check("test_utf8_tokens_equivalence_haswell",
          [&](test_context_t &context) { check_utf8_tokens_equivalence_(context, utf8_tokens_haswell); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_haswell {"haswell", sz_utf8_delimiters_haswell};
    check("test_utf8_delimiters_unit_haswell", [&] { check_utf8_delimiters_unit_(utf8_delimiters_haswell); });
    check("test_utf8_delimiters_safety_haswell",
          [&](test_context_t &context) { check_utf8_delimiters_safety_(context, utf8_delimiters_haswell); });
    check("test_utf8_delimiters_equivalence_haswell",
          [&](test_context_t &context) { check_utf8_delimiters_equivalence_(context, utf8_delimiters_haswell); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_haswell {"haswell", sz_utf8_wordbreaks_haswell};
    check("test_utf8_wordbreaks_unit_haswell", [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_haswell); });
    check("test_utf8_wordbreaks_rules_haswell", [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_haswell); });
    check("test_utf8_wordbreaks_safety_haswell",
          [&](test_context_t &context) { check_utf8_wordbreaks_safety_(context, utf8_wordbreaks_haswell); });
    check("test_utf8_wordbreaks_equivalence_haswell",
          [&](test_context_t &context) { check_utf8_wordbreaks_equivalence_(context, utf8_wordbreaks_haswell); });

    constexpr utf8_segment_backend_t utf8_graphemes_haswell {"haswell", sz_utf8_graphemes_haswell};
    check("test_utf8_graphemes_unit_haswell", [&] { check_utf8_graphemes_unit_(utf8_graphemes_haswell); });
    check("test_utf8_graphemes_rules_haswell", [&] { check_utf8_graphemes_rules_(utf8_graphemes_haswell); });
    check("test_utf8_graphemes_safety_haswell",
          [&](test_context_t &context) { check_utf8_graphemes_safety_(context, utf8_graphemes_haswell); });
    check("test_utf8_graphemes_equivalence_haswell",
          [&](test_context_t &context) { check_utf8_graphemes_equivalence_(context, utf8_graphemes_haswell); });

    constexpr utf8_segment_backend_t utf8_sentences_haswell {"haswell", sz_utf8_sentences_haswell};
    check("test_utf8_sentences_unit_haswell", [&] { check_utf8_sentences_unit_(utf8_sentences_haswell); });
    check("test_utf8_sentences_rules_haswell", [&] { check_utf8_sentences_rules_(utf8_sentences_haswell); });
    check("test_utf8_sentences_safety_haswell",
          [&](test_context_t &context) { check_utf8_sentences_safety_(context, utf8_sentences_haswell); });
    check("test_utf8_sentences_equivalence_haswell",
          [&](test_context_t &context) { check_utf8_sentences_equivalence_(context, utf8_sentences_haswell); });

    constexpr utf8_segment_backend_t utf8_linebreaks_haswell {"haswell", sz_utf8_linebreaks_haswell};
    check("test_utf8_linebreaks_unit_haswell", [&] { check_utf8_linebreaks_unit_(utf8_linebreaks_haswell); });
    check("test_utf8_linebreaks_rules_haswell", [&] { check_utf8_linebreaks_rules_(utf8_linebreaks_haswell); });
    check("test_utf8_linebreaks_safety_haswell",
          [&](test_context_t &context) { check_utf8_linebreaks_safety_(context, utf8_linebreaks_haswell); });
    check("test_utf8_linebreaks_equivalence_haswell",
          [&](test_context_t &context) { check_utf8_linebreaks_equivalence_(context, utf8_linebreaks_haswell); });

    constexpr utf8_norm_kernels_t utf8_norm_haswell {sz_utf8_norm_haswell, sz_utf8_find_denormalized_haswell};
    check("test_utf8_norm_unit_haswell", [&] { check_utf8_norm_unit_(utf8_norm_haswell); });
    check("test_utf8_norm_equivalence_haswell",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_haswell); });
    check("test_utf8_norm_safety_haswell",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_haswell); });

    constexpr utf8_uncased_kernels_t utf8_uncased_haswell {sz_utf8_uncased_fold_haswell, sz_utf8_uncased_search_haswell,
                                                           sz_utf8_uncased_order_haswell, sz_utf8_find_cased_haswell};
    check("test_utf8_uncased_unit_haswell", [&] { check_utf8_uncased_unit_(utf8_uncased_haswell); });
    check("test_utf8_uncased_equivalence_haswell",
          [&](test_context_t &context) { check_utf8_uncased_equivalence_(context, utf8_uncased_haswell); });
    check("test_utf8_uncased_safety_haswell",
          [&](test_context_t &context) { check_utf8_uncased_safety_(context, utf8_uncased_haswell); });
#endif // STRINGZILLA_TARGET_HASWELL

#if STRINGZILLA_TARGET_SKYLAKE
    check.section("Cross Skylake", sz_cap_skylake_k);

    check("test_compare_unit_skylake", [] { check_compare_unit_({"skylake", sz_equal_skylake, sz_order_skylake}); });

    constexpr memory_backend_t memory_skylake {"skylake", sz_copy_skylake, sz_move_skylake, sz_fill_skylake};
    check("test_memory_unit_skylake", [&] { check_memory_unit_(memory_skylake); });
    check("test_memory_equivalence_skylake",
          [&](test_context_t &context) { check_memory_equivalence_(context, memory_skylake); });
    check("test_memory_safety_skylake", [&] { check_memory_safety_(memory_skylake); });

    constexpr find_backend_t find_skylake {
        .name = "skylake",
        .find = sz_find_skylake,
        .rfind = sz_rfind_skylake,
        .find_byte = sz_find_byte_skylake,
        .rfind_byte = sz_rfind_byte_skylake,
    };
    check("test_find_unit_skylake", [&] { check_find_unit_(find_skylake); });
    check("test_find_equivalence_skylake",
          [&](test_context_t &context) { check_find_equivalence_(context, find_skylake); });
    check("test_find_safety_skylake", [&] { check_find_safety_(find_skylake); });

    constexpr hash_backend_t hash_skylake {
        .hash_kernel = sz_hash_skylake,
        .init_kernel = sz_hash_state_init_skylake,
        .update_kernel = sz_hash_state_update_skylake,
        .digest_kernel = sz_hash_state_digest_skylake,
    };
    constexpr sha256_multistate_backend_t sha256_multistate_skylake {
        .update_kernel = sz_sha256_multistate_update_skylake,
        .digest_kernel = sz_sha256_multistate_digest_skylake,
    };
    check("test_hash_equivalence_skylake", [&](test_context_t &context) {
        check_bytesum_equivalence_(context, sz_bytesum_skylake);
        check_hash_equivalence_(context, hash_skylake);
        check_fill_random_equivalence_(context, sz_fill_random_skylake);
        check_sha256_multistate_equivalence_(context, sha256_multistate_skylake);
    });

    constexpr sort_backend_t sort_skylake {"skylake", sz_sequence_argsort_skylake, sz_sequence_argsort_uncased_skylake};
    check("test_sort_unit_skylake", [&] { check_sort_unit_(sort_skylake); });
    check("test_sort_equivalence_skylake",
          [&](test_context_t &context) { check_sort_equivalence_(context, sort_skylake); });
    check("test_sort_safety_skylake", [&] { check_sort_safety_(sort_skylake); });

    constexpr levenshtein_backend_t levenshtein_skylake {"skylake", sz_levenshtein_engine_init_skylake,
                                                         sz_levenshtein_distances_skylake};
    check("test_levenshtein_unit_skylake", [&] { check_levenshtein_unit_(levenshtein_skylake); });
    check("test_levenshtein_equivalence_skylake",
          [&](test_context_t &context) { check_levenshtein_equivalence_(context, levenshtein_skylake); });
    check("test_levenshtein_safety_skylake", [&] { check_levenshtein_safety_(levenshtein_skylake); });

#if STRINGZILLA_HEADER_ONLY // The step verbs are inline helpers, compiled only with the capability headers
    constexpr overlap_step_backend_t overlap_steps_skylake {
        .name = "skylake",
        .positions_per_step = sz_overlap_f64x8_positions_per_step_skylake_k,
        .prefix_hash_step = sz_overlap_f64x8_prefix_hash_step_skylake,
        .prefix_hash_step_tail = sz_overlap_f64x8_prefix_hash_step_tail_skylake,
        .window_hash_step = sz_overlap_f64x8_window_hash_step_skylake,
        .window_hash_step_tail = sz_overlap_f64x8_window_hash_step_tail_skylake,
        .btree_sort = sz_overlap_u32x16_btree_sort_skylake,
        .btree_probe = sz_overlap_u32x16_btree_probe_skylake,
    };
    check("test_overlap_steps_equivalence_skylake",
          [&](test_context_t &context) { check_overlap_steps_equivalence_(context, overlap_steps_skylake); });
#endif // STRINGZILLA_HEADER_ONLY
    constexpr overlap_backend_t overlap_skylake {"skylake", sz_overlap_engine_init_skylake, sz_overlap_scores_skylake};
    check("test_overlap_equivalence_skylake",
          [&](test_context_t &context) { check_overlap_equivalence_(context, overlap_skylake); });
    check("test_overlap_safety_skylake", [&] { check_overlap_safety_(overlap_skylake); });

    constexpr utf8_norm_kernels_t utf8_norm_skylake {sz_utf8_norm_skylake, sz_utf8_find_denormalized_skylake};
    check("test_utf8_norm_unit_skylake", [&] { check_utf8_norm_unit_(utf8_norm_skylake); });
    check("test_utf8_norm_equivalence_skylake",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_skylake); });
    check("test_utf8_norm_safety_skylake",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_skylake); });
#endif // STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_TARGET_ICELAKE
    check.section("Cross Ice Lake", sz_cap_icelake_k);

    constexpr lookup_backend_t lookup_icelake {"icelake", sz_lookup_icelake};
    check("test_lookup_unit_icelake", [&] { check_lookup_unit_(lookup_icelake); });
    check("test_lookup_equivalence_icelake",
          [&](test_context_t &context) { check_lookup_equivalence_(context, lookup_icelake); });
    check("test_lookup_safety_icelake", [&] { check_lookup_safety_(lookup_icelake); });

    constexpr find_backend_t find_icelake {
        .name = "icelake",
        .find_byteset = sz_find_byteset_icelake,
        .rfind_byteset = sz_rfind_byteset_icelake,
    };
    check("test_find_unit_icelake", [&] { check_find_unit_(find_icelake); });
    check("test_find_equivalence_icelake",
          [&](test_context_t &context) { check_find_equivalence_(context, find_icelake); });
    check("test_find_safety_icelake", [&] { check_find_safety_(find_icelake); });

    constexpr hash_backend_t hash_icelake {
        .hash_kernel = sz_hash_icelake,
        .init_kernel = sz_hash_state_init_icelake,
        .update_kernel = sz_hash_state_update_icelake,
        .digest_kernel = sz_hash_state_digest_icelake,
    };
    check("test_hash_unit_icelake", [] { check_bytesum_unit_(sz_bytesum_icelake); });
    check("test_hash_equivalence_icelake", [&](test_context_t &context) {
        check_bytesum_equivalence_(context, sz_bytesum_icelake);
        check_hash_equivalence_(context, hash_icelake);
        check_fill_random_equivalence_(context, sz_fill_random_icelake);
    });
    check("test_hash_multiseed_equivalence_icelake", [](test_context_t &context) {
        check_hash_multiseed_equivalence_(context, {sz_hash_multiseed_icelake, sz_hash_icelake});
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
    check("test_cipher_unit_icelake", [&] { check_cipher_unit_(ctr_icelake, gcm_icelake); });
    check("test_cipher_equivalence_icelake",
          [&](test_context_t &context) { check_cipher_equivalence_(context, ctr_icelake, gcm_icelake); });

    check("test_intersect_unit_icelake", [] { check_intersect_unit_(sz_sequence_intersect_icelake); });

    constexpr levenshtein_backend_t levenshtein_icelake {"icelake", sz_levenshtein_engine_init_icelake,
                                                         sz_levenshtein_distances_icelake};
    check("test_levenshtein_unit_icelake", [&] { check_levenshtein_unit_(levenshtein_icelake); });
    check("test_levenshtein_equivalence_icelake",
          [&](test_context_t &context) { check_levenshtein_equivalence_(context, levenshtein_icelake); });
    check("test_levenshtein_safety_icelake", [&] { check_levenshtein_safety_(levenshtein_icelake); });

    constexpr substrings_tier_t substrings_icelake {
        .init = sz_substrings_engine_init_icelake,
        .counts = sz_substrings_counts_icelake,
        .find = sz_substrings_find_icelake,
        .replace = sz_substrings_replace_icelake,
        .bm25_scores = sz_substrings_bm25_scores_icelake,
    };
    check("test_substrings_unit_icelake", [&] { check_substrings_unit_(substrings_icelake); });
    check("test_substrings_equivalence_icelake",
          [&](test_context_t &context) { check_substrings_equivalence_(context, substrings_icelake); });

    constexpr utf8_runes_backend_t utf8_runes_icelake {"icelake", sz_utf8_count_icelake, sz_utf8_seek_icelake,
                                                       sz_utf8_decode_icelake};
    check("test_utf8_runes_unit_icelake", [&] { check_utf8_runes_unit_(utf8_runes_icelake); });
    check("test_utf8_runes_safety_icelake",
          [&](test_context_t &context) { check_utf8_runes_safety_(context, utf8_runes_icelake); });
    check("test_utf8_runes_equivalence_icelake",
          [&](test_context_t &context) { check_utf8_runes_equivalence_(context, utf8_runes_icelake); });

    constexpr utf8_tokens_backend_t utf8_tokens_icelake {"icelake", sz_utf8_count_icelake, sz_utf8_newlines_icelake,
                                                         sz_utf8_whitespaces_icelake};
    check("test_utf8_tokens_unit_icelake", [&] { check_utf8_tokens_unit_(utf8_tokens_icelake); });
    check("test_utf8_tokens_safety_icelake",
          [&](test_context_t &context) { check_utf8_tokens_safety_(context, utf8_tokens_icelake); });
    check("test_utf8_tokens_equivalence_icelake",
          [&](test_context_t &context) { check_utf8_tokens_equivalence_(context, utf8_tokens_icelake); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_icelake {"icelake", sz_utf8_delimiters_icelake};
    check("test_utf8_delimiters_unit_icelake", [&] { check_utf8_delimiters_unit_(utf8_delimiters_icelake); });
    check("test_utf8_delimiters_safety_icelake",
          [&](test_context_t &context) { check_utf8_delimiters_safety_(context, utf8_delimiters_icelake); });
    check("test_utf8_delimiters_equivalence_icelake",
          [&](test_context_t &context) { check_utf8_delimiters_equivalence_(context, utf8_delimiters_icelake); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_icelake {"icelake", sz_utf8_wordbreaks_icelake};
    check("test_utf8_wordbreaks_unit_icelake", [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_icelake); });
    check("test_utf8_wordbreaks_rules_icelake", [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_icelake); });
    check("test_utf8_wordbreaks_safety_icelake",
          [&](test_context_t &context) { check_utf8_wordbreaks_safety_(context, utf8_wordbreaks_icelake); });
    check("test_utf8_wordbreaks_equivalence_icelake",
          [&](test_context_t &context) { check_utf8_wordbreaks_equivalence_(context, utf8_wordbreaks_icelake); });

    constexpr utf8_segment_backend_t utf8_graphemes_icelake {"icelake", sz_utf8_graphemes_icelake};
    check("test_utf8_graphemes_unit_icelake", [&] { check_utf8_graphemes_unit_(utf8_graphemes_icelake); });
    check("test_utf8_graphemes_rules_icelake", [&] { check_utf8_graphemes_rules_(utf8_graphemes_icelake); });
    check("test_utf8_graphemes_safety_icelake",
          [&](test_context_t &context) { check_utf8_graphemes_safety_(context, utf8_graphemes_icelake); });
    check("test_utf8_graphemes_equivalence_icelake",
          [&](test_context_t &context) { check_utf8_graphemes_equivalence_(context, utf8_graphemes_icelake); });

    constexpr utf8_segment_backend_t utf8_sentences_icelake {"icelake", sz_utf8_sentences_icelake};
    check("test_utf8_sentences_unit_icelake", [&] { check_utf8_sentences_unit_(utf8_sentences_icelake); });
    check("test_utf8_sentences_rules_icelake", [&] { check_utf8_sentences_rules_(utf8_sentences_icelake); });
    check("test_utf8_sentences_safety_icelake",
          [&](test_context_t &context) { check_utf8_sentences_safety_(context, utf8_sentences_icelake); });
    check("test_utf8_sentences_equivalence_icelake",
          [&](test_context_t &context) { check_utf8_sentences_equivalence_(context, utf8_sentences_icelake); });

    constexpr utf8_segment_backend_t utf8_linebreaks_icelake {"icelake", sz_utf8_linebreaks_icelake};
    check("test_utf8_linebreaks_unit_icelake", [&] { check_utf8_linebreaks_unit_(utf8_linebreaks_icelake); });
    check("test_utf8_linebreaks_rules_icelake", [&] { check_utf8_linebreaks_rules_(utf8_linebreaks_icelake); });
    check("test_utf8_linebreaks_safety_icelake",
          [&](test_context_t &context) { check_utf8_linebreaks_safety_(context, utf8_linebreaks_icelake); });
    check("test_utf8_linebreaks_equivalence_icelake",
          [&](test_context_t &context) { check_utf8_linebreaks_equivalence_(context, utf8_linebreaks_icelake); });

    constexpr utf8_norm_kernels_t utf8_norm_icelake {sz_utf8_norm_icelake, sz_utf8_find_denormalized_icelake};
    check("test_utf8_norm_unit_icelake", [&] { check_utf8_norm_unit_(utf8_norm_icelake); });
    check("test_utf8_norm_equivalence_icelake",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_icelake); });
    check("test_utf8_norm_safety_icelake",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_icelake); });

    constexpr utf8_uncased_kernels_t utf8_uncased_icelake {sz_utf8_uncased_fold_icelake, sz_utf8_uncased_search_icelake,
                                                           sz_utf8_uncased_order_icelake, sz_utf8_find_cased_icelake};
    check("test_utf8_uncased_unit_icelake", [&] { check_utf8_uncased_unit_(utf8_uncased_icelake); });
    check("test_utf8_uncased_equivalence_icelake",
          [&](test_context_t &context) { check_utf8_uncased_equivalence_(context, utf8_uncased_icelake); });
    check("test_utf8_uncased_safety_icelake",
          [&](test_context_t &context) { check_utf8_uncased_safety_(context, utf8_uncased_icelake); });
#endif // STRINGZILLA_TARGET_ICELAKE

    return check.failures;
}

} // namespace ashvardanian::stringzilla::test
