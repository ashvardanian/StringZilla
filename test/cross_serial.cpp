/**
 *  @file test/cross_serial.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - the serial kernels, against known answers.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

void test_cross_serial() {
    capability_gate_t const serial(sz_cap_serial_k);

    run_test_case("sz_kernel_named", serial, [] {
        for (int kind = sz_kernel_unknown_k; kind <= sz_kernel_subdocuments_find_k; ++kind) {
            char const *name = sz_kernel_name(static_cast<sz_kernel_kind_t>(kind));
            expect_equal(sz_kernel_named(name, std::strlen(name)), static_cast<sz_kernel_kind_t>(kind));
        }
        expect_equal(sz_kernel_named("finder", 6), sz_kernel_unknown_k);
        expect_equal(sz_kernel_named("find", 3), sz_kernel_unknown_k);
    });

    run_test_case("check_compare_unit_({sz_equal_serial, sz_order_serial})", serial,
                  [] { check_compare_unit_({"serial", sz_equal_serial, sz_order_serial}); });

    constexpr strings_backend_t strings_serial {"serial",
                                                sz_strings_to_tape_serial,
                                                sz_strings_to_rows_serial,
                                                sz_strings_escape_serial,
                                                sz_strings_unescape_serial,
                                                sz_strings_lookup_serial,
                                                sz_strings_normalize_serial,
                                                sz_strings_split_serial,
                                                sz_strings_split_lines_serial,
                                                sz_strings_join_serial};
    run_test_case("check_strings_unit_(strings_serial)", serial, [&] { check_strings_unit_(strings_serial); });
    run_test_case("check_strings_safety_(strings_serial)", serial, [&] { check_strings_safety_(strings_serial); });
    run_test_case("check_strings_split_equivalence_(strings_serial)", serial, [&](split_mix_generator_t &generator) {
        check_strings_split_equivalence_(generator, strings_serial);
    });

    constexpr find_backend_t find_serial {
        .name = "serial",
        .find = sz_find_serial,
        .rfind = sz_rfind_serial,
        .find_byte = sz_find_byte_serial,
        .rfind_byte = sz_rfind_byte_serial,
        .find_byteset = sz_find_byteset_serial,
        .rfind_byteset = sz_rfind_byteset_serial,
        .find_strings = sz_find_strings_serial,
    };
    run_test_case("check_find_unit_(find_serial)", serial, [&] { check_find_unit_(find_serial); });
    run_test_case("check_find_safety_(find_serial)", serial, [&] { check_find_safety_(find_serial); });
    run_test_case("check_find_strings_equivalence_(sz_find_strings_serial)", serial,
                  [](split_mix_generator_t &generator) {
                      check_find_strings_equivalence_(generator, "serial", sz_find_strings_serial);
                  });

    run_test_case("check_sha256_unit_({sz_sha256_state_init_serial, sz_sha256_state_update_serial, " //
                  "sz_sha256_state_digest_serial})",
                  serial, [] {
                      check_sha256_unit_({
                          .init_kernel = sz_sha256_state_init_serial,
                          .update_kernel = sz_sha256_state_update_serial,
                          .digest_kernel = sz_sha256_state_digest_serial,
                      });
                      check_sha256_multistate_unit_({
                          .update_kernel = sz_sha256_multistate_update_serial,
                          .digest_kernel = sz_sha256_multistate_digest_serial,
                      });
                      check_bytesum_unit_(sz_bytesum_serial);
                      check_hash_unit_(sz_hash_serial);
                  });
    run_test_case("check_hash_strings_equivalence_(sz_hash_strings_serial)", serial,
                  [](split_mix_generator_t &generator) {
                      check_hash_strings_equivalence_(generator, "serial", sz_hash_strings_serial);
                      check_hash_strings_safety_(sz_hash_strings_serial);
                  });
    run_test_case("check_hash_multiseed_equivalence_({sz_hash_multiseed_serial, sz_hash_serial})", serial,
                  [](split_mix_generator_t &generator) {
                      check_hash_multiseed_equivalence_(generator, {sz_hash_multiseed_serial, sz_hash_serial});
                  });

    run_test_case("check_cipher_unit_(ctr_serial_backend_(), gcm_serial_backend_())", serial,
                  [] { check_cipher_unit_(ctr_serial_backend_(), gcm_serial_backend_()); });
    run_test_case("check_cipher_equivalence_(ctr_serial_backend_(), gcm_serial_backend_())", serial,
                  [](split_mix_generator_t &generator) {
                      check_cipher_equivalence_(generator, ctr_serial_backend_(), gcm_serial_backend_());
                  });

    constexpr sort_backend_t sort_serial {"serial", sz_sort_order_serial, sz_sort_uncased_order_serial};
    run_test_case("check_sort_unit_(sort_serial)", serial, [&] { check_sort_unit_(sort_serial); });
    run_test_case("check_sort_safety_(sort_serial)", serial, [&] { check_sort_safety_(sort_serial); });

    run_test_case("check_intersect_unit_(sz_intersect_serial)", serial,
                  [] { check_intersect_unit_(sz_intersect_serial); });

    constexpr levenshtein_backend_t levenshtein_serial {"serial", sz_levenshtein_engine_init_serial,
                                                        sz_levenshtein_distances_serial};
    run_test_case("check_levenshtein_unit_(levenshtein_serial)", serial,
                  [&] { check_levenshtein_unit_(levenshtein_serial); });
    run_test_case("check_levenshtein_equivalence_(levenshtein_serial)", serial, [&](split_mix_generator_t &generator) {
        check_levenshtein_equivalence_(generator, levenshtein_serial);
    });
    run_test_case("check_levenshtein_safety_(levenshtein_serial)", serial,
                  [&] { check_levenshtein_safety_(levenshtein_serial); });

    constexpr ngrams_backend_t ngrams_serial {"serial",
                                              sz_ngrams_engine_init_serial,
                                              sz_ngrams_overlaps_serial,
                                              sz_ngrams_sketches_serial,
                                              sz_ngrams_residues_serial,
                                              sz_ngrams_sketch_serial};
    run_test_case("check_ngrams_overlaps_(ngrams_serial)", serial,
                  [&](split_mix_generator_t &generator) { check_ngrams_overlaps_(generator, ngrams_serial); });
    run_test_case("check_ngrams_residues_(ngrams_serial)", serial,
                  [&](split_mix_generator_t &generator) { check_ngrams_residues_(generator, ngrams_serial); });
    run_test_case("check_ngrams_sketches_(ngrams_serial)", serial,
                  [&](split_mix_generator_t &generator) { check_ngrams_sketches_(generator, ngrams_serial); });
    run_test_case("check_ngrams_jaccard_(ngrams_serial)", serial,
                  [&](split_mix_generator_t &generator) { check_ngrams_jaccard_(generator, ngrams_serial); });
    run_test_case("check_ngrams_safety_(ngrams_serial)", serial, [&] { check_ngrams_safety_(ngrams_serial); });

    constexpr substrings_tier_t substrings_serial {
        .init = sz_substrings_engine_init_serial,
        .counts = sz_substrings_counts_serial,
        .find = sz_substrings_find_serial,
        .replace = sz_substrings_replace_serial,
        .bm25_scores = sz_substrings_bm25_scores_serial,
    };
    run_test_case("check_substrings_unit_(substrings_serial)", serial,
                  [&] { check_substrings_unit_(substrings_serial); });
    run_test_case("check_substrings_equivalence_(substrings_serial)", serial, [&](split_mix_generator_t &generator) {
        check_substrings_equivalence_(generator, substrings_serial);
    });

    constexpr utf8_runes_backend_t utf8_runes_serial {"serial", sz_utf8_count_serial, sz_utf8_seek_serial,
                                                      sz_utf8_decode_serial, sz_utf8_count_strings_serial};
    run_test_case("check_utf8_runes_unit_(utf8_runes_serial)", serial,
                  [&] { check_utf8_runes_unit_(utf8_runes_serial); });
    run_test_case("check_utf8_runes_safety_(utf8_runes_serial)", serial,
                  [&](split_mix_generator_t &generator) { check_utf8_runes_safety_(generator, utf8_runes_serial); });
    run_test_case("check_utf8_count_strings_equivalence_(sz_utf8_count_strings_serial)", serial,
                  [](split_mix_generator_t &generator) {
                      check_utf8_count_strings_equivalence_(generator, "serial", sz_utf8_count_strings_serial);
                  });

    constexpr utf8_tokens_backend_t utf8_tokens_serial {"serial", sz_utf8_count_serial, sz_utf8_newlines_serial,
                                                        sz_utf8_whitespaces_serial};
    run_test_case("check_utf8_tokens_unit_(utf8_tokens_serial)", serial,
                  [&] { check_utf8_tokens_unit_(utf8_tokens_serial); });
    run_test_case("check_utf8_tokens_safety_(utf8_tokens_serial)", serial,
                  [&](split_mix_generator_t &generator) { check_utf8_tokens_safety_(generator, utf8_tokens_serial); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_serial {"serial", sz_utf8_delimiters_serial};
    run_test_case("check_utf8_delimiters_unit_(utf8_delimiters_serial)", serial,
                  [&] { check_utf8_delimiters_unit_(utf8_delimiters_serial); });
    run_test_case(
        "check_utf8_delimiters_safety_(utf8_delimiters_serial)", serial,
        [&](split_mix_generator_t &generator) { check_utf8_delimiters_safety_(generator, utf8_delimiters_serial); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_serial {"serial", sz_utf8_wordbreaks_serial};
    run_test_case("check_utf8_wordbreaks_unit_(utf8_wordbreaks_serial)", serial,
                  [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_serial); });
    run_test_case(
        "check_utf8_wordbreaks_safety_(utf8_wordbreaks_serial)", serial,
        [&](split_mix_generator_t &generator) { check_utf8_wordbreaks_safety_(generator, utf8_wordbreaks_serial); });
    run_test_case("check_utf8_wordbreaks_oracle_()", serial, [] { check_utf8_wordbreaks_oracle_(); });

    constexpr utf8_segment_backend_t utf8_graphemes_serial {"serial", sz_utf8_graphemes_serial};
    run_test_case("check_utf8_graphemes_unit_(utf8_graphemes_serial)", serial,
                  [&] { check_utf8_graphemes_unit_(utf8_graphemes_serial); });
    run_test_case("check_utf8_graphemes_safety_(utf8_graphemes_serial)", serial, [&](split_mix_generator_t &generator) {
        check_utf8_graphemes_safety_(generator, utf8_graphemes_serial);
    });
    run_test_case("check_utf8_graphemes_oracle_()", serial, [] { check_utf8_graphemes_oracle_(); });

    constexpr utf8_segment_backend_t utf8_sentences_serial {"serial", sz_utf8_sentences_serial};
    run_test_case("check_utf8_sentences_unit_(utf8_sentences_serial)", serial,
                  [&] { check_utf8_sentences_unit_(utf8_sentences_serial); });
    run_test_case("check_utf8_sentences_safety_(utf8_sentences_serial)", serial, [&](split_mix_generator_t &generator) {
        check_utf8_sentences_safety_(generator, utf8_sentences_serial);
    });

    constexpr utf8_segment_backend_t utf8_linebreaks_serial {"serial", sz_utf8_linebreaks_serial};
    run_test_case("check_utf8_linebreaks_unit_(utf8_linebreaks_serial)", serial,
                  [&] { check_utf8_linebreaks_unit_(utf8_linebreaks_serial); });
    run_test_case(
        "check_utf8_linebreaks_safety_(utf8_linebreaks_serial)", serial,
        [&](split_mix_generator_t &generator) { check_utf8_linebreaks_safety_(generator, utf8_linebreaks_serial); });

    constexpr utf8_norm_kernels_t utf8_norm_serial {sz_utf8_norm_serial, sz_utf8_find_denormalized_serial};
    run_test_case("check_utf8_norm_unit_(utf8_norm_serial)", serial, [&] { check_utf8_norm_unit_(utf8_norm_serial); });
    run_test_case("check_utf8_norm_safety_(utf8_norm_serial)", serial,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_safety_(generator, utf8_norm_serial); });

    constexpr utf8_uncased_kernels_t utf8_uncased_serial {sz_utf8_uncased_fold_serial, sz_utf8_uncased_search_serial,
                                                          sz_utf8_uncased_order_serial, sz_utf8_find_cased_serial};
    run_test_case("check_utf8_uncased_unit_(utf8_uncased_serial)", serial, [&] {
        check_utf8_uncased_unit_(utf8_uncased_serial);
        check_uncased_invariant_reference_();
    });
    run_test_case("check_utf8_uncased_safety_(utf8_uncased_serial)", serial, [&](split_mix_generator_t &generator) {
        check_utf8_uncased_safety_(generator, utf8_uncased_serial);
    });
}

} // namespace ashvardanian::stringzilla::test
