/**
 *  @file test/cross_loongarch64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - LoongArch family, LASX.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

void test_cross_loongarch64() {
#if STRINGZILLA_TARGET_LOONGSONASX
    capability_gate_t const loongsonasx(sz_cap_loongsonasx_k);

    run_test_case("check_compare_unit_({sz_equal_loongsonasx, sz_order_loongsonasx})", loongsonasx,
                  [] { check_compare_unit_({"loongsonasx", sz_equal_loongsonasx, sz_order_loongsonasx}); });

    constexpr strings_backend_t strings_loongsonasx {"loongsonasx",
                                                     sz_strings_to_tape_loongsonasx,
                                                     sz_strings_to_rows_loongsonasx,
                                                     sz_strings_escape_loongsonasx,
                                                     sz_strings_unescape_loongsonasx,
                                                     sz_strings_lookup_loongsonasx,
                                                     sz_strings_normalize_loongsonasx};
    run_test_case("check_strings_unit_(strings_loongsonasx)", loongsonasx,
                  [&] { check_strings_unit_(strings_loongsonasx); });
    run_test_case(
        "check_strings_equivalence_(strings_loongsonasx)", loongsonasx,
        [&](split_mix_generator_t &generator) { check_strings_equivalence_(generator, strings_loongsonasx); });
    run_test_case("check_strings_safety_(strings_loongsonasx)", loongsonasx,
                  [&] { check_strings_safety_(strings_loongsonasx); });

    constexpr find_backend_t find_loongsonasx {
        .name = "loongsonasx",
        .find = sz_find_loongsonasx,
        .rfind = sz_rfind_loongsonasx,
        .find_byte = sz_find_byte_loongsonasx,
        .rfind_byte = sz_rfind_byte_loongsonasx,
        .find_byteset = sz_find_byteset_loongsonasx,
        .rfind_byteset = sz_rfind_byteset_loongsonasx,
    };
    run_test_case("check_find_unit_(find_loongsonasx)", loongsonasx, [&] { check_find_unit_(find_loongsonasx); });
    run_test_case("check_find_equivalence_(find_loongsonasx)", loongsonasx,
                  [&](split_mix_generator_t &generator) { check_find_equivalence_(generator, find_loongsonasx); });
    run_test_case("check_find_safety_(find_loongsonasx)", loongsonasx, [&] { check_find_safety_(find_loongsonasx); });

    constexpr hash_backend_t hash_loongsonasx {
        .hash_kernel = sz_hash_loongsonasx,
        .init_kernel = sz_hash_state_init_loongsonasx,
        .update_kernel = sz_hash_state_update_loongsonasx,
        .digest_kernel = sz_hash_state_digest_loongsonasx,
        .strings_kernel = sz_hash_strings_loongsonasx,
    };
    constexpr sha256_backend_t sha256_loongsonasx {
        .init_kernel = sz_sha256_state_init_loongsonasx,
        .update_kernel = sz_sha256_state_update_loongsonasx,
        .digest_kernel = sz_sha256_state_digest_loongsonasx,
    };
    run_test_case("check_bytesum_equivalence_(sz_bytesum_loongsonasx)", loongsonasx,
                  [&](split_mix_generator_t &generator) {
                      check_bytesum_equivalence_(generator, sz_bytesum_loongsonasx);
                      check_hash_equivalence_(generator, hash_loongsonasx);
                      check_fill_random_equivalence_(sz_fill_random_loongsonasx);
                      check_sha256_equivalence_(generator, sha256_loongsonasx);
                  });

    constexpr substrings_tier_t substrings_loongsonasx {
        .init = sz_substrings_engine_init_loongsonasx,
        .counts = sz_substrings_counts_loongsonasx,
        .find = sz_substrings_find_loongsonasx,
        .replace = sz_substrings_replace_loongsonasx,
        .bm25_scores = sz_substrings_bm25_scores_loongsonasx,
    };
    run_test_case("check_substrings_unit_(substrings_loongsonasx)", loongsonasx,
                  [&] { check_substrings_unit_(substrings_loongsonasx); });
    run_test_case(
        "check_substrings_equivalence_(substrings_loongsonasx)", loongsonasx,
        [&](split_mix_generator_t &generator) { check_substrings_equivalence_(generator, substrings_loongsonasx); });

    constexpr subdocuments_tier_t subdocuments_loongsonasx {
        .init = sz_subdocuments_engine_init_loongsonasx,
        .counts = sz_subdocuments_counts_loongsonasx,
        .find = sz_subdocuments_find_loongsonasx,
    };
    run_test_case("check_subdocuments_unit_(subdocuments_loongsonasx)", loongsonasx,
                  [&] { check_subdocuments_unit_(subdocuments_loongsonasx); });
    run_test_case("check_subdocuments_safety_(subdocuments_loongsonasx)", loongsonasx,
                  [&] { check_subdocuments_safety_(subdocuments_loongsonasx); });
    run_test_case("check_subdocuments_equivalence_(subdocuments_loongsonasx)", loongsonasx,
                  [&](split_mix_generator_t &generator) {
                      check_subdocuments_equivalence_(generator, subdocuments_loongsonasx);
                  });

    constexpr utf8_runes_backend_t utf8_runes_loongsonasx {"loongsonasx", sz_utf8_count_loongsonasx,
                                                           sz_utf8_seek_loongsonasx, sz_utf8_decode_loongsonasx};
    run_test_case("check_utf8_runes_unit_(utf8_runes_loongsonasx)", loongsonasx,
                  [&] { check_utf8_runes_unit_(utf8_runes_loongsonasx); });
    run_test_case(
        "check_utf8_runes_safety_(utf8_runes_loongsonasx)", loongsonasx,
        [&](split_mix_generator_t &generator) { check_utf8_runes_safety_(generator, utf8_runes_loongsonasx); });
    run_test_case(
        "check_utf8_runes_equivalence_(utf8_runes_loongsonasx)", loongsonasx,
        [&](split_mix_generator_t &generator) { check_utf8_runes_equivalence_(generator, utf8_runes_loongsonasx); });

    constexpr utf8_tokens_backend_t utf8_tokens_loongsonasx {
        "loongsonasx", sz_utf8_count_loongsonasx, sz_utf8_newlines_loongsonasx, sz_utf8_whitespaces_loongsonasx};
    run_test_case("check_utf8_tokens_unit_(utf8_tokens_loongsonasx)", loongsonasx,
                  [&] { check_utf8_tokens_unit_(utf8_tokens_loongsonasx); });
    run_test_case(
        "check_utf8_tokens_safety_(utf8_tokens_loongsonasx)", loongsonasx,
        [&](split_mix_generator_t &generator) { check_utf8_tokens_safety_(generator, utf8_tokens_loongsonasx); });
    run_test_case(
        "check_utf8_tokens_equivalence_(utf8_tokens_loongsonasx)", loongsonasx,
        [&](split_mix_generator_t &generator) { check_utf8_tokens_equivalence_(generator, utf8_tokens_loongsonasx); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_loongsonasx {"loongsonasx", sz_utf8_delimiters_loongsonasx};
    run_test_case("check_utf8_delimiters_unit_(utf8_delimiters_loongsonasx)", loongsonasx,
                  [&] { check_utf8_delimiters_unit_(utf8_delimiters_loongsonasx); });
    run_test_case("check_utf8_delimiters_safety_(utf8_delimiters_loongsonasx)", loongsonasx,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_delimiters_safety_(generator, utf8_delimiters_loongsonasx);
                  });
    run_test_case("check_utf8_delimiters_equivalence_(utf8_delimiters_loongsonasx)", loongsonasx,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_delimiters_equivalence_(generator, utf8_delimiters_loongsonasx);
                  });

    constexpr utf8_segment_backend_t utf8_wordbreaks_loongsonasx {"loongsonasx", sz_utf8_wordbreaks_loongsonasx};
    run_test_case("check_utf8_wordbreaks_unit_(utf8_wordbreaks_loongsonasx)", loongsonasx,
                  [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_loongsonasx); });
    run_test_case("check_utf8_wordbreaks_rules_(utf8_wordbreaks_loongsonasx)", loongsonasx,
                  [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_loongsonasx); });
    run_test_case("check_utf8_wordbreaks_safety_(utf8_wordbreaks_loongsonasx)", loongsonasx,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_wordbreaks_safety_(generator, utf8_wordbreaks_loongsonasx);
                  });
    run_test_case("check_utf8_wordbreaks_equivalence_(utf8_wordbreaks_loongsonasx)", loongsonasx,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_wordbreaks_equivalence_(generator, utf8_wordbreaks_loongsonasx);
                  });

    constexpr utf8_norm_kernels_t utf8_norm_loongsonasx {sz_utf8_norm_loongsonasx,
                                                         sz_utf8_find_denormalized_loongsonasx};
    run_test_case("check_utf8_norm_unit_(utf8_norm_loongsonasx)", loongsonasx,
                  [&] { check_utf8_norm_unit_(utf8_norm_loongsonasx); });
    run_test_case(
        "check_utf8_norm_equivalence_(utf8_norm_loongsonasx)", loongsonasx,
        [&](split_mix_generator_t &generator) { check_utf8_norm_equivalence_(generator, utf8_norm_loongsonasx); });
    run_test_case("check_utf8_norm_safety_(utf8_norm_loongsonasx)", loongsonasx,
                  [&](split_mix_generator_t &generator) { check_utf8_norm_safety_(generator, utf8_norm_loongsonasx); });

    constexpr utf8_uncased_kernels_t utf8_uncased_loongsonasx {
        sz_utf8_uncased_fold_loongsonasx, sz_utf8_uncased_search_loongsonasx, sz_utf8_uncased_order_loongsonasx,
        sz_utf8_find_cased_loongsonasx};
    run_test_case("check_utf8_uncased_unit_(utf8_uncased_loongsonasx)", loongsonasx,
                  [&] { check_utf8_uncased_unit_(utf8_uncased_loongsonasx); });
    run_test_case("check_utf8_uncased_equivalence_(utf8_uncased_loongsonasx)", loongsonasx,
                  [&](split_mix_generator_t &generator) {
                      check_utf8_uncased_equivalence_(generator, utf8_uncased_loongsonasx);
                  });
    run_test_case(
        "check_utf8_uncased_safety_(utf8_uncased_loongsonasx)", loongsonasx,
        [&](split_mix_generator_t &generator) { check_utf8_uncased_safety_(generator, utf8_uncased_loongsonasx); });
#endif // STRINGZILLA_TARGET_LOONGSONASX
}

} // namespace ashvardanian::stringzilla::test
