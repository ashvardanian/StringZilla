/**
 *  @file test/cross_loongarch64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - LoongArch family, LASX.
 */
#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

std::size_t test_cross_loongarch64(environment_t const &env) {
    [[maybe_unused]] cross_section_t check(env);

#if STRINGZILLA_TARGET_LOONGSONASX
    check.section("Cross LoongArch LASX", sz_cap_loongsonasx_k);

    check("test_compare_unit_loongsonasx",
          [] { check_compare_unit_({"loongsonasx", sz_equal_loongsonasx, sz_order_loongsonasx}); });

    constexpr memory_backend_t memory_loongsonasx {"loongsonasx", sz_copy_loongsonasx, sz_move_loongsonasx,
                                                   sz_fill_loongsonasx};
    check("test_memory_unit_loongsonasx", [&] { check_memory_unit_(memory_loongsonasx); });
    check("test_memory_equivalence_loongsonasx",
          [&](test_context_t &context) { check_memory_equivalence_(context, memory_loongsonasx); });
    check("test_memory_safety_loongsonasx", [&] { check_memory_safety_(memory_loongsonasx); });

    constexpr lookup_backend_t lookup_loongsonasx {"loongsonasx", sz_lookup_loongsonasx};
    check("test_lookup_unit_loongsonasx", [&] { check_lookup_unit_(lookup_loongsonasx); });
    check("test_lookup_equivalence_loongsonasx",
          [&](test_context_t &context) { check_lookup_equivalence_(context, lookup_loongsonasx); });
    check("test_lookup_safety_loongsonasx", [&] { check_lookup_safety_(lookup_loongsonasx); });

    constexpr find_backend_t find_loongsonasx {
        .name = "loongsonasx",
        .find = sz_find_loongsonasx,
        .rfind = sz_rfind_loongsonasx,
        .find_byte = sz_find_byte_loongsonasx,
        .rfind_byte = sz_rfind_byte_loongsonasx,
        .find_byteset = sz_find_byteset_loongsonasx,
        .rfind_byteset = sz_rfind_byteset_loongsonasx,
    };
    check("test_find_unit_loongsonasx", [&] { check_find_unit_(find_loongsonasx); });
    check("test_find_equivalence_loongsonasx",
          [&](test_context_t &context) { check_find_equivalence_(context, find_loongsonasx); });
    check("test_find_safety_loongsonasx", [&] { check_find_safety_(find_loongsonasx); });

    constexpr hash_backend_t hash_loongsonasx {
        .hash_kernel = sz_hash_loongsonasx,
        .init_kernel = sz_hash_state_init_loongsonasx,
        .update_kernel = sz_hash_state_update_loongsonasx,
        .digest_kernel = sz_hash_state_digest_loongsonasx,
    };
    constexpr sha256_backend_t sha256_loongsonasx {
        .init_kernel = sz_sha256_state_init_loongsonasx,
        .update_kernel = sz_sha256_state_update_loongsonasx,
        .digest_kernel = sz_sha256_state_digest_loongsonasx,
    };
    check("test_hash_equivalence_loongsonasx", [&](test_context_t &context) {
        check_bytesum_equivalence_(context, sz_bytesum_loongsonasx);
        check_hash_equivalence_(context, hash_loongsonasx);
        check_fill_random_equivalence_(context, sz_fill_random_loongsonasx);
        check_sha256_equivalence_(context, sha256_loongsonasx);
    });

    constexpr utf8_runes_backend_t utf8_runes_loongsonasx {"loongsonasx", sz_utf8_count_loongsonasx,
                                                           sz_utf8_seek_loongsonasx, sz_utf8_decode_loongsonasx};
    check("test_utf8_runes_unit_loongsonasx", [&] { check_utf8_runes_unit_(utf8_runes_loongsonasx); });
    check("test_utf8_runes_safety_loongsonasx",
          [&](test_context_t &context) { check_utf8_runes_safety_(context, utf8_runes_loongsonasx); });
    check("test_utf8_runes_equivalence_loongsonasx",
          [&](test_context_t &context) { check_utf8_runes_equivalence_(context, utf8_runes_loongsonasx); });

    constexpr utf8_tokens_backend_t utf8_tokens_loongsonasx {
        "loongsonasx", sz_utf8_count_loongsonasx, sz_utf8_newlines_loongsonasx, sz_utf8_whitespaces_loongsonasx};
    check("test_utf8_tokens_unit_loongsonasx", [&] { check_utf8_tokens_unit_(utf8_tokens_loongsonasx); });
    check("test_utf8_tokens_safety_loongsonasx",
          [&](test_context_t &context) { check_utf8_tokens_safety_(context, utf8_tokens_loongsonasx); });
    check("test_utf8_tokens_equivalence_loongsonasx",
          [&](test_context_t &context) { check_utf8_tokens_equivalence_(context, utf8_tokens_loongsonasx); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_loongsonasx {"loongsonasx", sz_utf8_delimiters_loongsonasx};
    check("test_utf8_delimiters_unit_loongsonasx", [&] { check_utf8_delimiters_unit_(utf8_delimiters_loongsonasx); });
    check("test_utf8_delimiters_safety_loongsonasx",
          [&](test_context_t &context) { check_utf8_delimiters_safety_(context, utf8_delimiters_loongsonasx); });
    check("test_utf8_delimiters_equivalence_loongsonasx",
          [&](test_context_t &context) { check_utf8_delimiters_equivalence_(context, utf8_delimiters_loongsonasx); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_loongsonasx {"loongsonasx", sz_utf8_wordbreaks_loongsonasx};
    check("test_utf8_wordbreaks_unit_loongsonasx", [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_loongsonasx); });
    check("test_utf8_wordbreaks_rules_loongsonasx", [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_loongsonasx); });
    check("test_utf8_wordbreaks_safety_loongsonasx",
          [&](test_context_t &context) { check_utf8_wordbreaks_safety_(context, utf8_wordbreaks_loongsonasx); });
    check("test_utf8_wordbreaks_equivalence_loongsonasx",
          [&](test_context_t &context) { check_utf8_wordbreaks_equivalence_(context, utf8_wordbreaks_loongsonasx); });

    constexpr utf8_norm_kernels_t utf8_norm_loongsonasx {sz_utf8_norm_loongsonasx,
                                                         sz_utf8_find_denormalized_loongsonasx};
    check("test_utf8_norm_unit_loongsonasx", [&] { check_utf8_norm_unit_(utf8_norm_loongsonasx); });
    check("test_utf8_norm_equivalence_loongsonasx",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_loongsonasx); });
    check("test_utf8_norm_safety_loongsonasx",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_loongsonasx); });

    constexpr utf8_uncased_kernels_t utf8_uncased_loongsonasx {
        sz_utf8_uncased_fold_loongsonasx, sz_utf8_uncased_search_loongsonasx, sz_utf8_uncased_order_loongsonasx,
        sz_utf8_find_cased_loongsonasx};
    check("test_utf8_uncased_unit_loongsonasx", [&] { check_utf8_uncased_unit_(utf8_uncased_loongsonasx); });
    check("test_utf8_uncased_equivalence_loongsonasx",
          [&](test_context_t &context) { check_utf8_uncased_equivalence_(context, utf8_uncased_loongsonasx); });
    check("test_utf8_uncased_safety_loongsonasx",
          [&](test_context_t &context) { check_utf8_uncased_safety_(context, utf8_uncased_loongsonasx); });
#endif // STRINGZILLA_TARGET_LOONGSONASX

    return check.failures;
}

} // namespace ashvardanian::stringzilla::test
