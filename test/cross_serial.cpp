/**
 *  @file test/cross_serial.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - the serial kernels, against known answers.
 */
#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

std::size_t test_cross_serial(environment_t const &env) {
    cross_section_t check(env);
    check.section("Cross Serial", sz_cap_serial_k);

    check("test_kernel_names_unit", [] {
        for (int kind = sz_kernel_unknown_k; kind <= sz_kernel_substrings_bm25_scores_k; ++kind) {
            char const *name = sz_kernel_name(static_cast<sz_kernel_kind_t>(kind));
            verify(sz_kernel_named(name, std::strlen(name)) == static_cast<sz_kernel_kind_t>(kind));
        }
        verify(sz_kernel_named("finder", 6) == sz_kernel_unknown_k);
        verify(sz_kernel_named("find", 3) == sz_kernel_unknown_k);
    });

    check("test_compare_unit_serial", [] { check_compare_unit_({"serial", sz_equal_serial, sz_order_serial}); });

    constexpr memory_backend_t memory_serial {"serial", sz_copy_serial, sz_move_serial, sz_fill_serial};
    check("test_memory_unit_serial", [&] { check_memory_unit_(memory_serial); });
    check("test_memory_safety_serial", [&] { check_memory_safety_(memory_serial); });

    check("test_sequence_realloc_overflow_serial", [] {
        sz_allocator_t allocator;
        verify(sz_allocator_init_heap(&allocator) == sz_success_k);
        sz_sequence_t source {}, target {};
        sz_size_t bytes = 17;
        source.count = STRINGZILLA_SIZE_MAX;
        verify(sz_sequence_realloc_serial(&target, &source, &allocator, &bytes, nullptr) == sz_bad_alloc_k);
        source.count = 1;
        source.get_length = [](void const *, sz_size_t) { return STRINGZILLA_SIZE_MAX; };
        verify(sz_sequence_realloc_serial(&target, &source, &allocator, &bytes, nullptr) == sz_bad_alloc_k);
        verify(!target.handle && bytes == 17);
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
        bool rejected = false;
        try {
            unified_alloc<sz_u64_t, sz_cap_serial_k> wide;
            wide.allocate(STRINGZILLA_SIZE_MAX);
        }
        catch (std::bad_alloc const &) {
            rejected = true;
        }
        verify(rejected);
#endif
    });

    constexpr lookup_backend_t lookup_serial {"serial", sz_lookup_serial};
    check("test_lookup_unit_serial", [&] { check_lookup_unit_(lookup_serial); });
    check("test_lookup_safety_serial", [&] { check_lookup_safety_(lookup_serial); });

    constexpr find_backend_t find_serial {
        .name = "serial",
        .find = sz_find_serial,
        .rfind = sz_rfind_serial,
        .find_byte = sz_find_byte_serial,
        .rfind_byte = sz_rfind_byte_serial,
        .find_byteset = sz_find_byteset_serial,
        .rfind_byteset = sz_rfind_byteset_serial,
    };
    check("test_find_unit_serial", [&] { check_find_unit_(find_serial); });
    check("test_find_safety_serial", [&] { check_find_safety_(find_serial); });

    check("test_hash_unit_serial", [] {
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
    check("test_hash_multiseed_equivalence_serial", [](test_context_t &context) {
        check_hash_multiseed_equivalence_(context, {sz_hash_multiseed_serial, sz_hash_serial});
    });

    check("test_cipher_unit_serial", [] { check_cipher_unit_(ctr_serial_backend_(), gcm_serial_backend_()); });
    check("test_cipher_equivalence_serial", [](test_context_t &context) {
        check_cipher_equivalence_(context, ctr_serial_backend_(), gcm_serial_backend_());
    });

    constexpr sort_backend_t sort_serial {"serial", sz_sequence_argsort_serial, sz_sequence_argsort_uncased_serial};
    check("test_sort_unit_serial", [&] { check_sort_unit_(sort_serial); });
    check("test_sort_safety_serial", [&] { check_sort_safety_(sort_serial); });

    check("test_intersect_unit_serial", [] { check_intersect_unit_(sz_sequence_intersect_serial); });

    constexpr levenshtein_backend_t levenshtein_serial {"serial", sz_levenshtein_engine_init_serial,
                                                        sz_levenshtein_distances_serial};
    check("test_levenshtein_unit_serial", [&] { check_levenshtein_unit_(levenshtein_serial); });
    check("test_levenshtein_equivalence_serial",
          [&](test_context_t &context) { check_levenshtein_equivalence_(context, levenshtein_serial); });
    check("test_levenshtein_safety_serial", [&] { check_levenshtein_safety_(levenshtein_serial); });

    constexpr overlap_backend_t overlap_serial {"serial", sz_overlap_engine_init_serial, sz_overlap_scores_serial};
    check("test_overlap_equivalence_serial",
          [&](test_context_t &context) { check_overlap_equivalence_(context, overlap_serial); });
    check("test_overlap_safety_serial", [&] { check_overlap_safety_(overlap_serial); });

    constexpr substrings_tier_t substrings_serial {
        .init = sz_substrings_engine_init_serial,
        .counts = sz_substrings_counts_serial,
        .find = sz_substrings_find_serial,
        .replace = sz_substrings_replace_serial,
        .bm25_scores = sz_substrings_bm25_scores_serial,
    };
    check("test_substrings_unit_serial", [&] { check_substrings_unit_(substrings_serial); });
    check("test_substrings_equivalence_serial",
          [&](test_context_t &context) { check_substrings_equivalence_(context, substrings_serial); });

    constexpr utf8_runes_backend_t utf8_runes_serial {"serial", sz_utf8_count_serial, sz_utf8_seek_serial,
                                                      sz_utf8_decode_serial};
    check("test_utf8_runes_unit_serial", [&] { check_utf8_runes_unit_(utf8_runes_serial); });
    check("test_utf8_runes_safety_serial",
          [&](test_context_t &context) { check_utf8_runes_safety_(context, utf8_runes_serial); });

    constexpr utf8_tokens_backend_t utf8_tokens_serial {"serial", sz_utf8_count_serial, sz_utf8_newlines_serial,
                                                        sz_utf8_whitespaces_serial};
    check("test_utf8_tokens_unit_serial", [&] { check_utf8_tokens_unit_(utf8_tokens_serial); });
    check("test_utf8_tokens_safety_serial",
          [&](test_context_t &context) { check_utf8_tokens_safety_(context, utf8_tokens_serial); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_serial {"serial", sz_utf8_delimiters_serial};
    check("test_utf8_delimiters_unit_serial", [&] { check_utf8_delimiters_unit_(utf8_delimiters_serial); });
    check("test_utf8_delimiters_safety_serial",
          [&](test_context_t &context) { check_utf8_delimiters_safety_(context, utf8_delimiters_serial); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_serial {"serial", sz_utf8_wordbreaks_serial};
    check("test_utf8_wordbreaks_unit_serial", [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_serial); });
    check("test_utf8_wordbreaks_safety_serial",
          [&](test_context_t &context) { check_utf8_wordbreaks_safety_(context, utf8_wordbreaks_serial); });
    check("test_utf8_wordbreaks_equivalence_serial", [] { check_utf8_wordbreaks_oracle_(); });

    constexpr utf8_segment_backend_t utf8_graphemes_serial {"serial", sz_utf8_graphemes_serial};
    check("test_utf8_graphemes_unit_serial", [&] { check_utf8_graphemes_unit_(utf8_graphemes_serial); });
    check("test_utf8_graphemes_safety_serial",
          [&](test_context_t &context) { check_utf8_graphemes_safety_(context, utf8_graphemes_serial); });
    check("test_utf8_graphemes_equivalence_serial", [] { check_utf8_graphemes_oracle_(); });

    constexpr utf8_segment_backend_t utf8_sentences_serial {"serial", sz_utf8_sentences_serial};
    check("test_utf8_sentences_unit_serial", [&] { check_utf8_sentences_unit_(utf8_sentences_serial); });
    check("test_utf8_sentences_safety_serial",
          [&](test_context_t &context) { check_utf8_sentences_safety_(context, utf8_sentences_serial); });

    constexpr utf8_segment_backend_t utf8_linebreaks_serial {"serial", sz_utf8_linebreaks_serial};
    check("test_utf8_linebreaks_unit_serial", [&] { check_utf8_linebreaks_unit_(utf8_linebreaks_serial); });
    check("test_utf8_linebreaks_safety_serial",
          [&](test_context_t &context) { check_utf8_linebreaks_safety_(context, utf8_linebreaks_serial); });

    constexpr utf8_norm_kernels_t utf8_norm_serial {sz_utf8_norm_serial, sz_utf8_find_denormalized_serial};
    check("test_utf8_norm_unit_serial", [&] { check_utf8_norm_unit_(utf8_norm_serial); });
    check("test_utf8_norm_safety_serial",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_serial); });

    constexpr utf8_uncased_kernels_t utf8_uncased_serial {sz_utf8_uncased_fold_serial, sz_utf8_uncased_search_serial,
                                                          sz_utf8_uncased_order_serial, sz_utf8_find_cased_serial};
    check("test_utf8_uncased_unit_serial", [&] {
        check_utf8_uncased_unit_(utf8_uncased_serial);
        check_uncased_invariant_reference_();
    });
    check("test_utf8_uncased_safety_serial",
          [&](test_context_t &context) { check_utf8_uncased_safety_(context, utf8_uncased_serial); });

    return check.failures;
}

} // namespace ashvardanian::stringzilla::test
