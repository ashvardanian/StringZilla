/**
 *  @file test/main.cpp
 *  @author Ash Vardanian
 *  @date December 21, 2023
 *  @brief Test entry point and template instantiations; registers every per-domain unit and driver.
 */
#undef NDEBUG // ! Enable all assertions for testing

/** The Visual C++ run-time library detects incorrect iterator use, and asserts and displays a
 *  dialog box at run time on Windows. */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

/*  Overload the following with caution. Those parameters must never be explicitly set during
 *  releases, but they come handy during development, to validate different ISA-specific backends:
 *
 *      #define STRINGZILLA_TARGET_WESTMERE 0
 *      #define STRINGZILLA_TARGET_HASWELL 0
 *      #define STRINGZILLA_TARGET_GOLDMONT 0
 *      #define STRINGZILLA_TARGET_SKYLAKE 0
 *      #define STRINGZILLA_TARGET_ICELAKE 0
 *      #define STRINGZILLA_TARGET_NEON 0
 *      #define STRINGZILLA_TARGET_SVE 0
 *      #define STRINGZILLA_TARGET_SVE2 0 */
#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Include the StringZilla headers before anything else, to intercept missing @c #include
 *  directives and other issues. */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h> // We use ASAN API to poison memory addresses
#endif

#include <cstdio>  // `stderr`, `stdout`
#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::transform`
#include <iterator>      // `std::distance`
#include <map>           // `std::map`
#include <memory>        // `std::allocator`
#include <numeric>       // `std::accumulate`
#include <random>        // `std::random_device`
#include <set>           // `std::set`
#include <string>        // `std::string` baseline
#include <string_view>   // `std::string_view` baseline
#include <unordered_map> // `std::unordered_map`
#include <unordered_set> // `std::unordered_set`
#include <vector>        // `std::vector`

#include <fmt/format.h>

#include "harness.hpp" // `read_test_environment`, `run_test`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;
using sz::literals::operator""_sv; // for `sz::string_view_t`
using sz::literals::operator""_bs; // for `sz::byteset_t`

using namespace std::literals; // for ""sv

/*  Instantiate all the templates to make the symbols visible and also check for weird compilation
 *  errors on uncommon paths. */
template class std::basic_string_view<char>;
template class sz::basic_string_slice<char>;
template class std::basic_string<char>;
template class sz::basic_string<>;
template class sz::basic_string_slice<char const>;

template class std::vector<sz::string_t>;
template class std::map<sz::string_t, int>;
template class std::unordered_map<sz::string_t, int>;

template class std::vector<sz::string_view_t>;
template class std::map<sz::string_view_t, int>;
template class std::unordered_map<sz::string_view_t, int>;

int main(int, char const **argv) {
    test_environment_t const environment = read_test_environment(argv[0]);
    install_test_signal_handlers(); // Backtrace on fatal signals + line-buffered stdout for crash localization.
    log_environment();
    print_test_environment(environment);

    std::size_t failures = 0;

    failures += run_test(environment, "test_arithmetic_unit", test_arithmetic_unit);
    failures += run_test(environment, "test_sequence_unit", test_sequence_unit);
    failures += run_test(environment, "test_strings_tape_assign_unit", test_strings_tape_assign_unit);
    failures += run_test(environment, "test_strings_tape_overflow_unit", test_strings_tape_overflow_unit);
    failures += run_test(environment, "test_allocator_unit", test_allocator_unit);
    failures += run_test(environment, "test_byteset_unit", test_byteset_unit);

    failures += run_test(environment, "test_hash_unit", test_hash_unit);
    failures += run_test(environment, "test_hash_all", test_hash_all);
    failures += run_test(environment, "test_hash_multiseed_all", test_hash_multiseed_all);
    failures += run_test(environment, "test_hash_safety", test_hash_safety);

    failures += run_test(environment, "test_cipher_unit", test_cipher_unit);
    failures += run_test(environment, "test_cipher_safety", test_cipher_safety);
    failures += run_test(environment, "test_cipher_all", test_cipher_all);

    failures += run_test(environment, "test_sort_unit", test_sort_unit);
    failures += run_test(environment, "test_sort_reference_equivalence", test_sort_reference_equivalence);
    failures += run_test(environment, "test_sort_all", test_sort_all);
    failures += run_test(environment, "test_sort_safety", test_sort_safety);
    failures += run_test(environment, "test_intersect_unit", test_intersect_unit);
    failures += run_test(environment, "test_intersect_equivalence", test_intersect_equivalence);
    failures += run_test(environment, "test_levenshtein_unit", test_levenshtein_unit);
    failures += run_test(environment, "test_levenshtein_all", test_levenshtein_all);
    failures += run_test(environment, "test_levenshtein_safety", test_levenshtein_safety);
    failures += run_test(environment, "test_overlap_unit", test_overlap_unit);
    failures += run_test(environment, "test_overlap_all", test_overlap_all);
    failures += run_test(environment, "test_overlap_safety", test_overlap_safety);
    failures += run_test(environment, "test_substrings_unit", test_substrings_unit);
    failures += run_test(environment, "test_substrings_all", test_substrings_all);
    failures += run_test(environment, "test_substrings_safety", test_substrings_safety);

    failures += run_test(environment, "test_ascii_unit<sz::string_t>", test_ascii_unit<sz::string_t>);
    failures += run_test(environment, "test_ascii_unit<sz::string_view_t>", test_ascii_unit<sz::string_view_t>);
    failures += run_test(environment, "test_memory_unit", [] { test_memory_unit(); }); // ! Defaulted arg
    failures += run_test(environment, "test_memory_large_unit", test_memory_large_unit);
    failures += run_test(environment, "test_memory_all", test_memory_all);
    failures += run_test(environment, "test_memory_safety", test_memory_safety);

    failures += run_test(environment, "test_stl_reads_unit<std::string_view>", test_stl_reads_unit<std::string_view>);
    failures += run_test(environment, "test_stl_reads_unit<std::string>", test_stl_reads_unit<std::string>);
    failures += run_test(environment, "test_stl_reads_unit<sz::string_view_t>", test_stl_reads_unit<sz::string_view_t>);
    failures += run_test(environment, "test_stl_reads_unit<sz::string_t>", test_stl_reads_unit<sz::string_t>);
    failures += run_test(environment, "test_stl_updates_unit<std::string>", test_stl_updates_unit<std::string>);
    failures += run_test(environment, "test_stl_updates_unit<sz::string_t>", test_stl_updates_unit<sz::string_t>);
    failures += run_test(environment, "test_stl_conversions_unit", test_stl_conversions_unit);
    failures += run_test(environment, "test_stl_containers_unit", test_stl_containers_unit);

    failures += run_test(environment, "test_extensions_reads_unit<sz::string_view_t>",
                         test_extensions_reads_unit<sz::string_view_t>);
    failures += run_test(environment, "test_extensions_reads_unit<sz::string_t>",
                         test_extensions_reads_unit<sz::string_t>);
    failures += run_test(environment, "test_extensions_updates_unit", test_extensions_updates_unit);
    failures += run_test(environment, "test_extensions_ranges_unit", test_extensions_ranges_unit);

    failures += run_test(environment, "test_string_constructors_unit", test_string_constructors_unit);
    failures += run_test(environment, "test_string_reserve_unit", test_string_reserve_unit);
    failures += run_test(environment, "test_memory_stability_equivalence_1024",
                         [](test_context_t &context) { test_memory_stability_equivalence(context, 1024); });
    failures += run_test(environment, "test_memory_stability_equivalence_14",
                         [](test_context_t &context) { test_memory_stability_equivalence(context, 14); });
    failures += run_test(environment, "test_string_updates_equivalence",
                         [](test_context_t &context) { test_string_updates_equivalence(context); }); // ! Defaulted

    failures += run_test(environment, "test_compare_unit", test_compare_unit);
    failures += run_test(environment, "test_find_unit", test_find_unit);
    failures += run_test(environment, "test_find_all", test_find_all);
    failures += run_test(environment, "test_find_safety", test_find_safety);
    failures += run_test(environment, "test_lookup_equivalence",
                         [](test_context_t &context) { test_lookup_equivalence(context); }); // ! Defaulted args
    failures += run_test(environment, "test_find_misaligned_equivalence", test_find_misaligned_equivalence);

    failures += run_test(environment, "test_utf8_runes_unit", test_utf8_runes_unit);
    failures += run_test(environment, "test_utf8_runes_scripts_unit", test_utf8_runes_scripts_unit);
    failures += run_test(environment, "test_utf8_runes_safety", test_utf8_runes_safety);
    failures += run_test(environment, "test_utf8_runes_all", test_utf8_runes_all);
    failures += run_test(environment, "test_utf8_tokens_unit", test_utf8_tokens_unit);
    failures += run_test(environment, "test_utf8_tokens_scripts_unit", test_utf8_tokens_scripts_unit);
    failures += run_test(environment, "test_utf8_tokens_safety", test_utf8_tokens_safety);
    failures += run_test(environment, "test_utf8_tokens_all", test_utf8_tokens_all);
    failures += run_test(environment, "test_utf8_wordbreaks_unit", test_utf8_wordbreaks_unit);
    failures += run_test(environment, "test_utf8_wordbreaks_rules", test_utf8_wordbreaks_rules);
    failures += run_test(environment, "test_utf8_wordbreaks_safety", test_utf8_wordbreaks_safety);
    failures += run_test(environment, "test_utf8_wordbreaks_all", test_utf8_wordbreaks_all);
    failures += run_test(environment, "test_utf8_graphemes_unit", test_utf8_graphemes_unit);
    failures += run_test(environment, "test_utf8_graphemes_rules", test_utf8_graphemes_rules);
    failures += run_test(environment, "test_utf8_graphemes_safety", test_utf8_graphemes_safety);
    failures += run_test(environment, "test_utf8_graphemes_all", test_utf8_graphemes_all);
    failures += run_test(environment, "test_utf8_sentences_unit", test_utf8_sentences_unit);
    failures += run_test(environment, "test_utf8_sentences_rules", test_utf8_sentences_rules);
    failures += run_test(environment, "test_utf8_sentences_safety", test_utf8_sentences_safety);
    failures += run_test(environment, "test_utf8_sentences_all", test_utf8_sentences_all);
    failures += run_test(environment, "test_utf8_linebreaks_unit", test_utf8_linebreaks_unit);
    failures += run_test(environment, "test_utf8_linebreaks_rules", test_utf8_linebreaks_rules);
    failures += run_test(environment, "test_utf8_linebreaks_safety", test_utf8_linebreaks_safety);
    failures += run_test(environment, "test_utf8_linebreaks_all", test_utf8_linebreaks_all);
    failures += run_test(environment, "test_utf8_norm_unit", test_utf8_norm_unit);
    failures += run_test(environment, "test_utf8_norm_safety", test_utf8_norm_safety);
    failures += run_test(environment, "test_utf8_norm_all", test_utf8_norm_all);
    failures += run_test(environment, "test_utf8_delimiters_unit", test_utf8_delimiters_unit);
    failures += run_test(environment, "test_utf8_delimiters_safety", test_utf8_delimiters_safety);
    failures += run_test(environment, "test_utf8_delimiters_all", test_utf8_delimiters_all);

    failures += run_test(environment, "test_uncased_unit", test_uncased_unit);
    failures += run_test(environment, "test_uncased_scripts_unit", test_uncased_scripts_unit);
    failures += run_test(environment, "test_uncased_regressions_unit", test_uncased_regressions_unit);
    failures += run_test(environment, "test_uncased_all", test_uncased_all);
    failures += run_test(environment, "test_uncased_safety", test_uncased_safety);

    if (failures != 0) {
        fmt::println(stderr, "\n{} test(s) failed.", failures);
        return 1;
    }
    fmt::println("\nAll tests passed!");
    return 0;
}
