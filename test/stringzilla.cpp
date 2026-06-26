/**
 *  @brief  Test entry point and template instantiations; registers every per-domain unit and driver.
 *  @file   scripts/test_stringzilla.cpp
 *  @author Ash Vardanian
 *  @date June 16, 2026
 */
#undef NDEBUG // ! Enable all assertions for testing

/**
 *  The Visual C++ run-time library detects incorrect iterator use,
 *  and asserts and displays a dialog box at run time on Windows.
 */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

/**
 *  ! Overload the following with caution.
 *  ! Those parameters must never be explicitly set during releases,
 *  ! but they come handy during development, if you want to validate
 *  ! different ISA-specific implementations.

 #define SZ_USE_WESTMERE 0
 #define SZ_USE_HASWELL 0
 #define SZ_USE_GOLDMONT 0
 #define SZ_USE_SKYLAKE 0
 #define SZ_USE_ICELAKE 0
 #define SZ_USE_NEON 0
 #define SZ_USE_SVE 0
 #define SZ_USE_SVE2 0
 */
#define SZ_USE_MISALIGNED_LOADS 0
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/**
 *  Make sure to include the StringZilla headers before anything else,
 *  to intercept missing `#include` directives and other issues.
 */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h> // We use ASAN API to poison memory addresses
#endif

#include <cassert> // C-style assertions
#include <cstdio>  // `std::printf`
#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::transform`
#include <iterator>      // `std::distance`
#include <map>           // `std::map`
#include <memory>        // `std::allocator`
#include <numeric>       // `std::accumulate`
#include <random>        // `std::random_device`
#include <set>           // `std::set`
#include <sstream>       // `std::ostringstream`
#include <string>        // `std::string` baseline
#include <string_view>   // `std::string_view` baseline
#include <unordered_map> // `std::unordered_map`
#include <unordered_set> // `std::unordered_set`
#include <vector>        // `std::vector`

#if !SZ_IS_CPP11_
#error "This test requires C++11 or later."
#endif

#include "stringzilla.hpp" // `global_random_generator`, `random_string`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using sz::literals::operator""_sv; // for `sz::string_view`
using sz::literals::operator""_bs; // for `sz::byteset`

#if SZ_IS_CPP17_
using namespace std::literals; // for ""sv
#endif

/**
 *  Instantiate all the templates to make the symbols visible and also check
 *  for weird compilation errors on uncommon paths.
 */
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
template class std::basic_string_view<char>;
#endif
template class sz::basic_string_slice<char>;
template class std::basic_string<char>;
template class sz::basic_string<char>;
template class sz::basic_byteset<char>;

template class std::vector<sz::string>;
template class std::map<sz::string, int>;
template class std::unordered_map<sz::string, int>;

template class std::vector<sz::string_view>;
template class std::map<sz::string_view, int>;
template class std::unordered_map<sz::string_view, int>;

int main(int argc, char const **argv) {

    // Let's greet the user nicely
    sz_unused_(argc && argv);
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Hi, dear tester! You look nice today!\n");
    std::printf("- Uses Westmere: %s \n", SZ_USE_WESTMERE ? "yes" : "no");
    std::printf("- Uses Goldmont: %s \n", SZ_USE_GOLDMONT ? "yes" : "no");
    std::printf("- Uses Haswell: %s \n", SZ_USE_HASWELL ? "yes" : "no");
    std::printf("- Uses Goldmont: %s \n", SZ_USE_GOLDMONT ? "yes" : "no");
    std::printf("- Uses Skylake: %s \n", SZ_USE_SKYLAKE ? "yes" : "no");
    std::printf("- Uses Ice Lake: %s \n", SZ_USE_ICELAKE ? "yes" : "no");
    std::printf("- Uses NEON: %s \n", SZ_USE_NEON ? "yes" : "no");
    std::printf("- Uses NEON AES: %s \n", SZ_USE_NEONAES ? "yes" : "no");
    std::printf("- Uses NEON SHA: %s \n", SZ_USE_NEONSHA ? "yes" : "no");
    std::printf("- Uses SVE: %s \n", SZ_USE_SVE ? "yes" : "no");
    std::printf("- Uses SVE2: %s \n", SZ_USE_SVE2 ? "yes" : "no");
    std::printf("- Uses SVE2 AES: %s \n", SZ_USE_SVE2AES ? "yes" : "no");
    std::printf("- Uses WASM SIMD128: %s \n", SZ_USE_V128 ? "yes" : "no");
    std::printf("- Uses WASM relaxed SIMD: %s \n", SZ_USE_V128RELAXED ? "yes" : "no");
    std::printf("- Uses RISC-V RVV: %s \n", SZ_USE_RVV ? "yes" : "no");
    std::printf("- Uses LoongArch LASX: %s \n", SZ_USE_LASX ? "yes" : "no");
    std::printf("- Uses Power VSX: %s \n", SZ_USE_POWERVSX ? "yes" : "no");
    std::printf("- Uses CUDA: %s \n", SZ_USE_CUDA ? "yes" : "no");
    print_test_environment();

    int failures = 0;

    std::printf("\n=== Basic Utilities ===\n");
    failures += run_test("test_arithmetic_unit", test_arithmetic_unit);
    failures += run_test("test_sequence_unit", test_sequence_unit);
    failures += run_test("test_allocator_unit", test_allocator_unit);
    failures += run_test("test_byteset_unit", test_byteset_unit);

    std::printf("\n=== Hashing ===\n");
    failures += run_test("test_hash_unit", test_hash_unit);
    failures += run_test("test_hash_all", test_hash_all);
    failures += run_test("test_hash_multiseed_all", test_hash_multiseed_all);

    std::printf("\n=== Sequence Algorithms ===\n");
    failures += run_test("test_sort_unit", test_sort_unit);
    failures += run_test("test_sort_all", test_sort_all);
    failures += run_test("test_intersect_unit", test_intersect_unit);

    std::printf("\n=== Core APIs ===\n");
    failures += run_test("test_ascii_unit<sz::string>", test_ascii_unit<sz::string>);
    failures += run_test("test_ascii_unit<sz::string_view>", test_ascii_unit<sz::string_view>);
    failures += run_test("test_memory_unit", [] { test_memory_unit(); }); // ! Defaulted arg
    failures += run_test("test_memory_large_unit", test_memory_large_unit);
    failures += run_test("test_memory_all", test_memory_all);
    failures += run_test("test_memory_safety", test_memory_safety);

    std::printf("\n=== STL Compatibility ===\n");
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
    failures += run_test("test_stl_reads_unit<std::string_view>", test_stl_reads_unit<std::string_view>);
#endif
    failures += run_test("test_stl_reads_unit<std::string>", test_stl_reads_unit<std::string>);
    failures += run_test("test_stl_reads_unit<sz::string_view>", test_stl_reads_unit<sz::string_view>);
    failures += run_test("test_stl_reads_unit<sz::string>", test_stl_reads_unit<sz::string>);
    failures += run_test("test_stl_updates_unit<std::string>", test_stl_updates_unit<std::string>);
    failures += run_test("test_stl_updates_unit<sz::string>", test_stl_updates_unit<sz::string>);
    failures += run_test("test_stl_conversions_unit", test_stl_conversions_unit);
    failures += run_test("test_stl_containers_unit", test_stl_containers_unit);

    std::printf("\n=== StringZilla Extensions ===\n");
    failures += run_test("test_extensions_reads_unit<sz::string_view>", test_extensions_reads_unit<sz::string_view>);
    failures += run_test("test_extensions_reads_unit<sz::string>", test_extensions_reads_unit<sz::string>);
    failures += run_test("test_extensions_updates_unit", test_extensions_updates_unit);

    std::printf("\n=== String Class Implementation ===\n");
    failures += run_test("test_string_constructors_unit", test_string_constructors_unit);
    failures += run_test("test_memory_stability_unit(1024)", [] { test_memory_stability_unit(1024); });
    failures += run_test("test_memory_stability_unit(14)", [] { test_memory_stability_unit(14); });
    failures += run_test("test_string_updates_unit", [] { test_string_updates_unit(); }); // ! Defaulted arg

    std::printf("\n=== Search and Comparison ===\n");
    failures += run_test("test_compare_unit", test_compare_unit);
    failures += run_test("test_find_unit", test_find_unit);
    failures += run_test("test_find_all", test_find_all);
    failures += run_test("test_lookup_fuzz", [] { test_lookup_fuzz(); }); // ! Defaulted args
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
    // Overloaded name (clean as-is) - left outside `run_test`, which needs a non-overloaded bare name.
    std::printf("- test_find_misaligned_fuzz...\n");
    test_find_misaligned_fuzz();
#endif

    std::printf("\n=== UTF-8 ===\n");
    failures += run_test("test_utf8_runes_unit", test_utf8_runes_unit);
    failures += run_test("test_utf8_runes_safety", test_utf8_runes_safety);
    failures += run_test("test_utf8_runes_all", test_utf8_runes_all);
    failures += run_test("test_utf8_tokens_unit", test_utf8_tokens_unit);
    failures += run_test("test_utf8_tokens_safety", test_utf8_tokens_safety);
    failures += run_test("test_utf8_tokens_all", test_utf8_tokens_all);
    failures += run_test("test_utf8_words_unit", test_utf8_words_unit);
    failures += run_test("test_utf8_words_rules", test_utf8_words_rules);
    failures += run_test("test_utf8_words_safety", test_utf8_words_safety);
    failures += run_test("test_utf8_words_all", test_utf8_words_all);
    failures += run_test("test_utf8_graphemes_unit", test_utf8_graphemes_unit);
    failures += run_test("test_utf8_graphemes_rules", test_utf8_graphemes_rules);
    failures += run_test("test_utf8_graphemes_safety", test_utf8_graphemes_safety);
    failures += run_test("test_utf8_graphemes_all", test_utf8_graphemes_all);
    failures += run_test("test_utf8_sentences_unit", test_utf8_sentences_unit);
    failures += run_test("test_utf8_sentences_rules", test_utf8_sentences_rules);
    failures += run_test("test_utf8_sentences_safety", test_utf8_sentences_safety);
    failures += run_test("test_utf8_sentences_all", test_utf8_sentences_all);
    failures += run_test("test_utf8_linewraps_unit", test_utf8_linewraps_unit);
    failures += run_test("test_utf8_linewraps_rules", test_utf8_linewraps_rules);
    failures += run_test("test_utf8_linewraps_safety", test_utf8_linewraps_safety);
    failures += run_test("test_utf8_linewraps_all", test_utf8_linewraps_all);
    failures += run_test("test_utf8_norm_unit", test_utf8_norm_unit);
    failures += run_test("test_utf8_norm_safety", test_utf8_norm_safety);
    failures += run_test("test_utf8_norm_all", test_utf8_norm_all);
    failures += run_test("test_utf8_delimiters_unit", test_utf8_delimiters_unit);
    failures += run_test("test_utf8_delimiters_safety", test_utf8_delimiters_safety);
    failures += run_test("test_utf8_delimiters_all", test_utf8_delimiters_all);

    std::printf("\n=== Uncased UTF-8 ===\n");
    failures += run_test("test_uncased_unit", test_uncased_unit);
    failures += run_test("test_uncased_all", test_uncased_all);
    failures += run_test("test_uncased_safety", test_uncased_safety);

    if (failures != 0) {
        std::fprintf(stderr, "\n%d test(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nAll tests passed!\n");
    return 0;
}
