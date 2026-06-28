/**
 *  @brief  Comparisons, search/find_all/split, misaligned-repetition search, and replacement tests.
 *  @file   scripts/test_find.cpp
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
#include <string>        // Baseline
#include <string_view>   // Baseline
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

#pragma region Helpers

/**
 *  @brief Runs one substring-search known-answer case through the dispatched `sz_find`/`sz_rfind`,
 *         every natively-compiled backend kernel, and the C++ `sz::string_view` wrapper.
 *
 *  Asserts each backend resolves the @p needle to the expected offset within @p haystack, or to
 *  `SZ_NULL_CHAR` when @p expected_offset is `SZ_SIZE_MAX` (the not-found sentinel). The forward
 *  search is checked against `sz_find`, the backward search against `sz_rfind`, so the caller passes
 *  the forward and backward expectations independently.
 *
 *  @param haystack           The text to search within.
 *  @param haystack_length    The number of bytes in @p haystack.
 *  @param needle             The substring to search for.
 *  @param needle_length      The number of bytes in @p needle.
 *  @param forward_offset     The expected offset of the first occurrence, or `SZ_SIZE_MAX` if absent.
 *  @param backward_offset    The expected offset of the last occurrence, or `SZ_SIZE_MAX` if absent.
 */
static void check_find_unit_(                      //
    sz_cptr_t haystack, sz_size_t haystack_length, //
    sz_cptr_t needle, sz_size_t needle_length,     //
    sz_size_t forward_offset, sz_size_t backward_offset) {

    sz_cptr_t const forward_expected = forward_offset == SZ_SIZE_MAX ? SZ_NULL_CHAR : haystack + forward_offset;
    sz_cptr_t const backward_expected = backward_offset == SZ_SIZE_MAX ? SZ_NULL_CHAR : haystack + backward_offset;

    // Dispatched (automatic kernel resolution).
    assert(sz_find(haystack, haystack_length, needle, needle_length) == forward_expected);
    assert(sz_rfind(haystack, haystack_length, needle, needle_length) == backward_expected);

    // Manual propagation to each natively-compiled backend kernel.
    assert(sz_find_serial(haystack, haystack_length, needle, needle_length) == forward_expected);
    assert(sz_rfind_serial(haystack, haystack_length, needle, needle_length) == backward_expected);
#if SZ_USE_WESTMERE
    assert(sz_find_westmere(haystack, haystack_length, needle, needle_length) == forward_expected);
    assert(sz_rfind_westmere(haystack, haystack_length, needle, needle_length) == backward_expected);
#endif
#if SZ_USE_HASWELL
    assert(sz_find_haswell(haystack, haystack_length, needle, needle_length) == forward_expected);
    assert(sz_rfind_haswell(haystack, haystack_length, needle, needle_length) == backward_expected);
#endif
#if SZ_USE_SKYLAKE
    assert(sz_find_skylake(haystack, haystack_length, needle, needle_length) == forward_expected);
    assert(sz_rfind_skylake(haystack, haystack_length, needle, needle_length) == backward_expected);
#endif
#if SZ_USE_NEON
    assert(sz_find_neon(haystack, haystack_length, needle, needle_length) == forward_expected);
    assert(sz_rfind_neon(haystack, haystack_length, needle, needle_length) == backward_expected);
#endif
#if SZ_USE_SVE
    assert(sz_find_sve(haystack, haystack_length, needle, needle_length) == forward_expected);
#endif

    // The C++ `sz::string_view` wrapper resolves to the same offsets.
    sz::string_view const haystack_view(haystack, haystack_length);
    sz::string_view const needle_view(needle, needle_length);
    assert(haystack_view.find(needle_view) == (forward_offset == SZ_SIZE_MAX ? sz::string_view::npos : forward_offset));
    assert(haystack_view.rfind(needle_view) ==
           (backward_offset == SZ_SIZE_MAX ? sz::string_view::npos : backward_offset));
}

#pragma endregion // Helpers

#pragma region Unit

/**
 *  @brief Known-answer + coverage for the search & comparison family on simple, hand-verifiable inputs.
 *
 *  Begins with known-answer vectors exercising each function through the dispatched C API
 *  (automatic kernel resolution), through the natively-compiled backend kernels directly (manual
 *  propagation to a specific kernel), and - where it applies - through the C++ `sz::string_view`
 *  wrappers, so a regression that the serial-vs-SIMD agreement tests would miss - because both share a
 *  wrong constant - is still caught against an external ground truth. The remainder covers the string
 *  class search methods (`find`, `find_first_of`, `find_all`, `split`, …) over haystacks and needles of
 *  different lengths and character-sets.
 *
 *  The exact-substring `sz_find`/`sz_rfind` and the single-byte `sz_find_byte`/`sz_rfind_byte` ship serial,
 *  westmere, haswell, and skylake backends (no icelake). The byteset search ships serial, haswell, and
 *  icelake backends. The comparison family `sz_order`/`sz_equal` ships serial, haswell, and skylake.
 */
void test_find_unit() {
    std::printf("  - testing search & comparison known-answer vectors...\n");

    char const *hello = "hello world";
    sz_size_t const hello_length = (sz_size_t)std::strlen(hello); // 11 bytes

    // `sz_find` / `sz_rfind`: the substring "o" occurs at offsets 4 and 7; the multi-byte needle "wor"
    // sits at offset 6; and a missing needle "xyz" yields `SZ_NULL_CHAR` (encoded as the `SZ_SIZE_MAX`
    // not-found sentinel). Each case is checked across every backend through `check_find_unit_`.
    check_find_unit_(hello, hello_length, "o", 1, 4, 7);                       // Single-byte needle
    check_find_unit_(hello, hello_length, "wor", 3, 6, 6);                     // Present multi-byte needle
    check_find_unit_(hello, hello_length, "xyz", 3, SZ_SIZE_MAX, SZ_SIZE_MAX); // Missing needle

    // `sz_find_byte` / `sz_rfind_byte` in isolation: the byte 'l' occurs at offsets 2, 3, and 9.
    assert(sz_find_byte(hello, hello_length, "l") == hello + 2);           // Dispatched (automatic kernel)
    assert(sz_rfind_byte(hello, hello_length, "l") == hello + 9);          // Dispatched (automatic kernel)
    assert(sz_find_byte_serial(hello, hello_length, "l") == hello + 2);    // Manual propagation to the serial kernel
    assert(sz_rfind_byte_serial(hello, hello_length, "l") == hello + 9);   // Manual propagation to the serial kernel
    assert(sz_find_byte(hello, hello_length, "z") == SZ_NULL_CHAR);        // Missing byte
    assert(sz_rfind_byte(hello, hello_length, "z") == SZ_NULL_CHAR);       // Missing byte
    assert(sz_find_byte_serial(hello, hello_length, "z") == SZ_NULL_CHAR); // Missing byte, serial kernel
#if SZ_USE_WESTMERE
    assert(sz_find_byte_westmere(hello, hello_length, "l") == hello + 2);
    assert(sz_rfind_byte_westmere(hello, hello_length, "l") == hello + 9);
#endif
#if SZ_USE_HASWELL
    assert(sz_find_byte_haswell(hello, hello_length, "l") == hello + 2);
    assert(sz_rfind_byte_haswell(hello, hello_length, "l") == hello + 9);
#endif
#if SZ_USE_SKYLAKE
    assert(sz_find_byte_skylake(hello, hello_length, "l") == hello + 2);
    assert(sz_rfind_byte_skylake(hello, hello_length, "l") == hello + 9);
#endif

    // `sz_find_byteset` / `sz_rfind_byteset`: a set of vowels {a, e, i, o, u} first hits 'e' at offset 1
    // and last hits 'o' at offset 7 in "hello world".
    sz_byteset_t vowels;
    sz_byteset_init(&vowels);
    sz_byteset_add(&vowels, 'a');
    sz_byteset_add(&vowels, 'e');
    sz_byteset_add(&vowels, 'i');
    sz_byteset_add(&vowels, 'o');
    sz_byteset_add(&vowels, 'u');
    assert(sz_find_byteset(hello, hello_length, &vowels) == hello + 1);  // Dispatched (automatic kernel)
    assert(sz_rfind_byteset(hello, hello_length, &vowels) == hello + 7); // Dispatched (automatic kernel)
    assert(sz_find_byteset_serial(hello, hello_length, &vowels) ==
           hello + 1); // Manual propagation to the serial kernel
    assert(sz_rfind_byteset_serial(hello, hello_length, &vowels) ==
           hello + 7); // Manual propagation to the serial kernel
    // A set with none of the present bytes returns `SZ_NULL_CHAR`.
    sz_byteset_t digits;
    sz_byteset_init(&digits);
    sz_byteset_add(&digits, '0');
    sz_byteset_add(&digits, '9');
    assert(sz_find_byteset(hello, hello_length, &digits) == SZ_NULL_CHAR);        // No digit present
    assert(sz_find_byteset_serial(hello, hello_length, &digits) == SZ_NULL_CHAR); // No digit present, serial kernel

    // `sz_order` / `sz_equal`: lexicographic ordering and byte-equality on hand-verifiable pairs.
    assert(sz_order("abc", 3, "abc", 3) == sz_equal_k);       // Equal strings
    assert(sz_order("abc", 3, "abd", 3) == sz_less_k);        // Differ in the last byte
    assert(sz_order("abd", 3, "abc", 3) == sz_greater_k);     // Differ in the last byte
    assert(sz_order("ab", 2, "abc", 3) == sz_less_k);         // Prefix orders before the longer string
    assert(sz_order("abc", 3, "ab", 2) == sz_greater_k);      // Longer string orders after its prefix
    assert(sz_equal("abc", "abc", 3) == sz_true_k);           // Identical bytes
    assert(sz_equal("abc", "abd", 3) == sz_false_k);          // Differing bytes
    assert(sz_order_serial("abc", 3, "abd", 3) == sz_less_k); // Manual propagation to the serial kernel
    assert(sz_equal_serial("abc", "abc", 3) == sz_true_k);    // Manual propagation to the serial kernel
    assert(sz_equal_serial("abc", "abd", 3) == sz_false_k);   // Manual propagation to the serial kernel
#if SZ_USE_HASWELL
    assert(sz_order_haswell("abc", 3, "abd", 3) == sz_less_k);
    assert(sz_equal_haswell("abc", "abc", 3) == sz_true_k);
    assert(sz_equal_haswell("abc", "abd", 3) == sz_false_k);
#endif
#if SZ_USE_SKYLAKE
    assert(sz_order_skylake("abc", 3, "abd", 3) == sz_less_k);
    assert(sz_equal_skylake("abc", "abc", 3) == sz_true_k);
    assert(sz_equal_skylake("abc", "abd", 3) == sz_false_k);
#endif
    // And the same orderings through the C++ `sz::string_view` comparison operators.
    assert("abc"_sv == "abc"_sv); // Equality operator
    assert("abc"_sv != "abd"_sv); // Inequality operator
    assert("abc"_sv < "abd"_sv);  // Strictly-less operator
    assert("abd"_sv > "abc"_sv);  // Strictly-greater operator
    assert("ab"_sv < "abc"_sv);   // Prefix orders before the longer string

    // Searching for a set of characters
    assert(sz::string_view("a").find_first_of("az") == 0);
    assert(sz::string_view("a").find_last_of("az") == 0);
    assert(sz::string_view("a").find_first_of("xz") == sz::string_view::npos);
    assert(sz::string_view("a").find_last_of("xz") == sz::string_view::npos);

    assert(sz::string_view("a").find_first_not_of("xz") == 0);
    assert(sz::string_view("a").find_last_not_of("xz") == 0);
    assert(sz::string_view("a").find_first_not_of("az") == sz::string_view::npos);
    assert(sz::string_view("a").find_last_not_of("az") == sz::string_view::npos);

    assert(sz::string_view("aXbYaXbY").find_first_of("XY") == 1);
    assert(sz::string_view("axbYaxbY").find_first_of("Y") == 3);
    assert(sz::string_view("YbXaYbXa").find_last_of("XY") == 6);
    assert(sz::string_view("YbxaYbxa").find_last_of("Y") == 4);
    assert(sz::string_view(sz::base64(), sizeof(sz::base64())).find_first_of("_") == sz::string_view::npos);
    assert(sz::string_view(sz::base64(), sizeof(sz::base64())).find_first_of("+") == 62);
    assert(sz::string_view(sz::ascii_printables(), sizeof(sz::ascii_printables())).find_first_of("~") !=
           sz::string_view::npos);

    assert("aabaa"_sv.remove_prefix("a") == "abaa");
    assert("aabaa"_sv.remove_suffix("a") == "aaba");
    assert("aabaa"_sv.lstrip("a"_bs) == "baa");
    assert("aabaa"_sv.rstrip("a"_bs) == "aab");
    assert("aabaa"_sv.strip("a"_bs) == "b");

    // Check more advanced composite operations
    assert("abbccc"_sv.partition('b').before.size() == 1);
    assert("abbccc"_sv.partition("bb").before.size() == 1);
    assert("abbccc"_sv.partition("bb").match.size() == 2);
    assert("abbccc"_sv.partition("bb").after.size() == 3);
    assert("abbccc"_sv.partition("bb").before == "a");
    assert("abbccc"_sv.partition("bb").match == "bb");
    assert("abbccc"_sv.partition("bb").after == "ccc");
    assert("abb ccc"_sv.partition(sz::whitespaces_set()).after == "ccc");

    // Check ranges of search matches
    assert("hello"_sv.find_all("l").size() == 2);
    assert("hello"_sv.rfind_all("l").size() == 2);

    assert(""_sv.find_all(".", sz::include_overlaps_type {}).size() == 0);
    assert(""_sv.find_all(".", sz::exclude_overlaps_type {}).size() == 0);
    assert("."_sv.find_all(".", sz::include_overlaps_type {}).size() == 1);
    assert("."_sv.find_all(".", sz::exclude_overlaps_type {}).size() == 1);
    assert(".."_sv.find_all(".", sz::include_overlaps_type {}).size() == 2);
    assert(".."_sv.find_all(".", sz::exclude_overlaps_type {}).size() == 2);
    assert(""_sv.rfind_all(".", sz::include_overlaps_type {}).size() == 0);
    assert(""_sv.rfind_all(".", sz::exclude_overlaps_type {}).size() == 0);
    assert("."_sv.rfind_all(".", sz::include_overlaps_type {}).size() == 1);
    assert("."_sv.rfind_all(".", sz::exclude_overlaps_type {}).size() == 1);
    assert(".."_sv.rfind_all(".", sz::include_overlaps_type {}).size() == 2);
    assert(".."_sv.rfind_all(".", sz::exclude_overlaps_type {}).size() == 2);

    assert("a.b.c.d"_sv.find_all(".").size() == 3);
    assert("a.,b.,c.,d"_sv.find_all(".,").size() == 3);
    assert("a.,b.,c.,d"_sv.rfind_all(".,").size() == 3);
    assert("a.b,c.d"_sv.find_all(".,"_bs).size() == 3);
    assert("a...b...c"_sv.rfind_all("..").size() == 4);
    assert("a...b...c"_sv.rfind_all("..", sz::include_overlaps_type {}).size() == 4);
    assert("a...b...c"_sv.rfind_all("..", sz::exclude_overlaps_type {}).size() == 2);

    let_assert(auto finds = "a.b.c"_sv.find_all("abcd"_bs).template to<std::vector<std::string>>(),
               finds.size() == 3 && finds[0] == "a");
    let_assert(auto rfinds = "a.b.c"_sv.rfind_all("abcd"_bs).template to<std::vector<std::string>>(),
               rfinds.size() == 3 && rfinds[0] == "c");

    // Test propagating strings and their non-owning views into temporary ranges and iterators
    assert(sz::find_all("abc"_sv, "b"_sv).size() == 1);
    assert(sz::find_all("hello"_sv, "l"_sv).size() == 2);
    assert(sz::rfind_all("abc"_sv, "b"_sv).size() == 1);

    {
        sz::string h("abc"), n("b");
        assert(sz::find_all(h, n).size() == 1);
    }
    {
        sz::string h("hello"), n("l");
        assert(sz::find_all(h, n).size() == 2);
    }
    {
        sz::string h("abc"), n("b");
        assert(sz::rfind_all(h, n).size() == 1);
    }

    assert(sz::find_all(sz::string("abc"), sz::string("b")).size() == 1);
    assert(sz::find_all(sz::string("hello"), sz::string("l")).size() == 2);
    assert(sz::rfind_all(sz::string("abc"), sz::string("b")).size() == 1);

    // Check splitting - the inverse of `find_all` ranges
    let_assert(auto splits = ".a..c."_sv.split("."_bs).template to<std::vector<std::string>>(),
               splits.size() == 5 && splits[0] == "" && splits[1] == "a" && splits[4] == "");
    let_assert(auto line_splits = "line1\nline2\nline3"_sv.split("line3").template to<std::vector<std::string>>(),
               line_splits.size() == 2 && line_splits[0] == "line1\nline2\n" && line_splits[1] == "");

    assert(""_sv.split(".").size() == 1);
    assert(""_sv.rsplit(".").size() == 1);

    assert("hello"_sv.split("l").size() == 3);
    assert("hello"_sv.rsplit("l").size() == 3);
    assert(*advanced("hello"_sv.split("l").begin(), 0) == "he");
    assert(*advanced("hello"_sv.rsplit("l").begin(), 0) == "o");
    assert(*advanced("hello"_sv.split("l").begin(), 1) == "");
    assert(*advanced("hello"_sv.rsplit("l").begin(), 1) == "");
    assert(*advanced("hello"_sv.split("l").begin(), 2) == "o");
    assert(*advanced("hello"_sv.rsplit("l").begin(), 2) == "he");

    assert("a.b.c.d"_sv.split(".").size() == 4);
    assert("a.b.c.d"_sv.rsplit(".").size() == 4);
    assert(*("a.b.c.d"_sv.split(".").begin()) == "a");
    assert(*("a.b.c.d"_sv.rsplit(".").begin()) == "d");
    assert(*advanced("a.b.c.d"_sv.split(".").begin(), 1) == "b");
    assert(*advanced("a.b.c.d"_sv.rsplit(".").begin(), 1) == "c");
    assert(*advanced("a.b.c.d"_sv.split(".").begin(), 3) == "d");
    assert(*advanced("a.b.c.d"_sv.rsplit(".").begin(), 3) == "a");
    assert("a.b.,c,d"_sv.split(".,").size() == 2);
    assert("a.b,c.d"_sv.split(".,"_bs).size() == 4);

    let_assert(auto rsplits = ".a..c."_sv.rsplit("."_bs).template to<std::vector<std::string>>(),
               rsplits.size() == 5 && rsplits[0] == "" && rsplits[1] == "c" && rsplits[4] == "");
}

/**
 *  @brief Tests the correctness of the string class comparison methods, such as `compare` and `operator==`.
 */
void test_compare_unit() {
    // Comparing relative order of the strings
    assert("a"_sv.compare("a") == 0);
    assert("a"_sv.compare("ab") == -1);
    assert("ab"_sv.compare("a") == 1);
    assert("a"_sv.compare("a\0"_sv) == -1);
    assert("a\0"_sv.compare("a") == 1);
    assert("a\0"_sv.compare("a\0"_sv) == 0);
    assert("a"_sv == "a"_sv);
    assert("a"_sv != "a\0"_sv);
    assert("a\0"_sv == "a\0"_sv);
}

#pragma endregion // Unit

#pragma region Equivalence

/** @brief Wraps a substring-search backend (find or rfind) by its kernel pointer. */
template <sz_find_t kernel_>
struct search_from_sz_ {
    sz_cptr_t operator()(sz_cptr_t haystack, sz_size_t haystack_length, //
                         sz_cptr_t needle, sz_size_t needle_length) const noexcept {
        return kernel_(haystack, haystack_length, needle, needle_length);
    }
};

/** @brief Wraps a byteset-search backend (find or rfind) by its kernel pointer. */
template <sz_find_byteset_t kernel_>
struct byteset_search_from_sz_ {
    sz_cptr_t operator()(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *byteset) const noexcept {
        return kernel_(haystack, haystack_length, byteset);
    }
};

/**
 *  @brief Cross-checks two substring-search backends of the @b same operation (both forward or both
 *         backward) against each other on random and hand-picked edge-case inputs.
 *
 *  The candidate must resolve every needle to the identical pointer the reference does, or both must
 *  return `SZ_NULL`. Each haystack is replayed at every sub-cacheline alignment via
 *  `for_each_cacheline_offset_`, so a needle straddling a 64-byte boundary is always exercised.
 *
 *  @param reference  Reference search backend wrapper (the serial kernel in the drivers).
 *  @param candidate  Candidate search backend wrapper to validate against the reference.
 *  @param inputs     Number of random haystack/needle pairs to fuzz beyond the structured edge cases.
 */
template <typename reference_, typename candidate_>
void test_search_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    // Replays one haystack/needle pair at every intra-cacheline alignment and compares the backends.
    auto compare_on = [&](std::string const &haystack_pattern, std::string const &needle) {
        for_each_cacheline_offset_(
            haystack_pattern.size(), [&](sz_ptr_t haystack, [[maybe_unused]] std::size_t offset) {
                std::memcpy(haystack, haystack_pattern.data(), haystack_pattern.size());
                sz_size_t const haystack_length = (sz_size_t)haystack_pattern.size();
                sz_size_t const needle_length = (sz_size_t)needle.size();

                sz_cptr_t const result_reference = reference(haystack, haystack_length, needle.data(), needle_length);
                sz_cptr_t const result_candidate = candidate(haystack, haystack_length, needle.data(), needle_length);
                assert(result_reference == result_candidate);
            });
    };

    // Hand-picked edge cases: empty needle, not-found, needle at start, needle at end, needle == haystack,
    // repeated occurrences, and an embedded NUL byte.
    compare_on("hello world", "");                                 // Empty needle
    compare_on("hello world", "xyz");                              // Not found
    compare_on("hello world", "hello");                            // Needle at the start
    compare_on("hello world", "world");                            // Needle at the end
    compare_on("hello world", "hello world");                      // Needle equals the haystack
    compare_on("abababab", "ab");                                  // Repeated occurrences
    compare_on(std::string("a\0bc\0a", 6), std::string("\0a", 2)); // Embedded NUL byte

    // Random haystacks and needles of assorted lengths.
    for (sz_size_t iteration = 0; iteration != scale_iterations(inputs); ++iteration) {
        std::size_t const haystack_length = std::uniform_int_distribution<std::size_t>(0,
                                                                                       200)(global_random_generator());
        std::size_t const needle_length = std::uniform_int_distribution<std::size_t>(
            0, haystack_length + 4)(global_random_generator());
        std::string haystack(haystack_length, '\0');
        std::string needle(needle_length, '\0');
        // A small alphabet makes spurious and overlapping matches likely, stressing the kernels.
        randomize_string(&haystack[0], haystack_length, "abc", 3);
        randomize_string(&needle[0], needle_length, "abc", 3);
        compare_on(haystack, needle);
    }
}

/**
 *  @brief Cross-checks two byteset-search backends of the @b same operation (both forward or both
 *         backward) against each other on random and hand-picked edge-case inputs.
 *
 *  The candidate must resolve every byteset to the identical pointer the reference does, or both must
 *  return `SZ_NULL`. Each haystack is replayed at every sub-cacheline alignment via
 *  `for_each_cacheline_offset_`, so a match straddling a 64-byte boundary is always exercised.
 *
 *  @param reference  Reference byteset-search backend wrapper (the serial kernel in the drivers).
 *  @param candidate  Candidate byteset-search backend wrapper to validate against the reference.
 *  @param inputs     Number of random haystacks to fuzz beyond the structured edge cases.
 */
template <typename reference_, typename candidate_>
void test_byteset_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    // The byteset of ASCII vowels, used for the hand-picked structured cases.
    sz_byteset_t vowels;
    sz_byteset_init(&vowels);
    sz_byteset_add(&vowels, 'a');
    sz_byteset_add(&vowels, 'e');
    sz_byteset_add(&vowels, 'i');
    sz_byteset_add(&vowels, 'o');
    sz_byteset_add(&vowels, 'u');

    // Replays one haystack at every intra-cacheline alignment and compares the backends.
    auto compare_on = [&](std::string const &haystack_pattern, sz_byteset_t const &byteset) {
        for_each_cacheline_offset_(
            haystack_pattern.size(), [&](sz_ptr_t haystack, [[maybe_unused]] std::size_t offset) {
                std::memcpy(haystack, haystack_pattern.data(), haystack_pattern.size());
                sz_size_t const haystack_length = (sz_size_t)haystack_pattern.size();

                sz_cptr_t const result_reference = reference(haystack, haystack_length, &byteset);
                sz_cptr_t const result_candidate = candidate(haystack, haystack_length, &byteset);
                assert(result_reference == result_candidate);
            });
    };

    // Hand-picked edge cases: empty haystack, no member present, member at start, member at end,
    // all members, repeated members, and an embedded NUL byte.
    compare_on("", vowels);                         // Empty haystack
    compare_on("xyz wrld", vowels);                 // No member present
    compare_on("apple", vowels);                    // Member at the start
    compare_on("xyzo", vowels);                     // Member at the end
    compare_on("aeiou", vowels);                    // Every byte is a member
    compare_on("aaeeii", vowels);                   // Repeated members
    compare_on(std::string("x\0aey\0", 6), vowels); // Embedded NUL byte

    // Random haystacks of assorted lengths against a random byteset.
    for (sz_size_t iteration = 0; iteration != scale_iterations(inputs); ++iteration) {
        sz_byteset_t random_byteset;
        sz_byteset_init(&random_byteset);
        std::size_t const members = std::uniform_int_distribution<std::size_t>(0, 16)(global_random_generator());
        for (std::size_t member = 0; member != members; ++member)
            sz_byteset_add(&random_byteset,
                           (sz_u8_t)std::uniform_int_distribution<int>(0, 255)(global_random_generator()));

        std::size_t const haystack_length = std::uniform_int_distribution<std::size_t>(0,
                                                                                       200)(global_random_generator());
        std::string haystack(haystack_length, '\0');
        randomize_string(&haystack[0], haystack_length);
        compare_on(haystack, random_byteset);
    }
}

#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)

/**
 *  @brief Evaluates the correctness of a "matcher", searching for all the occurrences of the @p needle_stl
 *         in a haystack formed of @p haystack_pattern repeated from one to `max_repeats` times.
 *
 *  @param haystack_pattern The pattern repeated to synthesize the haystack.
 *  @param needle_stl The needle searched for, also used as the STL ground-truth needle.
 *  @param misalignment The number of bytes to misalign the haystack within the cacheline.
 */
template <typename stl_matcher_, typename sz_matcher_>
void test_find_misaligned_fuzz(std::string_view haystack_pattern, std::string_view needle_stl,
                               std::size_t misalignment) {
    std::size_t const max_repeats = scale_iterations(128);

    // Allocate a buffer to store the haystack with enough padding to mis-align it.
    std::size_t haystack_buffer_length = max_repeats * haystack_pattern.size() + 2 * SZ_CACHE_LINE_WIDTH;
    std::vector<char> haystack_buffer(haystack_buffer_length, 'x');
    char *haystack = haystack_buffer.data();

    // Skip the misaligned part.
    while (reinterpret_cast<std::uintptr_t>(haystack) % SZ_CACHE_LINE_WIDTH != misalignment) ++haystack;

    /// Helper container to store the offsets of the matches. Useful during debugging :)
    std::vector<std::size_t> offsets_stl, offsets_sz;

    for (std::size_t repeats = 0; repeats != max_repeats; ++repeats) {
        std::size_t haystack_length = (repeats + 1) * haystack_pattern.size();

#if defined(__SANITIZE_ADDRESS__)
        // Let's manually poison the prefix and the suffix.
        std::size_t poisoned_prefix_length = haystack - haystack_buffer.data();
        std::size_t poisoned_suffix_length = haystack_buffer_length - haystack_length - poisoned_prefix_length;
        ASAN_POISON_MEMORY_REGION(haystack_buffer.data(), poisoned_prefix_length);
        ASAN_POISON_MEMORY_REGION(haystack + haystack_length, poisoned_suffix_length);
#endif

        // Append the new repetition to our buffer.
        std::memcpy(haystack + repeats * haystack_pattern.size(), haystack_pattern.data(), haystack_pattern.size());

        // Convert to string views
        auto haystack_stl = std::string_view(haystack, haystack_length);
        auto haystack_sz = sz::string_view(haystack, haystack_length);
        auto needle_sz = sz::string_view(needle_stl.data(), needle_stl.size());

        // Wrap into ranges
        auto matches_stl = stl_matcher_(haystack_stl, {needle_stl});
        auto matches_sz = sz_matcher_(haystack_sz, {needle_sz});
        auto begin_stl = matches_stl.begin();
        auto begin_sz = matches_sz.begin();
        auto end_stl = matches_stl.end();
        auto end_sz = matches_sz.end();
        auto count_stl = std::distance(begin_stl, end_stl);
        auto count_sz = std::distance(begin_sz, end_sz);

        // To simplify debugging, let's first export all the match offsets, and only then compare them
        std::transform(begin_stl, end_stl, std::back_inserter(offsets_stl),
                       [&](auto const &match) { return match.data() - haystack_stl.data(); });
        std::transform(begin_sz, end_sz, std::back_inserter(offsets_sz),
                       [&](auto const &match) { return match.data() - haystack_sz.data(); });
        auto print_all_matches = [&]() {
            std::printf("Breakdown of found matches:\n");
            std::printf("- STL (%zu): ", offsets_stl.size());
            for (auto offset : offsets_stl) std::printf("%zu ", offset);
            std::printf("\n");
            std::printf("- StringZilla (%zu): ", offsets_sz.size());
            for (auto offset : offsets_sz) std::printf("%zu ", offset);
            std::printf("\n");
        };

        // Compare results
        for (std::size_t match_idx = 0; begin_stl != end_stl && begin_sz != end_sz;
             ++begin_stl, ++begin_sz, ++match_idx) {
            auto match_stl = *begin_stl;
            auto match_sz = *begin_sz;
            if (match_stl.data() != match_sz.data()) {
                std::printf("Mismatch at index #%zu: %zu != %zu\n", match_idx, match_stl.data() - haystack_stl.data(),
                            match_sz.data() - haystack_sz.data());
                print_all_matches();
                assert(false);
            }
        }

        // If one range is not finished, assert failure
        if (count_stl != count_sz) {
            print_all_matches();
            assert(false);
        }
        assert(begin_stl == end_stl && begin_sz == end_sz);

        offsets_stl.clear();
        offsets_sz.clear();

#if defined(__SANITIZE_ADDRESS__)
        // Don't forget to manually unpoison the prefix and the suffix.
        ASAN_UNPOISON_MEMORY_REGION(haystack_buffer.data(), poisoned_prefix_length);
        ASAN_UNPOISON_MEMORY_REGION(haystack + haystack_length, poisoned_suffix_length);
#endif
    }
}

/**
 *  @brief Evaluates the correctness of a "matcher", searching for all the occurrences of the @p needle_stl,
 *         as a substring, as a set of allowed characters, or as a set of disallowed characters, in a haystack.
 *
 *  @param haystack_pattern The pattern repeated to synthesize the haystack.
 *  @param needle_stl The needle searched for, also used as the STL ground-truth needle.
 *  @param misalignment The number of bytes to misalign the haystack within the cacheline.
 */
void test_find_misaligned_fuzz(std::string_view haystack_pattern, std::string_view needle_stl,
                               std::size_t misalignment) {

    test_find_misaligned_fuzz<                                                       //
        sz::find_matches_view<std::string_view, sz::matcher_find<std::string_view>>, //
        sz::find_matches_view<sz::string_view, sz::matcher_find<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_find_misaligned_fuzz<                                                         //
        sz::rfind_matches_view<std::string_view, sz::matcher_rfind<std::string_view>>, //
        sz::rfind_matches_view<sz::string_view, sz::matcher_rfind<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_find_misaligned_fuzz<                                                                //
        sz::find_matches_view<std::string_view, sz::matcher_find_first_of<std::string_view>>, //
        sz::find_matches_view<sz::string_view, sz::matcher_find_first_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_find_misaligned_fuzz<                                                                //
        sz::rfind_matches_view<std::string_view, sz::matcher_find_last_of<std::string_view>>, //
        sz::rfind_matches_view<sz::string_view, sz::matcher_find_last_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_find_misaligned_fuzz<                                                                    //
        sz::find_matches_view<std::string_view, sz::matcher_find_first_not_of<std::string_view>>, //
        sz::find_matches_view<sz::string_view, sz::matcher_find_first_not_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_find_misaligned_fuzz<                                                                    //
        sz::rfind_matches_view<std::string_view, sz::matcher_find_last_not_of<std::string_view>>, //
        sz::rfind_matches_view<sz::string_view, sz::matcher_find_last_not_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);
}

/**
 *  @brief Replays the misaligned-repetition search across a fixed sweep of intra-cacheline offsets.
 *
 *  @param haystack_pattern The pattern repeated to synthesize the haystack.
 *  @param needle_stl The needle searched for, also used as the STL ground-truth needle.
 */
void test_find_misaligned_fuzz(std::string_view haystack_pattern, std::string_view needle_stl) {
    test_find_misaligned_fuzz(haystack_pattern, needle_stl, 0);
    test_find_misaligned_fuzz(haystack_pattern, needle_stl, 1);
    test_find_misaligned_fuzz(haystack_pattern, needle_stl, 2);
    test_find_misaligned_fuzz(haystack_pattern, needle_stl, 3);
    test_find_misaligned_fuzz(haystack_pattern, needle_stl, 63);
    test_find_misaligned_fuzz(haystack_pattern, needle_stl, 24);
    test_find_misaligned_fuzz(haystack_pattern, needle_stl, 33);
}

/**
 *  @brief Extensively tests the correctness of the string class search methods, such as `find` and `find_first_of`.
 *         Covers different alignment cases within a cache line, repetitive patterns, and overlapping matches.
 */
void test_find_misaligned_fuzz() {
    // When haystack is only formed of needles:
    test_find_misaligned_fuzz("a", "a");
    test_find_misaligned_fuzz("ab", "ab");
    test_find_misaligned_fuzz("abc", "abc");
    test_find_misaligned_fuzz("abcd", "abcd");
    test_find_misaligned_fuzz({sz::base64(), sizeof(sz::base64())}, {sz::base64(), sizeof(sz::base64())});
    test_find_misaligned_fuzz({sz::ascii_lowercase(), sizeof(sz::ascii_lowercase())},
                              {sz::ascii_lowercase(), sizeof(sz::ascii_lowercase())});
    test_find_misaligned_fuzz({sz::ascii_printables(), sizeof(sz::ascii_printables())},
                              {sz::ascii_printables(), sizeof(sz::ascii_printables())});

    // When we are dealing with NULL characters inside the string
    test_find_misaligned_fuzz("\0", "\0");
    test_find_misaligned_fuzz("a\0", "a\0");
    test_find_misaligned_fuzz("ab\0", "ab");
    test_find_misaligned_fuzz("ab\0", "ab\0");
    test_find_misaligned_fuzz("abc\0", "abc");
    test_find_misaligned_fuzz("abc\0", "abc\0");
    test_find_misaligned_fuzz("abcd\0", "abcd");

    // When searching for all-null needles in a haystack with no null bytes.
    // This exercises the SIMD tail path where masked-off lanes are zeroed:
    // if the needle characters are also zero, spurious matches appear at
    // invalid offsets beyond the haystack, causing OOB reads.
    test_find_misaligned_fuzz("a", {"\0\0", 2});
    test_find_misaligned_fuzz("a", {"\0\0\0", 3});
    test_find_misaligned_fuzz("a", {"\0\0\0\0", 4});
    test_find_misaligned_fuzz("a", {"\0\0\0\0\0", 5});
    test_find_misaligned_fuzz("abcd", {"\0\0", 2});
    test_find_misaligned_fuzz("abcd", {"\0\0\0\0", 4});

    // When haystack is formed of equidistant needles:
    test_find_misaligned_fuzz("ab", "a");
    test_find_misaligned_fuzz("abc", "a");
    test_find_misaligned_fuzz("abcd", "a");

    // When matches occur in between pattern words:
    test_find_misaligned_fuzz("ab", "ba");
    test_find_misaligned_fuzz("abc", "ca");
    test_find_misaligned_fuzz("abcd", "da");

    // Examples targeted exactly against the Raita heuristic,
    // which matches the first, the last, and the middle characters with SIMD.
    test_find_misaligned_fuzz("aaabbccc", "aaabbccc");
    test_find_misaligned_fuzz("axabbcxc", "aaabbccc");
    test_find_misaligned_fuzz("axabbcxcaaabbccc", "aaabbccc");
}

#endif

/**
 *  @brief Evaluates the correctness of look-up table transforms using random lookup tables.
 *
 *  @param lookup_tables_to_try The number of random lookup tables to try.
 *  @param slices_per_table The number of random inputs to test per lookup table.
 */
void test_lookup_fuzz(std::size_t lookup_tables_to_try, std::size_t slices_per_table) {

    std::string body, transformed;
    body.resize(1024 * 1024); // 1MB
    transformed.resize(1024 * 1024);
    std::generate(body.begin(), body.end(), []() { return (char)(std::rand() % 256); });

    for (std::size_t lookup_table_variation = 0; lookup_table_variation != lookup_tables_to_try;
         ++lookup_table_variation) {
        sz::look_up_table lut;
        for (std::size_t i = 0; i < 256; ++i) lut[(char)i] = (char)(std::rand() % 256);

        for (std::size_t slice_idx = 0; slice_idx != slices_per_table; ++slice_idx) {
            std::size_t slice_offset = std::rand() % (body.length());
            std::size_t slice_length = std::rand() % (body.length() - slice_offset);

            sz::lookup<char>(sz::string_view(body.data() + slice_offset, slice_length), lut,
                             const_cast<char *>(transformed.data()) + slice_offset);
            for (std::size_t i = 0; i != slice_length; ++i) {
                assert(transformed[slice_offset + i] == lut[body[slice_offset + i]]);
            }
        }
    }
}

#pragma endregion // Equivalence

#pragma region Drivers

/**
 *  @brief Drives the serial-vs-SIMD substring-search and byteset-search differential tests across
 *         every search backend compiled on this target. The serial kernel is the reference.
 *
 *  Three ladders run back to back: forward substring search (`sz_find`, with serial, westmere, haswell,
 *  skylake, neon, sve, v128, v128relaxed, rvv, lasx, powervsx tiers), backward substring search
 *  (`sz_rfind`, identical tiers minus sve), and byteset search (`sz_find_byteset` and `sz_rfind_byteset`,
 *  with serial, haswell, icelake, neon, v128, v128relaxed, rvv, lasx, powervsx tiers - no westmere,
 *  skylake, or sve). Each tier is guarded by its matching `SZ_USE_<ISA>` macro.
 */
void test_find_all() {

    // Forward substring search: `sz_find` tiers.
    search_from_sz_<sz_find_serial> const find_serial;
#if SZ_USE_WESTMERE
    test_search_equivalence(find_serial, search_from_sz_<sz_find_westmere> {}, 200);
#endif
#if SZ_USE_HASWELL
    test_search_equivalence(find_serial, search_from_sz_<sz_find_haswell> {}, 200);
#endif
#if SZ_USE_SKYLAKE
    test_search_equivalence(find_serial, search_from_sz_<sz_find_skylake> {}, 200);
#endif
#if SZ_USE_NEON
    test_search_equivalence(find_serial, search_from_sz_<sz_find_neon> {}, 200);
#endif
#if SZ_USE_SVE
    test_search_equivalence(find_serial, search_from_sz_<sz_find_sve> {}, 200);
#endif
#if SZ_USE_V128
    test_search_equivalence(find_serial, search_from_sz_<sz_find_v128> {}, 200);
#endif
#if SZ_USE_V128RELAXED
    test_search_equivalence(find_serial, search_from_sz_<sz_find_v128relaxed> {}, 200);
#endif
#if SZ_USE_RVV
    test_search_equivalence(find_serial, search_from_sz_<sz_find_rvv> {}, 200);
#endif
#if SZ_USE_LASX
    test_search_equivalence(find_serial, search_from_sz_<sz_find_lasx> {}, 200);
#endif
#if SZ_USE_POWERVSX
    test_search_equivalence(find_serial, search_from_sz_<sz_find_powervsx> {}, 200);
#endif

    // Backward substring search: `sz_rfind` tiers (no sve backend).
    search_from_sz_<sz_rfind_serial> const rfind_serial;
#if SZ_USE_WESTMERE
    test_search_equivalence(rfind_serial, search_from_sz_<sz_rfind_westmere> {}, 200);
#endif
#if SZ_USE_HASWELL
    test_search_equivalence(rfind_serial, search_from_sz_<sz_rfind_haswell> {}, 200);
#endif
#if SZ_USE_SKYLAKE
    test_search_equivalence(rfind_serial, search_from_sz_<sz_rfind_skylake> {}, 200);
#endif
#if SZ_USE_NEON
    test_search_equivalence(rfind_serial, search_from_sz_<sz_rfind_neon> {}, 200);
#endif
#if SZ_USE_V128
    test_search_equivalence(rfind_serial, search_from_sz_<sz_rfind_v128> {}, 200);
#endif
#if SZ_USE_V128RELAXED
    test_search_equivalence(rfind_serial, search_from_sz_<sz_rfind_v128relaxed> {}, 200);
#endif
#if SZ_USE_RVV
    test_search_equivalence(rfind_serial, search_from_sz_<sz_rfind_rvv> {}, 200);
#endif
#if SZ_USE_LASX
    test_search_equivalence(rfind_serial, search_from_sz_<sz_rfind_lasx> {}, 200);
#endif
#if SZ_USE_POWERVSX
    test_search_equivalence(rfind_serial, search_from_sz_<sz_rfind_powervsx> {}, 200);
#endif

    // Forward byteset search: `sz_find_byteset` tiers (no westmere, skylake, or sve backend).
    byteset_search_from_sz_<sz_find_byteset_serial> const find_byteset_serial;
#if SZ_USE_HASWELL
    test_byteset_equivalence(find_byteset_serial, byteset_search_from_sz_<sz_find_byteset_haswell> {}, 200);
#endif
#if SZ_USE_ICELAKE
    test_byteset_equivalence(find_byteset_serial, byteset_search_from_sz_<sz_find_byteset_icelake> {}, 200);
#endif
#if SZ_USE_NEON
    test_byteset_equivalence(find_byteset_serial, byteset_search_from_sz_<sz_find_byteset_neon> {}, 200);
#endif
#if SZ_USE_V128
    test_byteset_equivalence(find_byteset_serial, byteset_search_from_sz_<sz_find_byteset_v128> {}, 200);
#endif
#if SZ_USE_V128RELAXED
    test_byteset_equivalence(find_byteset_serial, byteset_search_from_sz_<sz_find_byteset_v128relaxed> {}, 200);
#endif
#if SZ_USE_RVV
    test_byteset_equivalence(find_byteset_serial, byteset_search_from_sz_<sz_find_byteset_rvv> {}, 200);
#endif
#if SZ_USE_LASX
    test_byteset_equivalence(find_byteset_serial, byteset_search_from_sz_<sz_find_byteset_lasx> {}, 200);
#endif
#if SZ_USE_POWERVSX
    test_byteset_equivalence(find_byteset_serial, byteset_search_from_sz_<sz_find_byteset_powervsx> {}, 200);
#endif

    // Backward byteset search: `sz_rfind_byteset` tiers (identical to the forward byteset tiers).
    byteset_search_from_sz_<sz_rfind_byteset_serial> const rfind_byteset_serial;
#if SZ_USE_HASWELL
    test_byteset_equivalence(rfind_byteset_serial, byteset_search_from_sz_<sz_rfind_byteset_haswell> {}, 200);
#endif
#if SZ_USE_ICELAKE
    test_byteset_equivalence(rfind_byteset_serial, byteset_search_from_sz_<sz_rfind_byteset_icelake> {}, 200);
#endif
#if SZ_USE_NEON
    test_byteset_equivalence(rfind_byteset_serial, byteset_search_from_sz_<sz_rfind_byteset_neon> {}, 200);
#endif
#if SZ_USE_V128
    test_byteset_equivalence(rfind_byteset_serial, byteset_search_from_sz_<sz_rfind_byteset_v128> {}, 200);
#endif
#if SZ_USE_V128RELAXED
    test_byteset_equivalence(rfind_byteset_serial, byteset_search_from_sz_<sz_rfind_byteset_v128relaxed> {}, 200);
#endif
#if SZ_USE_RVV
    test_byteset_equivalence(rfind_byteset_serial, byteset_search_from_sz_<sz_rfind_byteset_rvv> {}, 200);
#endif
#if SZ_USE_LASX
    test_byteset_equivalence(rfind_byteset_serial, byteset_search_from_sz_<sz_rfind_byteset_lasx> {}, 200);
#endif
#if SZ_USE_POWERVSX
    test_byteset_equivalence(rfind_byteset_serial, byteset_search_from_sz_<sz_rfind_byteset_powervsx> {}, 200);
#endif

    // Silence "unused variable" diagnostics on targets where the reference is never paired with a candidate.
    sz_unused_(find_serial);
    sz_unused_(rfind_serial);
    sz_unused_(find_byteset_serial);
    sz_unused_(rfind_byteset_serial);
}

#pragma endregion // Drivers
