/**
 *  @brief  Comparisons, search/find_all/split, misaligned-repetition search, and replacement tests.
 *  @file   test/find.cpp
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

#include <cstdio>  // `std::printf`
#include <cstring> // `std::memcpy`

#include <algorithm>   // `std::transform`
#include <iterator>    // `std::distance`
#include <random>      // `std::uniform_int_distribution`
#include <string>      // Baseline
#include <string_view> // Baseline
#include <vector>      // `std::vector`

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
    verify(sz_find(haystack, haystack_length, needle, needle_length) == forward_expected);
    verify(sz_rfind(haystack, haystack_length, needle, needle_length) == backward_expected);

    // Manual propagation to each natively-compiled backend kernel.
    verify(sz_find_serial(haystack, haystack_length, needle, needle_length) == forward_expected);
    verify(sz_rfind_serial(haystack, haystack_length, needle, needle_length) == backward_expected);
#if SZ_USE_WESTMERE
    verify(sz_find_westmere(haystack, haystack_length, needle, needle_length) == forward_expected);
    verify(sz_rfind_westmere(haystack, haystack_length, needle, needle_length) == backward_expected);
#endif
#if SZ_USE_HASWELL
    verify(sz_find_haswell(haystack, haystack_length, needle, needle_length) == forward_expected);
    verify(sz_rfind_haswell(haystack, haystack_length, needle, needle_length) == backward_expected);
#endif
#if SZ_USE_SKYLAKE
    verify(sz_find_skylake(haystack, haystack_length, needle, needle_length) == forward_expected);
    verify(sz_rfind_skylake(haystack, haystack_length, needle, needle_length) == backward_expected);
#endif
#if SZ_USE_NEON
    verify(sz_find_neon(haystack, haystack_length, needle, needle_length) == forward_expected);
    verify(sz_rfind_neon(haystack, haystack_length, needle, needle_length) == backward_expected);
#endif
#if SZ_USE_SVE
    verify(sz_find_sve(haystack, haystack_length, needle, needle_length) == forward_expected);
#endif

    // The C++ `sz::string_view` wrapper resolves to the same offsets.
    sz::string_view const haystack_view(haystack, haystack_length);
    sz::string_view const needle_view(needle, needle_length);
    verify(haystack_view.find(needle_view) == (forward_offset == SZ_SIZE_MAX ? sz::string_view::npos : forward_offset));
    verify(haystack_view.rfind(needle_view) ==
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
    verify(sz_find_byte(hello, hello_length, "l") == hello + 2);           // Dispatched (automatic kernel)
    verify(sz_rfind_byte(hello, hello_length, "l") == hello + 9);          // Dispatched (automatic kernel)
    verify(sz_find_byte_serial(hello, hello_length, "l") == hello + 2);    // Manual propagation to the serial kernel
    verify(sz_rfind_byte_serial(hello, hello_length, "l") == hello + 9);   // Manual propagation to the serial kernel
    verify(sz_find_byte(hello, hello_length, "z") == SZ_NULL_CHAR);        // Missing byte
    verify(sz_rfind_byte(hello, hello_length, "z") == SZ_NULL_CHAR);       // Missing byte
    verify(sz_find_byte_serial(hello, hello_length, "z") == SZ_NULL_CHAR); // Missing byte, serial kernel
#if SZ_USE_WESTMERE
    verify(sz_find_byte_westmere(hello, hello_length, "l") == hello + 2);
    verify(sz_rfind_byte_westmere(hello, hello_length, "l") == hello + 9);
#endif
#if SZ_USE_HASWELL
    verify(sz_find_byte_haswell(hello, hello_length, "l") == hello + 2);
    verify(sz_rfind_byte_haswell(hello, hello_length, "l") == hello + 9);
#endif
#if SZ_USE_SKYLAKE
    verify(sz_find_byte_skylake(hello, hello_length, "l") == hello + 2);
    verify(sz_rfind_byte_skylake(hello, hello_length, "l") == hello + 9);
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
    verify(sz_find_byteset(hello, hello_length, &vowels) == hello + 1);  // Dispatched (automatic kernel)
    verify(sz_rfind_byteset(hello, hello_length, &vowels) == hello + 7); // Dispatched (automatic kernel)
    verify(sz_find_byteset_serial(hello, hello_length, &vowels) ==
           hello + 1); // Manual propagation to the serial kernel
    verify(sz_rfind_byteset_serial(hello, hello_length, &vowels) ==
           hello + 7); // Manual propagation to the serial kernel
    // A set with none of the present bytes returns `SZ_NULL_CHAR`.
    sz_byteset_t digits;
    sz_byteset_init(&digits);
    sz_byteset_add(&digits, '0');
    sz_byteset_add(&digits, '9');
    verify(sz_find_byteset(hello, hello_length, &digits) == SZ_NULL_CHAR);        // No digit present
    verify(sz_find_byteset_serial(hello, hello_length, &digits) == SZ_NULL_CHAR); // No digit present, serial kernel

    // `sz_order` / `sz_equal`: lexicographic ordering and byte-equality on hand-verifiable pairs.
    verify(sz_order("abc", 3, "abc", 3) == sz_equal_k);       // Equal strings
    verify(sz_order("abc", 3, "abd", 3) == sz_less_k);        // Differ in the last byte
    verify(sz_order("abd", 3, "abc", 3) == sz_greater_k);     // Differ in the last byte
    verify(sz_order("ab", 2, "abc", 3) == sz_less_k);         // Prefix orders before the longer string
    verify(sz_order("abc", 3, "ab", 2) == sz_greater_k);      // Longer string orders after its prefix
    verify(sz_equal("abc", "abc", 3) == sz_true_k);           // Identical bytes
    verify(sz_equal("abc", "abd", 3) == sz_false_k);          // Differing bytes
    verify(sz_order_serial("abc", 3, "abd", 3) == sz_less_k); // Manual propagation to the serial kernel
    verify(sz_equal_serial("abc", "abc", 3) == sz_true_k);    // Manual propagation to the serial kernel
    verify(sz_equal_serial("abc", "abd", 3) == sz_false_k);   // Manual propagation to the serial kernel
#if SZ_USE_HASWELL
    verify(sz_order_haswell("abc", 3, "abd", 3) == sz_less_k);
    verify(sz_equal_haswell("abc", "abc", 3) == sz_true_k);
    verify(sz_equal_haswell("abc", "abd", 3) == sz_false_k);
#endif
#if SZ_USE_SKYLAKE
    verify(sz_order_skylake("abc", 3, "abd", 3) == sz_less_k);
    verify(sz_equal_skylake("abc", "abc", 3) == sz_true_k);
    verify(sz_equal_skylake("abc", "abd", 3) == sz_false_k);
#endif
    // And the same orderings through the C++ `sz::string_view` comparison operators.
    verify("abc"_sv == "abc"_sv); // Equality operator
    verify("abc"_sv != "abd"_sv); // Inequality operator
    verify("abc"_sv < "abd"_sv);  // Strictly-less operator
    verify("abd"_sv > "abc"_sv);  // Strictly-greater operator
    verify("ab"_sv < "abc"_sv);   // Prefix orders before the longer string

    // Searching for a set of characters
    verify(sz::string_view("a").find_first_of("az") == 0);
    verify(sz::string_view("a").find_last_of("az") == 0);
    verify(sz::string_view("a").find_first_of("xz") == sz::string_view::npos);
    verify(sz::string_view("a").find_last_of("xz") == sz::string_view::npos);

    verify(sz::string_view("a").find_first_not_of("xz") == 0);
    verify(sz::string_view("a").find_last_not_of("xz") == 0);
    verify(sz::string_view("a").find_first_not_of("az") == sz::string_view::npos);
    verify(sz::string_view("a").find_last_not_of("az") == sz::string_view::npos);

    verify(sz::string_view("aXbYaXbY").find_first_of("XY") == 1);
    verify(sz::string_view("axbYaxbY").find_first_of("Y") == 3);
    verify(sz::string_view("YbXaYbXa").find_last_of("XY") == 6);
    verify(sz::string_view("YbxaYbxa").find_last_of("Y") == 4);
    verify(sz::string_view(sz::base64(), sizeof(sz::base64())).find_first_of("_") == sz::string_view::npos);
    verify(sz::string_view(sz::base64(), sizeof(sz::base64())).find_first_of("+") == 62);
    verify(sz::string_view(sz::ascii_printables(), sizeof(sz::ascii_printables())).find_first_of("~") !=
           sz::string_view::npos);

    verify("aabaa"_sv.remove_prefix("a") == "abaa");
    verify("aabaa"_sv.remove_suffix("a") == "aaba");
    verify("aabaa"_sv.lstrip("a"_bs) == "baa");
    verify("aabaa"_sv.rstrip("a"_bs) == "aab");
    verify("aabaa"_sv.strip("a"_bs) == "b");

    // Check more advanced composite operations
    verify("abbccc"_sv.partition('b').before.size() == 1);
    verify("abbccc"_sv.partition("bb").before.size() == 1);
    verify("abbccc"_sv.partition("bb").match.size() == 2);
    verify("abbccc"_sv.partition("bb").after.size() == 3);
    verify("abbccc"_sv.partition("bb").before == "a");
    verify("abbccc"_sv.partition("bb").match == "bb");
    verify("abbccc"_sv.partition("bb").after == "ccc");
    verify("abb ccc"_sv.partition(sz::whitespaces_set()).after == "ccc");

    // Check ranges of search matches
    verify("hello"_sv.find_all("l").size() == 2);
    verify("hello"_sv.rfind_all("l").size() == 2);

    verify(""_sv.find_all(".", sz::include_overlaps_type {}).size() == 0);
    verify(""_sv.find_all(".", sz::exclude_overlaps_type {}).size() == 0);
    verify("."_sv.find_all(".", sz::include_overlaps_type {}).size() == 1);
    verify("."_sv.find_all(".", sz::exclude_overlaps_type {}).size() == 1);
    verify(".."_sv.find_all(".", sz::include_overlaps_type {}).size() == 2);
    verify(".."_sv.find_all(".", sz::exclude_overlaps_type {}).size() == 2);
    verify(""_sv.rfind_all(".", sz::include_overlaps_type {}).size() == 0);
    verify(""_sv.rfind_all(".", sz::exclude_overlaps_type {}).size() == 0);
    verify("."_sv.rfind_all(".", sz::include_overlaps_type {}).size() == 1);
    verify("."_sv.rfind_all(".", sz::exclude_overlaps_type {}).size() == 1);
    verify(".."_sv.rfind_all(".", sz::include_overlaps_type {}).size() == 2);
    verify(".."_sv.rfind_all(".", sz::exclude_overlaps_type {}).size() == 2);

    verify("a.b.c.d"_sv.find_all(".").size() == 3);
    verify("a.,b.,c.,d"_sv.find_all(".,").size() == 3);
    verify("a.,b.,c.,d"_sv.rfind_all(".,").size() == 3);
    verify("a.b,c.d"_sv.find_all(".,"_bs).size() == 3);
    verify("a...b...c"_sv.rfind_all("..").size() == 4);
    verify("a...b...c"_sv.rfind_all("..", sz::include_overlaps_type {}).size() == 4);
    verify("a...b...c"_sv.rfind_all("..", sz::exclude_overlaps_type {}).size() == 2);

    let_verify(auto finds = "a.b.c"_sv.find_all("abcd"_bs).template to<std::vector<std::string>>(),
               finds.size() == 3 && finds[0] == "a");
    let_verify(auto rfinds = "a.b.c"_sv.rfind_all("abcd"_bs).template to<std::vector<std::string>>(),
               rfinds.size() == 3 && rfinds[0] == "c");

    // Test propagating strings and their non-owning views into temporary ranges and iterators
    verify(sz::find_all("abc"_sv, "b"_sv).size() == 1);
    verify(sz::find_all("hello"_sv, "l"_sv).size() == 2);
    verify(sz::rfind_all("abc"_sv, "b"_sv).size() == 1);

    {
        sz::string h("abc"), n("b");
        verify(sz::find_all(h, n).size() == 1);
    }
    {
        sz::string h("hello"), n("l");
        verify(sz::find_all(h, n).size() == 2);
    }
    {
        sz::string h("abc"), n("b");
        verify(sz::rfind_all(h, n).size() == 1);
    }

    verify(sz::find_all(sz::string("abc"), sz::string("b")).size() == 1);
    verify(sz::find_all(sz::string("hello"), sz::string("l")).size() == 2);
    verify(sz::rfind_all(sz::string("abc"), sz::string("b")).size() == 1);

    // Lvalue haystacks are borrowed, so slices land inside the caller's own buffer. A copied
    // haystack would offset into a private copy - and under SSO those offsets look plausible.
    {
        sz::string haystack("hello world, hello cpp");
        sz::string sso("a b a");
        let_verify(auto matches = sz::find_all(haystack, "hello").template to<std::vector<sz::string_view>>(),
                   matches.size() == 2 &&                          //
                       matches[0].data() - haystack.data() == 0 && //
                       matches[1].data() - haystack.data() == 13);
        let_verify(auto in_sso = sz::find_all(sso, "a").template to<std::vector<sz::string_view>>(),
                   in_sso.size() == 2 &&                     //
                       in_sso[0].data() - sso.data() == 0 && //
                       in_sso[1].data() - sso.data() == 4);
    }

    // Needles are copied into the matcher, so a temporary one outlives the expression that built it.
    verify(sz::find_all(sz::string("hello world, hello cpp"), sz::string("hello")).size() == 2);

    // Haystack and needle need not share a type - literals, views, and owning strings mix.
    {
        sz::string owning("a-b-c");
        sz::string_view view("a-b-c");
        sz::string needle("-");
        verify(sz::find_all(view, "-").size() == 2);
        verify(sz::find_all(owning, "-").size() == 2);
        verify(sz::find_all(owning, view.substr(1, 1)).size() == 2);
        verify(sz::find_all(view, needle).size() == 2);
        verify(sz::split(owning, "-").size() == 3);
        verify(sz::rsplit(view, needle).size() == 3);
        verify(sz::split_characters(owning, "-").size() == 3);
    }

    // Check splitting - the inverse of `find_all` ranges
    let_verify(auto splits = ".a..c."_sv.split("."_bs).template to<std::vector<std::string>>(),
               splits.size() == 5 && splits[0] == "" && splits[1] == "a" && splits[4] == "");
    let_verify(auto line_splits = "line1\nline2\nline3"_sv.split("line3").template to<std::vector<std::string>>(),
               line_splits.size() == 2 && line_splits[0] == "line1\nline2\n" && line_splits[1] == "");

    verify(""_sv.split(".").size() == 1);
    verify(""_sv.rsplit(".").size() == 1);

    verify("hello"_sv.split("l").size() == 3);
    verify("hello"_sv.rsplit("l").size() == 3);
    verify(*advanced("hello"_sv.split("l").begin(), 0) == "he");
    verify(*advanced("hello"_sv.rsplit("l").begin(), 0) == "o");
    verify(*advanced("hello"_sv.split("l").begin(), 1) == "");
    verify(*advanced("hello"_sv.rsplit("l").begin(), 1) == "");
    verify(*advanced("hello"_sv.split("l").begin(), 2) == "o");
    verify(*advanced("hello"_sv.rsplit("l").begin(), 2) == "he");

    verify("a.b.c.d"_sv.split(".").size() == 4);
    verify("a.b.c.d"_sv.rsplit(".").size() == 4);
    verify(*("a.b.c.d"_sv.split(".").begin()) == "a");
    verify(*("a.b.c.d"_sv.rsplit(".").begin()) == "d");
    verify(*advanced("a.b.c.d"_sv.split(".").begin(), 1) == "b");
    verify(*advanced("a.b.c.d"_sv.rsplit(".").begin(), 1) == "c");
    verify(*advanced("a.b.c.d"_sv.split(".").begin(), 3) == "d");
    verify(*advanced("a.b.c.d"_sv.rsplit(".").begin(), 3) == "a");
    verify("a.b.,c,d"_sv.split(".,").size() == 2);
    verify("a.b,c.d"_sv.split(".,"_bs).size() == 4);

    let_verify(auto rsplits = ".a..c."_sv.rsplit("."_bs).template to<std::vector<std::string>>(),
               rsplits.size() == 5 && rsplits[0] == "" && rsplits[1] == "c" && rsplits[4] == "");
}

/**
 *  @brief Tests the correctness of the string class comparison methods, such as `compare` and `operator==`.
 */
void test_compare_unit() {
    // Comparing relative order of the strings
    verify("a"_sv.compare("a") == 0);
    verify("a"_sv.compare("ab") == -1);
    verify("ab"_sv.compare("a") == 1);
    verify("a"_sv.compare("a\0"_sv) == -1);
    verify("a\0"_sv.compare("a") == 1);
    verify("a\0"_sv.compare("a\0"_sv) == 0);
    verify("a"_sv == "a"_sv);
    verify("a"_sv != "a\0"_sv);
    verify("a\0"_sv == "a\0"_sv);
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief One substring-search backend (find or rfind), stored by pointer so the differential driver can iterate a
 *         table. `reference(haystack, hlen, needle, nlen)` invokes the kernel via `operator()`.
 */
struct search_backend_t {
    char const *name;
    sz_find_t kernel;
    sz_cptr_t operator()(sz_cptr_t haystack, sz_size_t haystack_length, //
                         sz_cptr_t needle, sz_size_t needle_length) const noexcept {
        return kernel(haystack, haystack_length, needle, needle_length);
    }
};

/**
 *  @brief One byteset-search backend (find or rfind), stored by pointer; `reference(haystack, hlen, byteset)`
 *         invokes the kernel via `operator()`.
 */
struct byteset_backend_t {
    char const *name;
    sz_find_byteset_t kernel;
    sz_cptr_t operator()(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *byteset) const noexcept {
        return kernel(haystack, haystack_length, byteset);
    }
};

/**
 *  @brief Cross-checks two substring-search backends of the @b same operation (both forward or both
 *         backward) against each other on random and hand-picked edge-case inputs.
 *
 *  The candidate must resolve every needle to the identical pointer the reference does, or both must
 *  return `SZ_NULL`. Each haystack is replayed at every sub-cacheline alignment via
 *  `for_each_cacheline_offset_`, so a needle straddling a 64-byte boundary is always exercised.
 */
template <typename reference_, typename candidate_>
void test_find_search_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    // Replays one haystack/needle pair at every intra-cacheline alignment and compares the backends.
    auto compare_on = [&](std::string const &haystack_pattern, std::string const &needle) {
        for_each_cacheline_offset_(
            haystack_pattern.size(), [&](sz_ptr_t haystack, [[maybe_unused]] std::size_t offset) {
                std::memcpy(haystack, haystack_pattern.data(), haystack_pattern.size());
                sz_size_t const haystack_length = (sz_size_t)haystack_pattern.size();
                sz_size_t const needle_length = (sz_size_t)needle.size();

                sz_cptr_t const result_reference = reference(haystack, haystack_length, needle.data(), needle_length);
                sz_cptr_t const result_candidate = candidate(haystack, haystack_length, needle.data(), needle_length);
                verify(result_reference == result_candidate);
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
                verify(result_reference == result_candidate);
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

#pragma endregion // Equivalence

#pragma region Safety

#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)

/**
 *  @brief Evaluates the correctness of a "matcher", searching for all the occurrences of the @p needle_stl
 *         in a haystack formed of @p haystack_pattern repeated from one to `max_repeats` times,
 *         misaligned by @p misalignment bytes within the cacheline.
 */
template <typename stl_matcher_, typename sz_matcher_>
void check_find_misaligned_(std::string_view haystack_pattern, std::string_view needle_stl, std::size_t misalignment) {
    // Each repetition re-scans the whole growing haystack, so the work is quadratic in this count, and it is
    // multiplied again by every case, misalignment and matcher family.
    std::size_t const max_repeats = scale_iterations_quadratic(40);

    // Allocate a buffer to store the haystack with enough padding to mis-align it.
    std::size_t haystack_buffer_length = max_repeats * haystack_pattern.size() + 2 * SZ_CACHE_LINE_WIDTH;
    std::vector<char> haystack_buffer(haystack_buffer_length, 'x');
    char *haystack = haystack_buffer.data();

    while (reinterpret_cast<std::uintptr_t>(haystack) % SZ_CACHE_LINE_WIDTH != misalignment) ++haystack;

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

        std::memcpy(haystack + repeats * haystack_pattern.size(), haystack_pattern.data(), haystack_pattern.size());

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

        for (std::size_t match_idx = 0; begin_stl != end_stl && begin_sz != end_sz;
             ++begin_stl, ++begin_sz, ++match_idx) {
            auto match_stl = *begin_stl;
            auto match_sz = *begin_sz;
            if (match_stl.data() != match_sz.data()) {
                std::printf("Mismatch at index #%zu: %zu != %zu\n", match_idx, match_stl.data() - haystack_stl.data(),
                            match_sz.data() - haystack_sz.data());
                print_all_matches();
                verify(false);
            }
        }

        if (count_stl != count_sz) {
            print_all_matches();
            verify(false);
        }
        verify(begin_stl == end_stl && begin_sz == end_sz);

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
 */
void check_find_misaligned_(std::string_view haystack_pattern, std::string_view needle_stl, std::size_t misalignment) {

    check_find_misaligned_<                                                          //
        sz::find_matches_view<std::string_view, sz::matcher_find<std::string_view>>, //
        sz::find_matches_view<sz::string_view, sz::matcher_find<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    check_find_misaligned_<                                                            //
        sz::rfind_matches_view<std::string_view, sz::matcher_rfind<std::string_view>>, //
        sz::rfind_matches_view<sz::string_view, sz::matcher_rfind<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    check_find_misaligned_<                                                                   //
        sz::find_matches_view<std::string_view, sz::matcher_find_first_of<std::string_view>>, //
        sz::find_matches_view<sz::string_view, sz::matcher_find_first_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    check_find_misaligned_<                                                                   //
        sz::rfind_matches_view<std::string_view, sz::matcher_find_last_of<std::string_view>>, //
        sz::rfind_matches_view<sz::string_view, sz::matcher_find_last_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    check_find_misaligned_<                                                                       //
        sz::find_matches_view<std::string_view, sz::matcher_find_first_not_of<std::string_view>>, //
        sz::find_matches_view<sz::string_view, sz::matcher_find_first_not_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    check_find_misaligned_<                                                                       //
        sz::rfind_matches_view<std::string_view, sz::matcher_find_last_not_of<std::string_view>>, //
        sz::rfind_matches_view<sz::string_view, sz::matcher_find_last_not_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);
}

/** @brief Replays the misaligned-repetition search across a fixed sweep of intra-cacheline offsets. */
void check_find_misaligned_(std::string_view haystack_pattern, std::string_view needle_stl) {
    check_find_misaligned_(haystack_pattern, needle_stl, 0);
    check_find_misaligned_(haystack_pattern, needle_stl, 1);
    check_find_misaligned_(haystack_pattern, needle_stl, 2);
    check_find_misaligned_(haystack_pattern, needle_stl, 3);
    check_find_misaligned_(haystack_pattern, needle_stl, 63);
    check_find_misaligned_(haystack_pattern, needle_stl, 24);
    check_find_misaligned_(haystack_pattern, needle_stl, 33);
}

/**
 *  @brief Extensively tests the correctness of the string class search methods, such as `find` and `find_first_of`.
 *         Covers different alignment cases within a cache line, repetitive patterns, and overlapping matches.
 */
void test_find_misaligned_all() {
    // When haystack is only formed of needles:
    check_find_misaligned_("a", "a");
    check_find_misaligned_("ab", "ab");
    check_find_misaligned_("abc", "abc");
    check_find_misaligned_("abcd", "abcd");
    check_find_misaligned_({sz::base64(), sizeof(sz::base64())}, {sz::base64(), sizeof(sz::base64())});
    check_find_misaligned_({sz::ascii_lowercase(), sizeof(sz::ascii_lowercase())},
                           {sz::ascii_lowercase(), sizeof(sz::ascii_lowercase())});
    check_find_misaligned_({sz::ascii_printables(), sizeof(sz::ascii_printables())},
                           {sz::ascii_printables(), sizeof(sz::ascii_printables())});

    // When we are dealing with NULL characters inside the string
    check_find_misaligned_("\0", "\0");
    check_find_misaligned_("a\0", "a\0");
    check_find_misaligned_("ab\0", "ab");
    check_find_misaligned_("ab\0", "ab\0");
    check_find_misaligned_("abc\0", "abc");
    check_find_misaligned_("abc\0", "abc\0");
    check_find_misaligned_("abcd\0", "abcd");

    // When searching for all-null needles in a haystack with no null bytes.
    // This exercises the SIMD tail path where masked-off lanes are zeroed:
    // if the needle characters are also zero, spurious matches appear at
    // invalid offsets beyond the haystack, causing OOB reads.
    check_find_misaligned_("a", {"\0\0", 2});
    check_find_misaligned_("a", {"\0\0\0", 3});
    check_find_misaligned_("a", {"\0\0\0\0", 4});
    check_find_misaligned_("a", {"\0\0\0\0\0", 5});
    check_find_misaligned_("abcd", {"\0\0", 2});
    check_find_misaligned_("abcd", {"\0\0\0\0", 4});

    // When haystack is formed of equidistant needles:
    check_find_misaligned_("ab", "a");
    check_find_misaligned_("abc", "a");
    check_find_misaligned_("abcd", "a");

    // When matches occur in between pattern words:
    check_find_misaligned_("ab", "ba");
    check_find_misaligned_("abc", "ca");
    check_find_misaligned_("abcd", "da");

    // Examples targeted exactly against the Raita heuristic,
    // which matches the first, the last, and the middle characters with SIMD.
    check_find_misaligned_("aaabbccc", "aaabbccc");
    check_find_misaligned_("axabbcxc", "aaabbccc");
    check_find_misaligned_("axabbcxcaaabbccc", "aaabbccc");
}

#endif

/** @brief Evaluates the correctness of look-up table transforms using random lookup tables. */
void test_lookup_all(std::size_t lookup_tables_to_try, std::size_t slices_per_table) {

    std::size_t const body_length = 1024 * 1024;
    std::string body(body_length, '\0'), transformed(body_length, '\0');
    randomize_string(&body[0], body_length);
    std::uniform_int_distribution<int> byte_distribution(0, 255);

    // One scaled count over the whole slice budget, so the cost tracks the multiplier linearly rather than
    // compounding across nested loops. A fresh table every `slices_per_table` slices keeps the original mix.
    sz::look_up_table lut;
    for (std::size_t slice = 0; slice != scale_iterations(lookup_tables_to_try * slices_per_table); ++slice) {
        if (slice % slices_per_table == 0)
            for (std::size_t index = 0; index < 256; ++index)
                lut[(char)index] = (char)byte_distribution(global_random_generator());

        std::uniform_int_distribution<std::size_t> offset_distribution(0, body_length - 1);
        std::size_t const slice_offset = offset_distribution(global_random_generator());
        std::uniform_int_distribution<std::size_t> length_distribution(0, body_length - slice_offset - 1);
        std::size_t const slice_length = length_distribution(global_random_generator());

        sz::lookup<char>(sz::string_view(body.data() + slice_offset, slice_length), lut,
                         &transformed[0] + slice_offset);
        for (std::size_t index = 0; index != slice_length; ++index)
            verify(transformed[slice_offset + index] == lut[body[slice_offset + index]]);
    }
}

#pragma endregion // Safety

#pragma region Drivers

/** @brief Forward substring-search (`sz_find`) backends; `dispatched` first keeps the table non-empty on baseline. */
static search_backend_t const find_backends[] = {
    {"dispatched", sz_find},
#if SZ_USE_WESTMERE
    {"westmere", sz_find_westmere},
#endif
#if SZ_USE_HASWELL
    {"haswell", sz_find_haswell},
#endif
#if SZ_USE_SKYLAKE
    {"skylake", sz_find_skylake},
#endif
#if SZ_USE_NEON
    {"neon", sz_find_neon},
#endif
#if SZ_USE_SVE
    {"sve", sz_find_sve},
#endif
#if SZ_USE_V128
    {"v128", sz_find_v128},
#endif
#if SZ_USE_V128RELAXED
    {"v128relaxed", sz_find_v128relaxed},
#endif
#if SZ_USE_RVV
    {"rvv", sz_find_rvv},
#endif
#if SZ_USE_LASX
    {"lasx", sz_find_lasx},
#endif
#if SZ_USE_POWERVSX
    {"powervsx", sz_find_powervsx},
#endif
};

/** @brief Backward substring-search (`sz_rfind`) backends; same tiers as forward. */
static search_backend_t const rfind_backends[] = {
    {"dispatched", sz_rfind},
#if SZ_USE_SVE
    {"sve", sz_rfind_sve},
#endif
#if SZ_USE_WESTMERE
    {"westmere", sz_rfind_westmere},
#endif
#if SZ_USE_HASWELL
    {"haswell", sz_rfind_haswell},
#endif
#if SZ_USE_SKYLAKE
    {"skylake", sz_rfind_skylake},
#endif
#if SZ_USE_NEON
    {"neon", sz_rfind_neon},
#endif
#if SZ_USE_V128
    {"v128", sz_rfind_v128},
#endif
#if SZ_USE_V128RELAXED
    {"v128relaxed", sz_rfind_v128relaxed},
#endif
#if SZ_USE_RVV
    {"rvv", sz_rfind_rvv},
#endif
#if SZ_USE_LASX
    {"lasx", sz_rfind_lasx},
#endif
#if SZ_USE_POWERVSX
    {"powervsx", sz_rfind_powervsx},
#endif
};

/** @brief Forward byteset-search (`sz_find_byteset`) backends; no Westmere/Skylake/SVE tiers. */
static byteset_backend_t const find_byteset_backends[] = {
    {"dispatched", sz_find_byteset},
#if SZ_USE_HASWELL
    {"haswell", sz_find_byteset_haswell},
#endif
#if SZ_USE_ICELAKE
    {"icelake", sz_find_byteset_icelake},
#endif
#if SZ_USE_NEON
    {"neon", sz_find_byteset_neon},
#endif
#if SZ_USE_SVE2
    {"sve2", sz_find_byteset_sve2},
#endif
#if SZ_USE_V128
    {"v128", sz_find_byteset_v128},
#endif
#if SZ_USE_V128RELAXED
    {"v128relaxed", sz_find_byteset_v128relaxed},
#endif
#if SZ_USE_RVV
    {"rvv", sz_find_byteset_rvv},
#endif
#if SZ_USE_LASX
    {"lasx", sz_find_byteset_lasx},
#endif
#if SZ_USE_POWERVSX
    {"powervsx", sz_find_byteset_powervsx},
#endif
};

/** @brief Backward byteset-search (`sz_rfind_byteset`) backends; identical tiers to the forward byteset table. */
static byteset_backend_t const rfind_byteset_backends[] = {
    {"dispatched", sz_rfind_byteset},
#if SZ_USE_HASWELL
    {"haswell", sz_rfind_byteset_haswell},
#endif
#if SZ_USE_ICELAKE
    {"icelake", sz_rfind_byteset_icelake},
#endif
#if SZ_USE_NEON
    {"neon", sz_rfind_byteset_neon},
#endif
#if SZ_USE_SVE2
    {"sve2", sz_rfind_byteset_sve2},
#endif
#if SZ_USE_V128
    {"v128", sz_rfind_byteset_v128},
#endif
#if SZ_USE_V128RELAXED
    {"v128relaxed", sz_rfind_byteset_v128relaxed},
#endif
#if SZ_USE_RVV
    {"rvv", sz_rfind_byteset_rvv},
#endif
#if SZ_USE_LASX
    {"lasx", sz_rfind_byteset_lasx},
#endif
#if SZ_USE_POWERVSX
    {"powervsx", sz_rfind_byteset_powervsx},
#endif
};

/**
 *  @brief Drives the serial-vs-SIMD substring-search and byteset-search differential tests across every search
 *         backend compiled on this target (dispatched first). The serial kernel is the reference; four tables run
 *         back to back — forward/backward substring search and forward/backward byteset search.
 */
void test_find_all() {
    search_backend_t const find_serial {"serial", sz_find_serial};
    for (search_backend_t const &backend : find_backends) test_find_search_equivalence(find_serial, backend, 200);

    search_backend_t const rfind_serial {"serial", sz_rfind_serial};
    for (search_backend_t const &backend : rfind_backends) test_find_search_equivalence(rfind_serial, backend, 200);

    byteset_backend_t const find_byteset_serial {"serial", sz_find_byteset_serial};
    for (byteset_backend_t const &backend : find_byteset_backends)
        test_byteset_equivalence(find_byteset_serial, backend, 200);

    byteset_backend_t const rfind_byteset_serial {"serial", sz_rfind_byteset_serial};
    for (byteset_backend_t const &backend : rfind_byteset_backends)
        test_byteset_equivalence(rfind_byteset_serial, backend, 200);
}

#pragma endregion // Drivers
