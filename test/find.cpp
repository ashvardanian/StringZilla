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
    verify(haystack_view.find(needle_view) ==
               (forward_offset == SZ_SIZE_MAX ? sz::string_view::npos : forward_offset) &&
           "sz::string_view::find must agree with the C API's forward offset");
    verify(haystack_view.rfind(needle_view) ==
               (backward_offset == SZ_SIZE_MAX ? sz::string_view::npos : backward_offset) &&
           "sz::string_view::rfind must agree with the C API's backward offset");
}

/** @brief Lengths every comparison and byte-scan size ladder switches at, plus one either side of each. */
static sz_size_t const backend_ladder_lengths_[] = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255};

/**
 *  @brief Drives one comparison backend across the lengths its size ladder switches at.
 *
 *  Every backend delegates to serial below its vector width, so the three-byte literals above never reach the
 *  vectorized body at all. The overlapping-load boundaries at 8, 16, 32 and 64 are where these ladders break,
 *  and a mismatch is placed at the front, the middle and the last byte of each length to catch a tail that
 *  reads one byte too few or one too many.
 */
static void check_compare_unit_(char const *name, sz_equal_t equal, sz_order_t order) {
    std::string left, right;
    for (sz_size_t const length : span_over(backend_ladder_lengths_)) {
        left.assign((std::size_t)length, 'a');

        right = left;
        if (equal(left.data(), right.data(), length) != sz_true_k) {
            std::fprintf(stderr, "%s: equal() denied identical %zu-byte inputs\n", name, (std::size_t)length);
            verify(false && "Comparison backend must accept identical inputs at every ladder length");
        }
        verify(order(left.data(), length, right.data(), length) == sz_equal_k &&
               "Comparison backend must order two identical inputs as equal");

        sz_size_t const positions[] = {0, length / 2, length - 1};
        for (sz_size_t const position : span_over(positions)) {
            right = left, right[(std::size_t)position] = 'b'; // ? 'b' sorts after 'a', so `left` is the lesser
            if (equal(left.data(), right.data(), length) != sz_false_k) {
                std::fprintf(stderr, "%s: equal() missed a difference at byte %zu of %zu\n", name,
                             (std::size_t)position, (std::size_t)length);
                verify(false && "Comparison backend must see a single differing byte at every ladder length");
            }
            verify(order(left.data(), length, right.data(), length) == sz_less_k &&
                   "Comparison backend must order the byte-decreased input as lesser");
            verify(order(right.data(), length, left.data(), length) == sz_greater_k &&
                   "Comparison backend must order the byte-increased input as greater");
        }
    }
}

/**
 *  @brief Drives one byte-scanner pair across the same ladder, with the match at both ends and absent.
 *
 *  The reverse scanners are the risky half: each backend reaches "last match" differently - predicate
 *  reversal, gather-reversal, or a leading-zero count with a hand-written offset correction - and all of it
 *  is index arithmetic that an eleven-byte literal never exercises.
 */
static void check_find_byte_unit_(char const *name, sz_find_byte_t find_byte, sz_find_byte_t rfind_byte) {
    std::string haystack;
    for (sz_size_t const length : span_over(backend_ladder_lengths_)) {
        haystack.assign((std::size_t)length, 'a');
        verify(find_byte(haystack.data(), length, "z") == SZ_NULL_CHAR && "Forward byte scan must miss an absent byte");
        verify(rfind_byte(haystack.data(), length, "z") == SZ_NULL_CHAR &&
               "Reverse byte scan must miss an absent byte");

        sz_size_t const positions[] = {0, length / 2, length - 1};
        for (sz_size_t const position : span_over(positions)) {
            haystack.assign((std::size_t)length, 'a');
            haystack[(std::size_t)position] = 'z';
            sz_cptr_t const expected = haystack.data() + position;
            if (find_byte(haystack.data(), length, "z") != expected) {
                std::fprintf(stderr, "%s: find_byte missed the byte at %zu of %zu\n", name, (std::size_t)position,
                             (std::size_t)length);
                verify(false && "Forward byte scan must find a lone match at every ladder length");
            }
            if (rfind_byte(haystack.data(), length, "z") != expected) {
                std::fprintf(stderr, "%s: rfind_byte missed the byte at %zu of %zu\n", name, (std::size_t)position,
                             (std::size_t)length);
                verify(false && "Reverse byte scan must find a lone match at every ladder length");
            }
        }

        // Both ends occupied, so forward and reverse must disagree about which one they report.
        haystack.assign((std::size_t)length, 'a');
        haystack[0] = 'z', haystack[(std::size_t)length - 1] = 'z';
        verify(find_byte(haystack.data(), length, "z") == haystack.data() &&
               "Forward byte scan must report the first of two matches");
        verify(rfind_byte(haystack.data(), length, "z") == haystack.data() + length - 1 &&
               "Reverse byte scan must report the last of two matches");
    }
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
    // Dispatched (automatic kernel resolution).
    verify(sz_find_byte(hello, hello_length, "l") == hello + 2);
    verify(sz_rfind_byte(hello, hello_length, "l") == hello + 9);
    // Manual propagation to the serial kernel.
    verify(sz_find_byte_serial(hello, hello_length, "l") == hello + 2);
    verify(sz_rfind_byte_serial(hello, hello_length, "l") == hello + 9);
    verify(sz_find_byte(hello, hello_length, "z") == SZ_NULL_CHAR);        // Missing byte
    verify(sz_rfind_byte(hello, hello_length, "z") == SZ_NULL_CHAR);       // Missing byte
    verify(sz_find_byte_serial(hello, hello_length, "z") == SZ_NULL_CHAR); // Missing byte
    // Every compiled tier, swept across the lengths its ladder switches at - "hello world" is eleven bytes,
    // so the literals above only ever reach each backend's scalar tail.
    check_find_byte_unit_("dispatched", sz_find_byte, sz_rfind_byte);
    check_find_byte_unit_("serial", sz_find_byte_serial, sz_rfind_byte_serial);
#if SZ_USE_WESTMERE
    check_find_byte_unit_("westmere", sz_find_byte_westmere, sz_rfind_byte_westmere);
#endif
#if SZ_USE_HASWELL
    check_find_byte_unit_("haswell", sz_find_byte_haswell, sz_rfind_byte_haswell);
#endif
#if SZ_USE_SKYLAKE
    check_find_byte_unit_("skylake", sz_find_byte_skylake, sz_rfind_byte_skylake);
#endif
#if SZ_USE_NEON
    check_find_byte_unit_("neon", sz_find_byte_neon, sz_rfind_byte_neon);
#endif
#if SZ_USE_SVE
    check_find_byte_unit_("sve", sz_find_byte_sve, sz_rfind_byte_sve);
#endif
#if SZ_USE_V128
    check_find_byte_unit_("v128", sz_find_byte_v128, sz_rfind_byte_v128);
#endif
#if SZ_USE_V128RELAXED
    check_find_byte_unit_("v128relaxed", sz_find_byte_v128relaxed, sz_rfind_byte_v128relaxed);
#endif
#if SZ_USE_RVV
    check_find_byte_unit_("rvv", sz_find_byte_rvv, sz_rfind_byte_rvv);
#endif
#if SZ_USE_LASX
    check_find_byte_unit_("lasx", sz_find_byte_lasx, sz_rfind_byte_lasx);
#endif
#if SZ_USE_POWERVSX
    check_find_byte_unit_("powervsx", sz_find_byte_powervsx, sz_rfind_byte_powervsx);
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
    // Dispatched (automatic kernel resolution).
    verify(sz_find_byteset(hello, hello_length, &vowels) == hello + 1);
    verify(sz_rfind_byteset(hello, hello_length, &vowels) == hello + 7);
    // Manual propagation to the serial kernel.
    verify(sz_find_byteset_serial(hello, hello_length, &vowels) == hello + 1);
    verify(sz_rfind_byteset_serial(hello, hello_length, &vowels) == hello + 7);
    // A set with none of the present bytes returns `SZ_NULL_CHAR`.
    sz_byteset_t digits;
    sz_byteset_init(&digits);
    sz_byteset_add(&digits, '0');
    sz_byteset_add(&digits, '9');
    verify(sz_find_byteset(hello, hello_length, &digits) == SZ_NULL_CHAR);        // No digit present
    verify(sz_find_byteset_serial(hello, hello_length, &digits) == SZ_NULL_CHAR); // No digit present

    // `sz_find_byte_from` / `sz_find_byte_not_from` / `sz_rfind_byte_from` / `sz_rfind_byte_not_from`:
    // the `_from` family takes the needle bytes as the accepted set (the byteset built out of them), and
    // the `_not_from` family inverts that set, so they are the byteset family above spelled with a needle
    // string in place of an `sz_byteset_t`. Against "hello world" and the needle "helo" (accepts h, e, l, o):
    // the first accepted byte is 'h' at offset 0 and the last is 'l' at offset 9; the first byte NOT in the
    // set is ' ' at offset 5, and the last byte not in the set is 'd' at offset 10.
    verify(sz_find_byte_from(hello, hello_length, "helo", 4) == hello + 0);       // First accepted: 'h'
    verify(sz_rfind_byte_from(hello, hello_length, "helo", 4) == hello + 9);      // Last accepted: 'l'
    verify(sz_find_byte_not_from(hello, hello_length, "helo", 4) == hello + 5);   // First rejected: ' '
    verify(sz_rfind_byte_not_from(hello, hello_length, "helo", 4) == hello + 10); // Last rejected: 'd'
    // An empty needle accepts nothing, so `_from` must miss everywhere and `_not_from` must hit immediately.
    verify(sz_find_byte_from(hello, hello_length, "", 0) == SZ_NULL_CHAR);
    verify(sz_rfind_byte_from(hello, hello_length, "", 0) == SZ_NULL_CHAR);
    verify(sz_find_byte_not_from(hello, hello_length, "", 0) == hello + 0);
    verify(sz_rfind_byte_not_from(hello, hello_length, "", 0) == hello + hello_length - 1);
    // A needle covering the whole alphabet present in the haystack leaves `_not_from` with nothing to reject.
    verify(sz_find_byte_not_from(hello, hello_length, hello, hello_length) == SZ_NULL_CHAR);
    verify(sz_rfind_byte_not_from(hello, hello_length, hello, hello_length) == SZ_NULL_CHAR);

    // `sz_order` / `sz_equal`: lexicographic ordering and byte-equality on hand-verifiable pairs.
    verify(sz_order("abc", 3, "abc", 3) == sz_equal_k);   // Equal strings
    verify(sz_order("abc", 3, "abd", 3) == sz_less_k);    // Differ in the last byte
    verify(sz_order("abd", 3, "abc", 3) == sz_greater_k); // Differ in the last byte
    verify(sz_order("ab", 2, "abc", 3) == sz_less_k);     // Prefix orders before the longer string
    verify(sz_order("abc", 3, "ab", 2) == sz_greater_k);  // Longer string orders after its prefix
    verify(sz_equal("abc", "abc", 3) == sz_true_k);       // Identical bytes
    verify(sz_equal("abc", "abd", 3) == sz_false_k);      // Differing bytes
    // Manual propagation to the serial kernel.
    verify(sz_order_serial("abc", 3, "abd", 3) == sz_less_k);
    verify(sz_equal_serial("abc", "abc", 3) == sz_true_k);
    verify(sz_equal_serial("abc", "abd", 3) == sz_false_k);
    // As above: the three-byte literals never reach a vectorized body, so every tier is swept across the
    // lengths its ladder switches at.
    check_compare_unit_("dispatched", sz_equal, sz_order);
    check_compare_unit_("serial", sz_equal_serial, sz_order_serial);
#if SZ_USE_WESTMERE
    check_compare_unit_("westmere", sz_equal_westmere, sz_order_westmere);
#endif
#if SZ_USE_HASWELL
    check_compare_unit_("haswell", sz_equal_haswell, sz_order_haswell);
#endif
#if SZ_USE_SKYLAKE
    check_compare_unit_("skylake", sz_equal_skylake, sz_order_skylake);
#endif
#if SZ_USE_NEON
    check_compare_unit_("neon", sz_equal_neon, sz_order_neon);
#endif
#if SZ_USE_SVE
    check_compare_unit_("sve", sz_equal_sve, sz_order_sve);
#endif
#if SZ_USE_V128
    check_compare_unit_("v128", sz_equal_v128, sz_order_v128);
#endif
#if SZ_USE_V128RELAXED
    check_compare_unit_("v128relaxed", sz_equal_v128relaxed, sz_order_v128relaxed);
#endif
#if SZ_USE_RVV
    check_compare_unit_("rvv", sz_equal_rvv, sz_order_rvv);
#endif
#if SZ_USE_LASX
    check_compare_unit_("lasx", sz_equal_lasx, sz_order_lasx);
#endif
#if SZ_USE_POWERVSX
    check_compare_unit_("powervsx", sz_equal_powervsx, sz_order_powervsx);
#endif
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

    // The relational operators over the same orderings.
    verify("abc"_sv == "abc"_sv); // Equality operator
    verify("abc"_sv != "abd"_sv); // Inequality operator
    verify("abc"_sv < "abd"_sv);  // Strictly-less operator
    verify("abd"_sv > "abc"_sv);  // Strictly-greater operator
    verify("ab"_sv < "abc"_sv);   // Prefix orders before the longer string
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief One substring-search backend (find or rfind), stored by pointer so the differential driver can iterate a
 *         table. `reference(haystack, hlen, needle, nlen)` invokes the kernel via `operator()`.
 */
struct find_backend_t {
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
void check_find_search_equivalence_(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    // Replays one haystack/needle pair at every intra-cacheline alignment and compares the backends.
    auto compare_on = [&](std::string const &haystack_pattern, std::string const &needle) {
        for_each_cacheline_offset_(
            haystack_pattern.size(), [&](sz_ptr_t haystack, [[maybe_unused]] std::size_t offset) {
                std::memcpy(haystack, haystack_pattern.data(), haystack_pattern.size());
                sz_size_t const haystack_length = (sz_size_t)haystack_pattern.size();
                sz_size_t const needle_length = (sz_size_t)needle.size();

                sz_cptr_t const result_reference = reference(haystack, haystack_length, needle.data(), needle_length);
                sz_cptr_t const result_candidate = candidate(haystack, haystack_length, needle.data(), needle_length);
                if (result_reference != result_candidate) {
                    std::fprintf(
                        stderr, "%s vs %s: substring search disagreed on a %zu-byte needle in a %zu-byte haystack\n",
                        reference.name, candidate.name, (std::size_t)needle_length, (std::size_t)haystack_length);
                    verify(false && "Candidate backend must resolve every needle to the same offset as the reference");
                }
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
void check_byteset_equivalence_(reference_ reference, candidate_ candidate, sz_size_t inputs) {

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
                if (result_reference != result_candidate) {
                    std::fprintf(stderr, "%s vs %s: byteset search disagreed on a %zu-byte haystack\n", reference.name,
                                 candidate.name, (std::size_t)haystack_length);
                    verify(false && "Candidate backend must resolve every byteset to the same offset as the reference");
                }
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
                verify(false && "StringZilla must land on the same match offset as the STL reference matcher");
            }
        }

        if (count_stl != count_sz) {
            print_all_matches();
            verify(false && "StringZilla must report the same match count as the STL reference matcher");
        }
        verify(begin_stl == end_stl && begin_sz == end_sz &&
               "Both matchers must exhaust their ranges together, not leave one with unconsumed matches");

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
void test_find_misaligned_equivalence() {
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
void test_lookup_equivalence(std::size_t lookup_tables_to_try, std::size_t slices_per_table) {

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

/**
 *  @brief Degenerate and boundary shapes for the search family, asserting survival and in-bounds results.
 *
 *  Answers are not the subject here - a needle longer than its haystack, a zero length, or an empty byteset
 *  each has one defensible reply, and what matters is that every backend gives it without reading a byte it
 *  was not handed. The scanners run at every sub-cache-line alignment, since a misaligned tail is where an
 *  overlapping load reaches past the end.
 */
void test_find_safety() {
    std::printf("  - testing degenerate and boundary inputs of the search kernels...\n");

    char const *body = "the quick brown fox";
    sz_size_t const body_length = (sz_size_t)std::strlen(body);

    // A zero-length haystack, and a needle longer than what it is searched in, must both miss rather than
    // read - and every compiled backend has to say so, not merely whichever one the dispatcher picks here.
    auto check_degenerate_ = [&](char const *name, sz_find_byte_t find_byte, sz_find_byte_t rfind_byte) {
        if (find_byte(body, 0, "a") != SZ_NULL_CHAR) {
            std::fprintf(stderr, "%s: find_byte reported a match in a zero-length haystack\n", name);
            verify(false && "A zero-length haystack holds no byte to find");
        }
        if (rfind_byte(body, 0, "a") != SZ_NULL_CHAR) {
            std::fprintf(stderr, "%s: rfind_byte reported a match in a zero-length haystack\n", name);
            verify(false && "A zero-length haystack holds no byte to find");
        }
    };
    check_degenerate_("dispatched", sz_find_byte, sz_rfind_byte);
    check_degenerate_("serial", sz_find_byte_serial, sz_rfind_byte_serial);
#if SZ_USE_WESTMERE
    check_degenerate_("westmere", sz_find_byte_westmere, sz_rfind_byte_westmere);
#endif
#if SZ_USE_HASWELL
    check_degenerate_("haswell", sz_find_byte_haswell, sz_rfind_byte_haswell);
#endif
#if SZ_USE_SKYLAKE
    check_degenerate_("skylake", sz_find_byte_skylake, sz_rfind_byte_skylake);
#endif
#if SZ_USE_NEON
    check_degenerate_("neon", sz_find_byte_neon, sz_rfind_byte_neon);
#endif
#if SZ_USE_SVE
    check_degenerate_("sve", sz_find_byte_sve, sz_rfind_byte_sve);
#endif
#if SZ_USE_V128
    check_degenerate_("v128", sz_find_byte_v128, sz_rfind_byte_v128);
#endif
#if SZ_USE_V128RELAXED
    check_degenerate_("v128relaxed", sz_find_byte_v128relaxed, sz_rfind_byte_v128relaxed);
#endif
#if SZ_USE_RVV
    check_degenerate_("rvv", sz_find_byte_rvv, sz_rfind_byte_rvv);
#endif
#if SZ_USE_LASX
    check_degenerate_("lasx", sz_find_byte_lasx, sz_rfind_byte_lasx);
#endif
#if SZ_USE_POWERVSX
    check_degenerate_("powervsx", sz_find_byte_powervsx, sz_rfind_byte_powervsx);
#endif

    verify(sz_find(body, 0, "a", 1) == SZ_NULL_CHAR);
    verify(sz_rfind(body, 0, "a", 1) == SZ_NULL_CHAR);
    verify(sz_find(body, 3, body, body_length) == SZ_NULL_CHAR);
    verify(sz_rfind(body, 3, body, body_length) == SZ_NULL_CHAR);

    // An empty byteset matches nothing; a full one matches the first and last byte.
    sz_byteset_t empty_set, full_set;
    sz_byteset_init(&empty_set);
    sz_byteset_init(&full_set);
    for (int byte_value = 0; byte_value != 256; ++byte_value) sz_byteset_add_u8(&full_set, (sz_u8_t)byte_value);
    verify(sz_find_byteset(body, body_length, &empty_set) == SZ_NULL_CHAR);
    verify(sz_rfind_byteset(body, body_length, &empty_set) == SZ_NULL_CHAR);
    verify(sz_find_byteset(body, body_length, &full_set) == body);
    verify(sz_rfind_byteset(body, body_length, &full_set) == body + body_length - 1);
    verify(sz_find_byteset(body, 0, &full_set) == SZ_NULL_CHAR);

    // Every scanner, at every sub-cache-line alignment, over a buffer with no match and then one at the end.
    for (sz_size_t length : span_over(backend_ladder_lengths_)) {
        for_each_cacheline_offset_((std::size_t)length, [&](sz_ptr_t buffer, std::size_t) {
            std::memset(buffer, 'a', (std::size_t)length);
            verify(sz_find_byte(buffer, length, "z") == SZ_NULL_CHAR);
            verify(sz_rfind_byte(buffer, length, "z") == SZ_NULL_CHAR);
            verify(sz_find(buffer, length, "zz", 2) == SZ_NULL_CHAR);
            buffer[length - 1] = 'z';
            verify(sz_find_byte(buffer, length, "z") == buffer + length - 1);
            verify(sz_rfind_byte(buffer, length, "z") == buffer + length - 1);
        });
    }

    std::printf("    boundary-input safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/** @brief Forward substring-search (`sz_find`) backends; `dispatched` first keeps the table non-empty on baseline. */
static find_backend_t const find_backends[] = {
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
static find_backend_t const rfind_backends[] = {
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
    find_backend_t const find_serial {"serial", sz_find_serial};
    for (find_backend_t const &backend : find_backends) check_find_search_equivalence_(find_serial, backend, 200);

    find_backend_t const rfind_serial {"serial", sz_rfind_serial};
    for (find_backend_t const &backend : rfind_backends) check_find_search_equivalence_(rfind_serial, backend, 200);

    byteset_backend_t const find_byteset_serial {"serial", sz_find_byteset_serial};
    for (byteset_backend_t const &backend : find_byteset_backends)
        check_byteset_equivalence_(find_byteset_serial, backend, 200);

    byteset_backend_t const rfind_byteset_serial {"serial", sz_rfind_byteset_serial};
    for (byteset_backend_t const &backend : rfind_byteset_backends)
        check_byteset_equivalence_(rfind_byteset_serial, backend, 200);
}

#pragma endregion // Drivers
