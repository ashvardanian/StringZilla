/**
 *  @file test/find.cpp
 *  @author Ash Vardanian
 *  @date December 21, 2023
 *  @brief Comparisons, search/find_all/split, misaligned-repetition search, and replacement tests.
 */
#undef NDEBUG // ! Enable all assertions for testing

/** The Visual C++ run-time library detects incorrect iterator use, and asserts and displays a
 *  dialog box at run time on Windows. */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

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

#include <cstdio>  // `stderr`
#include <cstring> // `std::memcpy`

#include <iterator>    // `std::distance`
#include <random>      // `std::uniform_int_distribution`
#include <string>      // Baseline
#include <string_view> // Baseline
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include "cross.hpp"   // `check_find_unit_`, `check_compare_unit_`, `find_backend_t`
#include "harness.hpp" // `randomize_string`, `test_context_t`

namespace ashvardanian::stringzilla::test {
using sz::literals::operator""_sv; // for `sz::string_view_t`
using sz::literals::operator""_bs; // for `sz::byteset_t`

using namespace std::literals; // for ""sv

#pragma region Helpers

/** The search dispatch points, in the shape of their kernels. */
static find_backend_t const find_dispatched {
    .name = "dispatched",
    .find = cpu_best<sz_find_best>,
    .rfind = cpu_best<sz_rfind_best>,
    .find_byte = cpu_best<sz_find_byte_best>,
    .rfind_byte = cpu_best<sz_rfind_byte_best>,
    .find_byteset = cpu_best<sz_find_byteset_best>,
    .rfind_byteset = cpu_best<sz_rfind_byteset_best>,
};

/** The comparison dispatch points, in the shape of their kernels. */
static compare_backend_t const compare_dispatched {"dispatched", cpu_best<sz_equal_best>, cpu_best<sz_order_best>};

/**
 *  @brief Runs one substring-search known-answer case through the C++ @c sz::string_view_t wrapper.
 *  @param[in] forward_offset Expected first offset, or @c STRINGZILLA_SIZE_MAX if absent.
 *  @param[in] backward_offset Expected last offset, or @c STRINGZILLA_SIZE_MAX if absent.
 */
static void check_find_views_unit_(sz::string_view_t haystack, sz::string_view_t needle, sz_size_t forward_offset,
                                   sz_size_t backward_offset) {
    verify(haystack.find(needle) ==
               (forward_offset == STRINGZILLA_SIZE_MAX ? sz::string_view_t::npos : forward_offset) &&
           "sz::string_view_t::find must agree with the C API's forward offset");
    verify(haystack.rfind(needle) ==
               (backward_offset == STRINGZILLA_SIZE_MAX ? sz::string_view_t::npos : backward_offset) &&
           "sz::string_view_t::rfind must agree with the C API's backward offset");
}

#pragma endregion Helpers

#pragma region Unit

/**
 *  @brief Known-answer tests for the search and comparison dispatch points on simple inputs.
 *
 *  Begins with known-answer vectors exercising each dispatch point and the C++ @c sz::string_view_t
 *  wrappers, so a regression that the serial-vs-SIMD agreement tests would miss - because both
 *  share a wrong constant - is still caught against an external ground truth. The kernels of each
 *  capability face the same vectors in `cross_<arch>.cpp`.
 */
void test_find_unit() {
    char const *hello = "hello world";
    sz_size_t const hello_length = (sz_size_t)std::strlen(hello); // 11 bytes

    // `sz_find_best` / `sz_rfind_best`: "o" occurs at offsets 4 and 7, the multi-byte needle "wor"
    // at 6, and a missing "xyz" yields `STRINGZILLA_NULL_CHAR`, which the views spell as `npos`,
    // encoded here as the `STRINGZILLA_SIZE_MAX` not-found sentinel.
    check_find_unit_(find_dispatched);
    check_find_views_unit_({hello, hello_length}, "o"_sv, 4, 7);   // Single-byte needle
    check_find_views_unit_({hello, hello_length}, "wor"_sv, 6, 6); // Multi-byte needle
    check_find_views_unit_({hello, hello_length}, "xyz"_sv, STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX); // Missing

    // A mask of no CPU capability finds no kernel; the finder picks what the dispatch point runs.
    sz_cptr_t unused = nullptr;
    verify(sz_find_best(hello, hello_length, "wor", 3, &unused, sz_cap_metal_k, nullptr) == sz_missing_kernel_k);
    sz_kernel_punned_t kernel = nullptr;
    sz_capability_t capability = 0;
    verify(sz_find_kernel_punned(sz_kernel_find_k, sz_cap_serial_k, &kernel, &capability) == sz_success_k);
    verify(capability == sz_cap_serial_k);
    verify(kernel_result<sz_cptr_t>((sz_kernel_find_t)kernel, hello, hello_length, "wor", 3) == hello + 6);
    verify(sz_find_kernel_punned(sz_kernel_find_k, sz::default_capabilities(), &kernel, &capability) == sz_success_k);
    verify(kernel_result<sz_cptr_t>((sz_kernel_find_t)kernel, hello, hello_length, "wor", 3) ==
           kernel_result<sz_cptr_t>(cpu_best<sz_find_best>, hello, hello_length, "wor", 3));

    // `sz_find_byte_from` / `sz_find_byte_not_from` / `sz_rfind_byte_from` /
    // `sz_rfind_byte_not_from`: the `_from` family takes the needle bytes as the accepted set (the
    // byteset built out of them), and the `_not_from` family inverts that set, so they are the
    // byteset family above spelled with a needle string in place of an `sz_byteset_t`. Against
    // "hello world" and the needle "helo" (accepts h, e, l, o): the first accepted byte is 'h' at
    // offset 0 and the last is 'l' at offset 9; the first byte not in the set is ' ' at offset 5,
    // and the last byte not in the set is 'd' at offset 10.
    verify(sz_find_byte_from(hello, hello_length, "helo", 4) == hello + 0);       // First accepted: 'h'
    verify(sz_rfind_byte_from(hello, hello_length, "helo", 4) == hello + 9);      // Last accepted: 'l'
    verify(sz_find_byte_not_from(hello, hello_length, "helo", 4) == hello + 5);   // First rejected: ' '
    verify(sz_rfind_byte_not_from(hello, hello_length, "helo", 4) == hello + 10); // Last rejected: 'd'
    // An empty needle accepts nothing, so `_from` must miss everywhere and `_not_from` must hit immediately.
    verify(sz_find_byte_from(hello, hello_length, "", 0) == STRINGZILLA_NULL_CHAR);
    verify(sz_rfind_byte_from(hello, hello_length, "", 0) == STRINGZILLA_NULL_CHAR);
    verify(sz_find_byte_not_from(hello, hello_length, "", 0) == hello + 0);
    verify(sz_rfind_byte_not_from(hello, hello_length, "", 0) == hello + hello_length - 1);
    // A needle covering the whole alphabet present in the haystack leaves `_not_from` with nothing to reject.
    verify(sz_find_byte_not_from(hello, hello_length, hello, hello_length) == STRINGZILLA_NULL_CHAR);
    verify(sz_rfind_byte_not_from(hello, hello_length, hello, hello_length) == STRINGZILLA_NULL_CHAR);

    // `sz_order_best` / `sz_equal_best`: ordering and byte-equality on hand-verifiable pairs.
    check_compare_unit_(compare_dispatched);
}

/** Tests the string class comparison methods, such as @c compare and `operator==`. */
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

#pragma endregion Unit

#pragma region Safety

/** The STL reference for a byteset matcher: an STL character-set search over the set's members. */
template <std::size_t (*search_)(std::string_view, std::string_view)>
struct stl_matcher_bytes_ {
    using size_type = std::size_t;
    std::string_view members_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(std::string_view haystack) const noexcept { return search_(haystack, members_); }
};

static std::size_t stl_first_of_(std::string_view h, std::string_view n) { return h.find_first_of(n); }
static std::size_t stl_last_of_(std::string_view h, std::string_view n) { return h.find_last_of(n); }
static std::size_t stl_first_not_of_(std::string_view h, std::string_view n) { return h.find_first_not_of(n); }
static std::size_t stl_last_not_of_(std::string_view h, std::string_view n) { return h.find_last_not_of(n); }

/** Evaluates the correctness of a "matcher", searching for all the occurrences of a needle in a
 *  haystack formed of @p haystack_pattern repeated from one to @c max_repeats times, misaligned by
 *  @p misalignment bytes within the cacheline. */
template <typename stl_matches_, typename sz_matches_>
void check_find_misaligned_(test_context_t &context, std::string_view haystack_pattern,
                            typename stl_matches_::matcher_type const &stl_matcher,
                            typename sz_matches_::matcher_type const &sz_matcher, std::size_t misalignment) {
    // Each repetition re-scans the whole growing haystack, so the work is quadratic in this count, and it is
    // multiplied again by every case, misalignment and matcher family.
    std::size_t const max_repeats = context.iterations_quadratic(40);

    // Allocate a buffer to store the haystack with enough padding to mis-align it.
    std::size_t haystack_buffer_length = max_repeats * haystack_pattern.size() + 2 * sz_default_alignment_k;
    std::vector<char> haystack_buffer(haystack_buffer_length, 'x');
    char *haystack = haystack_buffer.data();

    while (reinterpret_cast<std::uintptr_t>(haystack) % sz_default_alignment_k != misalignment) ++haystack;

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
        auto haystack_sz = sz::string_view_t(haystack, haystack_length);

        // Wrap into ranges
        auto matches_stl = stl_matches_(haystack_stl, stl_matcher);
        auto matches_sz = sz_matches_(haystack_sz, sz_matcher);
        auto begin_stl = matches_stl.begin();
        auto begin_sz = matches_sz.begin();
        auto end_stl = matches_stl.end();
        auto end_sz = matches_sz.end();
        auto count_stl = std::distance(begin_stl, end_stl);
        auto count_sz = std::distance(begin_sz, end_sz);

        // To simplify debugging, let's first export all the match offsets, and only then compare them
        auto const offsets_stl = offsets_within(haystack_stl, matches_stl);
        auto const offsets_sz = offsets_within(haystack_sz, matches_sz);
        auto print_all_matches = [&]() {
            fmt::println(stderr, "Breakdown of found matches:\n- STL ({}): {}\n- StringZilla ({}): {}",
                         offsets_stl.size(), fmt::join(offsets_stl, " "), offsets_sz.size(),
                         fmt::join(offsets_sz, " "));
        };

        for (std::size_t match_idx = 0; begin_stl != end_stl && begin_sz != end_sz;
             ++begin_stl, ++begin_sz, ++match_idx) {
            auto match_stl = *begin_stl;
            auto match_sz = *begin_sz;
            if (match_stl.data() != match_sz.data()) {
                fmt::println(stderr, "Mismatch at index #{}: {} != {}", match_idx,
                             match_stl.data() - haystack_stl.data(), match_sz.data() - haystack_sz.data());
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

#if defined(__SANITIZE_ADDRESS__)
        // Don't forget to manually unpoison the prefix and the suffix.
        ASAN_UNPOISON_MEMORY_REGION(haystack_buffer.data(), poisoned_prefix_length);
        ASAN_UNPOISON_MEMORY_REGION(haystack + haystack_length, poisoned_suffix_length);
#endif
    }
}

/** Evaluates the correctness of a "matcher", searching for all the occurrences of the
 *  @p needle_stl, as a substring, as a set of allowed characters, or as a set of disallowed
 *  characters, in a haystack. */
void check_find_misaligned_(test_context_t &context, std::string_view haystack_pattern, std::string_view needle_stl,
                            std::size_t misalignment) {
    sz::string_view_t const needle_sz(needle_stl.data(), needle_stl.size());
    sz::byteset_t const set(needle_stl.data(), needle_stl.size());

    check_find_misaligned_<                                                             //
        sz::find_matches_view<std::string_view, sz::matcher_find<std::string_view>>,    //
        sz::find_matches_view<sz::string_view_t, sz::matcher_find<sz::string_view_t>>>( //
        context, haystack_pattern, {needle_stl}, {needle_sz}, misalignment);

    check_find_misaligned_<                                                               //
        sz::rfind_matches_view<std::string_view, sz::matcher_rfind<std::string_view>>,    //
        sz::rfind_matches_view<sz::string_view_t, sz::matcher_rfind<sz::string_view_t>>>( //
        context, haystack_pattern, {needle_stl}, {needle_sz}, misalignment);

    check_find_misaligned_<                                                                      //
        sz::find_matches_view<std::string_view, stl_matcher_bytes_<stl_first_of_>>,              //
        sz::find_matches_view<sz::string_view_t, sz::matcher_find_first_of<sz::string_view_t>>>( //
        context, haystack_pattern, {needle_stl}, {set}, misalignment);

    check_find_misaligned_<                                                                      //
        sz::rfind_matches_view<std::string_view, stl_matcher_bytes_<stl_last_of_>>,              //
        sz::rfind_matches_view<sz::string_view_t, sz::matcher_find_last_of<sz::string_view_t>>>( //
        context, haystack_pattern, {needle_stl}, {set}, misalignment);

    check_find_misaligned_<                                                                      //
        sz::find_matches_view<std::string_view, stl_matcher_bytes_<stl_first_not_of_>>,          //
        sz::find_matches_view<sz::string_view_t, sz::matcher_find_first_of<sz::string_view_t>>>( //
        context, haystack_pattern, {needle_stl}, {set.inverted()}, misalignment);

    check_find_misaligned_<                                                                      //
        sz::rfind_matches_view<std::string_view, stl_matcher_bytes_<stl_last_not_of_>>,          //
        sz::rfind_matches_view<sz::string_view_t, sz::matcher_find_last_of<sz::string_view_t>>>( //
        context, haystack_pattern, {needle_stl}, {set.inverted()}, misalignment);
}

/** Replays the misaligned-repetition search across a fixed sweep of intra-cacheline offsets. */
void check_find_misaligned_(test_context_t &context, std::string_view haystack_pattern, std::string_view needle_stl) {
    check_find_misaligned_(context, haystack_pattern, needle_stl, 0);
    check_find_misaligned_(context, haystack_pattern, needle_stl, 1);
    check_find_misaligned_(context, haystack_pattern, needle_stl, 2);
    check_find_misaligned_(context, haystack_pattern, needle_stl, 3);
    check_find_misaligned_(context, haystack_pattern, needle_stl, 63);
    check_find_misaligned_(context, haystack_pattern, needle_stl, 24);
    check_find_misaligned_(context, haystack_pattern, needle_stl, 33);
}

/** Extensively tests the string class search methods, such as @c find and @c find_first_of,
 *  covering alignment cases within a cache line, repetitive patterns, and overlapping matches. */
void test_find_misaligned_equivalence(test_context_t &context) {
    // When haystack is only formed of needles:
    check_find_misaligned_(context, "a", "a");
    check_find_misaligned_(context, "ab", "ab");
    check_find_misaligned_(context, "abc", "abc");
    check_find_misaligned_(context, "abcd", "abcd");
    check_find_misaligned_(context, {sz::base64(), sizeof(sz::base64())}, {sz::base64(), sizeof(sz::base64())});
    check_find_misaligned_(context, {sz::ascii_lowercase(), sizeof(sz::ascii_lowercase())},
                           {sz::ascii_lowercase(), sizeof(sz::ascii_lowercase())});
    check_find_misaligned_(context, {sz::ascii_printables(), sizeof(sz::ascii_printables())},
                           {sz::ascii_printables(), sizeof(sz::ascii_printables())});

    // When we are dealing with NULL characters inside the string
    check_find_misaligned_(context, "\0", "\0");
    check_find_misaligned_(context, "a\0", "a\0");
    check_find_misaligned_(context, "ab\0", "ab");
    check_find_misaligned_(context, "ab\0", "ab\0");
    check_find_misaligned_(context, "abc\0", "abc");
    check_find_misaligned_(context, "abc\0", "abc\0");
    check_find_misaligned_(context, "abcd\0", "abcd");

    // When searching for all-null needles in a haystack with no null bytes.
    // This exercises the SIMD tail path where masked-off lanes are zeroed:
    // if the needle characters are also zero, spurious matches appear at
    // invalid offsets beyond the haystack, causing OOB reads.
    check_find_misaligned_(context, "a", {"\0\0", 2});
    check_find_misaligned_(context, "a", {"\0\0\0", 3});
    check_find_misaligned_(context, "a", {"\0\0\0\0", 4});
    check_find_misaligned_(context, "a", {"\0\0\0\0\0", 5});
    check_find_misaligned_(context, "abcd", {"\0\0", 2});
    check_find_misaligned_(context, "abcd", {"\0\0\0\0", 4});

    // When haystack is formed of equidistant needles:
    check_find_misaligned_(context, "ab", "a");
    check_find_misaligned_(context, "abc", "a");
    check_find_misaligned_(context, "abcd", "a");

    // When matches occur in between pattern words:
    check_find_misaligned_(context, "ab", "ba");
    check_find_misaligned_(context, "abc", "ca");
    check_find_misaligned_(context, "abcd", "da");

    // Examples targeted exactly against the Raita heuristic,
    // which matches the first, the last, and the middle characters with SIMD.
    check_find_misaligned_(context, "aaabbccc", "aaabbccc");
    check_find_misaligned_(context, "axabbcxc", "aaabbccc");
    check_find_misaligned_(context, "axabbcxcaaabbccc", "aaabbccc");
}

/** Evaluates the correctness of look-up table transforms using random lookup tables. */
void test_lookup_equivalence(test_context_t &context, std::size_t lookup_tables_to_try, std::size_t slices_per_table) {

    std::size_t const body_length = 1024 * 1024;
    std::string body(body_length, '\0'), transformed(body_length, '\0');
    randomize_string(context.generator, body);
    std::uniform_int_distribution<int> byte_distribution(0, 255);

    // One scaled count over the whole slice budget, so the cost tracks the multiplier linearly rather than
    // compounding across nested loops. A fresh table every `slices_per_table` slices keeps the original mix.
    sz::look_up_table_t lut;
    for (std::size_t slice = 0; slice != context.iterations(lookup_tables_to_try * slices_per_table); ++slice) {
        if (slice % slices_per_table == 0)
            for (std::size_t index = 0; index < 256; ++index)
                lut[(char)index] = (char)byte_distribution(context.generator);

        std::uniform_int_distribution<std::size_t> offset_distribution(0, body_length - 1);
        std::size_t const slice_offset = offset_distribution(context.generator);
        std::uniform_int_distribution<std::size_t> length_distribution(0, body_length - slice_offset - 1);
        std::size_t const slice_length = length_distribution(context.generator);

        verify(sz::succeeded(sz::lookup(sz::string_view_t(body.data() + slice_offset, slice_length), lut,
                                        &transformed[0] + slice_offset)));
        for (std::size_t index = 0; index != slice_length; ++index)
            verify(transformed[slice_offset + index] == lut[body[slice_offset + index]]);
    }
}

/** Degenerate and boundary shapes for the search dispatch points, asserting survival and bounds. */
void test_find_safety() { check_find_safety_(find_dispatched); }

#pragma endregion Safety

#pragma region Drivers

/** Drives the serial-vs-dispatched substring-search and byteset-search differential tests, each
 *  forward and backward; the kernels of each capability face them in `cross_<arch>.cpp`. */
void test_find_all(test_context_t &context) { check_find_equivalence_(context, find_dispatched); }

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test
