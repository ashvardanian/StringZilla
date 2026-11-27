/**
 *  @file cpp/test.cpp
 *  @author Ash Vardanian
 *  @date November 26, 2026
 *  @brief The smoke test of the C++ SDK: an @c sz::string built and grown past its inline buffer,
 *      a find and an rfind, a hash checked against the C ABI, an argsort, an intersect, one round
 *      through each batch engine, and what the SDK refuses to compile.
 *
 *  It links the archive a @c stringzilla::cpp consumer links, not the header-only engine, so every
 *  call dispatches through the compiled kernels. The suites in @c test/ hold each kernel to the
 *  serial one on every capability; this one only checks that the public entry points are wired.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`, `std::uint8_t`
#include <cstdio>  // `std::printf`, `std::fprintf`

#include <algorithm>       // `std::equal` - the two sketches
#include <source_location> // `std::source_location` - where a check failed
#include <string>          // `std::string` - the owning temporary the ranges refuse
#include <string_view>     // `std::string_view` - the borrowed view the ranges accept
#include <vector>          // `std::vector` - the argsort input

#include "stringzilla/stringzilla.h"   // `sz_hash_best`, the C ABI the hash case cross-checks
#include "stringzilla/stringzilla.hpp" // `sz::string`, `sz::argsort`, `sz::intersect`, the engines

namespace sz = ashvardanian::stringzilla;

#pragma region Run

/** The run in progress, the one mutable global: the checks that failed in any case. */
struct test_run_t {
    std::size_t failed_checks = 0;
};

inline test_run_t test_run;

/** Holds when @p is_satisfied does; otherwise counts a failed check at the caller's line. */
inline bool expect_true(bool is_satisfied, std::source_location where = std::source_location::current()) noexcept {
    if (is_satisfied) return true;
    std::fprintf(stderr, "  check failed at %s:%u\n", where.file_name(), static_cast<unsigned>(where.line()));
    ++test_run.failed_checks;
    return false;
}

/** Runs one case and prints whether its checks held. */
inline void run_test_case(char const *label, void (*case_function)() noexcept) noexcept {
    std::size_t const failed_before = test_run.failed_checks;
    case_function();
    if (test_run.failed_checks == failed_before) std::printf("- %s ... ok\n", label);
    else std::fprintf(stderr, "- %s ... FAILED\n", label);
}

#pragma endregion Run

#pragma region Compile Time Refusals

/** Whether @p string_type_ answers a member @c find, as STL strings do and StringZilla containers
 *  refuse, naming @c sz::find instead. */
template <typename string_type_>
concept a_string_with_member_find = requires(string_type_ const &text) { text.find("x"); };

static_assert(a_string_with_member_find<std::string_view>);
static_assert(!a_string_with_member_find<sz::string_view_t>);
static_assert(!a_string_with_member_find<sz::string_t>);

/** Whether a lazy split accepts @p haystack_type_ as its argument's type, which an owning temporary
 *  fails, as it would die before the range is walked. */
template <typename haystack_type_>
concept a_splittable_haystack = requires(haystack_type_ &&haystack) {
    sz::split(static_cast<haystack_type_ &&>(haystack), " ");
};

static_assert(a_splittable_haystack<sz::string_view_t>);
static_assert(a_splittable_haystack<std::string_view>);
static_assert(a_splittable_haystack<sz::string_t &>);
static_assert(!a_splittable_haystack<std::string>);
static_assert(!a_splittable_haystack<sz::string_t>);

#pragma endregion Compile Time Refusals

#pragma region Compile Time Views

/** Whether a view built at compile time sizes, indexes, slices and walks backwards. */
consteval bool a_view_reads_at_compile_time() {
    sz::string_view_t const hello = "hello";
    return hello.size() == 5 && hello[1] == 'e' && hello.sub(1, -1) == "ell" && hello.substr(3) == "lo" &&
           *hello.rbegin() == 'o';
}

static_assert(a_view_reads_at_compile_time());
static_assert(sz::string_view_t("ab") < "abc" && sz::string_view_t("abc") < "abd" && sz::string_view_t("\xFF") > "a" &&
              sz::string_view_t {} == "" && sz::string_view_t {} < "a");
static_assert(sz::less_t {}("abc", "abd") && sz::equal_to_t {}("abc", "abc") && !sz::equal_to_t {}("abc", "ab"));

/** Whether an Apache Arrow column built at compile time reads its strings in place. */
consteval bool a_tape_indexes_at_compile_time() {
    char const bytes[] = "abcde";
    sz::u32_t const offsets[] = {0, 2, 5};
    sz::strings_view_t const column(bytes, offsets, 2);
    return column.size() == 2 && *column.begin() == "ab" && column[1] == "cde";
}

static_assert(a_tape_indexes_at_compile_time());

#pragma endregion Compile Time Views

/** An @c sz::string starts empty, holds what it is made from, and keeps its prefix as it grows past
 *  the inline buffer onto the heap. */
static void test_string_append() noexcept {
    sz::string_t fresh;
    expect_true(fresh.empty() && fresh.size() == 0 && fresh == "");

    auto [text, status] = sz::string_t::make("hello");
    if (!expect_true(sz::succeeded(status))) return;
    expect_true(text.size() == 5 && text == "hello");

    std::size_t const inline_capacity = text.capacity();
    for (std::size_t round = 0; round != 32; ++round) expect_true(sz::succeeded(text.append(", world")));
    expect_true(text.size() == 5 + 32 * 7);
    expect_true(text.capacity() > inline_capacity && text.capacity() >= text.size());
    expect_true(text.view().substr(0, 12) == "hello, world");
    expect_true(sz::succeeded(text.push_back('!')) && text.view().back() == '!');
}

/** The first and last occurrences in "abracadabra", of substrings and of single characters. */
static void test_string_find() noexcept {
    auto [text, status] = sz::string_t::make("abracadabra");
    if (!expect_true(sz::succeeded(status))) return;
    expect_true(sz::find(text, "abra") == 0 && sz::rfind(text, "abra") == 7);
    expect_true(sz::find(text, "cad") == 4 && sz::find(text, "abra", 1) == 7);
    expect_true(sz::find(text, 'c') == 4 && sz::rfind(text, 'a') == 10);
    expect_true(sz::find(text, "xyz") == sz::string_t::npos && sz::rfind(text, "xyz") == sz::string_t::npos);
}

/** The SDK hash equals the serial kernel's through the C ABI, depends on the seed, and is the same
 *  for equal bytes wherever they live. */
static void test_string_hash() noexcept {
    sz::string_view_t const view = "The quick brown fox jumps over the lazy dog";
    auto [text, status] = sz::string_t::make(view);
    if (!expect_true(sz::succeeded(status))) return;

    for (std::uint64_t const seed : {std::uint64_t {0}, std::uint64_t {42}}) {
        sz_u64_t serial = 0;
        expect_true(sz_hash_best(view.data(), view.size(), seed, &serial, sz_cap_serial_k, nullptr) == sz_success_k);
        expect_true(sz::hash(view, seed) == serial);
    }
    expect_true(sz::hash(text) == sz::hash(view) && sz::hash_t {}(view) == sz::hash(view));
    expect_true(sz::hash(view, 0) != sz::hash(view, 42));

    std::uint64_t const seeds[2] {0, 42};
    std::uint64_t hashes[2] {};
    expect_true(sz::succeeded(sz::hash_multiseed(view, seeds, hashes)));
    expect_true(hashes[0] == sz::hash(view, 0) && hashes[1] == sz::hash(view, 42));
    expect_true(sz::hash_multiseed(view, seeds, {hashes, 1}) == sz::status_t::unexpected_dimensions_k);
}

/** The permutation that sorts four words, two of them equal. */
static void test_argsort() noexcept {
    std::vector<sz::string_view_t> const words {"banana", "apple", "cherry", "apple"};
    auto [order, status] = sz::argsort(words);
    if (!expect_true(sz::succeeded(status) && order.size() == words.size())) return;
    expect_true((order[0] == 1 && order[1] == 3) || (order[0] == 3 && order[1] == 1));
    expect_true(order[2] == 0 && order[3] == 2);

    auto [descending, descending_status] = sz::argsort(words, 0, sz::sort_direction_t::descending_k);
    if (!expect_true(sz::succeeded(descending_status) && descending.size() == words.size())) return;
    expect_true(descending[0] == 2 && descending[1] == 0);
}

/** The positions of the two words both lists hold, whatever order the hash table finds them in. */
static void test_intersect() noexcept {
    std::vector<sz::string_view_t> const first {"apple", "banana", "cherry"};
    std::vector<sz::string_view_t> const second {"cherry", "durian", "apple", "elderberry"};
    auto [offsets, status] = sz::intersect(first, second, 42);
    if (!expect_true(sz::succeeded(status) && offsets.first_offsets.size() == 2)) return;
    for (std::size_t match = 0; match != 2; ++match)
        expect_true(first[offsets.first_offsets[match]] == second[offsets.second_offsets[match]]);
}

/** Two queries against two candidates into a block with a padded row, a block one entry short, and
 *  the same block from an engine an empty mask prepared for the serial kernels. */
static void test_levenshtein_engine() noexcept {
    sz::string_view_t const queries[] {"kitten", "flaw"}, candidates[] {"sitting", "lawn"};
    auto [engine, status] = sz::levenshtein_engine_t::make(sz::strings_view_t(queries), sz_levenshtein_bytes_k);
    if (!expect_true(sz::succeeded(status) && engine.queries_count() == 2)) return;
    std::size_t distances[5] {};
    expect_true(
        sz::succeeded(engine.distances(sz::strings_view_t(candidates), SIZE_MAX, distances, 3 * sizeof(std::size_t))));
    expect_true(distances[0] == 3 && distances[1] == 5 && distances[3] == 7 && distances[4] == 2);
    expect_true(engine.distances(sz::strings_view_t(candidates), SIZE_MAX, {distances, 4}, 3 * sizeof(std::size_t)) ==
                sz::status_t::unexpected_dimensions_k);

    auto [serial, serial_status] = sz::levenshtein_engine_t::make(sz::strings_view_t(queries), sz_levenshtein_bytes_k,
                                                                  nullptr, sz_cap_serial_k);
    if (!expect_true(sz::succeeded(serial_status))) return;
    std::size_t serial_distances[4] {};
    expect_true(sz::succeeded(
        serial.distances(sz::strings_view_t(candidates), SIZE_MAX, serial_distances, 2 * sizeof(std::size_t))));
    expect_true(serial_distances[0] == 3 && serial_distances[1] == 5 && serial_distances[2] == 7 &&
                serial_distances[3] == 2);
}

/** The share of a query's bigrams each candidate holds, and equal sketches of equal texts. */
static void test_ngrams_engine() noexcept {
    sz::string_view_t const queries[] {"abcd"}, candidates[] {"xabcdx", "zzz"};
    std::size_t const widths[] {2};
    auto [engine, status] = sz::ngrams_engine_t::make(sz::strings_view_t(queries), widths, 0);
    if (!expect_true(sz::succeeded(status) && engine.queries_count() == 1 && engine.widths_count() == 1)) return;
    float shares[2] {-1.0f, -1.0f};
    expect_true(
        sz::succeeded(engine.overlaps(sz::strings_view_t(candidates), 1.0f, shares, 2 * sizeof(float), sizeof(float))));
    expect_true(shares[0] == 1.0f && shares[1] == 0.0f);
    expect_true(engine.overlaps(sz::strings_view_t(candidates), 1.0f, {shares, 1}, 2 * sizeof(float), sizeof(float)) ==
                sz::status_t::unexpected_dimensions_k);

    sz::string_view_t const texts[] {"the same text", "the same text"};
    std::uint8_t sketches[16] {};
    expect_true(sz::succeeded(engine.sketches(sz::strings_view_t(texts), 8, sketches, 8)));
    expect_true(std::equal(sketches, sketches + 8, sketches + 8));
    expect_true(engine.sketches(sz::strings_view_t(texts), 8, {sketches, 15}, 8) ==
                sz::status_t::unexpected_dimensions_k);
}

/** Overlapping "he" and "she" in "ushers" and nothing in "none", marked, counted and found. */
static void test_substrings_engine() noexcept {
    sz::string_view_t const needles[] {"he", "she"}, haystacks[] {"ushers", "none"};
    auto [engine, status] = sz::substrings_engine_t::make(sz::strings_view_t(needles), sz_cased_k,
                                                          sz_substrings_overlapping_k,
                                                          STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0);
    if (!expect_true(sz::succeeded(status) && engine.needles_count() == 2)) return;
    sz_u1x8_t contained[1] {};
    expect_true(sz::succeeded(engine.contains(sz::strings_view_t(haystacks), contained)) && contained[0] == 1);
    std::size_t counts[2] {};
    expect_true(sz::succeeded(engine.counts(sz::strings_view_t(haystacks), counts, sizeof(std::size_t))));
    expect_true(counts[0] == 2 && counts[1] == 0);
    sz_match_t matches[2] {};
    std::size_t offsets[3] {};
    expect_true(sz::succeeded(engine.find(sz::strings_view_t(haystacks), matches, offsets)));
    expect_true(offsets[1] == 2 && offsets[2] == 2 && engine.report().matches_stored == 2);
    expect_true(engine.find(sz::strings_view_t(haystacks), matches, {offsets, 2}) ==
                sz::status_t::unexpected_dimensions_k);
}

/** One JSON path counted in a document holding it and in one without. */
static void test_subdocuments_engine() noexcept {
    sz::string_view_t const paths[] {"$.a"}, documents[] {R"({"a": 1})", R"({"b": 2})"};
    std::size_t malformed_index = 0;
    auto [engine, status] = sz::subdocuments_engine_t::make(sz::strings_view_t(paths), sz_subdocuments_json_k,
                                                            sz_subdocuments_options_none_k, malformed_index);
    if (!expect_true(sz::succeeded(status) && engine.paths_count() == 1)) return;
    std::size_t counts[2] {};
    expect_true(
        sz::succeeded(engine.counts(sz::strings_view_t(documents), counts, sizeof(std::size_t), malformed_index)));
    expect_true(counts[0] == 1 && counts[1] == 0);
    expect_true(engine.counts(sz::strings_view_t(documents), {counts, 1}, sizeof(std::size_t), malformed_index) ==
                sz::status_t::unexpected_dimensions_k);
}

/** Runs of digits counted and rewritten, and a malformed pattern named by its index. */
static void test_regex_engine() noexcept {
    sz::string_view_t const patterns[] {"[0-9]+"}, haystacks[] {"a1b22", "none"}, replacements[] {"#"};
    std::size_t malformed_index = 0;
    auto [engine, status] = sz::regex_engine_t::make(sz::strings_view_t(patterns), sz_cased_k, malformed_index);
    if (!expect_true(sz::succeeded(status) && engine.patterns_count() == 1)) return;
    std::size_t counts[2] {};
    expect_true(sz::succeeded(engine.counts(sz::strings_view_t(haystacks), counts, sizeof(std::size_t))));
    expect_true(counts[0] == 2 && counts[1] == 0);
    char target[16] {};
    std::size_t offsets[3] {};
    expect_true(sz::succeeded(
        engine.replace(sz::strings_view_t(haystacks), sz::strings_view_t(replacements), target, offsets)));
    expect_true(offsets[2] == 8 && sz::string_view_t(target, offsets[2]) == "a#b#none");

    sz::string_view_t const broken[] {"ok", "(unclosed"};
    auto [refused, refused_status] = sz::regex_engine_t::make(sz::strings_view_t(broken), sz_cased_k, malformed_index);
    expect_true(refused_status == sz::status_t::invalid_syntax_k && malformed_index == 1);
}

int main() {
    std::printf("StringZilla %d.%d.%d through the C++ SDK\n", sz_version_major(), sz_version_minor(),
                sz_version_patch());
    run_test_case("sz::string::append", test_string_append);
    run_test_case("sz::string::find", test_string_find);
    run_test_case("sz::string_view_t::hash", test_string_hash);
    run_test_case("sz::argsort", test_argsort);
    run_test_case("sz::intersect", test_intersect);
    run_test_case("sz::levenshtein_engine_t", test_levenshtein_engine);
    run_test_case("sz::ngrams_engine_t", test_ngrams_engine);
    run_test_case("sz::substrings_engine_t", test_substrings_engine);
    run_test_case("sz::subdocuments_engine_t", test_subdocuments_engine);
    run_test_case("sz::regex_engine_t", test_regex_engine);

    if (test_run.failed_checks != 0) {
        std::fprintf(stderr, "%zu check(s) failed\n", test_run.failed_checks);
        return 1;
    }
    std::printf("All checks passed\n");
    return 0;
}
