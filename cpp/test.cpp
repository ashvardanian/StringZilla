/**
 *  @file cpp/test.cpp
 *  @author Ash Vardanian
 *  @date November 26, 2026
 *  @brief The smoke test of the C++ SDK: an @c sz::string built and grown past its inline buffer,
 *      a find and an rfind, a hash checked against the C ABI, and an argsort.
 *
 *  It links the archive a @c stringzilla::cpp consumer links, not the header-only engine, so every
 *  call dispatches through the compiled kernels. The suites in @c test/ hold each kernel to the
 *  serial one on every capability; this one only checks that the public entry points are wired.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstdio>  // `std::printf`, `std::fprintf`

#include <source_location> // `std::source_location` - where a check failed
#include <vector>          // `std::vector` - the argsort input

#include "stringzilla/stringzilla.h"   // `sz_hash_best`, the C ABI the hash case cross-checks
#include "stringzilla/stringzilla.hpp" // `sz::string`, `sz::string_view_t`, `sz::argsort`

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
    expect_true(text.find("abra") == 0 && text.rfind("abra") == 7);
    expect_true(text.find("cad") == 4 && text.find("abra", 1) == 7);
    expect_true(text.find('c') == 4 && text.rfind('a') == 10);
    expect_true(text.find("xyz") == sz::string_t::npos && text.rfind("xyz") == sz::string_t::npos);
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
        expect_true(view.hash(seed) == serial);
    }
    expect_true(text.hash() == view.hash() && sz::hash {}(view) == view.hash());
    expect_true(view.hash(0) != view.hash(42));
}

/** The permutation that sorts four words, two of them equal. */
static void test_argsort() noexcept {
    std::vector<sz::string_view_t> const words {"banana", "apple", "cherry", "apple"};
    auto [order, status] = sz::argsort(words);
    if (!expect_true(sz::succeeded(status) && order.size() == words.size())) return;
    expect_true((order[0] == 1 && order[1] == 3) || (order[0] == 3 && order[1] == 1));
    expect_true(order[2] == 0 && order[3] == 2);
}

int main() {
    std::printf("StringZilla %d.%d.%d through the C++ SDK\n", sz_version_major(), sz_version_minor(),
                sz_version_patch());
    run_test_case("sz::string::append", test_string_append);
    run_test_case("sz::string::find", test_string_find);
    run_test_case("sz::string_view_t::hash", test_string_hash);
    run_test_case("sz::argsort", test_argsort);

    if (test_run.failed_checks != 0) {
        std::fprintf(stderr, "%zu check(s) failed\n", test_run.failed_checks);
        return 1;
    }
    std::printf("All checks passed\n");
    return 0;
}
