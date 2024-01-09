#include <algorithm> // `std::transform`
#include <cassert>   // assertions
#include <cstdio>    // `std::printf`
#include <cstring>   // `std::memcpy`
#include <iterator>  // `std::distance`
#include <memory>    // `std::allocator`
#include <random>    // `std::random_device`
#include <vector>    // `std::vector`

// Overload the following with caution.
// Those parameters must never be explicitly set during releases,
// but they come handy during development, if you want to validate
// different ISA-specific implementations.
// #define SZ_USE_X86_AVX2 0
// #define SZ_USE_X86_AVX512 0
// #define SZ_USE_ARM_NEON 0
// #define SZ_USE_ARM_SVE 0

#include <string>                      // Baseline
#include <string_view>                 // Baseline
#include <stringzilla/stringzilla.hpp> // Contender

#include <test.hpp> // `levenshtein_baseline`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using sz::literals::operator""_sz;

/**
 *  @brief  Several string processing operations rely on computing integer logarithms.
 *          Failures in such operations will result in wrong `resize` outcomes and heap corruption.
 */
static void test_arithmetical_utilities() {

    assert(sz_u64_clz(0x0000000000000001ull) == 63);
    assert(sz_u64_clz(0x0000000000000002ull) == 62);
    assert(sz_u64_clz(0x0000000000000003ull) == 62);
    assert(sz_u64_clz(0x0000000000000004ull) == 61);
    assert(sz_u64_clz(0x0000000000000007ull) == 61);
    assert(sz_u64_clz(0x8000000000000001ull) == 0);
    assert(sz_u64_clz(0xffffffffffffffffull) == 0);
    assert(sz_u64_clz(0x4000000000000000ull) == 1);

    assert(sz_size_log2i_nonzero(1) == 0);
    assert(sz_size_log2i_nonzero(2) == 1);
    assert(sz_size_log2i_nonzero(3) == 1);

    assert(sz_size_log2i_nonzero(4) == 2);
    assert(sz_size_log2i_nonzero(5) == 2);
    assert(sz_size_log2i_nonzero(7) == 2);

    assert(sz_size_log2i_nonzero(8) == 3);
    assert(sz_size_log2i_nonzero(9) == 3);

    assert(sz_size_bit_ceil(0) == 0);
    assert(sz_size_bit_ceil(1) == 1);

    assert(sz_size_bit_ceil(2) == 2);
    assert(sz_size_bit_ceil(3) == 4);
    assert(sz_size_bit_ceil(4) == 4);

    assert(sz_size_bit_ceil(77) == 128);
    assert(sz_size_bit_ceil(127) == 128);
    assert(sz_size_bit_ceil(128) == 128);

    assert(sz_size_bit_ceil(uint64_t(1e6)) == (1ull << 20));
    assert(sz_size_bit_ceil(uint64_t(2e6)) == (1ull << 21));
    assert(sz_size_bit_ceil(uint64_t(4e6)) == (1ull << 22));
    assert(sz_size_bit_ceil(uint64_t(8e6)) == (1ull << 23));

    assert(sz_size_bit_ceil(uint64_t(1.6e7)) == (1ull << 24));
    assert(sz_size_bit_ceil(uint64_t(3.2e7)) == (1ull << 25));
    assert(sz_size_bit_ceil(uint64_t(6.4e7)) == (1ull << 26));

    assert(sz_size_bit_ceil(uint64_t(1.28e8)) == (1ull << 27));
    assert(sz_size_bit_ceil(uint64_t(2.56e8)) == (1ull << 28));
    assert(sz_size_bit_ceil(uint64_t(5.12e8)) == (1ull << 29));

    assert(sz_size_bit_ceil(uint64_t(1e9)) == (1ull << 30));
    assert(sz_size_bit_ceil(uint64_t(2e9)) == (1ull << 31));
    assert(sz_size_bit_ceil(uint64_t(4e9)) == (1ull << 32));
    assert(sz_size_bit_ceil(uint64_t(8e9)) == (1ull << 33));

    assert(sz_size_bit_ceil(uint64_t(1.6e10)) == (1ull << 34));

    assert(sz_size_bit_ceil((1ull << 62)) == (1ull << 62));
    assert(sz_size_bit_ceil((1ull << 62) + 1) == (1ull << 63));
    assert(sz_size_bit_ceil((1ull << 63)) == (1ull << 63));
}

#define assert_scoped(init, operation, condition) \
    {                                             \
        init;                                     \
        operation;                                \
        assert(condition);                        \
    }

/**
 *  @brief  Invokes different C++ member methods of the string class to make sure they all pass compilation.
 *          This test guarantees API compatibility with STL `std::basic_string` template.
 */
template <typename string_type>
static void test_compilation() {

    using str = string_type;

    // Constructors
    assert(str().empty());                             // Test default constructor
    assert(str("hello").size() == 5);                  // Test constructor with c-string
    assert(str("hello", 4) == "hell");                 // Construct from substring
    assert(str(5, 'a') == "aaaaa");                    // Construct with count and character
    assert(str({'h', 'e', 'l', 'l', 'o'}) == "hello"); // Construct from initializer list
    assert(str(str("hello"), 2) == "llo");             // Construct from another string suffix
    assert(str(str("hello"), 2, 2) == "ll");           // Construct from another string range

    // Assignments
    assert_scoped(str s, s = "hello", s == "hello");
    assert_scoped(str s, s.assign("hello"), s == "hello");
    assert_scoped(str s, s.assign("hello", 4), s == "hell");
    assert_scoped(str s, s.assign(5, 'a'), s == "aaaaa");
    assert_scoped(str s, s.assign({'h', 'e', 'l', 'l', 'o'}), s == "hello");
    assert_scoped(str s, s.assign(str("hello")), s == "hello");
    assert_scoped(str s, s.assign(str("hello"), 2), s == "llo");
    assert_scoped(str s, s.assign(str("hello"), 2, 2), s == "ll");

    // Comparisons
    assert(str("a") != str("b"));
    assert(std::strcmp(str("c_str").c_str(), "c_str") == 0);
    assert(str("a") < str("b"));
    assert(str("a") <= str("b"));
    assert(str("b") > str("a"));
    assert(str("b") >= str("a"));
    assert(str("a") < str("aa"));

    // Allocations, capacity and memory management
    assert_scoped(str s, s.reserve(10), s.capacity() >= 10);
    assert_scoped(str s, s.resize(10), s.size() == 10);
    assert_scoped(str s, s.resize(10, 'a'), s.size() == 10 && s == "aaaaaaaaaa");
    assert(str("size").size() == 4 && str("length").length() == 6);
    assert(str().max_size() > 0);
    assert(str().get_allocator() == std::allocator<char>());

    // Incremental construction
    assert(str().append("test") == "test");
    assert(str("test") + "ing" == "testing");
    assert(str("test") + str("ing") == "testing");
    assert(str("test") + str("ing") + str("123") == "testing123");
    assert_scoped(str s = "__", s.insert(1, "test"), s == "_test_");
    assert_scoped(str s = "__", s.insert(1, "test", 2), s == "_te_");
    assert_scoped(str s = "__", s.insert(1, 5, 'a'), s == "_aaaaa_");
    assert_scoped(str s = "__", s.insert(1, {'a', 'b', 'c'}), s == "_abc_");
    assert_scoped(str s = "__", s.insert(1, str("test")), s == "_test_");
    assert_scoped(str s = "__", s.insert(1, str("test"), 2), s == "_st_");
    assert_scoped(str s = "__", s.insert(1, str("test"), 2, 1), s == "_s_");
    assert_scoped(str s = "test", s.erase(1, 2), s == "tt");
    assert_scoped(str s = "test", s.erase(1), s == "t");
    assert_scoped(str s = "test", s.erase(s.begin() + 1), s == "tst");
    assert_scoped(str s = "test", s.erase(s.begin() + 1, s.begin() + 2), s == "tst");
    assert_scoped(str s = "test", s.erase(s.begin() + 1, s.begin() + 3), s == "tt");
    assert_scoped(str s = "!?", s.push_back('a'), s == "!?a");
    assert_scoped(str s = "!?", s.pop_back(), s == "!");

    // Following are missing in strings, but are present in vectors.
    // assert_scoped(str s = "!?", s.push_front('a'), s == "a!?");
    // assert_scoped(str s = "!?", s.pop_front(), s == "?");

    // Element access
    assert(str("test")[0] == 't');
    assert(str("test").at(1) == 'e');
    assert(str("front").front() == 'f');
    assert(str("back").back() == 'k');
    assert(*str("data").data() == 'd');

    // Iterators
    assert(*str("begin").begin() == 'b' && *str("cbegin").cbegin() == 'c');
    assert(*str("rbegin").rbegin() == 'n' && *str("crbegin").crbegin() == 'n');

    // Slices
    assert(str("hello world").substr(0, 5) == "hello");
    assert(str("hello world").substr(6, 5) == "world");
    assert(str("hello world").substr(6) == "world");
    assert(str("hello world").substr(6, 100) == "world");

    // Substring and character search in normal and reverse directions
    assert(str("hello").find("ell") == 1);
    assert(str("hello").find("ell", 1) == 1);
    assert(str("hello").find("ell", 2) == str::npos);
    assert(str("hello").find("ell", 1, 2) == 1);
    assert(str("hello").rfind("l") == 3);
    assert(str("hello").rfind("l", 2) == 2);
    assert(str("hello").rfind("l", 1) == str::npos);

    // ! `rfind` and `find_last_of` are not consitent in meaning of their arguments.
    assert(str("hello").find_first_of("le") == 1);
    assert(str("hello").find_first_of("le", 1) == 1);
    assert(str("hello").find_last_of("le") == 3);
    assert(str("hello").find_last_of("le", 2) == 2);
    assert(str("hello").find_first_not_of("hel") == 4);
    assert(str("hello").find_first_not_of("hel", 1) == 4);
    assert(str("hello").find_last_not_of("hel") == 4);
    assert(str("hello").find_last_not_of("hel", 4) == 4);

    // Substitutions
    assert(str("hello").replace(1, 2, "123") == "h123lo");
    assert(str("hello").replace(1, 2, str("123"), 1) == "h23lo");
    assert(str("hello").replace(1, 2, "123", 1) == "h1lo");
    assert(str("hello").replace(1, 2, "123", 1, 1) == "h2lo");
    assert(str("hello").replace(1, 2, str("123"), 1, 1) == "h2lo");
    assert(str("hello").replace(1, 2, 3, 'a') == "haaalo");
    assert(str("hello").replace(1, 2, {'a', 'b'}) == "hablo");
}

/**
 *  @brief  Tests copy constructor and copy-assignment constructor of `sz::string`.
 */
static void test_constructors() {
    std::string alphabet {sz::ascii_printables, sizeof(sz::ascii_printables)};
    std::vector<sz::string> strings;
    for (std::size_t alphabet_slice = 0; alphabet_slice != alphabet.size(); ++alphabet_slice)
        strings.push_back(alphabet.substr(0, alphabet_slice));
    std::vector<sz::string> copies {strings};
    assert(copies.size() == strings.size());
    for (size_t i = 0; i < copies.size(); i++) {
        assert(copies[i].size() == strings[i].size());
        assert(copies[i] == strings[i]);
        for (size_t j = 0; j < strings[i].size(); j++) { assert(copies[i][j] == strings[i][j]); }
    }
    std::vector<sz::string> assignments = strings;
    for (size_t i = 0; i < assignments.size(); i++) {
        assert(assignments[i].size() == strings[i].size());
        assert(assignments[i] == strings[i]);
        for (size_t j = 0; j < strings[i].size(); j++) { assert(assignments[i][j] == strings[i][j]); }
    }
    assert(std::equal(strings.begin(), strings.end(), copies.begin()));
    assert(std::equal(strings.begin(), strings.end(), assignments.begin()));
}

struct accounting_allocator : public std::allocator<char> {
    inline static bool verbose = false;
    inline static std::size_t current_bytes_alloced = 0;

    template <typename... args_types>
    static void print_if_verbose(char const *fmt, args_types... args) {
        if (!verbose) return;
        std::printf(fmt, args...);
    }

    char *allocate(std::size_t n) {
        current_bytes_alloced += n;
        print_if_verbose("alloc %zd -> %zd\n", n, current_bytes_alloced);
        return std::allocator<char>::allocate(n);
    }

    void deallocate(char *val, std::size_t n) {
        assert(n <= current_bytes_alloced);
        current_bytes_alloced -= n;
        print_if_verbose("dealloc: %zd -> %zd\n", n, current_bytes_alloced);
        std::allocator<char>::deallocate(val, n);
    }

    template <typename callback_type>
    static std::size_t account_block(callback_type callback) {
        auto before = accounting_allocator::current_bytes_alloced;
        print_if_verbose("starting block: %zd\n", before);
        callback();
        auto after = accounting_allocator::current_bytes_alloced;
        print_if_verbose("ending block: %zd\n", after);
        return after - before;
    }
};

template <typename callback_type>
void assert_balanced_memory(callback_type callback) {
    auto bytes = accounting_allocator::account_block(callback);
    assert(bytes == 0);
}

static void test_memory_stability_for_length(std::size_t len = 1ull << 10) {
    std::size_t iterations = 4;

    assert(accounting_allocator::current_bytes_alloced == 0);
    using string = sz::basic_string<accounting_allocator>;
    string base;

    for (std::size_t i = 0; i < len; i++) base.push_back('c');
    assert(base.length() == len);

    // Do copies leak?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; i++) {
            string copy(base);
            assert(copy.length() == len);
            assert(copy == base);
        }
    });

    // How about assignments?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; i++) {
            string copy;
            copy = base;
            assert(copy.length() == len);
            assert(copy == base);
        }
    });

    // How about the move ctor?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; i++) {
            string unique_item(base);
            assert(unique_item.length() == len);
            assert(unique_item == base);
            string copy(std::move(unique_item));
            assert(copy.length() == len);
            assert(copy == base);
        }
    });

    // And the move assignment operator with an empty target payload?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; i++) {
            string unique_item(base);
            string copy;
            copy = std::move(unique_item);
            assert(copy.length() == len);
            assert(copy == base);
        }
    });

    // And move assignment where the target had a payload?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; i++) {
            string unique_item(base);
            string copy;
            for (std::size_t j = 0; j < 317; j++) copy.push_back('q');
            copy = std::move(unique_item);
            assert(copy.length() == len);
            assert(copy == base);
        }
    });

    // Now let's clear the base and check that we're back to zero
    base = string();
    assert(accounting_allocator::current_bytes_alloced == 0);
}

/**
 *  @brief  Tests the correctness of the string class update methods, such as `append` and `erase`.
 */
static void test_updates() {
    // Compare STL and StringZilla strings append functionality.
    char const alphabet_chars[] = "abcdefghijklmnopqrstuvwxyz";
    std::string stl_string;
    sz::string sz_string;
    for (std::size_t length = 1; length != 200; ++length) {
        char c = alphabet_chars[std::rand() % 26];
        stl_string.push_back(c);
        sz_string.push_back(c);
        assert(sz::string_view(stl_string) == sz::string_view(sz_string));
    }

    // Compare STL and StringZilla strings erase functionality.
    while (stl_string.length()) {
        std::size_t offset_to_erase = std::rand() % stl_string.length();
        std::size_t chars_to_erase = std::rand() % (stl_string.length() - offset_to_erase) + 1;
        stl_string.erase(offset_to_erase, chars_to_erase);
        sz_string.erase(offset_to_erase, chars_to_erase);
        assert(sz::string_view(stl_string) == sz::string_view(sz_string));
    }
}

/**
 *  @brief  Tests the correctness of the string class comparison methods, such as `compare` and `operator==`.
 */
static void test_comparisons() {
    // Comparing relative order of the strings
    assert("a"_sz.compare("a") == 0);
    assert("a"_sz.compare("ab") == -1);
    assert("ab"_sz.compare("a") == 1);
    assert("a"_sz.compare("a\0"_sz) == -1);
    assert("a\0"_sz.compare("a") == 1);
    assert("a\0"_sz.compare("a\0"_sz) == 0);
    assert("a"_sz == "a"_sz);
    assert("a"_sz != "a\0"_sz);
    assert("a\0"_sz == "a\0"_sz);
}

/**
 *  @brief  Tests the correctness of the string class search methods, such as `find` and `find_first_of`.
 *          This covers haystacks and needles of different lengths, as well as character-sets.
 */
static void test_search() {

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
    assert(sz::string_view(sz::base64).find_first_of("_") == sz::string_view::npos);
    assert(sz::string_view(sz::base64).find_first_of("+") == 62);
    assert(sz::string_view(sz::ascii_printables).find_first_of("~") != sz::string_view::npos);

    // Check more advanced composite operations:
    assert("abbccc"_sz.partition("bb").before.size() == 1);
    assert("abbccc"_sz.partition("bb").match.size() == 2);
    assert("abbccc"_sz.partition("bb").after.size() == 3);
    assert("abbccc"_sz.partition("bb").before == "a");
    assert("abbccc"_sz.partition("bb").match == "bb");
    assert("abbccc"_sz.partition("bb").after == "ccc");

    // Check ranges of search matches
    assert(""_sz.find_all(".").size() == 0);
    assert("a.b.c.d"_sz.find_all(".").size() == 3);
    assert("a.,b.,c.,d"_sz.find_all(".,").size() == 3);
    assert("a.,b.,c.,d"_sz.rfind_all(".,").size() == 3);
    assert("a.b,c.d"_sz.find_all(sz::character_set(".,")).size() == 3);
    assert("a...b...c"_sz.rfind_all("..", true).size() == 4);

    auto finds = "a.b.c"_sz.find_all(sz::character_set("abcd")).template to<std::vector<std::string>>();
    assert(finds.size() == 3);
    assert(finds[0] == "a");

    auto rfinds = "a.b.c"_sz.rfind_all(sz::character_set("abcd")).template to<std::vector<std::string>>();
    assert(rfinds.size() == 3);
    assert(rfinds[0] == "c");

    auto splits = ".a..c."_sz.split(sz::character_set(".")).template to<std::vector<std::string>>();
    assert(splits.size() == 5);
    assert(splits[0] == "");
    assert(splits[1] == "a");
    assert(splits[4] == "");

    assert(""_sz.split(".").size() == 1);
    assert(""_sz.rsplit(".").size() == 1);
    assert("a.b.c.d"_sz.split(".").size() == 4);
    assert("a.b.c.d"_sz.rsplit(".").size() == 4);
    assert("a.b.,c,d"_sz.split(".,").size() == 2);
    assert("a.b,c.d"_sz.split(sz::character_set(".,")).size() == 4);

    auto rsplits = ".a..c."_sz.rsplit(sz::character_set(".")).template to<std::vector<std::string>>();
    assert(rsplits.size() == 5);
    assert(rsplits[0] == "");
    assert(rsplits[1] == "c");
    assert(rsplits[4] == "");
}

/**
 *  Evaluates the correctness of a "matcher", searching for all the occurences of the `needle_stl`
 *  in a haystack formed of `haystack_pattern` repeated from one to `max_repeats` times.
 *
 *  @param misalignment The number of bytes to misalign the haystack within the cacheline.
 */
template <typename stl_matcher_, typename sz_matcher_>
void test_search_with_misaligned_repetitions(std::string_view haystack_pattern, std::string_view needle_stl,
                                             std::size_t misalignment) {
    constexpr std::size_t max_repeats = 128;
    alignas(64) char haystack[misalignment + max_repeats * haystack_pattern.size()];
    std::vector<std::size_t> offsets_stl;
    std::vector<std::size_t> offsets_sz;

    for (std::size_t repeats = 0; repeats != 128; ++repeats) {
        std::size_t haystack_length = (repeats + 1) * haystack_pattern.size();
        std::memcpy(haystack + misalignment + repeats * haystack_pattern.size(), haystack_pattern.data(),
                    haystack_pattern.size());

        // Convert to string views
        auto haystack_stl = std::string_view(haystack + misalignment, haystack_length);
        auto haystack_sz = sz::string_view(haystack + misalignment, haystack_length);
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
    }
}

/**
 *  Evaluates the correctness of a "matcher", searching for all the occurences of the `needle_stl`,
 *  as a substring, as a set of allowed characters, or as a set of disallowed characters, in a haystack.
 */
void test_search_with_misaligned_repetitions(std::string_view haystack_pattern, std::string_view needle_stl,
                                             std::size_t misalignment) {

    test_search_with_misaligned_repetitions<                   //
        sz::range_matches<std::string_view, sz::matcher_find>, //
        sz::range_matches<sz::string_view, sz::matcher_find>>( //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                     //
        sz::range_rmatches<std::string_view, sz::matcher_rfind>, //
        sz::range_rmatches<sz::string_view, sz::matcher_rfind>>( //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                            //
        sz::range_matches<std::string_view, sz::matcher_find_first_of>, //
        sz::range_matches<sz::string_view, sz::matcher_find_first_of>>( //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                            //
        sz::range_rmatches<std::string_view, sz::matcher_find_last_of>, //
        sz::range_rmatches<sz::string_view, sz::matcher_find_last_of>>( //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                //
        sz::range_matches<std::string_view, sz::matcher_find_first_not_of>, //
        sz::range_matches<sz::string_view, sz::matcher_find_first_not_of>>( //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                //
        sz::range_rmatches<std::string_view, sz::matcher_find_last_not_of>, //
        sz::range_rmatches<sz::string_view, sz::matcher_find_last_not_of>>( //
        haystack_pattern, needle_stl, misalignment);
}

void test_search_with_misaligned_repetitions(std::string_view haystack_pattern, std::string_view needle_stl) {
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 0);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 1);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 2);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 3);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 63);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 24);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 33);
}

/**
 *  @brief  Extensively tests the correctness of the string class search methods, such as `find` and `find_first_of`.
 *          Covers different alignment cases within a cache line, repetitive patterns, and overlapping matches.
 */
static void test_search_with_misaligned_repetitions() {
    // When haystack is only formed of needles:
    test_search_with_misaligned_repetitions("a", "a");
    test_search_with_misaligned_repetitions("ab", "ab");
    test_search_with_misaligned_repetitions("abc", "abc");
    test_search_with_misaligned_repetitions("abcd", "abcd");
    test_search_with_misaligned_repetitions({sz::ascii_lowercase, sizeof(sz::ascii_lowercase)},
                                            {sz::ascii_lowercase, sizeof(sz::ascii_lowercase)});
    test_search_with_misaligned_repetitions({sz::ascii_printables, sizeof(sz::ascii_printables)},
                                            {sz::ascii_printables, sizeof(sz::ascii_printables)});

    // When we are dealing with NULL characters inside the string
    test_search_with_misaligned_repetitions("\0", "\0");
    test_search_with_misaligned_repetitions("a\0", "a\0");
    test_search_with_misaligned_repetitions("ab\0", "ab");
    test_search_with_misaligned_repetitions("ab\0", "ab\0");
    test_search_with_misaligned_repetitions("abc\0", "abc");
    test_search_with_misaligned_repetitions("abc\0", "abc\0");
    test_search_with_misaligned_repetitions("abcd\0", "abcd");

    // When haystack is formed of equidistant needles:
    test_search_with_misaligned_repetitions("ab", "a");
    test_search_with_misaligned_repetitions("abc", "a");
    test_search_with_misaligned_repetitions("abcd", "a");

    // When matches occur in between pattern words:
    test_search_with_misaligned_repetitions("ab", "ba");
    test_search_with_misaligned_repetitions("abc", "ca");
    test_search_with_misaligned_repetitions("abcd", "da");
}

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance computation,
 *          as well as TODO: the similarity scoring functions for bioinformatics-like workloads.
 */
static void test_levenshtein_distances() {
    struct {
        char const *left;
        char const *right;
        std::size_t distance;
    } explicit_cases[] = {
        {"", "", 0},
        {"", "abc", 3},
        {"abc", "", 3},
        {"abc", "ac", 1},                   // one deletion
        {"abc", "a_bc", 1},                 // one insertion
        {"abc", "adc", 1},                  // one substitution
        {"ggbuzgjux{}l", "gbuzgjux{}l", 1}, // one insertion (prepended
    };

    auto print_failure = [&](sz::string const &l, sz::string const &r, std::size_t expected, std::size_t received) {
        char const *ellipsis = l.length() > 22 || r.length() > 22 ? "..." : "";
        std::printf("Levenshtein distance error: distance(\"%.22s%s\", \"%.22s%s\"); got %zd, expected %zd\n", //
                    l.c_str(), ellipsis, r.c_str(), ellipsis, received, expected);
    };

    auto test_distance = [&](sz::string const &l, sz::string const &r, std::size_t expected) {
        auto received = l.edit_distance(r);
        if (received != expected) print_failure(l, r, expected, received);
        // The distance relation commutes
        received = r.edit_distance(l);
        if (received != expected) print_failure(r, l, expected, received);
    };

    for (auto explicit_case : explicit_cases)
        test_distance(sz::string(explicit_case.left), sz::string(explicit_case.right), explicit_case.distance);

    // Randomized tests
    // TODO: Add bounded distance tests
    struct {
        std::size_t length_upper_bound;
        std::size_t iterations;
    } fuzzy_cases[] = {
        {10, 1000},
        {100, 100},
        {1000, 10},
    };
    std::random_device random_device;
    std::mt19937 generator(random_device());
    sz::string first, second;
    for (auto fuzzy_case : fuzzy_cases) {
        char alphabet[2] = {'a', 'b'};
        std::uniform_int_distribution<std::size_t> length_distribution(0, fuzzy_case.length_upper_bound);
        for (std::size_t i = 0; i != fuzzy_case.iterations; ++i) {
            std::size_t first_length = length_distribution(generator);
            std::size_t second_length = length_distribution(generator);
            std::generate_n(std::back_inserter(first), first_length, [&]() { return alphabet[generator() % 2]; });
            std::generate_n(std::back_inserter(second), second_length, [&]() { return alphabet[generator() % 2]; });
            test_distance(first, second, levenshtein_baseline(first, second));
            first.clear();
            second.clear();
        }
    }
}

int main(int argc, char const **argv) {

    // Let's greet the user nicely
    static const char *USER_NAME =
#define str(s) #s
#define xstr(s) str(s)
        xstr(DEV_USER_NAME);
    std::printf("Hi " xstr(DEV_USER_NAME) "! You look nice today!\n");
#undef str
#undef xstr

    // Basic utilities
    test_arithmetical_utilities();

    // Compatibility with STL
    test_compilation<std::string>(); // Make sure the test itself is reasonable
    // test_compilation<sz::string>(); // To early for this...

    // The string class implementation
    test_constructors();
    test_memory_stability_for_length(1024);
    test_memory_stability_for_length(14);
    test_updates();

    // Advanced search operations
    test_comparisons();
    test_search();
    test_search_with_misaligned_repetitions();
    test_levenshtein_distances();

    return 0;
}
