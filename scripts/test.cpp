#undef NDEBUG      // Enable all assertions

// Enable assertions for iterators
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#include <cassert> // assertions

// Overload the following with caution.
// Those parameters must never be explicitly set during releases,
// but they come handy during development, if you want to validate
// different ISA-specific implementations.
// #define SZ_USE_X86_AVX2 0
// #define SZ_USE_X86_AVX512 0
// #define SZ_USE_ARM_NEON 0
// #define SZ_USE_ARM_SVE 0
#define SZ_DEBUG 1 // Enforce aggressive logging for this unit.

// Put this at the top to make sure it pulls all the right dependencies
#include <stringzilla/stringzilla.hpp>

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h> // ASAN
#endif

#include <algorithm> // `std::transform`
#include <cstdio>    // `std::printf`
#include <cstring>   // `std::memcpy`
#include <iterator>  // `std::distance`
#include <memory>    // `std::allocator`
#include <random>    // `std::random_device`
#include <sstream>   // `std::ostringstream`
#include <vector>    // `std::vector`

#include <string>      // Baseline
#include <string_view> // Baseline

#if !SZ_DETECT_CPP_11
#error "This test requires C++11 or later."
#endif

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

    assert(sz_size_bit_ceil(1000000ull) == (1ull << 20));
    assert(sz_size_bit_ceil(2000000ull) == (1ull << 21));
    assert(sz_size_bit_ceil(4000000ull) == (1ull << 22));
    assert(sz_size_bit_ceil(8000000ull) == (1ull << 23));

    assert(sz_size_bit_ceil(16000000ull) == (1ull << 24));
    assert(sz_size_bit_ceil(32000000ull) == (1ull << 25));
    assert(sz_size_bit_ceil(64000000ull) == (1ull << 26));

    assert(sz_size_bit_ceil(128000000ull) == (1ull << 27));
    assert(sz_size_bit_ceil(256000000ull) == (1ull << 28));
    assert(sz_size_bit_ceil(512000000ull) == (1ull << 29));

    assert(sz_size_bit_ceil(1000000000ull) == (1ull << 30));
    assert(sz_size_bit_ceil(2000000000ull) == (1ull << 31));

#if SZ_DETECT_64_BIT
    assert(sz_size_bit_ceil(4000000000ull) == (1ull << 32));
    assert(sz_size_bit_ceil(8000000000ull) == (1ull << 33));
    assert(sz_size_bit_ceil(16000000000ull) == (1ull << 34));

    assert(sz_size_bit_ceil((1ull << 62)) == (1ull << 62));
    assert(sz_size_bit_ceil((1ull << 62) + 1) == (1ull << 63));
    assert(sz_size_bit_ceil((1ull << 63)) == (1ull << 63));
#endif

    for (sz_u16_t number = 0; number != 256; ++number)
        for (sz_u16_t divisor = 2; divisor != 256; ++divisor)
            assert(sz_u8_divide(static_cast<sz_u8_t>(number), static_cast<sz_u8_t>(divisor)) ==
                   (static_cast<sz_u8_t>(number) / static_cast<sz_u8_t>(divisor)));
}

/**
 *  @brief  Validates that `sz_move` and `sz_copy` work as expected,
 *          comparing them to `std::memmove` and `std::memcpy`.
 */
static void test_memory_utilities() {
    constexpr std::size_t size = 1024;
    char body_stl[size];
    char body_sz[size];

    auto &gen = global_random_generator();
    uniform_uint8_distribution_t distribution;
    std::generate(body_stl, body_stl + size, [&]() { return distribution(gen); });
    std::copy(body_stl, body_stl + size, body_sz);

    // Move the contents of both strings around, validating overall
    // equivalency after every random iteration.
    for (std::size_t i = 0; i < size; i++) {
        std::size_t offset = gen() % size;
        std::size_t length = gen() % (size - offset);
        std::size_t destination = gen() % (size - length);

        std::memmove(body_stl + destination, body_stl + offset, length);
        sz_move(body_sz + destination, body_sz + offset, length);
        assert(std::memcmp(body_stl, body_sz, size) == 0);
    }
}

#define assert_scoped(init, operation, condition) \
    {                                             \
        init;                                     \
        operation;                                \
        assert(condition);                        \
    }

#define assert_throws(expression, exception_type) \
    {                                             \
        bool threw = false;                       \
        try {                                     \
            sz_unused(expression);                \
        }                                         \
        catch (exception_type const &) {          \
            threw = true;                         \
        }                                         \
        assert(threw);                            \
    }

/**
 *  @brief  Invokes different C++ member methods of immutable strings to cover all STL APIs.
 *          This test guarantees API compatibility with STL `std::basic_string` template.
 */
template <typename string_type>
static void test_api_readonly() {

    using str = string_type;

    // Constructors.
    assert(str().empty());             // Test default constructor
    assert(str().size() == 0);         // Test default constructor
    assert(str("").empty());           // Test default constructor
    assert(str("").size() == 0);       // Test default constructor
    assert(str("hello").size() == 5);  // Test constructor with c-string
    assert(str("hello", 4) == "hell"); // Construct from substring

    // Element access.
    assert(str("test")[0] == 't');
    assert(str("test").at(1) == 'e');
    assert(str("front").front() == 'f');
    assert(str("back").back() == 'k');
    assert(*str("data").data() == 'd');

    // Iterators.
    assert(*str("begin").begin() == 'b' && *str("cbegin").cbegin() == 'c');
    assert(*str("rbegin").rbegin() == 'n' && *str("crbegin").crbegin() == 'n');
    assert(str("size").size() == 4 && str("length").length() == 6);

    // Slices... out-of-bounds exceptions are asymmetric!
    // Moreover, `std::string` has no `remove_prefix` and `remove_suffix` methods.
    // assert_scoped(str s = "hello", s.remove_prefix(1), s == "ello");
    // assert_scoped(str s = "hello", s.remove_suffix(1), s == "hell");
    assert(str("hello world").substr(0, 5) == "hello");
    assert(str("hello world").substr(6, 5) == "world");
    assert(str("hello world").substr(6) == "world");
    assert(str("hello world").substr(6, 100) == "world"); // 106 is beyond the length of the string, but its OK
    assert_throws(str("hello world").substr(100), std::out_of_range);   // 100 is beyond the length of the string
    assert_throws(str("hello world").substr(20, 5), std::out_of_range); // 20 is beyond the length of the string
    assert_throws(str("hello world").substr(-1, 5), std::out_of_range); // -1 casts to unsigned without any warnings...
    assert(str("hello world").substr(0, -1) == "hello world");          // -1 casts to unsigned without any warnings...

    // Character search in normal and reverse directions.
    assert(str("hello").find('e') == 1);
    assert(str("hello").find('e', 1) == 1);
    assert(str("hello").find('e', 2) == str::npos);
    assert(str("hello").rfind('l') == 3);
    assert(str("hello").rfind('l', 2) == 2);
    assert(str("hello").rfind('l', 1) == str::npos);

    // Substring search in normal and reverse directions.
    assert(str("hello").find("ell") == 1);
    assert(str("hello").find("ell", 1) == 1);
    assert(str("hello").find("ell", 2) == str::npos);
    assert(str("hello").find("el", 1) == 1);
    assert(str("hello").find("ell", 1, 2) == 1);
    assert(str("hello").rfind("l") == 3);
    assert(str("hello").rfind("l", 2) == 2);
    assert(str("hello").rfind("l", 1) == str::npos);

    // The second argument is the last possible value of the returned offset.
    assert(str("hello").rfind("el", 1) == 1);
    assert(str("hello").rfind("ell", 1) == 1);
    assert(str("hello").rfind("ello", 1) == 1);
    assert(str("hello").rfind("ell", 1, 2) == 1);

    // More complex queries.
    assert(str("abbabbaaaaaa").find("aa") == 6);
    assert(str("abcdabcd").substr(2, 4).find("abc") == str::npos);

    // ! `rfind` and `find_last_of` are not consistent in meaning of their arguments.
    assert(str("hello").find_first_of("le") == 1);
    assert(str("hello").find_first_of("le", 1) == 1);
    assert(str("hello").find_last_of("le") == 3);
    assert(str("hello").find_last_of("le", 2) == 2);
    assert(str("hello").find_first_not_of("hel") == 4);
    assert(str("hello").find_first_not_of("hel", 1) == 4);
    assert(str("hello").find_last_not_of("hel") == 4);
    assert(str("hello").find_last_not_of("hel", 4) == 4);

    // Try longer strings to enforce SIMD.
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find('x') == 23);  // first byte
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find('X') == 49);  // first byte
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind('x') == 23); // last byte
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind('X') == 49); // last byte

    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xy") == 23);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XY") == 49);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("yz") == 24);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("YZ") == 50);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xy") == 23); // last match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XY") == 49); // last match

    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xyz") == 23);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XYZ") == 49);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xyz") == 23); // last match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XYZ") == 49); // last match

    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xyzA") == 23);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XYZ0") == 49);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xyzA") == 23); // last match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XYZ0") == 49); // last match

    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_first_of("xyz") == 23); // sets
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_first_of("XYZ") == 49); // sets
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_last_of("xyz") == 25);  // sets
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_last_of("XYZ") == 51);  // sets

    // clang-format off
    // Using single-byte non-ASCII values, e.g., À (0xC0), Æ (0xC6)
    assert(str("abcdefgh" "\x01" "\xC6" "ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" "\xC0" "\xFA" "0123456789+-", 68).find_first_of("\xC6\xC7") == 9);  // sets
    assert(str("abcdefgh" "\x01" "\xC6" "ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" "\xC0" "\xFA" "0123456789+-", 68).find_first_of("\xC0\xC1") == 54); // sets
    assert(str("abcdefgh" "\x01" "\xC6" "ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" "\xC0" "\xFA" "0123456789+-", 68).find_last_of("\xC6\xC7") == 9);   // sets
    assert(str("abcdefgh" "\x01" "\xC6" "ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" "\xC0" "\xFA" "0123456789+-", 68).find_last_of("\xC0\xC1") == 54);  // sets
    // clang-format on

    // Boundary consitions.
    assert(str("hello").find_first_of("ox", 4) == 4);
    assert(str("hello").find_first_of("ox", 5) == str::npos);
    assert(str("hello").find_last_of("ox", 4) == 4);
    assert(str("hello").find_last_of("ox", 5) == 4);
    assert(str("hello").find_first_of("hx", 0) == 0);
    assert(str("hello").find_last_of("hx", 0) == 0);

    // Comparisons.
    assert(str("a") != str("b"));
    assert(str("a") < str("b"));
    assert(str("a") <= str("b"));
    assert(str("b") > str("a"));
    assert(str("b") >= str("a"));
    assert(str("a") < str("aa"));

#if SZ_DETECT_CPP20 && __cpp_lib_three_way_comparison
    // Spaceship operator instead of conventional comparions.
    assert((str("a") <=> str("b")) == std::strong_ordering::less);
    assert((str("b") <=> str("a")) == std::strong_ordering::greater);
    assert((str("b") <=> str("b")) == std::strong_ordering::equal);
    assert((str("a") <=> str("aa")) == std::strong_ordering::less);
#endif

    // Compare with another `str`.
    assert(str("test").compare(str("test")) == 0);   // Equal strings
    assert(str("apple").compare(str("banana")) < 0); // "apple" is less than "banana"
    assert(str("banana").compare(str("apple")) > 0); // "banana" is greater than "apple"

    // Compare with a C-string.
    assert(str("test").compare("test") == 0); // Equal to C-string "test"
    assert(str("alpha").compare("beta") < 0); // "alpha" is less than C-string "beta"
    assert(str("beta").compare("alpha") > 0); // "beta" is greater than C-string "alpha"

    // Compare substring with another `str`.
    assert(str("hello world").compare(0, 5, str("hello")) == 0); // Substring "hello" is equal to "hello"
    assert(str("hello world").compare(6, 5, str("earth")) > 0);  // Substring "world" is greater than "earth"
    assert(str("hello world").compare(6, 5, str("worlds")) < 0); // Substring "world" is less than "worlds"
    assert_throws(str("hello world").compare(20, 5, str("worlds")), std::out_of_range);

    // Compare substring with another `str`'s substring.
    assert(str("hello world").compare(0, 5, str("say hello"), 4, 5) == 0);      // Substring "hello" in both strings
    assert(str("hello world").compare(6, 5, str("world peace"), 0, 5) == 0);    // Substring "world" in both strings
    assert(str("hello world").compare(6, 5, str("a better world"), 9, 5) == 0); // Both substrings are "world"

    // Out of bounds cases for both compared strings.
    assert_throws(str("hello world").compare(20, 5, str("a better world"), 9, 5), std::out_of_range);
    assert_throws(str("hello world").compare(6, 5, str("a better world"), 90, 5), std::out_of_range);

    // Compare substring with a C-string.
    assert(str("hello world").compare(0, 5, "hello") == 0); // Substring "hello" is equal to C-string "hello"
    assert(str("hello world").compare(6, 5, "earth") > 0);  // Substring "world" is greater than C-string "earth"
    assert(str("hello world").compare(6, 5, "worlds") < 0); // Substring "world" is greater than C-string "worlds"

    // Compare substring with a C-string's prefix.
    assert(str("hello world").compare(0, 5, "hello Ash", 5) == 0); // Substring "hello" in both strings
    assert(str("hello world").compare(6, 5, "worlds", 5) == 0);    // Substring "world" in both strings
    assert(str("hello world").compare(6, 5, "worlds", 6) < 0);     // Substring "world" is less than "worlds"

#if SZ_DETECT_CPP20 && __cpp_lib_starts_ends_with
    // Prefix and suffix checks against strings.
    assert(str("https://cppreference.com").starts_with(str("http")) == true);
    assert(str("https://cppreference.com").starts_with(str("ftp")) == false);
    assert(str("https://cppreference.com").ends_with(str("com")) == true);
    assert(str("https://cppreference.com").ends_with(str("org")) == false);

    // Prefix and suffix checks against characters.
    assert(str("C++20").starts_with('C') == true);
    assert(str("C++20").starts_with('J') == false);
    assert(str("C++20").ends_with('0') == true);
    assert(str("C++20").ends_with('3') == false);

    // Prefix and suffix checks against C-style strings.
    assert(str("string_view").starts_with("string") == true);
    assert(str("string_view").starts_with("String") == false);
    assert(str("string_view").ends_with("view") == true);
    assert(str("string_view").ends_with("View") == false);
#endif

#if SZ_DETECT_CPP_23 && __cpp_lib_string_contains
    // Checking basic substring presence.
    assert(str("hello").contains(str("ell")) == true);
    assert(str("hello").contains(str("oll")) == false);
    assert(str("hello").contains('l') == true);
    assert(str("hello").contains('x') == false);
    assert(str("hello").contains("lo") == true);
    assert(str("hello").contains("lx") == false);
#endif

    // Exporting the contents of the string using the `str::copy` method.
    assert_scoped(char buf[5 + 1] = {0}, str("hello").copy(buf, 5), std::strcmp(buf, "hello") == 0);
    assert_scoped(char buf[4 + 1] = {0}, str("hello").copy(buf, 4, 1), std::strcmp(buf, "ello") == 0);
    assert_throws(str("hello").copy((char *)"", 1, 100), std::out_of_range);

    // Swaps.
    for (str const first : {"", "hello", "hellohellohellohellohellohellohellohellohellohellohellohello"}) {
        for (str const second : {"", "world", "worldworldworldworldworldworldworldworldworldworldworldworld"}) {
            str first_copy = first;
            str second_copy = second;
            first_copy.swap(second_copy);
            assert(first_copy == second && second_copy == first);
            first_copy.swap(first_copy); // Swapping with itself.
            assert(first_copy == second);
        }
    }

    // Make sure the standard hash and function-objects instantiate just fine.
    assert(std::hash<str> {}("hello") != 0);
    assert_scoped(std::ostringstream os, os << str("hello"), os.str() == "hello");

#if SZ_DETECT_CPP14
    // Comparison function objects are a C++14 feature.
    assert(std::equal_to<str> {}("hello", "world") == false);
    assert(std::less<str> {}("hello", "world") == true);
#endif
}

/**
 *  @brief  Invokes different C++ member methods of the memory-owning string class to make sure they all pass
 *          compilation. This test guarantees API compatibility with STL `std::basic_string` template.
 */
template <typename string_type>
static void test_api_mutable() {

    using str = string_type;

    // Constructors.
    assert(str().empty());                             // Test default constructor
    assert(str().size() == 0);                         // Test default constructor
    assert(str("").empty());                           // Test default constructor
    assert(str("").size() == 0);                       // Test default constructor
    assert(str("hello").size() == 5);                  // Test constructor with c-string
    assert(str("hello", 4) == "hell");                 // Construct from substring
    assert(str(5, 'a') == "aaaaa");                    // Construct with count and character
    assert(str({'h', 'e', 'l', 'l', 'o'}) == "hello"); // Construct from initializer list
    assert(str(str("hello"), 2) == "llo");             // Construct from another string suffix
    assert(str(str("hello"), 2, 2) == "ll");           // Construct from another string range

    // Assignments.
    assert_scoped(str s = "obsolete", s = "hello", s == "hello");
    assert_scoped(str s = "obsolete", s.assign("hello"), s == "hello");
    assert_scoped(str s = "obsolete", s.assign("hello", 4), s == "hell");
    assert_scoped(str s = "obsolete", s.assign(5, 'a'), s == "aaaaa");
    assert_scoped(str s = "obsolete", s.assign(32, 'a'), s == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    assert_scoped(str s = "obsolete", s.assign({'h', 'e', 'l', 'l', 'o'}), s == "hello");
    assert_scoped(str s = "obsolete", s.assign(str("hello")), s == "hello");
    assert_scoped(str s = "obsolete", s.assign(str("hello"), 2), s == "llo");
    assert_scoped(str s = "obsolete", s.assign(str("hello"), 2, 2), s == "ll");
    assert_scoped(str s = "obsolete", s.assign(str("hello"), 2, 2), s == "ll");
    assert_scoped(str s = "obsolete", s.assign(s), s == "obsolete");                  // Self-assignment
    assert_scoped(str s = "obsolete", s.assign(s.begin(), s.end()), s == "obsolete"); // Self-assignment
    assert_scoped(str s = "obsolete", s.assign(s, 4), s == "lete");                   // Partial self-assignment
    assert_scoped(str s = "obsolete", s.assign(s, 4, 3), s == "let");                 // Partial self-assignment

    // Self-assignment is a special case of assignment.
    assert_scoped(str s = "obsolete", s = s, s == "obsolete");
    assert_scoped(str s = "obsolete", s.assign(s), s == "obsolete");
    assert_scoped(str s = "obsolete", s.assign(s.data(), 2), s == "ob");
    assert_scoped(str s = "obsolete", s.assign(s.data(), s.size()), s == "obsolete");

    // Allocations, capacity and memory management.
    assert_scoped(str s, s.reserve(10), s.capacity() >= 10);
    assert_scoped(str s, s.resize(10), s.size() == 10);
    assert_scoped(str s, s.resize(10, 'a'), s.size() == 10 && s == "aaaaaaaaaa");
    assert(str().max_size() > 0);
    assert(str().get_allocator() == std::allocator<char>());
    assert(std::strcmp(str("c_str").c_str(), "c_str") == 0);
    assert_scoped(str s = "hello", s.shrink_to_fit(), s.capacity() <= sz::string::min_capacity);

    // Concatenation.
    // Following are missing in strings, but are present in vectors.
    // assert_scoped(str s = "!?", s.push_front('a'), s == "a!?");
    // assert_scoped(str s = "!?", s.pop_front(), s == "?");
    assert(str().append("test") == "test");
    assert(str("test") + "ing" == "testing");
    assert(str("test") + str("ing") == "testing");
    assert(str("test") + str("ing") + str("123") == "testing123");
    assert_scoped(str s = "!?", s.push_back('a'), s == "!?a");
    assert_scoped(str s = "!?", s.pop_back(), s == "!");

    // Incremental construction.
    assert(str("__").insert(1, "test") == "_test_");
    assert(str("__").insert(1, "test", 2) == "_te_");
    assert(str("__").insert(1, 5, 'a') == "_aaaaa_");
    assert(str("__").insert(1, str("test")) == "_test_");
    assert(str("__").insert(1, str("test"), 2) == "_st_");
    assert(str("__").insert(1, str("test"), 2, 1) == "_s_");

    // Inserting at a given iterator position yields back an iterator.
    assert_scoped(str s = "__", s.insert(s.begin() + 1, 5, 'a'), s == "_aaaaa_");
    assert_scoped(str s = "__", s.insert(s.begin() + 1, {'a', 'b', 'c'}), s == "_abc_");
    assert_scoped(str s = "__", (void)0, s.insert(s.begin() + 1, 5, 'a') == (s.begin() + 1));
    assert_scoped(str s = "__", (void)0, s.insert(s.begin() + 1, {'a', 'b', 'c'}) == (s.begin() + 1));

    // Handle exceptions.
    // The `length_error` might be difficult to catch due to a large `max_size()`.
    // assert_throws(large_string.insert(large_string.size() - 1, large_number_of_chars, 'a'), std::length_error);
    assert_throws(str("hello").insert(6, "world"), std::out_of_range);         // `index > size()` case from STL
    assert_throws(str("hello").insert(5, str("world"), 6), std::out_of_range); // `s_index > str.size()` case from STL

    // Erasure.
    assert(str("").erase(0, 3) == "");
    assert(str("test").erase(1, 2) == "tt");
    assert(str("test").erase(1) == "t");
    assert_scoped(str s = "test", s.erase(s.begin() + 1), s == "tst");
    assert_scoped(str s = "test", s.erase(s.begin() + 1, s.begin() + 2), s == "tst");
    assert_scoped(str s = "test", s.erase(s.begin() + 1, s.begin() + 3), s == "tt");
    assert_scoped(str s = "test", (void)0, s.erase(s.begin() + 1) == (s.begin() + 1));
    assert_scoped(str s = "test", (void)0, s.erase(s.begin() + 1, s.begin() + 2) == (s.begin() + 1));
    assert_scoped(str s = "test", (void)0, s.erase(s.begin() + 1, s.begin() + 3) == (s.begin() + 1));

    // Substitutions.
    assert(str("hello").replace(1, 2, "123") == "h123lo");
    assert(str("hello").replace(1, 2, str("123"), 1) == "h23lo");
    assert(str("hello").replace(1, 2, "123", 1) == "h1lo");
    assert(str("hello").replace(1, 2, "123", 1, 1) == "h2lo");
    assert(str("hello").replace(1, 2, str("123"), 1, 1) == "h2lo");
    assert(str("hello").replace(1, 2, 3, 'a') == "haaalo");

    // Substitutions with iterators.
    assert_scoped(str s = "hello", s.replace(s.begin() + 1, s.begin() + 3, 3, 'a'), s == "haaalo");
    assert_scoped(str s = "hello", s.replace(s.begin() + 1, s.begin() + 3, {'a', 'b'}), s == "hablo");

    // Some nice "tweetable" examples :)
    assert(str("Loose").replace(2, 2, str("vath"), 1) == "Loathe");
    assert(str("Loose").replace(2, 2, "vath", 1) == "Love");

    // Insertion is a special case of replacement.
    // Appending and assigning are special cases of insertion.
    // Still, we test them separately to make sure they are not broken.
    assert(str("hello").append("123") == "hello123");
    assert(str("hello").append(str("123")) == "hello123");
    assert(str("hello").append(str("123"), 1) == "hello23");
    assert(str("hello").append(str("123"), 1, 1) == "hello2");
    assert(str("hello").append({'1', '2'}) == "hello12");
    assert(str("hello").append(2, '!') == "hello!!");
    assert_scoped(str s = "123", (void)0, str("hello").append(s.begin(), s.end()) == "hello123");
}

static void test_stl_conversion_api() {
    // From a mutable STL string to StringZilla and vice-versa.
    {
        std::string stl {"hello"};
        sz::string sz = stl;
        sz::string_view szv = stl;
        sz::string_span szs = stl;
        stl = sz;
        stl = szv;
        stl = szs;
    }
    // From an immutable STL string to StringZilla.
    {
        std::string const stl {"hello"};
        sz::string sz = stl;
        sz::string_view szv = stl;
        sz_unused(sz);
        sz_unused(szv);
    }
#if SZ_DETECT_CPP_17 && __cpp_lib_string_view
    // From STL `string_view` to StringZilla and vice-versa.
    {
        std::string_view stl {"hello"};
        sz::string sz = stl;
        sz::string_view szv = stl;
        stl = sz;
        stl = szv;
    }
#endif
}

/**
 *  @brief  Invokes different C++ member methods of immutable strings to cover
 *          extensions beyond the STL API.
 */
template <typename string_type>
static void test_api_readonly_extensions() {
    using str = string_type;

    // Signed offset lokups and slices.
    assert(str("hello").sat(0) == 'h');
    assert(str("hello").sat(-1) == 'o');
    assert(str("hello").sub(1) == "ello");
    assert(str("hello").sub(-1) == "o");
    assert(str("hello").sub(1, 2) == "e");
    assert(str("hello").sub(1, 100) == "ello");
    assert(str("hello").sub(100, 100) == "");
    assert(str("hello").sub(-2, -1) == "l");
    assert(str("hello").sub(-2, -2) == "");
    assert(str("hello").sub(100, -100) == "");

    // Passing initializer lists to `operator[]`.
    // Put extra braces to correctly estimate the number of macro arguments :)
    assert((str("hello")[{1, 2}] == "e"));
    assert((str("hello")[{1, 100}] == "ello"));
    assert((str("hello")[{100, 100}] == ""));
    assert((str("hello")[{100, -100}] == ""));
    assert((str("hello")[{-100, -100}] == ""));

    // Computing edit-distances.
    assert(sz::hamming_distance(str("hello"), str("hello")) == 0);
    assert(sz::hamming_distance(str("hello"), str("hell")) == 1);
    assert(sz::hamming_distance(str("abc"), str("adc")) == 1);                // one substitution
    assert(sz::hamming_distance(str("αβγδ"), str("αxxγδ")) == 2);             // replace Beta UTF8 codepoint
    assert(sz::hamming_distance_utf8(str("abcdefgh"), str("_bcdefg_")) == 2); // replace ASCI prefix and suffix
    assert(sz::hamming_distance_utf8(str("αβγδ"), str("αγγδ")) == 1);         // replace Beta UTF8 codepoint

    assert(sz::edit_distance(str("hello"), str("hello")) == 0);
    assert(sz::edit_distance(str("hello"), str("hell")) == 1);
    assert(sz::edit_distance(str(""), str("")) == 0);
    assert(sz::edit_distance(str(""), str("abc")) == 3);
    assert(sz::edit_distance(str("abc"), str("")) == 3);
    assert(sz::edit_distance(str("abc"), str("ac")) == 1);                   // one deletion
    assert(sz::edit_distance(str("abc"), str("a_bc")) == 1);                 // one insertion
    assert(sz::edit_distance(str("abc"), str("adc")) == 1);                  // one substitution
    assert(sz::edit_distance(str("ggbuzgjux{}l"), str("gbuzgjux{}l")) == 1); // one insertion (prepended)
    assert(sz::edit_distance(str("abcdefgABCDEFG"), str("ABCDEFGabcdefg")) == 14);

    assert(sz::edit_distance_utf8(str("hello"), str("hell")) == 1);           // no unicode symbols, just ASCII
    assert(sz::edit_distance_utf8(str("𠜎 𠜱 𠝹 𠱓"), str("𠜎𠜱𠝹𠱓")) == 3); // add 3 whitespaces in Chinese
    assert(sz::edit_distance_utf8(str("💖"), str("💗")) == 1);

    assert(sz::edit_distance_utf8(str("αβγδ"), str("αγδ")) == 1);      // insert Beta
    assert(sz::edit_distance_utf8(str("école"), str("école")) == 2);   // etter "é" as a single character vs "e" + "´"
    assert(sz::edit_distance_utf8(str("façade"), str("facade")) == 1); // "ç" with cedilla vs. plain
    assert(sz::edit_distance_utf8(str("Schön"), str("Scho\u0308n")) == 2); // "ö" represented as "o" + "¨"
    assert(sz::edit_distance_utf8(str("München"), str("Muenchen")) == 2);  // German with umlaut vs. transcription
    assert(sz::edit_distance_utf8(str("こんにちは世界"), str("こんばんは世界")) == 2);

    // Computing alignment scores.
    using matrix_t = std::int8_t[256][256];
    std::vector<std::int8_t> costs_vector = unary_substitution_costs();
    matrix_t &costs = *reinterpret_cast<matrix_t *>(costs_vector.data());

    assert(sz::alignment_score(str("listen"), str("silent"), costs, -1) == -4);
    assert(sz::alignment_score(str("abcdefgABCDEFG"), str("ABCDEFGabcdefg"), costs, -1) == -14);
    assert(sz::alignment_score(str("hello"), str("hello"), costs, -1) == 0);
    assert(sz::alignment_score(str("hello"), str("hell"), costs, -1) == -1);

    assert(sz::hashes_fingerprint<512>(str("aaaa"), 3).count() == 1);

    // Computing rolling fingerprints.
    assert(sz::hashes_fingerprint<512>(str("hello"), 4).count() == 2);
    assert(sz::hashes_fingerprint<512>(str("hello"), 3).count() == 3);

    // No matter how many times one repeats a character, the hash should only contain at most one set bit.
    assert(sz::hashes_fingerprint<512>(str("a"), 3).count() == 0);
    assert(sz::hashes_fingerprint<512>(str("aa"), 3).count() == 0);
    assert(sz::hashes_fingerprint<512>(str("aaa"), 3).count() == 1);
    assert(sz::hashes_fingerprint<512>(str("aaaa"), 3).count() == 1);
    assert(sz::hashes_fingerprint<512>(str("aaaaa"), 3).count() == 1);

    // Computing fuzzy search results.
}

void test_api_mutable_extensions() {
    using str = sz::string;

    // Try methods.
    assert(str("obsolete").try_assign("hello"));
    assert(str().try_reserve(10));
    assert(str().try_resize(10));
    assert(str("__").try_insert(1, "test"));
    assert(str("test").try_erase(1, 2));
    assert(str("test").try_clear());
    assert(str("test").try_replace(1, 2, "aaaa"));
    assert(str("test").try_push_back('a'));
    assert(str("test").try_shrink_to_fit());

    // Self-referencing methods.
    assert_scoped(str s = "test", s.try_assign(s.view()), s == "test");
    assert_scoped(str s = "test", s.try_assign(s.view().sub(1, 2)), s == "e");
    assert_scoped(str s = "test", s.try_append(s.view().sub(1, 2)), s == "teste");

    // Try methods going beyond and beneath capacity threshold.
    assert_scoped(str s = "0123456789012345678901234567890123456789012345678901234567890123", // 64 symbols at start
                  s.try_append(s) && s.try_append(s) && s.try_append(s) && s.try_append(s) && s.try_clear() &&
                      s.try_shrink_to_fit(),
                  s.capacity() < sz::string::min_capacity);

    // Same length replacements.
    assert_scoped(str s = "hello", s.replace_all("xx", "xx"), s == "hello");
    assert_scoped(str s = "hello", s.replace_all("l", "1"), s == "he11o");
    assert_scoped(str s = "hello", s.replace_all("he", "al"), s == "alllo");
    assert_scoped(str s = "hello", s.replace_all(sz::char_set("x"), "!"), s == "hello");
    assert_scoped(str s = "hello", s.replace_all(sz::char_set("o"), "!"), s == "hell!");
    assert_scoped(str s = "hello", s.replace_all(sz::char_set("ho"), "!"), s == "!ell!");

    // Shorter replacements.
    assert_scoped(str s = "hello", s.replace_all("xx", "x"), s == "hello");
    assert_scoped(str s = "hello", s.replace_all("l", ""), s == "heo");
    assert_scoped(str s = "hello", s.replace_all("h", ""), s == "ello");
    assert_scoped(str s = "hello", s.replace_all("o", ""), s == "hell");
    assert_scoped(str s = "hello", s.replace_all("llo", "!"), s == "he!");
    assert_scoped(str s = "hello", s.replace_all(sz::char_set("x"), ""), s == "hello");
    assert_scoped(str s = "hello", s.replace_all(sz::char_set("lo"), ""), s == "he");

    // Longer replacements.
    assert_scoped(str s = "hello", s.replace_all("xx", "xxx"), s == "hello");
    assert_scoped(str s = "hello", s.replace_all("l", "ll"), s == "hellllo");
    assert_scoped(str s = "hello", s.replace_all("h", "hh"), s == "hhello");
    assert_scoped(str s = "hello", s.replace_all("o", "oo"), s == "helloo");
    assert_scoped(str s = "hello", s.replace_all("llo", "llo!"), s == "hello!");
    assert_scoped(str s = "hello", s.replace_all(sz::char_set("x"), "xx"), s == "hello");
    assert_scoped(str s = "hello", s.replace_all(sz::char_set("lo"), "lo"), s == "helololo");

    // Concatenation.
    assert(str(str("a") | str("b")) == "ab");
    assert(str(str("a") | str("b") | str("ab")) == "abab");

    assert(str(sz::concatenate("a"_sz, "b"_sz)) == "ab");
    assert(str(sz::concatenate("a"_sz, "b"_sz, "c"_sz)) == "abc");

    // Randomization.
    assert(str::random(0).empty());
    assert(str::random(4, "a") == "aaaa");
    assert(str::random(4, "aaaa") == "aaaa");
    assert(str::random(global_random_generator(), 4, "aaaa") == "aaaa");
    assert_scoped(str s = str::random(128, "ACGT"), (void)s,
                  s.contains('A') && s.contains('C') && s.contains('G') && s.contains('T'));
}

/**
 *  @brief  Tests copy constructor and copy-assignment constructor of `sz::string`.
 */
static void test_constructors() {
    std::string alphabet {sz::ascii_printables(), sizeof(sz::ascii_printables())};
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
    inline static bool &verbose_ref() {
        static bool global_value = false;
        return global_value;
    }
    inline static std::size_t &counter_ref() {
        static std::size_t global_value = 0ul;
        return global_value;
    }

    template <typename... args_types>
    static void print_if_verbose(char const *fmt, args_types... args) {
        if (!verbose_ref()) return;
        std::printf(fmt, args...);
    }

    char *allocate(std::size_t n) {
        counter_ref() += n;
        print_if_verbose("alloc %zd -> %zd\n", n, counter_ref());
        return std::allocator<char>::allocate(n);
    }

    void deallocate(char *val, std::size_t n) {
        assert(n <= counter_ref());
        counter_ref() -= n;
        print_if_verbose("dealloc: %zd -> %zd\n", n, counter_ref());
        std::allocator<char>::deallocate(val, n);
    }

    template <typename callback_type>
    static std::size_t account_block(callback_type callback) {
        auto before = accounting_allocator::counter_ref();
        print_if_verbose("starting block: %zd\n", before);
        callback();
        auto after = accounting_allocator::counter_ref();
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

    assert(accounting_allocator::counter_ref() == 0);
    using string = sz::basic_string<char, accounting_allocator>;
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
    assert(accounting_allocator::counter_ref() == 0);
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
    assert(sz::string_view(sz::base64(), sizeof(sz::base64())).find_first_of("_") == sz::string_view::npos);
    assert(sz::string_view(sz::base64(), sizeof(sz::base64())).find_first_of("+") == 62);
    assert(sz::string_view(sz::ascii_printables(), sizeof(sz::ascii_printables())).find_first_of("~") !=
           sz::string_view::npos);

    assert("aabaa"_sz.remove_prefix("a") == "abaa");
    assert("aabaa"_sz.remove_suffix("a") == "aaba");
    assert("aabaa"_sz.lstrip(sz::char_set {"a"}) == "baa");
    assert("aabaa"_sz.rstrip(sz::char_set {"a"}) == "aab");
    assert("aabaa"_sz.strip(sz::char_set {"a"}) == "b");

    // Check more advanced composite operations
    assert("abbccc"_sz.partition("bb").before.size() == 1);
    assert("abbccc"_sz.partition("bb").match.size() == 2);
    assert("abbccc"_sz.partition("bb").after.size() == 3);
    assert("abbccc"_sz.partition("bb").before == "a");
    assert("abbccc"_sz.partition("bb").match == "bb");
    assert("abbccc"_sz.partition("bb").after == "ccc");

    // Check ranges of search matches
    assert("hello"_sz.find_all("l").size() == 2);
    assert("hello"_sz.rfind_all("l").size() == 2);

    assert(""_sz.find_all(".", sz::include_overlaps_type {}).size() == 0);
    assert(""_sz.find_all(".", sz::exclude_overlaps_type {}).size() == 0);
    assert("."_sz.find_all(".", sz::include_overlaps_type {}).size() == 1);
    assert("."_sz.find_all(".", sz::exclude_overlaps_type {}).size() == 1);
    assert(".."_sz.find_all(".", sz::include_overlaps_type {}).size() == 2);
    assert(".."_sz.find_all(".", sz::exclude_overlaps_type {}).size() == 2);
    assert(""_sz.rfind_all(".", sz::include_overlaps_type {}).size() == 0);
    assert(""_sz.rfind_all(".", sz::exclude_overlaps_type {}).size() == 0);
    assert("."_sz.rfind_all(".", sz::include_overlaps_type {}).size() == 1);
    assert("."_sz.rfind_all(".", sz::exclude_overlaps_type {}).size() == 1);
    assert(".."_sz.rfind_all(".", sz::include_overlaps_type {}).size() == 2);
    assert(".."_sz.rfind_all(".", sz::exclude_overlaps_type {}).size() == 2);

    assert("a.b.c.d"_sz.find_all(".").size() == 3);
    assert("a.,b.,c.,d"_sz.find_all(".,").size() == 3);
    assert("a.,b.,c.,d"_sz.rfind_all(".,").size() == 3);
    assert("a.b,c.d"_sz.find_all(sz::char_set(".,")).size() == 3);
    assert("a...b...c"_sz.rfind_all("..").size() == 4);
    assert("a...b...c"_sz.rfind_all("..", sz::include_overlaps_type {}).size() == 4);
    assert("a...b...c"_sz.rfind_all("..", sz::exclude_overlaps_type {}).size() == 2);

    auto finds = "a.b.c"_sz.find_all(sz::char_set("abcd")).template to<std::vector<std::string>>();
    assert(finds.size() == 3);
    assert(finds[0] == "a");

    auto rfinds = "a.b.c"_sz.rfind_all(sz::char_set("abcd")).template to<std::vector<std::string>>();
    assert(rfinds.size() == 3);
    assert(rfinds[0] == "c");

    auto splits = ".a..c."_sz.split(sz::char_set(".")).template to<std::vector<std::string>>();
    assert(splits.size() == 5);
    assert(splits[0] == "");
    assert(splits[1] == "a");
    assert(splits[4] == "");

    assert(""_sz.split(".").size() == 1);
    assert(""_sz.rsplit(".").size() == 1);

    assert("hello"_sz.split("l").size() == 3);
    assert("hello"_sz.rsplit("l").size() == 3);
    assert(*advanced("hello"_sz.split("l").begin(), 0) == "he");
    assert(*advanced("hello"_sz.rsplit("l").begin(), 0) == "o");
    assert(*advanced("hello"_sz.split("l").begin(), 1) == "");
    assert(*advanced("hello"_sz.rsplit("l").begin(), 1) == "");
    assert(*advanced("hello"_sz.split("l").begin(), 2) == "o");
    assert(*advanced("hello"_sz.rsplit("l").begin(), 2) == "he");

    assert("a.b.c.d"_sz.split(".").size() == 4);
    assert("a.b.c.d"_sz.rsplit(".").size() == 4);
    assert(*("a.b.c.d"_sz.split(".").begin()) == "a");
    assert(*("a.b.c.d"_sz.rsplit(".").begin()) == "d");
    assert(*advanced("a.b.c.d"_sz.split(".").begin(), 1) == "b");
    assert(*advanced("a.b.c.d"_sz.rsplit(".").begin(), 1) == "c");
    assert(*advanced("a.b.c.d"_sz.split(".").begin(), 3) == "d");
    assert(*advanced("a.b.c.d"_sz.rsplit(".").begin(), 3) == "a");
    assert("a.b.,c,d"_sz.split(".,").size() == 2);
    assert("a.b,c.d"_sz.split(sz::char_set(".,")).size() == 4);

    auto rsplits = ".a..c."_sz.rsplit(sz::char_set(".")).template to<std::vector<std::string>>();
    assert(rsplits.size() == 5);
    assert(rsplits[0] == "");
    assert(rsplits[1] == "c");
    assert(rsplits[4] == "");
}

#if SZ_DETECT_CPP_17 && __cpp_lib_string_view

/**
 *  Evaluates the correctness of a "matcher", searching for all the occurrences of the `needle_stl`
 *  in a haystack formed of `haystack_pattern` repeated from one to `max_repeats` times.
 *
 *  @param misalignment The number of bytes to misalign the haystack within the cacheline.
 */
template <typename stl_matcher_, typename sz_matcher_>
void test_search_with_misaligned_repetitions(std::string_view haystack_pattern, std::string_view needle_stl,
                                             std::size_t misalignment) {
    constexpr std::size_t max_repeats = 128;

    // Allocate a buffer to store the haystack with enough padding to misalign it.
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
 *  Evaluates the correctness of a "matcher", searching for all the occurrences of the `needle_stl`,
 *  as a substring, as a set of allowed characters, or as a set of disallowed characters, in a haystack.
 */
void test_search_with_misaligned_repetitions(std::string_view haystack_pattern, std::string_view needle_stl,
                                             std::size_t misalignment) {

    test_search_with_misaligned_repetitions<                                     //
        sz::range_matches<std::string_view, sz::matcher_find<std::string_view>>, //
        sz::range_matches<sz::string_view, sz::matcher_find<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                       //
        sz::range_rmatches<std::string_view, sz::matcher_rfind<std::string_view>>, //
        sz::range_rmatches<sz::string_view, sz::matcher_rfind<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                              //
        sz::range_matches<std::string_view, sz::matcher_find_first_of<std::string_view>>, //
        sz::range_matches<sz::string_view, sz::matcher_find_first_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                              //
        sz::range_rmatches<std::string_view, sz::matcher_find_last_of<std::string_view>>, //
        sz::range_rmatches<sz::string_view, sz::matcher_find_last_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                                  //
        sz::range_matches<std::string_view, sz::matcher_find_first_not_of<std::string_view>>, //
        sz::range_matches<sz::string_view, sz::matcher_find_first_not_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                                  //
        sz::range_rmatches<std::string_view, sz::matcher_find_last_not_of<std::string_view>>, //
        sz::range_rmatches<sz::string_view, sz::matcher_find_last_not_of<sz::string_view>>>(  //
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
    test_search_with_misaligned_repetitions({sz::base64(), sizeof(sz::base64())}, {sz::base64(), sizeof(sz::base64())});
    test_search_with_misaligned_repetitions({sz::ascii_lowercase(), sizeof(sz::ascii_lowercase())},
                                            {sz::ascii_lowercase(), sizeof(sz::ascii_lowercase())});
    test_search_with_misaligned_repetitions({sz::ascii_printables(), sizeof(sz::ascii_printables())},
                                            {sz::ascii_printables(), sizeof(sz::ascii_printables())});

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

    // Examples targeted exactly against the Raita heuristic,
    // which matches the first, the last, and the middle characters with SIMD.
    test_search_with_misaligned_repetitions("aaabbccc", "aaabbccc");
    test_search_with_misaligned_repetitions("axabbcxc", "aaabbccc");
    test_search_with_misaligned_repetitions("axabbcxcaaabbccc", "aaabbccc");
}

#endif

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance computation,
 *          as well as the similarity scoring functions for bioinformatics-like workloads.
 */
static void test_levenshtein_distances() {
    struct {
        char const *left;
        char const *right;
        std::size_t distance;
    } explicit_cases[] = {
        {"listen", "silent", 4},
        {"", "", 0},
        {"", "abc", 3},
        {"abc", "", 3},
        {"abc", "ac", 1},                   // one deletion
        {"abc", "a_bc", 1},                 // one insertion
        {"abc", "adc", 1},                  // one substitution
        {"abc", "abc", 0},                  // same string
        {"ggbuzgjux{}l", "gbuzgjux{}l", 1}, // one insertion (prepended)
    };

    using matrix_t = std::int8_t[256][256];
    std::vector<std::int8_t> costs_vector = unary_substitution_costs();
    matrix_t &costs = *reinterpret_cast<matrix_t *>(costs_vector.data());

    auto print_failure = [&](char const *name, sz::string const &l, sz::string const &r, std::size_t expected,
                             std::size_t received) {
        char const *ellipsis = l.length() > 22 || r.length() > 22 ? "..." : "";
        std::printf("%s error: distance(\"%.22s%s\", \"%.22s%s\"); got %zd, expected %zd\n", //
                    name, l.c_str(), ellipsis, r.c_str(), ellipsis, received, expected);
    };

    auto test_distance = [&](sz::string const &l, sz::string const &r, std::size_t expected) {
        auto received = sz::edit_distance(l, r);
        auto received_score = sz::alignment_score(l, r, costs, -1);
        if (received != expected) print_failure("Levenshtein", l, r, expected, received);
        if ((std::size_t)(-received_score) != expected) print_failure("Scoring", l, r, expected, received_score);
        // The distance relation commutes
        received = sz::edit_distance(r, l);
        received_score = sz::alignment_score(r, l, costs, -1);
        if (received != expected) print_failure("Levenshtein", r, l, expected, received);
        if ((std::size_t)(-received_score) != expected) print_failure("Scoring", r, l, expected, received_score);
    };

    for (auto explicit_case : explicit_cases)
        test_distance(sz::string(explicit_case.left), sz::string(explicit_case.right), explicit_case.distance);

    // Gradually increasing the length of the strings.
    for (std::size_t length = 0; length != 1000; ++length) {
        sz::string left, right;
        for (std::size_t i = 0; i != length; ++i) left.push_back('a'), right.push_back('b');
        test_distance(left, right, length);
    }

    // Randomized tests
    struct {
        std::size_t length_upper_bound;
        std::size_t iterations;
    } fuzzy_cases[] = {
        {10, 1000},
        {100, 100},
        {1000, 10},
    };
    std::mt19937 &generator = global_random_generator();
    sz::string first, second;
    for (auto fuzzy_case : fuzzy_cases) {
        char alphabet[4] = {'a', 'c', 'g', 't'};
        std::uniform_int_distribution<std::size_t> length_distribution(0, fuzzy_case.length_upper_bound);
        for (std::size_t i = 0; i != fuzzy_case.iterations; ++i) {
            std::size_t first_length = length_distribution(generator);
            std::size_t second_length = length_distribution(generator);
            std::generate_n(std::back_inserter(first), first_length, [&]() { return alphabet[generator() % 4]; });
            std::generate_n(std::back_inserter(second), second_length, [&]() { return alphabet[generator() % 4]; });
            test_distance(first, second,
                          levenshtein_baseline(first.c_str(), first.length(), second.c_str(), second.length()));

            // Try computing the distance on equal-length chunks of those strings.
            first.resize(std::min(first_length, second_length));
            second.resize(std::min(first_length, second_length));
            test_distance(first, second,
                          levenshtein_baseline(first.c_str(), first.length(), second.c_str(), second.length()));

            // Discard before the next iteration.
            first.clear();
            second.clear();
        }
    }
}

/**
 *  @brief  Tests sorting functionality.
 */
static void test_sequence_algorithms() {
    using strs_t = std::vector<std::string>;
    using order_t = std::vector<sz::sorted_idx_t>;

    assert_scoped(strs_t x({"a", "b", "c", "d"}), (void)0, sz::sorted_order(x) == order_t({0u, 1u, 2u, 3u}));
    assert_scoped(strs_t x({"b", "c", "d", "a"}), (void)0, sz::sorted_order(x) == order_t({3u, 0u, 1u, 2u}));
    assert_scoped(strs_t x({"b", "a", "d", "c"}), (void)0, sz::sorted_order(x) == order_t({1u, 0u, 3u, 2u}));

    // Generate random strings of different lengths.
    for (std::size_t dataset_size : {10, 100, 1000, 10000}) {
        // Build the dataset.
        strs_t dataset;
        for (std::size_t i = 0; i != dataset_size; ++i)
            dataset.push_back(sz::scripts::random_string(i % 32, "abcdefghijklmnopqrstuvwxyz", 26));

        // Run several iterations of fuzzy tests.
        for (std::size_t experiment_idx = 0; experiment_idx != 10; ++experiment_idx) {
            std::shuffle(dataset.begin(), dataset.end(), global_random_generator());
            auto order = sz::sorted_order(dataset);
            for (std::size_t i = 1; i != dataset_size; ++i) { assert(dataset[order[i - 1]] <= dataset[order[i]]); }
        }
    }
}

int main(int argc, char const **argv) {

    // Let's greet the user nicely
    sz_unused(argc && argv);
    std::printf("Hi, dear tester! You look nice today!\n");
    std::printf("- Uses AVX2: %s \n", SZ_USE_X86_AVX2 ? "yes" : "no");
    std::printf("- Uses AVX512: %s \n", SZ_USE_X86_AVX512 ? "yes" : "no");
    std::printf("- Uses NEON: %s \n", SZ_USE_ARM_NEON ? "yes" : "no");
    std::printf("- Uses SVE: %s \n", SZ_USE_ARM_SVE ? "yes" : "no");

    // Basic utilities
    test_arithmetical_utilities();
    test_memory_utilities();

// Compatibility with STL
#if SZ_DETECT_CPP_17 && __cpp_lib_string_view
    test_api_readonly<std::string_view>();
#endif
    test_api_readonly<std::string>();
    test_api_readonly<sz::string_view>();
    test_api_readonly<sz::string>();

    test_api_mutable<std::string>(); // Make sure the test itself is reasonable
    test_api_mutable<sz::string>();  // The fact that this compiles is already a miracle :)

    // Cover the non-STL interfaces
    test_api_readonly_extensions<sz::string_view>();
    test_api_readonly_extensions<sz::string>();
    test_api_mutable_extensions();

    // The string class implementation
    test_constructors();
    test_memory_stability_for_length(1024);
    test_memory_stability_for_length(14);
    test_updates();

    // Advanced search operations
    test_stl_conversion_api();
    test_comparisons();
    test_search();
#if SZ_DETECT_CPP_17 && __cpp_lib_string_view
    test_search_with_misaligned_repetitions();
#endif

    // Similarity measures and fuzzy search
    test_levenshtein_distances();

    // Sequences of strings
    test_sequence_algorithms();

    std::printf("All tests passed... Unbelievable!\n");
    return 0;
}
