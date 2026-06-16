/**
 *  @brief  Arithmetic/struct plumbing, ASCII utilities, memory, STL-compat, conversions, extensions, and the string class.
 *  @file   scripts/test_string.cpp
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
#include <unordered_map> // `std::unordered_map`
#include <unordered_set> // `std::unordered_set`
#include <vector>        // `std::vector`

#include <string>      // Baseline
#include <string_view> // Baseline

#if !SZ_IS_CPP11_
#error "This test requires C++11 or later."
#endif

#include "test_stringzilla.hpp" // `global_random_generator`, `random_string`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using sz::literals::operator""_sv; // for `sz::string_view`
using sz::literals::operator""_bs; // for `sz::byteset`

#if SZ_IS_CPP17_
using namespace std::literals; // for ""sv
#endif

#pragma region Helpers

/**
 *  @brief Compares two byte ranges and aborts with a localized diagnostic on the first mismatch.
 *  @param first First byte range to compare.
 *  @param second Second byte range to compare.
 *  @param size Number of bytes to compare in both ranges.
 */
inline void expect_equality(char const *first, char const *second, std::size_t size) {
    if (std::memcmp(first, second, size) == 0) return;
    std::size_t mismatch_position = 0;
    for (; mismatch_position < size; ++mismatch_position)
        if (first[mismatch_position] != second[mismatch_position]) break;
    std::fprintf(stderr, "Mismatch at position %zu: %c != %c\n", mismatch_position, first[mismatch_position],
                 second[mismatch_position]);
    assert(false);
}

/**
 *  @brief The sum of an arithmetic progression.
 *  @see https://en.wikipedia.org/wiki/Arithmetic_progression
 *  @param first First term of the progression.
 *  @param last Last term of the progression.
 *  @param step Distance between consecutive terms.
 */
inline std::size_t arithmetic_sum(std::size_t first, std::size_t last, std::size_t step = 1) {
    std::size_t n = (last >= first) ? ((last - first) / step + 1) : 0;
    // Return 0 if there are no terms
    if (n == 0) return 0;
    // Compute the sum using the arithmetic sequence formula
    std::size_t sum = n / 2 * (2 * first + (n - 1) * step);
    // If n is odd, handle the remaining term separately to avoid overflow
    if (n % 2 == 1) sum += (2 * first + (n - 1) * step) / 2;
    return sum;
}

/** @brief Allocator wrapper that counts the number of allocated and deallocated bytes. */
struct accounting_allocator : public std::allocator<char> {
    inline static bool &verbose_ref() {
        static bool global_value = false;
        return global_value;
    }
    inline static std::size_t &counter_ref() {
        static std::size_t global_value = 0ul;
        return global_value;
    }

    template <typename... args_types_>
    static void print_if_verbose(char const *fmt, args_types_... args) {
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

/** @brief Runs @p callback and asserts that it leaves the global allocation counter unchanged. */
template <typename callback_type>
void assert_balanced_memory(callback_type callback) {
    auto bytes = accounting_allocator::account_block(callback);
    assert(bytes == 0);
}

/**
 *  @brief Runs one movement backend (copy/move/fill) through hand-verifiable known-answer vectors.
 *
 *  Mirrors the SHA256 known-answer helper in `test_hash.cpp`: each ISA tier feeds its kernel pointers here,
 *  so the dispatched C API and every natively-compiled backend share a single ground-truth check. Guard bytes
 *  past `length` catch stray writes.
 */
static void check_memory_unit_(sz_copy_t copy, sz_move_t move, sz_fill_t fill) {

    // `copy` duplicates a known buffer byte-for-byte. We over-allocate the target so a stray write
    // past `length` is visible as a corrupted guard byte.
    {
        char const source[] = "The quick brown fox"; // 19 bytes + terminator
        sz_size_t const length = (sz_size_t)(sizeof(source) - 1);
        char target[sizeof(source) + 1];
        std::memset(target, '#', sizeof(target));
        copy(target, source, length);
        assert(std::memcmp(target, source, length) == 0);
        assert(target[length] == '#'); // No overwrite past `length`
    }

    // `move` handles overlapping regions. Shifting "abcdef" left-into-itself by two yields "cdef" at the front.
    {
        char const expected[] = "cdef"; // After moving "cdef" (offset 2, 4 bytes) to offset 0
        char buffer[] = "abcdef";
        move(buffer, buffer + 2, 4);
        assert(std::memcmp(buffer, expected, 4) == 0);
    }

    // `fill` writes a known byte across a known span, leaving a guard byte untouched.
    {
        char const expected[] = "*****"; // Five asterisks
        char target[5 + 1];
        std::memset(target, '#', sizeof(target));
        fill(target, 5, (sz_u8_t)'*');
        assert(std::memcmp(target, expected, 5) == 0);
        assert(target[5] == '#'); // No overwrite past `length`
    }
}

/**
 *  @brief Runs one byte-lookup backend through a hand-verifiable known-answer vector.
 *
 *  The upper-casing table maps "Hello, World!" to "HELLO, WORLD!" while leaving punctuation and digits intact;
 *  a guard byte past `length` catches stray writes.
 */
static void check_lookup_unit_(sz_lookup_t lookup) {
    // An ASCII upper-casing table, built locally so the known-answer is verified against an external ground truth.
    char upper_table[256];
    for (sz_size_t byte_value = 0; byte_value != 256; ++byte_value) {
        char const character = (char)(unsigned char)byte_value;
        upper_table[byte_value] = (character >= 'a' && character <= 'z') ? (char)(character - 'a' + 'A') : character;
    }
    char const source[] = "Hello, World!"; // 13 bytes
    char const expected[] = "HELLO, WORLD!";
    sz_size_t const length = (sz_size_t)(sizeof(source) - 1);
    char target[sizeof(source) + 1];
    std::memset(target, '#', sizeof(target));
    lookup(target, length, source, upper_table);
    assert(std::memcmp(target, expected, length) == 0);
    assert(target[length] == '#'); // No overwrite past `length`
}

#pragma endregion // Helpers

#pragma region Unit

#pragma region Arithmetic

/**
 *  @brief Several string processing operations rely on computing integer logarithms.
 *         Failures in such operations will result in wrong `resize` outcomes and heap corruption.
 */
void test_arithmetic_unit() {

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

#if SZ_IS_64BIT_
    assert(sz_size_bit_ceil(4000000000ull) == (1ull << 32));
    assert(sz_size_bit_ceil(8000000000ull) == (1ull << 33));
    assert(sz_size_bit_ceil(16000000000ull) == (1ull << 34));

    assert(sz_size_bit_ceil((1ull << 62)) == (1ull << 62));
    assert(sz_size_bit_ceil((1ull << 62) + 1) == (1ull << 63));
    assert(sz_size_bit_ceil((1ull << 63)) == (1ull << 63));
#endif
}

#pragma endregion // Arithmetic

#pragma region Sequence

/** @brief Validates `sz_sequence_t` and related construction utilities. */
void test_sequence_unit() {
    // Make sure the sequence helper functions work as expected
    // for both trivial c-style arrays and more complicated STL containers.
    {
        sz_sequence_t sequence;
        sz_cptr_t strings[] = {"banana", "apple", "cherry"};
        sz_sequence_from_null_terminated_strings(strings, 3, &sequence);
        assert(sequence.count == 3);
        assert("banana"_sv == sequence.get_start(sequence.handle, 0));
        assert("apple"_sv == sequence.get_start(sequence.handle, 1));
        assert("cherry"_sv == sequence.get_start(sequence.handle, 2));
    }
    // Do the same for STL:
    {
        using strings_vector_t = std::vector<std::string>;
        strings_vector_t strings = {"banana", "apple", "cherry"};
        sz_sequence_t sequence;
        sequence.handle = &strings;
        sequence.count = strings.size();
        sequence.get_start = reinterpret_cast<sz_sequence_member_start_t>(
            +[](void *handle, sz_size_t index) noexcept -> sz_cptr_t {
                auto const &strings = *static_cast<strings_vector_t *>(handle);
                return strings[index].c_str();
            });
        sequence.get_length = reinterpret_cast<sz_sequence_member_length_t>(
            +[](void *handle, sz_size_t index) noexcept -> sz_size_t {
                auto const &strings = *static_cast<strings_vector_t *>(handle);
                return strings[index].size();
            });

        assert(sequence.count == 3);
        assert("banana"_sv == sequence.get_start(sequence.handle, 0));
        assert("apple"_sv == sequence.get_start(sequence.handle, 1));
        assert("cherry"_sv == sequence.get_start(sequence.handle, 2));
    }
}

#pragma endregion // Sequence

#pragma region Allocator

/** @brief Validates `sz_memory_allocator_t` and related construction utilities. */
void test_allocator_unit() {
    // Our behavior for `malloc(0)` is to return a NULL pointer,
    // while the standard is implementation-defined.
    {
        sz_memory_allocator_t alloc;
        sz_memory_allocator_init_default(&alloc);
        assert(alloc.allocate(0, alloc.handle) == nullptr);
    }

    // Non-NULL allocation
    {
        sz_memory_allocator_t alloc;
        sz_memory_allocator_init_default(&alloc);
        void *byte = alloc.allocate(1, alloc.handle);
        assert(byte != nullptr);
        alloc.free(byte, 1, alloc.handle);
    }

    // Use a fixed buffer
    {
        char buffer[1024];
        sz_memory_allocator_t alloc;
        sz_memory_allocator_init_fixed(&alloc, buffer, sizeof(buffer));
        void *byte = alloc.allocate(1, alloc.handle);
        assert(byte != nullptr);
        alloc.free(byte, 1, alloc.handle);
    }
}

#pragma endregion // Allocator

#pragma region Byteset

/** @brief Validates `sz_byteset_t` and related construction utilities. */
void test_byteset_unit() {
    sz_byteset_t s;
    sz_byteset_init(&s);
    assert(sz_byteset_contains(&s, 'a') == sz_false_k);
    sz_byteset_add(&s, 'a');
    assert(sz_byteset_contains(&s, 'a') == sz_true_k);
    sz_byteset_add(&s, 'z');
    assert(sz_byteset_contains(&s, 'z') == sz_true_k);
    sz_byteset_invert(&s);
    assert(sz_byteset_contains(&s, 'a') == sz_false_k);
    assert(sz_byteset_contains(&s, 'z') == sz_false_k);
    assert(sz_byteset_contains(&s, 'b') == sz_true_k);
    sz_byteset_init_ascii(&s);
    assert(sz_byteset_contains(&s, 'A') == sz_true_k);
}

/**
 *  @brief Tests various ASCII-based methods (e.g., `is_alpha`, `is_digit`)
 *         provided by `sz::string` and `sz::string_view`.
 */
template <typename string_type>
void test_ascii_unit() {

    using str = string_type;

    assert("aaa"_bs.size() == 1ull);
    assert("\0\0"_bs.size() == 1ull);
    assert("abc"_bs.size() == 3ull);
    assert("a\0bc"_bs.size() == 4ull);

    assert(!"abc"_bs.contains('\0'));
    assert(str("bca").contains_only("abc"_bs));

    assert(!str("").is_alpha());
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ").is_alpha());
    assert(!str("abc9").is_alpha());

    assert(!str("").is_alnum());
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789").is_alnum());
    assert(!str("abc!").is_alnum());

    assert(str("").is_ascii());
    assert(str("\x00x7F").is_ascii());
    assert(!str("abc123🔥").is_ascii());

    assert(!str("").is_digit());
    assert(str("0123456789").is_digit());
    assert(!str("012a").is_digit());

    assert(!str("").is_lower());
    assert(str("abcdefghijklmnopqrstuvwxyz").is_lower());
    assert(!str("abcA").is_lower());
    assert(!str("abc\n").is_lower());

    assert(!str("").is_space());
    assert(str(" \t\n\r\f\v").is_space());
    assert(!str(" \t\r\na").is_space());

    assert(!str("").is_upper());
    assert(str("ABCDEFGHIJKLMNOPQRSTUVWXYZ").is_upper());
    assert(!str("ABCa").is_upper());

    assert(str("").is_printable());
    assert(str("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+").is_printable());
    assert(!str("012🔥").is_printable());

    assert(str("").contains_only("abc"_bs));
    assert(str("abc").contains_only("abc"_bs));
    assert(!str("abcd").contains_only("abc"_bs));
}

#pragma endregion // Byteset

#pragma region Memory

/**
 *  @brief Known-answer + coverage for the memory primitives - the C-level building blocks of the string class.
 *
 *  Starts with known-answer vectors that exercise each function through the dispatched C API (automatic
 *  kernel resolution), through the natively-compiled backend kernels directly (manual propagation to a specific
 *  kernel), and through the C++ `sz::` wrappers, so a regression that the serial-vs-SIMD agreement tests would miss -
 *  because both share a wrong constant - is still caught against an external ground truth. It then mirrors a large set
 *  of `sz::memcpy`, `sz::memset`, and `sz::memmove` operations against their `std::` counterparts, using a large
 *  heap-allocated buffer to cover the larger-than-L2-cache code paths, various chunk sizes, overlapping regions, and
 *  both forward and backward traversals.
 *
 *  @param max_l2_size Size of the working buffers, chosen to exceed the L2 cache.
 */
void test_memory_unit(std::size_t max_l2_size) {

    std::printf("  - testing memory primitive known-answer vectors...\n");

    // Movement known-answers, through the dispatched C API and every natively-compiled backend.
    check_memory_unit_(sz_copy, sz_move, sz_fill);                      // Dispatched (automatic kernel)
    check_memory_unit_(sz_copy_serial, sz_move_serial, sz_fill_serial); // Manual: serial kernel
#if SZ_USE_HASWELL
    check_memory_unit_(sz_copy_haswell, sz_move_haswell, sz_fill_haswell); // Manual: Haswell kernel
#endif
#if SZ_USE_SKYLAKE
    check_memory_unit_(sz_copy_skylake, sz_move_skylake, sz_fill_skylake); // Manual: Skylake kernel
#endif
#if SZ_USE_NEON
    check_memory_unit_(sz_copy_neon, sz_move_neon, sz_fill_neon); // Manual: NEON kernel
#endif
#if SZ_USE_SVE
    check_memory_unit_(sz_copy_sve, sz_move_sve, sz_fill_sve); // Manual: SVE kernel
#endif
#if SZ_USE_V128
    check_memory_unit_(sz_copy_v128, sz_move_v128, sz_fill_v128); // Manual: WASM SIMD128 kernel
#endif
#if SZ_USE_V128RELAXED
    check_memory_unit_(sz_copy_v128relaxed, sz_move_v128relaxed, sz_fill_v128relaxed); // Manual: relaxed SIMD128 kernel
#endif
#if SZ_USE_RVV
    check_memory_unit_(sz_copy_rvv, sz_move_rvv, sz_fill_rvv); // Manual: RISC-V RVV kernel
#endif
#if SZ_USE_LASX
    check_memory_unit_(sz_copy_lasx, sz_move_lasx, sz_fill_lasx); // Manual: LoongArch LASX kernel
#endif
#if SZ_USE_POWERVSX
    check_memory_unit_(sz_copy_powervsx, sz_move_powervsx, sz_fill_powervsx); // Manual: Power VSX kernel
#endif

    // Lookup known-answers, through the dispatched C API and every natively-compiled backend.
    check_lookup_unit_(sz_lookup);        // Dispatched (automatic kernel)
    check_lookup_unit_(sz_lookup_serial); // Manual: serial kernel
#if SZ_USE_HASWELL
    check_lookup_unit_(sz_lookup_haswell); // Manual: Haswell kernel
#endif
#if SZ_USE_ICELAKE
    check_lookup_unit_(sz_lookup_icelake); // Manual: Ice Lake kernel
#endif
#if SZ_USE_NEON
    check_lookup_unit_(sz_lookup_neon); // Manual: NEON kernel
#endif
#if SZ_USE_SVE
    check_lookup_unit_(sz_lookup_sve); // Manual: SVE kernel
#endif
#if SZ_USE_V128
    check_lookup_unit_(sz_lookup_v128); // Manual: WASM SIMD128 kernel
#endif
#if SZ_USE_V128RELAXED
    check_lookup_unit_(sz_lookup_v128relaxed); // Manual: relaxed SIMD128 kernel
#endif
#if SZ_USE_RVV
    check_lookup_unit_(sz_lookup_rvv); // Manual: RISC-V RVV kernel
#endif
#if SZ_USE_LASX
    check_lookup_unit_(sz_lookup_lasx); // Manual: LoongArch LASX kernel
#endif
#if SZ_USE_POWERVSX
    check_lookup_unit_(sz_lookup_powervsx); // Manual: Power VSX kernel
#endif

    // C++ wrapper sanity: a couple of `sz::string` / `sz::string_view` known-answer reads alongside the C API.
    {
        sz::string_view const view = "Hello, World!"_sv;
        assert(view.size() == 13u);              // Length matches the literal
        assert(view.substr(7, 5) == "World"_sv); // Slice extracts the expected token
        assert(view.front() == 'H' && view.back() == '!');

        sz::string const owned = "Hello, World!";
        assert(owned.size() == 13u);                       // Owning string reports the same length
        assert(owned == view);                             // Owning string equals the view
        assert(sz::string("apple").compare("banana") < 0); // "apple" sorts before "banana"
    }

    // The C++ movement wrappers must agree with the known-answers, including overlapping `memmove`.
    {
        char const fox[] = "The quick brown fox";
        char target[sizeof(fox) + 1];
        let_assert(std::memset(target, '#', sizeof(target)), //
                   (sz::memcpy(target, fox, sizeof(fox) - 1), std::memcmp(target, fox, sizeof(fox) - 1) == 0) &&
                       target[sizeof(fox) - 1] == '#');

        char overlap[] = "abcdef";
        let_assert(sz::memmove(overlap, overlap + 2, 4), std::memcmp(overlap, "cdef", 4) == 0);

        char asterisks[5 + 1];
        let_assert(std::memset(asterisks, '#', sizeof(asterisks)), //
                   (sz::memset(asterisks, '*', 5), std::memcmp(asterisks, "*****", 5) == 0) && asterisks[5] == '#');
    }

    // Embedded NUL must be preserved verbatim by a stored `sz::string`: the size is the full byte length, and
    // indexing past the interior NUL reaches the trailing bytes rather than stopping at the C-string boundary.
    {
        char const with_nul[] = {'a', 'b', '\0', 'c', 'd'};
        sz::string const owned(with_nul, sizeof(with_nul));
        assert(owned.size() == sizeof(with_nul));   // Full length, NUL is a stored byte
        assert(owned[2] == '\0');                   // The interior NUL survives
        assert(owned[3] == 'c' && owned[4] == 'd'); // Indexing past the NUL works
        assert(owned == sz::string_view(with_nul, sizeof(with_nul)));
    }

    // We will be mirroring the operations on both standard and StringZilla strings.
    std::string text_stl(max_l2_size, '-');
    std::string text_sz(max_l2_size, '-');
    expect_equality(text_stl.data(), text_sz.data(), max_l2_size);

    // The traditional `memset` and `memcpy` functions are undefined for zero-length buffers and NULL pointers
    // for older C standards.  However, with the N3322 proposal for C2y, that issue has been resolved.
    // https://developers.redhat.com/articles/2024/12/11/making-memcpynull-null-0-well-defined
    //
    // Let's make sure, that our versions don't trigger any undefined behavior.
    sz::memset(NULL, 0, 0);
    sz::memcpy(NULL, NULL, 0);
    sz::memmove(NULL, NULL, 0);

    // First start with simple deterministic tests.
    // Let's use `memset` to fill the strings with a pattern like "122333444455555...00000000000011111111111..."
    std::size_t count_groups = 0;
    for (std::size_t offset = 0, fill_length = 1; offset < max_l2_size;
         offset += fill_length, ++fill_length, ++count_groups) {
        char fill_value = '0' + fill_length % 10;
        fill_length = offset + fill_length > max_l2_size ? max_l2_size - offset : fill_length;
        std::memset((void *)(text_stl.data() + offset), fill_value, fill_length);
        sz::memset((void *)(text_sz.data() + offset), fill_value, fill_length);
        expect_equality(text_stl.data(), text_sz.data(), max_l2_size);
    }

    // Let's copy those chunks to an empty buffer one by one, validating the overall equivalency after every copy.
    std::string copy_stl(max_l2_size, '-');
    std::string copy_sz(max_l2_size, '-');
    for (std::size_t offset = 0, fill_length = 1; offset < max_l2_size; offset += fill_length, ++fill_length) {
        fill_length = offset + fill_length > max_l2_size ? max_l2_size - offset : fill_length;
        std::memcpy((void *)(copy_stl.data() + offset), (void *)(text_stl.data() + offset), fill_length);
        sz::memcpy((void *)(copy_sz.data() + offset), (void *)(text_sz.data() + offset), fill_length);
        expect_equality(copy_stl.data(), copy_sz.data(), max_l2_size);
    }
    expect_equality(text_stl.data(), copy_stl.data(), max_l2_size);
    expect_equality(text_sz.data(), copy_sz.data(), max_l2_size);

    // Let's simulate a realistic `memmove` workloads, compacting parts of this buffer, removing all odd values,
    // so the buffer will look like "224444666666..."
    for (std::size_t offset = 0, fill_length = 1; offset < max_l2_size; offset += fill_length, ++fill_length) {
        if (fill_length % 2 == 0) continue;             // Skip even chunks
        if (offset + fill_length >= max_l2_size) break; // This is the last & there are no more even chunks to shift

        // Make sure we don't overflow the buffer
        std::size_t next_offset = offset + fill_length;
        std::size_t next_fill_length = fill_length + 1;
        next_fill_length = next_offset + next_fill_length > max_l2_size ? max_l2_size - next_offset : next_fill_length;

        std::memmove((void *)(text_stl.data() + offset), (void *)(text_stl.data() + next_offset), next_fill_length);
        sz::memmove((void *)(text_sz.data() + offset), (void *)(text_sz.data() + next_offset), next_fill_length);
        expect_equality(text_stl.data(), text_sz.data(), max_l2_size);
    }

    // Now the opposite workload, expanding the buffer, inserting a dash "-" before every group of equal characters.
    // We will need to navigate right-to left to avoid overwriting the groups.
    std::size_t dashed_capacity = copy_stl.size() + count_groups;
    std::size_t dashed_length = 0;
    copy_stl.resize(dashed_capacity);
    copy_sz.resize(dashed_capacity);
    for (std::size_t reverse_offset = 0; reverse_offset < max_l2_size;) {

        // Walk backwards to find the length of the current group
        std::size_t offset = max_l2_size - reverse_offset - 1;
        std::size_t fill_length = 1;
        while (offset > 0 && copy_stl[offset - 1] == copy_stl[offset]) --offset, ++fill_length;

        std::size_t new_offset = dashed_capacity - dashed_length - fill_length;
        std::memmove((void *)(copy_stl.data() + new_offset), (void *)(copy_stl.data() + offset), fill_length);
        sz::memmove((void *)(copy_sz.data() + new_offset), (void *)(copy_sz.data() + offset), fill_length);
        expect_equality(copy_stl.data(), copy_sz.data(), max_l2_size);

        // Put the delimiter
        copy_stl[new_offset] = '-';
        copy_sz[new_offset] = '-';
        dashed_length += fill_length + 1;
        reverse_offset += fill_length;
    }
}

/**
 *  @brief Tests memory utilities on large buffers (>1MB) that trigger special code paths
 *         in AVX2/AVX512 implementations. This specifically tests the bidirectional
 *         traversal optimization used for huge buffers.
 */
void test_memory_large_unit() {
    // Test sizes that trigger the "huge buffer" path (> 1MB)
    std::vector<std::size_t> test_sizes = {
        1024ull * 1024ull + 1,       // Just over 1MB
        1024ull * 10ull * 103ull,    // From GitHub issue #228: 1,055,360 bytes
        2ull * 1024ull * 1024ull,    // 2MB
        3ull * 1024ull * 1024ull + 7 // 3MB + 7 (unaligned size)
    };

    for (std::size_t size : test_sizes) {
        // Test memcpy with aligned buffers
        {
            std::vector<char> source(size);
            std::vector<char> target_std(size);
            std::vector<char> target_sz(size);

            // Fill source with pattern to detect copying errors
            for (std::size_t i = 0; i < size; i++) { source[i] = static_cast<char>('A' + (i % 26)); }

            std::memcpy(target_std.data(), source.data(), size);
            sz::memcpy(target_sz.data(), source.data(), size);

            expect_equality(target_std.data(), target_sz.data(), size);
        }

        // Test memcpy with unaligned buffers
        {
            std::vector<char> source_buffer(size + 64);
            std::vector<char> target_std_buffer(size + 64);
            std::vector<char> target_sz_buffer(size + 64);

            // Use unaligned pointers
            char *source = source_buffer.data() + 7;
            char *target_std = target_std_buffer.data() + 11;
            char *target_sz = target_sz_buffer.data() + 11;

            for (std::size_t i = 0; i < size; i++) { source[i] = static_cast<char>('a' + (i % 26)); }

            std::memcpy(target_std, source, size);
            sz::memcpy(target_sz, source, size);

            expect_equality(target_std, target_sz, size);
        }

        // Test memset
        {
            std::vector<char> buf_std(size);
            std::vector<char> buf_sz(size);

            std::memset(buf_std.data(), 'Z', size);
            sz::memset(buf_sz.data(), 'Z', size);

            expect_equality(buf_std.data(), buf_sz.data(), size);
        }

        // Test memmove with overlapping regions
        {
            std::vector<char> buf_std(size);
            std::vector<char> buf_sz(size);

            // Initialize both buffers identically
            for (std::size_t i = 0; i < size; i++) { buf_std[i] = buf_sz[i] = static_cast<char>('0' + (i % 10)); }

            // Move overlapping region forward
            std::size_t overlap_size = size / 2;
            std::memmove(buf_std.data() + 100, buf_std.data(), overlap_size);
            sz::memmove(buf_sz.data() + 100, buf_sz.data(), overlap_size);

            expect_equality(buf_std.data(), buf_sz.data(), size);
        }
    }
}

#pragma endregion // Memory

#pragma region STL Reads

/**
 *  @brief Invokes different C++ member methods of immutable strings to cover all STL APIs.
 *         This test guarantees API @b compatibility with STL `std::basic_string` template.
 */
template <typename string_type>
void test_stl_reads_unit() {

    using str = string_type;

    // Constructors.
    assert(str().empty());             // Test default constructor
    assert(str().size() == 0);         // Test default constructor
    assert(str("").empty());           // Test default constructor
    assert(str("").size() == 0);       // Test default constructor
    assert(str("hello").size() == 5);  // Test constructor with c-string
    assert(str("hello", 4) == "hell"); // Construct from substring

    // Element access.
    assert(str("rest")[0] == 'r');
    assert(str("rest").at(1) == 'e');
    assert(*str("rest").data() == 'r');
    assert(str("front").front() == 'f');
    assert(str("back").back() == 'k');

    // Iterators.
    assert(*str("begin").begin() == 'b' && *str("cbegin").cbegin() == 'c');
    assert(*str("rbegin").rbegin() == 'n' && *str("crbegin").crbegin() == 'n');
    assert(str("size").size() == 4 && str("length").length() == 6);

    // Slices... out-of-bounds exceptions are asymmetric!
    // Moreover, `std::string` has no `remove_prefix` and `remove_suffix` methods.
    // scope_assert(str s = "hello", s.remove_prefix(1), s == "ello");
    // scope_assert(str s = "hello", s.remove_suffix(1), s == "hell");
    assert(str("hello world").substr(0, 5) == "hello");
    assert(str("hello world").substr(6, 5) == "world");
    assert(str("hello world").substr(6) == "world");
    assert(str("hello world").substr(6, 100) == "world"); // 106 is beyond the length of the string, but its OK
    assert_throws(str("hello world").substr(100), std::out_of_range);   // 100 is beyond the length of the string
    assert_throws(str("hello world").substr(20, 5), std::out_of_range); // 20 is beyond the length of the string
#if defined(__GNUC__) && !defined(__NVCC__) // -1 casts to unsigned without warnings on GCC, but not NVCC
    assert_throws(str("hello world").substr(-1, 5), std::out_of_range);
    assert(str("hello world").substr(0, -1) == "hello world");
#endif

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
    assert(str("abbabbaaaaaa").find("ba") == 2);
    assert(str("abbabbaaaaaa").find("bb") == 1);
    assert(str("abbabbaaaaaa").find("bab") == 2);
    assert(str("abbabbaaaaaa").find("babb") == 2);
    assert(str("abbabbaaaaaa").find("babba") == 2);
    assert(str("abcdabcd").substr(2, 4).find("abc") == str::npos);
    assert(str("hello, world!").substr(0, 11).find("world") == str::npos);
    assert(str("axabbcxcaaabbccc").find("aaabbccc") == 8);
    assert(str("abcdabcdabc________").find("abcd") == 0);
    assert(str("________abcdabcdabc").find("abcd") == 8);

    // Cover every SWAR case for unique string sequences.
    auto lowercase_alphabet = str("abcdefghijklmnopqrstuvwxyz");
    for (std::size_t one_byte_offset = 0; one_byte_offset + 1 <= lowercase_alphabet.size(); ++one_byte_offset)
        assert(lowercase_alphabet.find(lowercase_alphabet.substr(one_byte_offset, 1)) == one_byte_offset);
    for (std::size_t two_byte_offset = 0; two_byte_offset + 2 <= lowercase_alphabet.size(); ++two_byte_offset)
        assert(lowercase_alphabet.find(lowercase_alphabet.substr(two_byte_offset, 2)) == two_byte_offset);
    for (std::size_t four_byte_offset = 0; four_byte_offset + 4 <= lowercase_alphabet.size(); ++four_byte_offset)
        assert(lowercase_alphabet.find(lowercase_alphabet.substr(four_byte_offset, 4)) == four_byte_offset);
    for (std::size_t three_byte_offset = 0; three_byte_offset + 3 <= lowercase_alphabet.size(); ++three_byte_offset)
        assert(lowercase_alphabet.find(lowercase_alphabet.substr(three_byte_offset, 3)) == three_byte_offset);
    for (std::size_t five_byte_offset = 0; five_byte_offset + 5 <= lowercase_alphabet.size(); ++five_byte_offset)
        assert(lowercase_alphabet.find(lowercase_alphabet.substr(five_byte_offset, 5)) == five_byte_offset);

    // Simple repeating patterns - with one "almost match" before an actual match in each direction.
    assert(str("_ab_abc_").find("abc") == 4);
    assert(str("_abc_ab_").rfind("abc") == 1);
    assert(str("_abc_abcd_").find("abcd") == 5);
    assert(str("_abcd_abc_").rfind("abcd") == 1);
    assert(str("_abcd_abcde_").find("abcde") == 6);
    assert(str("_abcde_abcd_").rfind("abcde") == 1);
    assert(str("_abcde_abcdef_").find("abcdef") == 7);
    assert(str("_abcdef_abcde_").rfind("abcdef") == 1);
    assert(str("_abcdef_abcdefg_").find("abcdefg") == 8);
    assert(str("_abcdefg_abcdef_").rfind("abcdefg") == 1);

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

    // Using single-byte non-ASCII values, e.g., À (0xC0), Æ (0xC6). The `\xFA`/`0` boundary is
    // load-bearing: a literal hex digit after `\xFA` would extend the escape, so keep it split.
    {
        char const *non_ascii_set = "abcdefgh\x01\xC6ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\xC0\xFA" //
                                    "0123456789+-";                                                        // 68 bytes
        assert(str(non_ascii_set, 68).find_first_of("\xC6\xC7") == 9);                                     // sets
        assert(str(non_ascii_set, 68).find_first_of("\xC0\xC1") == 54);                                    // sets
        assert(str(non_ascii_set, 68).find_last_of("\xC6\xC7") == 9);                                      // sets
        assert(str(non_ascii_set, 68).find_last_of("\xC0\xC1") == 54);                                     // sets
    }

    // Boundary conditions.
    assert(str("hello").find_first_of("ox", 4) == 4);
    assert(str("hello").find_first_of("ox", 5) == str::npos);
    assert(str("hello").find_last_of("ox", 4) == 4);
    assert(str("hello").find_last_of("ox", 5) == 4);
    assert(str("hello").find_first_of("hx", 0) == 0);
    assert(str("hello").find_last_of("hx", 0) == 0);

    // More complex relative patterns
    assert(str("0123456789012345678901234567890123456789012345678901234567890123") <=
           str("0123456789012345678901234567890123456789012345678901234567890123"));
    assert(str("0123456789012345678901234567890123456789012345678901234567890123") <=
           str("0223456789012345678901234567890123456789012345678901234567890123"));
    assert(str("0123456789012345678901234567890123456789012345678901234567890123") <=
           str("0213456789012345678901234567890123456789012345678901234567890123"));
    assert(str("12341234") <= str("12341234"));
    assert(str("12341234") > str("12241224"));
    assert(str("12341234") < str("13241324"));
    assert(str("0123456789012345678901234567890123456789012345678901234567890123") ==
           str("0123456789012345678901234567890123456789012345678901234567890123"));
    assert(str("0123456789012345678901234567890123456789012345678901234567890123") !=
           str("0223456789012345678901234567890123456789012345678901234567890123"));

    // Comparisons.
    assert(str("a") != str("b"));
    assert(str("a") < str("b"));
    assert(str("a") <= str("b"));
    assert(str("b") > str("a"));
    assert(str("b") >= str("a"));
    assert(str("a") < str("aa"));

#if SZ_IS_CPP20_ && defined(__cpp_lib_three_way_comparison)
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

#if SZ_IS_CPP20_ && defined(__cpp_lib_starts_ends_with)
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

#if SZ_IS_CPP23_ && defined(__cpp_lib_string_contains)
    // Checking basic substring presence.
    assert(str("hello").contains(str("ell")) == true);
    assert(str("hello").contains(str("oll")) == false);
    assert(str("hello").contains('l') == true);
    assert(str("hello").contains('x') == false);
    assert(str("hello").contains("lo") == true);
    assert(str("hello").contains("lx") == false);
#endif

    // Exporting the contents of the string using the `str::copy` method.
    scope_assert(char buf[5 + 1] = {0}, str("hello").copy(buf, 5), std::strcmp(buf, "hello") == 0);
    scope_assert(char buf[4 + 1] = {0}, str("hello").copy(buf, 4, 1), std::strcmp(buf, "ello") == 0);
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
    scope_assert(std::ostringstream os, os << str("hello"), os.str() == "hello");

#if SZ_IS_CPP14_
    // Comparison function objects are a C++14 feature.
    assert(std::equal_to<str> {}("hello", "world") == false);
    assert(std::less<str> {}("hello", "world") == true);
#endif
}

#pragma endregion // STL Reads

#pragma region STL Updates

/**
 *  @brief Invokes different C++ member methods of the memory-owning string class to make sure they all pass
 *         compilation. This test guarantees API compatibility with STL `std::basic_string` template.
 */
template <typename string_type>
void test_stl_updates_unit() {

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

    // Corner case constructors and search behaviors for long strings
    assert(str(258, '0').find(str(256, '1')) == str::npos);

    // Assignments.
    scope_assert(str s = "obsolete", s = "hello", s == "hello");
    scope_assert(str s = "obsolete", s.assign("hello"), s == "hello");
    scope_assert(str s = "obsolete", s.assign("hello", 4), s == "hell");
    scope_assert(str s = "obsolete", s.assign(5, 'a'), s == "aaaaa");
    scope_assert(str s = "obsolete", s.assign(32, 'a'), s == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    scope_assert(str s = "obsolete", s.assign({'h', 'e', 'l', 'l', 'o'}), s == "hello");
    scope_assert(str s = "obsolete", s.assign(str("hello")), s == "hello");
    scope_assert(str s = "obsolete", s.assign(str("hello"), 2), s == "llo");
    scope_assert(str s = "obsolete", s.assign(str("hello"), 2, 2), s == "ll");
    scope_assert(str s = "obsolete", s.assign(str("hello"), 2, 2), s == "ll");
    scope_assert(str s = "obsolete", s.assign(s), s == "obsolete");                  // Self-assignment
    scope_assert(str s = "obsolete", s.assign(s.begin(), s.end()), s == "obsolete"); // Self-assignment
    scope_assert(str s = "obsolete", s.assign(s, 4), s == "lete");                   // Partial self-assignment
    scope_assert(str s = "obsolete", s.assign(s, 4, 3), s == "let");                 // Partial self-assignment

    // Self-assignment is a special case of assignment.
    scope_assert(str s = "obsolete", s = s, s == "obsolete");
    scope_assert(str s = "obsolete", s.assign(s), s == "obsolete");
    scope_assert(str s = "obsolete", s.assign(s.data(), 2), s == "ob");
    scope_assert(str s = "obsolete", s.assign(s.data(), s.size()), s == "obsolete");

    // Allocations, capacity and memory management.
    scope_assert(str s, s.reserve(10), s.capacity() >= 10);
    scope_assert(str s, s.resize(10), s.size() == 10);
    scope_assert(str s, s.resize(10, 'a'), s.size() == 10 && s == "aaaaaaaaaa");
    assert(str().max_size() > 0);
    assert(str().get_allocator() == std::allocator<char>());
    assert(std::strcmp(str("c_str").c_str(), "c_str") == 0);

#if SZ_IS_CPP23_ && defined(__cpp_lib_string_resize_and_overwrite)
    // Test C++23 resize and overwrite functionality
    scope_assert(str s("hello"),
                 s.resize_and_overwrite(10,
                                        [](char *p, std::size_t count) noexcept {
                                            std::memset(p, 'X', count);
                                            return count;
                                        }),
                 s.size() == 10 && s == "XXXXXXXXXX");

    scope_assert(str s("test"),
                 s.resize_and_overwrite(8,
                                        [](char *p, std::size_t) noexcept {
                                            std::strcpy(p, "ABCDE");
                                            return 5;
                                        }),
                 s.size() == 5 && s == "ABCDE");

    scope_assert(str s("orig"),
                 s.try_resize_and_overwrite(6,
                                            [](char *p, std::size_t count) noexcept {
                                                std::strcpy(p, "works!");
                                                return count;
                                            }),
                 s.size() == 6 && s == "works!");
#endif

    // On 32-bit systems the base capacity can be larger than our `z::string::min_capacity`.
    // It's true for MSVC: https://github.com/ashvardanian/StringZilla/issues/168
    if (SZ_IS_64BIT_) scope_assert(str s = "hello", s.shrink_to_fit(), s.capacity() <= sz::string::min_capacity);

    // Concatenation.
    // Following are missing in strings, but are present in vectors.
    // scope_assert(str s = "!?", s.push_front('a'), s == "a!?");
    // scope_assert(str s = "!?", s.pop_front(), s == "?");
    assert(str().append("test") == "test");
    assert(str("test") + "ing" == "testing");
    assert(str("test") + str("ing") == "testing");
    assert(str("test") + str("ing") + str("123") == "testing123");
    scope_assert(str s = "!?", s.push_back('a'), s == "!?a");
    scope_assert(str s = "!?", s.pop_back(), s == "!");

    // Incremental construction.
    assert(str("__").insert(1, "test") == "_test_");
    assert(str("__").insert(1, "test", 2) == "_te_");
    assert(str("__").insert(1, 5, 'a') == "_aaaaa_");
    assert(str("__").insert(1, str("test")) == "_test_");
    assert(str("__").insert(1, str("test"), 2) == "_st_");
    assert(str("__").insert(1, str("test"), 2, 1) == "_s_");

    // Inserting at a given iterator position yields back an iterator.
    scope_assert(str s = "__", s.insert(s.begin() + 1, 5, 'a'), s == "_aaaaa_");
    scope_assert(str s = "__", s.insert(s.begin() + 1, {'a', 'b', 'c'}), s == "_abc_");
    let_assert(str s = "__", s.insert(s.begin() + 1, 5, 'a') == (s.begin() + 1));
    let_assert(str s = "__", s.insert(s.begin() + 1, {'a', 'b', 'c'}) == (s.begin() + 1));

    // Handle exceptions.
    // The `length_error` might be difficult to catch due to a large `max_size()`.
    // assert_throws(large_string.insert(large_string.size() - 1, large_number_of_chars, 'a'), std::length_error);
    assert_throws(str("hello").insert(6, "world"), std::out_of_range);         // `index > size()` case from STL
    assert_throws(str("hello").insert(5, str("world"), 6), std::out_of_range); // `s_index > str.size()` case from STL

    // Erasure.
    assert(str("").erase(0, 3) == "");
    assert(str("test").erase(1, 2) == "tt");
    assert(str("test").erase(1) == "t");
    scope_assert(str s = "test", s.erase(s.begin() + 1), s == "tst");
    scope_assert(str s = "test", s.erase(s.begin() + 1, s.begin() + 2), s == "tst");
    scope_assert(str s = "test", s.erase(s.begin() + 1, s.begin() + 3), s == "tt");
    let_assert(str s = "test", s.erase(s.begin() + 1) == (s.begin() + 1));
    let_assert(str s = "test", s.erase(s.begin() + 1, s.begin() + 2) == (s.begin() + 1));
    let_assert(str s = "test", s.erase(s.begin() + 1, s.begin() + 3) == (s.begin() + 1));

    // Substitutions.
    assert(str("hello").replace(1, 2, "123") == "h123lo");
    assert(str("hello").replace(1, 2, str("123"), 1) == "h23lo");
    assert(str("hello").replace(1, 2, "123", 1) == "h1lo");
    assert(str("hello").replace(1, 2, "123", 1, 1) == "h2lo");
    assert(str("hello").replace(1, 2, str("123"), 1, 1) == "h2lo");
    assert(str("hello").replace(1, 2, 3, 'a') == "haaalo");

    // Substitutions with iterators.
    scope_assert(str s = "hello", s.replace(s.begin() + 1, s.begin() + 3, 3, 'a'), s == "haaalo");
    scope_assert(str s = "hello", s.replace(s.begin() + 1, s.begin() + 3, {'a', 'b'}), s == "hablo");

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
    let_assert(str s = "123", str("hello").append(s.begin(), s.end()) == "hello123");
}

/** @brief Constructs StringZilla classes from STL and vice-versa to ensure that the conversions are working. */
void test_stl_conversions_unit() {
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
        [[maybe_unused]] sz::string const sz = stl;
        [[maybe_unused]] sz::string_view const szv = stl;
    }
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
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

/** @brief Tests constructing STL containers with StringZilla strings. */
void test_stl_containers_unit() {
    std::map<sz::string, int> sorted_words_sz;
    std::unordered_map<sz::string, int> words_sz;
    assert(sorted_words_sz.empty());
    assert(words_sz.empty());

    std::map<std::string, int, sz::less> sorted_words_stl;
    std::unordered_map<std::string, int, sz::hash, sz::equal_to> words_stl;
    assert(sorted_words_stl.empty());
    assert(words_stl.empty());
}

#pragma endregion // STL Updates

#pragma region Extensions

/**
 *  @brief Invokes different C++ member methods of immutable strings to cover
 *         extensions beyond the STL API.
 */
template <typename string_type>
void test_extensions_reads_unit() {
    using str = string_type;

    // Signed offset lookups and slices.
    assert(str("hello").sat(0) == 'h');
    assert(str("hello").sat(-1) == 'o');
    assert(str("rest").sat(1) == 'e');
    assert(str("rest").sat(-1) == 't');
    assert(str("rest").sat(-4) == 'r');

    assert(str("front").front() == 'f');
    assert(str("front").front(1) == "f");
    assert(str("front").front(2) == "fr");
    assert(str("front").front(2) == "fr");
    assert(str("front").front(-2) == "fro");
    assert(str("front").front(0) == "");
    assert(str("front").front(5) == "front");
    assert(str("front").front(-5) == "");

    assert(str("back").back() == 'k');
    assert(str("back").back(1) == "ack");
    assert(str("back").back(2) == "ck");
    assert(str("back").back(-1) == "k");
    assert(str("back").back(-2) == "ck");
    assert(str("back").back(-4) == "back");
    assert(str("back").back(4) == "");

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

    // Checksums
    auto accumulate_bytes = [](str const &s) -> std::size_t {
        return std::accumulate(s.begin(), s.end(), (std::size_t)0,
                               [](std::size_t sum, char c) { return sum + static_cast<unsigned char>(c); });
    };
    assert(str("a").bytesum() == (std::size_t)'a');
    assert(str("0").bytesum() == (std::size_t)'0');
    assert(str("0123456789").bytesum() == arithmetic_sum('0', '9'));
    assert(str("abcdefghijklmnopqrstuvwxyz").bytesum() == arithmetic_sum('a', 'z'));
    assert(str("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz").bytesum() ==
           arithmetic_sum('a', 'z') * 3);
    let_assert(str s = "近来，加文出席微博之夜时对着镜头频繁摆出假笑表情、一度累" //
                       "瘫睡倒在沙发上的照片被广泛转发，引发对他失去童年、被过度" //
                       "消费的担忧。八岁的加文，已当网红近六年了，可以说，自懂事" //
                       "以来，他没有过过一天没有名气的日子。",
               s.bytesum() == accumulate_bytes(s));
}

/** @brief Exercises StringZilla's non-STL mutating string extensions on `sz::string`. */
void test_extensions_updates_unit() {
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
    scope_assert(str s = "test", s.try_assign(s.view()), s == "test");
    scope_assert(str s = "test", s.try_assign(s.view().sub(1, 2)), s == "e");
    scope_assert(str s = "test", s.try_append(s.view().sub(1, 2)), s == "teste");

    // Try methods going beyond and beneath capacity threshold.
    scope_assert(str s = "0123456789012345678901234567890123456789012345678901234567890123", // 64 symbols at start
                 s.try_append(s) && s.try_append(s) && s.try_append(s) && s.try_append(s) && s.try_clear() &&
                     s.try_shrink_to_fit(),
                 s.capacity() < sz::string::min_capacity);

    // Same length replacements.
    scope_assert(str s = "hello", s.replace_all("xx", "xx"), s == "hello");
    scope_assert(str s = "hello", s.replace_all("l", "1"), s == "he11o");
    scope_assert(str s = "hello", s.replace_all("he", "al"), s == "alllo");
    scope_assert(str s = "hello", s.replace_all("x"_bs, "!"), s == "hello");
    scope_assert(str s = "hello", s.replace_all("o"_bs, "!"), s == "hell!");
    scope_assert(str s = "hello", s.replace_all("ho"_bs, "!"), s == "!ell!");

    // Shorter replacements.
    scope_assert(str s = "hello", s.replace_all("xx", "x"), s == "hello");
    scope_assert(str s = "hello", s.replace_all("l", ""), s == "heo");
    scope_assert(str s = "hello", s.replace_all("h", ""), s == "ello");
    scope_assert(str s = "hello", s.replace_all("o", ""), s == "hell");
    scope_assert(str s = "hello", s.replace_all("llo", "!"), s == "he!");
    scope_assert(str s = "hello", s.replace_all("x"_bs, ""), s == "hello");
    scope_assert(str s = "hello", s.replace_all("lo"_bs, ""), s == "he");

    // Longer replacements.
    scope_assert(str s = "hello", s.replace_all("xx", "xxx"), s == "hello");
    scope_assert(str s = "hello", s.replace_all("l", "ll"), s == "hellllo");
    scope_assert(str s = "hello", s.replace_all("h", "hh"), s == "hhello");
    scope_assert(str s = "hello", s.replace_all("o", "oo"), s == "helloo");
    scope_assert(str s = "hello", s.replace_all("llo", "llo!"), s == "hello!");
    scope_assert(str s = "hello", s.replace_all("x"_bs, "xx"), s == "hello");
    scope_assert(str s = "hello", s.replace_all("lo"_bs, "lo"), s == "helololo");

    // Directly mapping bytes using a Look-Up Table.
    sz::look_up_table invert_case = sz::look_up_table::identity();
    for (char c = 'a'; c <= 'z'; c++) invert_case[c] = c - 'a' + 'A';
    for (char c = 'A'; c <= 'Z'; c++) invert_case[c] = c - 'A' + 'a';
    scope_assert(str s = "hello", s.lookup(invert_case), s == "HELLO");
    scope_assert(str s = "HeLLo", s.lookup(invert_case), s == "hEllO");
    scope_assert(str s = "H-lL0", s.lookup(invert_case), s == "h-Ll0");

    // Concatenation.
    assert(str(str("a") | str("b")) == "ab");
    assert(str(str("a") | str("b") | str("ab")) == "abab");

    assert(str(sz::concatenate("a"_sv, "b"_sv)) == "ab");
    assert(str(sz::concatenate("a"_sv, "b"_sv, "c"_sv)) == "abc");

    // Randomization.
    assert(str::random(0).empty());
    assert(str::random(4).size() == 4);
    assert(str::random(4, 42).size() == 4);
}

#pragma endregion // Extensions

#pragma region String Class

/** @brief Tests copy constructor and copy-assignment constructor of `sz::string`. */
void test_string_constructors_unit() {
    std::string alphabet {sz::ascii_printables(), sizeof(sz::ascii_printables())};
    std::vector<sz::string> strings;
    for (std::size_t alphabet_slice = 0; alphabet_slice != alphabet.size(); ++alphabet_slice)
        strings.push_back(alphabet.substr(0, alphabet_slice));
    std::vector<sz::string> copies {strings};
    assert(copies.size() == strings.size());
    for (size_t i = 0; i < copies.size(); ++i) {
        assert(copies[i].size() == strings[i].size());
        assert(copies[i] == strings[i]);
        for (size_t j = 0; j < strings[i].size(); j++) { assert(copies[i][j] == strings[i][j]); }
    }
    std::vector<sz::string> assignments = strings;
    for (size_t i = 0; i < assignments.size(); ++i) {
        assert(assignments[i].size() == strings[i].size());
        assert(assignments[i] == strings[i]);
        for (size_t j = 0; j < strings[i].size(); j++) { assert(assignments[i][j] == strings[i][j]); }
    }
    assert(std::equal(strings.begin(), strings.end(), copies.begin()));
    assert(std::equal(strings.begin(), strings.end(), assignments.begin()));
}

/**
 *  @brief Checks for memory leaks in the string class using the `accounting_allocator`.
 *
 *  @param length Number of characters in the base string under test.
 *  @param iterations Number of repetitions per allocation-balance block.
 *  @note The baseline iteration count (100) is scaled by `SZ_TEST_ITERATIONS_MULTIPLIER`.
 */
void test_memory_stability_unit(std::size_t length, std::size_t iterations) {

    assert(accounting_allocator::counter_ref() == 0);
    using string = sz::basic_string<char, accounting_allocator>;
    string base;

    for (std::size_t i = 0; i < length; ++i) base.push_back('c');
    assert(base.length() == length);

    // Do copies leak?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string copy(base);
            assert(copy.length() == length);
            assert(copy == base);
        }
    });

    // How about assignments?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string copy;
            copy = base;
            assert(copy.length() == length);
            assert(copy == base);
        }
    });

    // How about the move constructor?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string unique_item(base);
            assert(unique_item.length() == length);
            assert(unique_item == base);
            string copy(std::move(unique_item));
            assert(copy.length() == length);
            assert(copy == base);
        }
    });

    // And the move assignment operator with an empty target payload?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string unique_item(base);
            string copy;
            copy = std::move(unique_item);
            assert(copy.length() == length);
            assert(copy == base);
        }
    });

    // And move assignment where the target had a payload?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string unique_item(base);
            string copy;
            for (std::size_t j = 0; j < 317; j++) copy.push_back('q');
            copy = std::move(unique_item);
            assert(copy.length() == length);
            assert(copy == base);
        }
    });

    // Now let's clear the base and check that we're back to zero
    base = string();
    assert(accounting_allocator::counter_ref() == 0);
}

/**
 *  @brief Tests the correctness of the string class update methods, such as `push_back` and `erase`.
 *  @param repetitions Number of randomized append-then-erase cycles to run.
 */
void test_string_updates_unit(std::size_t repetitions) {
    // Compare STL and StringZilla strings append functionality.
    char const alphabet_chars[] = "abcdefghijklmnopqrstuvwxyz";
    for (std::size_t repetition = 0; repetition != repetitions; ++repetition) {
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
}

#pragma endregion // String Class

#pragma endregion // Unit

#pragma region Equivalence

/** @brief Wraps the memory-movement primitives (copy/move/fill) of one backend by their pointers. */
template <sz_copy_t copy_, sz_move_t move_, sz_fill_t fill_>
struct memory_from_sz_ {
    void copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length) const noexcept { copy_(target, source, length); }
    void move(sz_ptr_t target, sz_cptr_t source, sz_size_t length) const noexcept { move_(target, source, length); }
    void fill(sz_ptr_t target, sz_size_t length, sz_u8_t value) const noexcept { fill_(target, length, value); }
};

/** @brief Wraps a byte-lookup (transform) backend by its kernel pointer. */
template <sz_lookup_t lookup_>
struct lookup_from_sz_ {
    void lookup(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lookup_table) const noexcept {
        lookup_(target, length, source, lookup_table);
    }
};

/**
 *  @brief A representative spread of lengths covering 0, tiny, the SWAR/SIMD-width neighborhood, and larger,
 *         so a kernel's head/body/tail handling is exercised on every backend.
 */
inline std::vector<sz_size_t> memory_equivalence_lengths() noexcept {
    return {0,  1,  2,  3,  7,  8,  9,   15,  16,  17,  31,  32,  33,   47,
            48, 63, 64, 65, 95, 96, 127, 128, 129, 255, 256, 257, 1024, 4096};
}

/**
 *  @brief Copies/moves/fills a buffer and compares the output between a reference and a candidate movement backend.
 *
 *  Runs over `for_each_cacheline_offset_` so the destination (and source) buffers are exercised at every
 *  sub-cache-line alignment, across the representative length set, with embedded-NUL content and overlapping
 *  `move` regions, so a misaligned head/tail bug on any backend is caught against the reference.
 *
 *  @param reference  Reference movement backend wrapper (copy/move/fill).
 *  @param candidate  Candidate movement backend wrapper to validate against the reference.
 *  @param inputs     Number of random source patterns to fuzz at each length.
 */
template <typename reference_, typename candidate_>
void test_memory_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    std::vector<sz_size_t> const lengths = memory_equivalence_lengths();
    sz_size_t const max_length = lengths.back();

    for (sz_size_t length : lengths) {
        for (sz_size_t input = 0; input != inputs; ++input) {

            // A randomized source with embedded NULs - the byte primitives must stay length-driven.
            // The source itself is read from a cache-line-shifted span so the load alignment varies too.
            std::vector<char> source_storage(length + SZ_CACHE_LINE_WIDTH, '\0');
            sz_cptr_t const source = source_storage.data() + (input % SZ_CACHE_LINE_WIDTH);
            if (length) randomize_string(const_cast<char *>(source), length);

            // `copy` and `fill`: place the destination at every sub-cache-line alignment, comparing the
            // candidate output against a serial reference run at the same alignment.
            sz_u8_t const fill_value = (sz_u8_t)(0xA5u ^ (sz_u8_t)length);
            for_each_cacheline_offset_(max_length, [&](sz_ptr_t target, std::size_t) {
                std::vector<char> reference_output(length, '\0');
                reference.copy(reference_output.data(), source, length);
                candidate.copy(target, source, length);
                assert(std::memcmp(reference_output.data(), target, length) == 0);

                reference.fill(reference_output.data(), length, fill_value);
                candidate.fill(target, length, fill_value);
                assert(std::memcmp(reference_output.data(), target, length) == 0);
            });

            // `move` with overlapping regions: shift the source pattern within one buffer by a small offset,
            // both forwards and backwards, at every alignment of the buffer.
            for (sz_size_t shift : {(sz_size_t)1, (sz_size_t)7, (sz_size_t)16}) {
                if (length <= shift) continue;
                sz_size_t const moved = length - shift;
                for_each_cacheline_offset_(max_length + shift, [&](sz_ptr_t buffer, std::size_t) {
                    std::vector<char> reference_buffer(length + shift, '\0');

                    // Forward overlap: destination ahead of the source.
                    std::memcpy(buffer, source, length);
                    std::memcpy(reference_buffer.data(), source, length);
                    candidate.move(buffer + shift, buffer, moved);
                    reference.move(reference_buffer.data() + shift, reference_buffer.data(), moved);
                    assert(std::memcmp(buffer, reference_buffer.data(), length) == 0);

                    // Backward overlap: destination behind the source.
                    std::memcpy(buffer, source, length);
                    std::memcpy(reference_buffer.data(), source, length);
                    candidate.move(buffer, buffer + shift, moved);
                    reference.move(reference_buffer.data(), reference_buffer.data() + shift, moved);
                    assert(std::memcmp(buffer, reference_buffer.data(), length) == 0);
                });
            }
        }
    }
}

/**
 *  @brief Applies a byte-lookup table and compares the output between a reference and a candidate backend.
 *
 *  Runs over `for_each_cacheline_offset_` so the destination and source buffers are exercised at every
 *  sub-cache-line alignment, across the representative length set, against a shared case-mapping table.
 *
 *  @param reference  Reference lookup backend wrapper.
 *  @param candidate  Candidate lookup backend wrapper to validate against the reference.
 *  @param inputs     Number of random source patterns to fuzz at each length.
 */
template <typename reference_, typename candidate_>
void test_lookup_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    char lookup_table[256];
    sz_lookup_init_upper(lookup_table);

    std::vector<sz_size_t> const lengths = memory_equivalence_lengths();
    sz_size_t const max_length = lengths.back();

    for (sz_size_t length : lengths) {
        for (sz_size_t input = 0; input != inputs; ++input) {

            std::vector<char> source_storage(length + SZ_CACHE_LINE_WIDTH, '\0');
            sz_cptr_t const source = source_storage.data() + (input % SZ_CACHE_LINE_WIDTH);
            if (length) randomize_string(const_cast<char *>(source), length);

            for_each_cacheline_offset_(max_length, [&](sz_ptr_t target, std::size_t) {
                std::vector<char> reference_output(length, '\0');
                reference.lookup(reference_output.data(), length, source, lookup_table);
                candidate.lookup(target, length, source, lookup_table);
                assert(std::memcmp(reference_output.data(), target, length) == 0);
            });
        }
    }
}

#pragma endregion // Equivalence

#pragma region Safety

/**
 *  @brief Runs one movement backend through adversarial inputs guarded by canary bytes, asserting no
 *         out-of-bounds write occurs (the canaries stay intact) and the operation does not crash.
 */
static void check_memory_safety_(sz_copy_t copy, sz_move_t move, sz_fill_t fill) {

    // Zero-length: copy/move/fill must touch nothing, including NULL targets.
    copy(nullptr, nullptr, 0);
    move(nullptr, nullptr, 0);
    fill(nullptr, 0, (sz_u8_t)'!');

    // A canary-guarded destination: writes outside [0, length) corrupt a guard byte.
    for (std::size_t length : {(std::size_t)1, (std::size_t)8, (std::size_t)64, (std::size_t)257})
        with_guarded_buffer_(length, [&](sz_ptr_t destination, std::size_t usable_length) {
            std::vector<char> source(usable_length, (char)0xC3);
            copy(destination, source.data(), usable_length);
            fill(destination, usable_length, (sz_u8_t)0x7E);
            move(destination, source.data(), usable_length); // Non-overlapping move
        });

    // Overlapping move inside one canary-guarded buffer, plus embedded-NUL content. The usable window
    // spans `length + shift` so both the shifted-forward and shifted-back overlaps stay inside the guards.
    {
        std::size_t const length = 257;
        std::size_t const shift = 16;
        with_guarded_buffer_(length + shift, [&](sz_ptr_t buffer, std::size_t) {
            for (std::size_t byte = 0; byte != length; ++byte) buffer[byte] = (char)((byte % 2) ? (byte & 0xFF) : 0);
            move(buffer + shift, buffer, length); // Forward overlap
            move(buffer, buffer + shift, length); // Backward overlap
        });
    }
}

/**
 *  @brief Runs one lookup backend through adversarial inputs guarded by canary bytes, asserting no
 *         out-of-bounds write occurs (the canaries stay intact) and the operation does not crash.
 */
static void check_lookup_safety_(sz_lookup_t lookup) {

    char lookup_table[256];
    sz_lookup_init_upper(lookup_table);

    lookup(nullptr, 0, nullptr, lookup_table); // Zero-length must touch nothing

    for (std::size_t length : {(std::size_t)1, (std::size_t)8, (std::size_t)64, (std::size_t)257})
        with_guarded_buffer_(length, [&](sz_ptr_t destination, std::size_t usable_length) {
            std::vector<char> source(usable_length, '\0');
            for (std::size_t byte = 0; byte != usable_length; ++byte)
                source[byte] = (char)((byte % 3) ? 'a' + (byte % 26) : 0);
            lookup(destination, usable_length, source.data(), lookup_table);
        });
}

/**
 *  @brief Adversarial safety driver: feeds zero-length, tiny, overlapping, and embedded-NUL inputs through
 *         the dispatched, serial, and every natively-compiled movement/lookup kernel, asserting that canary
 *         bytes guarding both sides of the destination remain intact and that nothing crashes.
 */
void test_memory_safety() {

    check_memory_safety_(sz_copy, sz_move, sz_fill);                      // Dispatched (automatic kernel)
    check_memory_safety_(sz_copy_serial, sz_move_serial, sz_fill_serial); // Manual: serial kernel
#if SZ_USE_HASWELL
    check_memory_safety_(sz_copy_haswell, sz_move_haswell, sz_fill_haswell);
#endif
#if SZ_USE_SKYLAKE
    check_memory_safety_(sz_copy_skylake, sz_move_skylake, sz_fill_skylake);
#endif
#if SZ_USE_NEON
    check_memory_safety_(sz_copy_neon, sz_move_neon, sz_fill_neon);
#endif
#if SZ_USE_SVE
    check_memory_safety_(sz_copy_sve, sz_move_sve, sz_fill_sve);
#endif
#if SZ_USE_V128
    check_memory_safety_(sz_copy_v128, sz_move_v128, sz_fill_v128);
#endif
#if SZ_USE_V128RELAXED
    check_memory_safety_(sz_copy_v128relaxed, sz_move_v128relaxed, sz_fill_v128relaxed);
#endif
#if SZ_USE_RVV
    check_memory_safety_(sz_copy_rvv, sz_move_rvv, sz_fill_rvv);
#endif
#if SZ_USE_LASX
    check_memory_safety_(sz_copy_lasx, sz_move_lasx, sz_fill_lasx);
#endif
#if SZ_USE_POWERVSX
    check_memory_safety_(sz_copy_powervsx, sz_move_powervsx, sz_fill_powervsx);
#endif

    check_lookup_safety_(sz_lookup);        // Dispatched (automatic kernel)
    check_lookup_safety_(sz_lookup_serial); // Manual: serial kernel
#if SZ_USE_HASWELL
    check_lookup_safety_(sz_lookup_haswell);
#endif
#if SZ_USE_ICELAKE
    check_lookup_safety_(sz_lookup_icelake);
#endif
#if SZ_USE_NEON
    check_lookup_safety_(sz_lookup_neon);
#endif
#if SZ_USE_SVE
    check_lookup_safety_(sz_lookup_sve);
#endif
#if SZ_USE_V128
    check_lookup_safety_(sz_lookup_v128);
#endif
#if SZ_USE_V128RELAXED
    check_lookup_safety_(sz_lookup_v128relaxed);
#endif
#if SZ_USE_RVV
    check_lookup_safety_(sz_lookup_rvv);
#endif
#if SZ_USE_LASX
    check_lookup_safety_(sz_lookup_lasx);
#endif
#if SZ_USE_POWERVSX
    check_lookup_safety_(sz_lookup_powervsx);
#endif
}

#pragma endregion // Safety

#pragma region Drivers

/**
 *  @brief Drives the serial-vs-SIMD movement and lookup differential tests across every memory backend
 *         compiled on this target. Copy/move/fill share one tier set; lookup has its own (icelake, no skylake).
 */
void test_memory_all() {

    using memory_serial_t = memory_from_sz_<sz_copy_serial, sz_move_serial, sz_fill_serial>;
    memory_serial_t const memory_serial;

    sz_size_t const inputs = (sz_size_t)scale_iterations(2);

#if SZ_USE_HASWELL
    test_memory_equivalence(memory_serial, memory_from_sz_<sz_copy_haswell, sz_move_haswell, sz_fill_haswell> {},
                            inputs);
#endif
#if SZ_USE_SKYLAKE
    test_memory_equivalence(memory_serial, memory_from_sz_<sz_copy_skylake, sz_move_skylake, sz_fill_skylake> {},
                            inputs);
#endif
#if SZ_USE_NEON
    test_memory_equivalence(memory_serial, memory_from_sz_<sz_copy_neon, sz_move_neon, sz_fill_neon> {}, inputs);
#endif
#if SZ_USE_SVE
    test_memory_equivalence(memory_serial, memory_from_sz_<sz_copy_sve, sz_move_sve, sz_fill_sve> {}, inputs);
#endif
#if SZ_USE_V128
    test_memory_equivalence(memory_serial, memory_from_sz_<sz_copy_v128, sz_move_v128, sz_fill_v128> {}, inputs);
#endif
#if SZ_USE_V128RELAXED
    test_memory_equivalence(memory_serial,
                            memory_from_sz_<sz_copy_v128relaxed, sz_move_v128relaxed, sz_fill_v128relaxed> {}, inputs);
#endif
#if SZ_USE_RVV
    test_memory_equivalence(memory_serial, memory_from_sz_<sz_copy_rvv, sz_move_rvv, sz_fill_rvv> {}, inputs);
#endif
#if SZ_USE_LASX
    test_memory_equivalence(memory_serial, memory_from_sz_<sz_copy_lasx, sz_move_lasx, sz_fill_lasx> {}, inputs);
#endif
#if SZ_USE_POWERVSX
    test_memory_equivalence(memory_serial, memory_from_sz_<sz_copy_powervsx, sz_move_powervsx, sz_fill_powervsx> {},
                            inputs);
#endif

    using lookup_serial_t = lookup_from_sz_<sz_lookup_serial>;
    lookup_serial_t const lookup_serial;

#if SZ_USE_HASWELL
    test_lookup_equivalence(lookup_serial, lookup_from_sz_<sz_lookup_haswell> {}, inputs);
#endif
#if SZ_USE_ICELAKE
    test_lookup_equivalence(lookup_serial, lookup_from_sz_<sz_lookup_icelake> {}, inputs);
#endif
#if SZ_USE_NEON
    test_lookup_equivalence(lookup_serial, lookup_from_sz_<sz_lookup_neon> {}, inputs);
#endif
#if SZ_USE_SVE
    test_lookup_equivalence(lookup_serial, lookup_from_sz_<sz_lookup_sve> {}, inputs);
#endif
#if SZ_USE_V128
    test_lookup_equivalence(lookup_serial, lookup_from_sz_<sz_lookup_v128> {}, inputs);
#endif
#if SZ_USE_V128RELAXED
    test_lookup_equivalence(lookup_serial, lookup_from_sz_<sz_lookup_v128relaxed> {}, inputs);
#endif
#if SZ_USE_RVV
    test_lookup_equivalence(lookup_serial, lookup_from_sz_<sz_lookup_rvv> {}, inputs);
#endif
#if SZ_USE_LASX
    test_lookup_equivalence(lookup_serial, lookup_from_sz_<sz_lookup_lasx> {}, inputs);
#endif
#if SZ_USE_POWERVSX
    test_lookup_equivalence(lookup_serial, lookup_from_sz_<sz_lookup_powervsx> {}, inputs);
#endif
}

#pragma endregion // Drivers

// Explicit template instantiations for the entry points invoked from `main()` (see `test_stringzilla.cpp`).
template void test_ascii_unit<sz::string>();
template void test_ascii_unit<sz::string_view>();
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
template void test_stl_reads_unit<std::string_view>();
#endif
template void test_stl_reads_unit<std::string>();
template void test_stl_reads_unit<sz::string_view>();
template void test_stl_reads_unit<sz::string>();
template void test_stl_updates_unit<std::string>();
template void test_stl_updates_unit<sz::string>();
template void test_extensions_reads_unit<sz::string_view>();
template void test_extensions_reads_unit<sz::string>();
