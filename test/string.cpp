/**
 *  @brief  Arithmetic/struct plumbing, ASCII utilities, memory, STL-compat, conversions, extensions, and the string class.
 *  @file   test/string.cpp
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

#include <algorithm>     // `std::transform`
#include <forward_list>  // `std::forward_list`
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

#include "stringzilla.hpp" // `global_random_generator`, `random_string`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using sz::literals::operator""_sv; // for `sz::string_view`
using sz::literals::operator""_bs; // for `sz::byteset`

#if SZ_IS_CPP17_
using namespace std::literals; // for ""sv
#endif

#pragma region Helpers

/** @brief Compares two byte ranges and aborts with a localized diagnostic on the first mismatch. */
inline void expect_equality(char const *first, char const *second, std::size_t size) {
    if (std::memcmp(first, second, size) == 0) return;
    std::size_t mismatch_position = 0;
    for (; mismatch_position < size; ++mismatch_position)
        if (first[mismatch_position] != second[mismatch_position]) break;
    std::fprintf(stderr, "Mismatch at position %zu: %c != %c\n", mismatch_position, first[mismatch_position],
                 second[mismatch_position]);
    verify(false);
}

/**
 *  @brief The sum of an arithmetic progression.
 *  @see https://en.wikipedia.org/wiki/Arithmetic_progression
 */
inline std::size_t arithmetic_sum(std::size_t first, std::size_t last, std::size_t step = 1) {
    std::size_t n = (last >= first) ? ((last - first) / step + 1) : 0;
    if (n == 0) return 0;
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
        verify(n <= counter_ref() && "Deallocated more bytes than were tracked as allocated");
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
    verify(bytes == 0 && "Callback leaked or double-freed tracked allocator bytes");
}

/**
 *  @brief Runs one movement backend (copy/move/fill) through hand-verifiable known-answer vectors.
 *
 *  Mirrors the SHA256 known-answer helper in `hash.cpp`: each ISA tier feeds its kernel pointers here,
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
        verify(std::memcmp(target, source, length) == 0 && "Copy backend diverged from the known-answer source");
        verify(target[length] == '#' && "Copy backend wrote past the requested length");
    }

    // `move` handles overlapping regions. Shifting "abcdef" left-into-itself by two yields "cdef" at the front.
    {
        char const expected[] = "cdef"; // After moving "cdef" (offset 2, 4 bytes) to offset 0
        char buffer[] = "abcdef";
        move(buffer, buffer + 2, 4);
        verify(std::memcmp(buffer, expected, 4) == 0 && "Move backend produced wrong bytes for overlapping shift");
    }

    // `fill` writes a known byte across a known span, leaving a guard byte untouched.
    {
        char const expected[] = "*****"; // Five asterisks
        char target[5 + 1];
        std::memset(target, '#', sizeof(target));
        fill(target, 5, (sz_u8_t)'*');
        verify(std::memcmp(target, expected, 5) == 0 && "Fill backend produced wrong bytes for the known pattern");
        verify(target[5] == '#' && "Fill backend wrote past the requested length");
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
    verify(std::memcmp(target, expected, length) == 0 && "Lookup backend diverged from the known-answer upper-casing");
    verify(target[length] == '#' && "Lookup backend wrote past the requested length");
}

#pragma endregion // Helpers

#pragma region Arithmetic

/**
 *  @brief Several string processing operations rely on computing integer logarithms.
 *         Failures in such operations will result in wrong `resize` outcomes and heap corruption.
 */
void test_arithmetic_unit() {

    verify(sz_u64_clz(0x0000000000000001ull) == 63);
    verify(sz_u64_clz(0x0000000000000002ull) == 62);
    verify(sz_u64_clz(0x0000000000000003ull) == 62);
    verify(sz_u64_clz(0x0000000000000004ull) == 61);
    verify(sz_u64_clz(0x0000000000000007ull) == 61);
    verify(sz_u64_clz(0x8000000000000001ull) == 0);
    verify(sz_u64_clz(0xffffffffffffffffull) == 0);
    verify(sz_u64_clz(0x4000000000000000ull) == 1);

    verify(sz_size_log2i_nonzero(1) == 0);
    verify(sz_size_log2i_nonzero(2) == 1);
    verify(sz_size_log2i_nonzero(3) == 1);

    verify(sz_size_log2i_nonzero(4) == 2);
    verify(sz_size_log2i_nonzero(5) == 2);
    verify(sz_size_log2i_nonzero(7) == 2);

    verify(sz_size_log2i_nonzero(8) == 3);
    verify(sz_size_log2i_nonzero(9) == 3);

    verify(sz_size_bit_ceil(0) == 0);
    verify(sz_size_bit_ceil(1) == 1);

    verify(sz_size_bit_ceil(2) == 2);
    verify(sz_size_bit_ceil(3) == 4);
    verify(sz_size_bit_ceil(4) == 4);

    verify(sz_size_bit_ceil(77) == 128);
    verify(sz_size_bit_ceil(127) == 128);
    verify(sz_size_bit_ceil(128) == 128);

    verify(sz_size_bit_ceil(1000000ull) == (1ull << 20));
    verify(sz_size_bit_ceil(2000000ull) == (1ull << 21));
    verify(sz_size_bit_ceil(4000000ull) == (1ull << 22));
    verify(sz_size_bit_ceil(8000000ull) == (1ull << 23));

    verify(sz_size_bit_ceil(16000000ull) == (1ull << 24));
    verify(sz_size_bit_ceil(32000000ull) == (1ull << 25));
    verify(sz_size_bit_ceil(64000000ull) == (1ull << 26));

    verify(sz_size_bit_ceil(128000000ull) == (1ull << 27));
    verify(sz_size_bit_ceil(256000000ull) == (1ull << 28));
    verify(sz_size_bit_ceil(512000000ull) == (1ull << 29));

    verify(sz_size_bit_ceil(1000000000ull) == (1ull << 30));
    verify(sz_size_bit_ceil(2000000000ull) == (1ull << 31));

#if SZ_IS_64BIT_
    verify(sz_size_bit_ceil(4000000000ull) == (1ull << 32));
    verify(sz_size_bit_ceil(8000000000ull) == (1ull << 33));
    verify(sz_size_bit_ceil(16000000000ull) == (1ull << 34));

    verify(sz_size_bit_ceil((1ull << 62)) == (1ull << 62));
    verify(sz_size_bit_ceil((1ull << 62) + 1) == (1ull << 63));
    verify(sz_size_bit_ceil((1ull << 63)) == (1ull << 63));
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
        verify(sequence.count == 3);
        verify("banana"_sv == sequence.get_start(sequence.handle, 0));
        verify("apple"_sv == sequence.get_start(sequence.handle, 1));
        verify("cherry"_sv == sequence.get_start(sequence.handle, 2));
        verify(sequence.get_length(sequence.handle, 0) == 6);
        verify(sequence.get_length(sequence.handle, 1) == 5);
        verify(sequence.get_length(sequence.handle, 2) == 6);
    }
    // Empty sequences, empty members, and duplicates are all legal.
    {
        sz_sequence_t sequence;
        sz_cptr_t strings[] = {"", "apple", "apple", ""};
        sz_sequence_from_null_terminated_strings(strings, 0, &sequence);
        verify(sequence.count == 0);
        sz_sequence_from_null_terminated_strings(strings, 4, &sequence);
        verify(sequence.count == 4);
        verify(sequence.get_length(sequence.handle, 0) == 0);
        verify(sequence.get_length(sequence.handle, 3) == 0);
        verify("apple"_sv == sequence.get_start(sequence.handle, 1));
        verify("apple"_sv == sequence.get_start(sequence.handle, 2));
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

        verify(sequence.count == 3);
        verify("banana"_sv == sequence.get_start(sequence.handle, 0));
        verify("apple"_sv == sequence.get_start(sequence.handle, 1));
        verify("cherry"_sv == sequence.get_start(sequence.handle, 2));
    }
}

/**
 *  @brief Validates that `arrow_strings_tape::try_assign` works with multi-pass forward iterators.
 *         It walks the range twice, once to measure and once to copy, so single-pass input
 *         iterators like `std::istream_iterator` are rejected at compile time.
 */
void test_strings_tape_assign_unit() {
    sz::arrow_strings_tape<char, std::uint32_t, std::allocator<char>> tape;

    // A forward list can only be walked forward, but any number of times - exactly what `try_assign` needs.
    std::forward_list<std::string> strings {"alpha", "", "gamma"};
    verify(tape.try_assign(strings.begin(), strings.end()) == sz::status_t::success_k);
    verify(tape.size() == 3);
    verify(sz::string_view(tape[0].data(), tape[0].size()) == "alpha"_sv);
    verify(tape[1].size() == 0);
    verify(sz::string_view(tape[2].data(), tape[2].size()) == "gamma"_sv);
}

/** @brief Validates that `arrow_strings_tape` refuses to grow past the range of its offset type. */
void test_strings_tape_overflow_unit() {
    // 8-bit offsets hit the same code path as 32-bit offsets past 4 GiB, but already at 256 bytes.
    using tape_t = sz::arrow_strings_tape<char, std::uint8_t, std::allocator<char>>;

    // Appending past the offset range must fail cleanly and leave the stored strings untouched.
    {
        tape_t tape;
        std::string const oversized_string(200, 'x');
        verify(tape.try_append(sz::to_view(oversized_string)) == sz::status_t::success_k);
        // Two 200-byte strings need 402 bytes of buffer, past the 255 maximum of 8-bit offsets.
        verify(tape.try_append(sz::to_view(oversized_string)) == sz::status_t::overflow_risk_k);
        verify(tape.size() == 1);
        // The first string must still sit at offset 0, ending at 201 with its NULL terminator.
        verify(tape.offsets()[0] == 0);
        verify(tape.offsets()[1] == 201);
        verify(std::memcmp(tape.buffer().data(), oversized_string.data(), oversized_string.size()) == 0);
    }

    // Same for bulk assignment: the combined size must fit the offset range.
    {
        tape_t tape;
        std::string const stored_string(10, 'z');
        verify(tape.try_append(sz::to_view(stored_string)) == sz::status_t::success_k);
        std::vector<std::string> strings {std::string(200, 'x'), std::string(200, 'y')};
        verify(tape.try_assign(strings.begin(), strings.end()) == sz::status_t::overflow_risk_k);
        // A rejected assignment releases the old contents, so the tape must not keep reporting them.
        verify(tape.size() == 0);
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
        verify(alloc.allocate(0, alloc.handle) == nullptr);
    }

    // Non-NULL allocation
    {
        sz_memory_allocator_t alloc;
        sz_memory_allocator_init_default(&alloc);
        void *byte = alloc.allocate(1, alloc.handle);
        verify(byte != nullptr && "Default allocator returned NULL for a non-zero-length allocation");
        alloc.free(byte, 1, alloc.handle);
    }

    // Use a fixed buffer
    {
        char buffer[1024];
        sz_memory_allocator_t alloc;
        sz_memory_allocator_init_fixed(&alloc, buffer, sizeof(buffer));
        void *byte = alloc.allocate(1, alloc.handle);
        verify(byte != nullptr && "Fixed-buffer allocator returned NULL for an allocation that should fit");
        alloc.free(byte, 1, alloc.handle);
    }
}

#pragma endregion // Allocator

#pragma region Byteset

/** @brief Validates `sz_byteset_t` and related construction utilities. */
void test_byteset_unit() {
    sz_byteset_t s;
    sz_byteset_init(&s);
    verify(sz_byteset_contains(&s, 'a') == sz_false_k);
    sz_byteset_add(&s, 'a');
    verify(sz_byteset_contains(&s, 'a') == sz_true_k);
    sz_byteset_add(&s, 'z');
    verify(sz_byteset_contains(&s, 'z') == sz_true_k);
    sz_byteset_invert(&s);
    verify(sz_byteset_contains(&s, 'a') == sz_false_k);
    verify(sz_byteset_contains(&s, 'z') == sz_false_k);
    verify(sz_byteset_contains(&s, 'b') == sz_true_k);
    sz_byteset_init_ascii(&s);
    verify(sz_byteset_contains(&s, 'A') == sz_true_k);
}

/**
 *  @brief Tests various ASCII-based methods (e.g., `is_alpha`, `is_digit`)
 *         provided by `sz::string` and `sz::string_view`.
 */
/** @brief Known-answer coverage for ASCII classification methods (`is_alpha`, `is_digit`, `contains_only`, ...). */
template <typename string_type>
void test_ascii_unit() {

    using str = string_type;

    verify("aaa"_bs.size() == 1ull);
    verify("\0\0"_bs.size() == 1ull);
    verify("abc"_bs.size() == 3ull);
    verify("a\0bc"_bs.size() == 4ull);

    verify(!"abc"_bs.contains('\0'));
    verify(str("bca").contains_only("abc"_bs));

    verify(!str("").is_alpha());
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ").is_alpha());
    verify(!str("abc9").is_alpha());

    verify(!str("").is_alnum());
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789").is_alnum());
    verify(!str("abc!").is_alnum());

    verify(str("").is_ascii());
    verify(str("\x00x7F").is_ascii());
    verify(!str("abc123🔥").is_ascii());

    verify(!str("").is_digit());
    verify(str("0123456789").is_digit());
    verify(!str("012a").is_digit());

    verify(!str("").is_lower());
    verify(str("abcdefghijklmnopqrstuvwxyz").is_lower());
    verify(!str("abcA").is_lower());
    verify(!str("abc\n").is_lower());

    verify(!str("").is_space());
    verify(str(" \t\n\r\f\v").is_space());
    verify(!str(" \t\r\na").is_space());

    verify(!str("").is_upper());
    verify(str("ABCDEFGHIJKLMNOPQRSTUVWXYZ").is_upper());
    verify(!str("ABCa").is_upper());

    verify(str("").is_printable());
    verify(str("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+").is_printable());
    verify(!str("012🔥").is_printable());

    verify(str("").contains_only("abc"_bs));
    verify(str("abc").contains_only("abc"_bs));
    verify(!str("abcd").contains_only("abc"_bs));
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
 */
void test_memory_unit(std::size_t max_l2_size) {

    std::printf("  - testing memory primitive known-answer vectors...\n");

    // Movement known-answers, through the dispatched C API and every natively-compiled backend.
    check_memory_unit_(sz_copy, sz_move, sz_fill);
    check_memory_unit_(sz_copy_serial, sz_move_serial, sz_fill_serial);
#if SZ_USE_HASWELL
    check_memory_unit_(sz_copy_haswell, sz_move_haswell, sz_fill_haswell);
#endif
#if SZ_USE_SKYLAKE
    check_memory_unit_(sz_copy_skylake, sz_move_skylake, sz_fill_skylake);
#endif
#if SZ_USE_NEON
    check_memory_unit_(sz_copy_neon, sz_move_neon, sz_fill_neon);
#endif
#if SZ_USE_SVE
    check_memory_unit_(sz_copy_sve, sz_move_sve, sz_fill_sve);
#endif
#if SZ_USE_V128
    check_memory_unit_(sz_copy_v128, sz_move_v128, sz_fill_v128);
#endif
#if SZ_USE_V128RELAXED
    check_memory_unit_(sz_copy_v128relaxed, sz_move_v128relaxed, sz_fill_v128relaxed);
#endif
#if SZ_USE_RVV
    check_memory_unit_(sz_copy_rvv, sz_move_rvv, sz_fill_rvv);
#endif
#if SZ_USE_LASX
    check_memory_unit_(sz_copy_lasx, sz_move_lasx, sz_fill_lasx);
#endif
#if SZ_USE_POWERVSX
    check_memory_unit_(sz_copy_powervsx, sz_move_powervsx, sz_fill_powervsx);
#endif

    // Lookup known-answers, through the dispatched C API and every natively-compiled backend.
    check_lookup_unit_(sz_lookup);
    check_lookup_unit_(sz_lookup_serial);
#if SZ_USE_HASWELL
    check_lookup_unit_(sz_lookup_haswell);
#endif
#if SZ_USE_ICELAKE
    check_lookup_unit_(sz_lookup_icelake);
#endif
#if SZ_USE_NEON
    check_lookup_unit_(sz_lookup_neon);
#endif
#if SZ_USE_SVE
    check_lookup_unit_(sz_lookup_sve);
#endif
#if SZ_USE_V128
    check_lookup_unit_(sz_lookup_v128);
#endif
#if SZ_USE_V128RELAXED
    check_lookup_unit_(sz_lookup_v128relaxed);
#endif
#if SZ_USE_RVV
    check_lookup_unit_(sz_lookup_rvv);
#endif
#if SZ_USE_LASX
    check_lookup_unit_(sz_lookup_lasx);
#endif
#if SZ_USE_POWERVSX
    check_lookup_unit_(sz_lookup_powervsx);
#endif

    // C++ wrapper sanity: a couple of `sz::string` / `sz::string_view` known-answer reads alongside the C API.
    {
        sz::string_view const view = "Hello, World!"_sv;
        verify(view.size() == 13u);
        verify(view.substr(7, 5) == "World"_sv);
        verify(view.front() == 'H' && view.back() == '!');

        sz::string const owned = "Hello, World!";
        verify(owned.size() == 13u);
        verify(owned == view);
        verify(sz::string("apple").compare("banana") < 0);
    }

    // The C++ movement wrappers must agree with the known-answers, including overlapping `memmove`.
    {
        char const fox[] = "The quick brown fox";
        char target[sizeof(fox) + 1];
        let_verify(std::memset(target, '#', sizeof(target)), //
                   (sz::memcpy(target, fox, sizeof(fox) - 1), std::memcmp(target, fox, sizeof(fox) - 1) == 0) &&
                       target[sizeof(fox) - 1] == '#');

        char overlap[] = "abcdef";
        let_verify(sz::memmove(overlap, overlap + 2, 4), std::memcmp(overlap, "cdef", 4) == 0);

        char asterisks[5 + 1];
        let_verify(std::memset(asterisks, '#', sizeof(asterisks)), //
                   (sz::memset(asterisks, '*', 5), std::memcmp(asterisks, "*****", 5) == 0) && asterisks[5] == '#');
    }

    // Embedded NUL must be preserved verbatim by a stored `sz::string`: the size is the full byte length, and
    // indexing past the interior NUL reaches the trailing bytes rather than stopping at the C-string boundary.
    {
        char const with_nul[] = {'a', 'b', '\0', 'c', 'd'};
        sz::string const owned(with_nul, sizeof(with_nul));
        verify(owned.size() == sizeof(with_nul));   // Full length, NUL is a stored byte
        verify(owned[2] == '\0');                   // The interior NUL survives
        verify(owned[3] == 'c' && owned[4] == 'd'); // Indexing past the NUL works
        verify(owned == sz::string_view(with_nul, sizeof(with_nul)));
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
    verify(str().empty());
    verify(str().size() == 0);
    verify(str("").empty());
    verify(str("").size() == 0);
    verify(str("hello").size() == 5);
    verify(str("hello", 4) == "hell");

    // Element access.
    verify(str("rest")[0] == 'r');
    verify(str("rest").at(1) == 'e');
    verify(*str("rest").data() == 'r');
    verify(str("front").front() == 'f');
    verify(str("back").back() == 'k');

    // Iterators.
    verify(*str("begin").begin() == 'b' && *str("cbegin").cbegin() == 'c');
    verify(*str("rbegin").rbegin() == 'n' && *str("crbegin").crbegin() == 'n');
    verify(str("size").size() == 4 && str("length").length() == 6);

    // Slices... out-of-bounds exceptions are asymmetric!
    // Moreover, `std::string` has no `remove_prefix` and `remove_suffix` methods.
    // scope_verify(str s = "hello", s.remove_prefix(1), s == "ello");
    // scope_verify(str s = "hello", s.remove_suffix(1), s == "hell");
    verify(str("hello world").substr(0, 5) == "hello");
    verify(str("hello world").substr(6, 5) == "world");
    verify(str("hello world").substr(6) == "world");
    verify(str("hello world").substr(6, 100) == "world"); // 106 is beyond the length of the string, but its OK
    throws_verify(str("hello world").substr(100), std::out_of_range);   // 100 is beyond the length of the string
    throws_verify(str("hello world").substr(20, 5), std::out_of_range); // 20 is beyond the length of the string
#if defined(__GNUC__) && !defined(__NVCC__) // -1 casts to unsigned without warnings on GCC, but not NVCC
    throws_verify(str("hello world").substr(-1, 5), std::out_of_range);
    verify(str("hello world").substr(0, -1) == "hello world");
#endif

    // Character search in normal and reverse directions.
    verify(str("hello").find('e') == 1);
    verify(str("hello").find('e', 1) == 1);
    verify(str("hello").find('e', 2) == str::npos);
    verify(str("hello").rfind('l') == 3);
    verify(str("hello").rfind('l', 2) == 2);
    verify(str("hello").rfind('l', 1) == str::npos);

    // Substring search in normal and reverse directions.
    verify(str("hello").find("ell") == 1);
    verify(str("hello").find("ell", 1) == 1);
    verify(str("hello").find("ell", 2) == str::npos);
    verify(str("hello").find("el", 1) == 1);
    verify(str("hello").find("ell", 1, 2) == 1);
    verify(str("hello").rfind("l") == 3);
    verify(str("hello").rfind("l", 2) == 2);
    verify(str("hello").rfind("l", 1) == str::npos);

    // The second argument is the last possible value of the returned offset.
    verify(str("hello").rfind("el", 1) == 1);
    verify(str("hello").rfind("ell", 1) == 1);
    verify(str("hello").rfind("ello", 1) == 1);
    verify(str("hello").rfind("ell", 1, 2) == 1);

    // More complex queries.
    verify(str("abbabbaaaaaa").find("aa") == 6);
    verify(str("abbabbaaaaaa").find("ba") == 2);
    verify(str("abbabbaaaaaa").find("bb") == 1);
    verify(str("abbabbaaaaaa").find("bab") == 2);
    verify(str("abbabbaaaaaa").find("babb") == 2);
    verify(str("abbabbaaaaaa").find("babba") == 2);
    verify(str("abcdabcd").substr(2, 4).find("abc") == str::npos);
    verify(str("hello, world!").substr(0, 11).find("world") == str::npos);
    verify(str("axabbcxcaaabbccc").find("aaabbccc") == 8);
    verify(str("abcdabcdabc________").find("abcd") == 0);
    verify(str("________abcdabcdabc").find("abcd") == 8);

    // Cover every SWAR case for unique string sequences.
    auto lowercase_alphabet = str("abcdefghijklmnopqrstuvwxyz");
    for (std::size_t one_byte_offset = 0; one_byte_offset + 1 <= lowercase_alphabet.size(); ++one_byte_offset)
        verify(lowercase_alphabet.find(lowercase_alphabet.substr(one_byte_offset, 1)) == one_byte_offset &&
               "1-byte SWAR needle matched at the wrong offset");
    for (std::size_t two_byte_offset = 0; two_byte_offset + 2 <= lowercase_alphabet.size(); ++two_byte_offset)
        verify(lowercase_alphabet.find(lowercase_alphabet.substr(two_byte_offset, 2)) == two_byte_offset &&
               "2-byte SWAR needle matched at the wrong offset");
    for (std::size_t four_byte_offset = 0; four_byte_offset + 4 <= lowercase_alphabet.size(); ++four_byte_offset)
        verify(lowercase_alphabet.find(lowercase_alphabet.substr(four_byte_offset, 4)) == four_byte_offset &&
               "4-byte SWAR needle matched at the wrong offset");
    for (std::size_t three_byte_offset = 0; three_byte_offset + 3 <= lowercase_alphabet.size(); ++three_byte_offset)
        verify(lowercase_alphabet.find(lowercase_alphabet.substr(three_byte_offset, 3)) == three_byte_offset &&
               "3-byte SWAR needle matched at the wrong offset");
    for (std::size_t five_byte_offset = 0; five_byte_offset + 5 <= lowercase_alphabet.size(); ++five_byte_offset)
        verify(lowercase_alphabet.find(lowercase_alphabet.substr(five_byte_offset, 5)) == five_byte_offset &&
               "5-byte SWAR needle matched at the wrong offset");

    // Simple repeating patterns - with one "almost match" before an actual match in each direction.
    verify(str("_ab_abc_").find("abc") == 4);
    verify(str("_abc_ab_").rfind("abc") == 1);
    verify(str("_abc_abcd_").find("abcd") == 5);
    verify(str("_abcd_abc_").rfind("abcd") == 1);
    verify(str("_abcd_abcde_").find("abcde") == 6);
    verify(str("_abcde_abcd_").rfind("abcde") == 1);
    verify(str("_abcde_abcdef_").find("abcdef") == 7);
    verify(str("_abcdef_abcde_").rfind("abcdef") == 1);
    verify(str("_abcdef_abcdefg_").find("abcdefg") == 8);
    verify(str("_abcdefg_abcdef_").rfind("abcdefg") == 1);

    // ! `rfind` and `find_last_of` are not consistent in meaning of their arguments.
    verify(str("hello").find_first_of("le") == 1);
    verify(str("hello").find_first_of("le", 1) == 1);
    verify(str("hello").find_last_of("le") == 3);
    verify(str("hello").find_last_of("le", 2) == 2);
    verify(str("hello").find_first_not_of("hel") == 4);
    verify(str("hello").find_first_not_of("hel", 1) == 4);
    verify(str("hello").find_last_not_of("hel") == 4);
    verify(str("hello").find_last_not_of("hel", 4) == 4);

    // Try longer strings to enforce SIMD.
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find('x') == 23);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find('X') == 49);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind('x') == 23);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind('X') == 49);

    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xy") == 23);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XY") == 49);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("yz") == 24);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("YZ") == 50);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xy") == 23);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XY") == 49);

    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xyz") == 23);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XYZ") == 49);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xyz") == 23);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XYZ") == 49);

    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xyzA") == 23);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XYZ0") == 49);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xyzA") == 23);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XYZ0") == 49);

    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_first_of("xyz") == 23);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_first_of("XYZ") == 49);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_last_of("xyz") == 25);
    verify(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_last_of("XYZ") == 51);

    // Using single-byte non-ASCII values, e.g., À (0xC0), Æ (0xC6). The `\xFA`/`0` boundary is
    // load-bearing: a literal hex digit after `\xFA` would extend the escape, so keep it split.
    {
        char const *non_ascii_set = "abcdefgh\x01\xC6ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\xC0\xFA" //
                                    "0123456789+-";                                                        // 68 bytes
        verify(str(non_ascii_set, 68).find_first_of("\xC6\xC7") == 9);
        verify(str(non_ascii_set, 68).find_first_of("\xC0\xC1") == 54);
        verify(str(non_ascii_set, 68).find_last_of("\xC6\xC7") == 9);
        verify(str(non_ascii_set, 68).find_last_of("\xC0\xC1") == 54);
    }

    // Boundary conditions.
    verify(str("hello").find_first_of("ox", 4) == 4);
    verify(str("hello").find_first_of("ox", 5) == str::npos);
    verify(str("hello").find_last_of("ox", 4) == 4);
    verify(str("hello").find_last_of("ox", 5) == 4);
    verify(str("hello").find_first_of("hx", 0) == 0);
    verify(str("hello").find_last_of("hx", 0) == 0);

    // More complex relative patterns
    verify(str("0123456789012345678901234567890123456789012345678901234567890123") <=
           str("0123456789012345678901234567890123456789012345678901234567890123"));
    verify(str("0123456789012345678901234567890123456789012345678901234567890123") <=
           str("0223456789012345678901234567890123456789012345678901234567890123"));
    verify(str("0123456789012345678901234567890123456789012345678901234567890123") <=
           str("0213456789012345678901234567890123456789012345678901234567890123"));
    verify(str("12341234") <= str("12341234"));
    verify(str("12341234") > str("12241224"));
    verify(str("12341234") < str("13241324"));
    verify(str("0123456789012345678901234567890123456789012345678901234567890123") ==
           str("0123456789012345678901234567890123456789012345678901234567890123"));
    verify(str("0123456789012345678901234567890123456789012345678901234567890123") !=
           str("0223456789012345678901234567890123456789012345678901234567890123"));

    // Comparisons.
    verify(str("a") != str("b"));
    verify(str("a") < str("b"));
    verify(str("a") <= str("b"));
    verify(str("b") > str("a"));
    verify(str("b") >= str("a"));
    verify(str("a") < str("aa"));

#if SZ_IS_CPP20_ && defined(__cpp_lib_three_way_comparison)
    // Spaceship operator instead of conventional comparions.
    verify((str("a") <=> str("b")) == std::strong_ordering::less);
    verify((str("b") <=> str("a")) == std::strong_ordering::greater);
    verify((str("b") <=> str("b")) == std::strong_ordering::equal);
    verify((str("a") <=> str("aa")) == std::strong_ordering::less);
#endif

    // Compare with another `str`.
    verify(str("test").compare(str("test")) == 0);
    verify(str("apple").compare(str("banana")) < 0);
    verify(str("banana").compare(str("apple")) > 0);

    // Compare with a C-string.
    verify(str("test").compare("test") == 0);
    verify(str("alpha").compare("beta") < 0);
    verify(str("beta").compare("alpha") > 0);

    // Compare substring with another `str`.
    verify(str("hello world").compare(0, 5, str("hello")) == 0);
    verify(str("hello world").compare(6, 5, str("earth")) > 0);
    verify(str("hello world").compare(6, 5, str("worlds")) < 0);
    throws_verify(str("hello world").compare(20, 5, str("worlds")), std::out_of_range);

    // Compare substring with another `str`'s substring.
    verify(str("hello world").compare(0, 5, str("say hello"), 4, 5) == 0);
    verify(str("hello world").compare(6, 5, str("world peace"), 0, 5) == 0);
    verify(str("hello world").compare(6, 5, str("a better world"), 9, 5) == 0);

    // Out of bounds cases for both compared strings.
    throws_verify(str("hello world").compare(20, 5, str("a better world"), 9, 5), std::out_of_range);
    throws_verify(str("hello world").compare(6, 5, str("a better world"), 90, 5), std::out_of_range);

    // Compare substring with a C-string.
    verify(str("hello world").compare(0, 5, "hello") == 0);
    verify(str("hello world").compare(6, 5, "earth") > 0);
    verify(str("hello world").compare(6, 5, "worlds") < 0);

    // Compare substring with a C-string's prefix.
    verify(str("hello world").compare(0, 5, "hello Ash", 5) == 0);
    verify(str("hello world").compare(6, 5, "worlds", 5) == 0);
    verify(str("hello world").compare(6, 5, "worlds", 6) < 0);

#if SZ_IS_CPP20_ && defined(__cpp_lib_starts_ends_with)
    // Prefix and suffix checks against strings.
    verify(str("https://cppreference.com").starts_with(str("http")) == true);
    verify(str("https://cppreference.com").starts_with(str("ftp")) == false);
    verify(str("https://cppreference.com").ends_with(str("com")) == true);
    verify(str("https://cppreference.com").ends_with(str("org")) == false);

    // Prefix and suffix checks against characters.
    verify(str("C++20").starts_with('C') == true);
    verify(str("C++20").starts_with('J') == false);
    verify(str("C++20").ends_with('0') == true);
    verify(str("C++20").ends_with('3') == false);

    // Prefix and suffix checks against C-style strings.
    verify(str("string_view").starts_with("string") == true);
    verify(str("string_view").starts_with("String") == false);
    verify(str("string_view").ends_with("view") == true);
    verify(str("string_view").ends_with("View") == false);
#endif

#if SZ_IS_CPP23_ && defined(__cpp_lib_string_contains)
    // Checking basic substring presence.
    verify(str("hello").contains(str("ell")) == true);
    verify(str("hello").contains(str("oll")) == false);
    verify(str("hello").contains('l') == true);
    verify(str("hello").contains('x') == false);
    verify(str("hello").contains("lo") == true);
    verify(str("hello").contains("lx") == false);
#endif

    // Exporting the contents of the string using the `str::copy` method.
    scope_verify(char buf[5 + 1] = {0}, str("hello").copy(buf, 5), std::strcmp(buf, "hello") == 0);
    scope_verify(char buf[4 + 1] = {0}, str("hello").copy(buf, 4, 1), std::strcmp(buf, "ello") == 0);
    throws_verify(str("hello").copy((char *)"", 1, 100), std::out_of_range);

    // Swaps.
    for (str const first : {"", "hello", "hellohellohellohellohellohellohellohellohellohellohellohello"}) {
        for (str const second : {"", "world", "worldworldworldworldworldworldworldworldworldworldworldworld"}) {
            str first_copy = first;
            str second_copy = second;
            first_copy.swap(second_copy);
            verify(first_copy == second && second_copy == first &&
                   "swap(other) did not exchange contents for this first/second pair");
            first_copy.swap(first_copy);
            verify(first_copy == second && "Self-swap mutated the string for this first/second pair");
        }
    }

    // Make sure the standard hash and function-objects instantiate just fine.
    verify(std::hash<str> {}("hello") != 0);
    scope_verify(std::ostringstream os, os << str("hello"), os.str() == "hello");

#if SZ_IS_CPP14_
    // Comparison function objects are a C++14 feature.
    verify(std::equal_to<str> {}("hello", "world") == false);
    verify(std::less<str> {}("hello", "world") == true);
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
    verify(str().empty());
    verify(str().size() == 0);
    verify(str("").empty());
    verify(str("").size() == 0);
    verify(str("hello").size() == 5);
    verify(str("hello", 4) == "hell");
    verify(str(5, 'a') == "aaaaa");
    verify(str({'h', 'e', 'l', 'l', 'o'}) == "hello");
    verify(str(str("hello"), 2) == "llo");
    verify(str(str("hello"), 2, 2) == "ll");

    // Corner case constructors and search behaviors for long strings
    verify(str(258, '0').find(str(256, '1')) == str::npos);

    // Assignments.
    scope_verify(str s = "obsolete", s = "hello", s == "hello");
    scope_verify(str s = "obsolete", s.assign("hello"), s == "hello");
    scope_verify(str s = "obsolete", s.assign("hello", 4), s == "hell");
    scope_verify(str s = "obsolete", s.assign(5, 'a'), s == "aaaaa");
    scope_verify(str s = "obsolete", s.assign(32, 'a'), s == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    scope_verify(str s = "obsolete", s.assign({'h', 'e', 'l', 'l', 'o'}), s == "hello");
    scope_verify(str s = "obsolete", s.assign(str("hello")), s == "hello");
    scope_verify(str s = "obsolete", s.assign(str("hello"), 2), s == "llo");
    scope_verify(str s = "obsolete", s.assign(str("hello"), 2, 2), s == "ll");
    scope_verify(str s = "obsolete", s.assign(str("hello"), 2, 2), s == "ll");
    scope_verify(str s = "obsolete", s.assign(s), s == "obsolete");
    scope_verify(str s = "obsolete", s.assign(s.begin(), s.end()), s == "obsolete");
    scope_verify(str s = "obsolete", s.assign(s, 4), s == "lete");
    scope_verify(str s = "obsolete", s.assign(s, 4, 3), s == "let");

    // Self-assignment is a special case of assignment.
    scope_verify(str s = "obsolete", s = s, s == "obsolete");
    scope_verify(str s = "obsolete", s.assign(s), s == "obsolete");
    scope_verify(str s = "obsolete", s.assign(s.data(), 2), s == "ob");
    scope_verify(str s = "obsolete", s.assign(s.data(), s.size()), s == "obsolete");

    // Allocations, capacity and memory management.
    scope_verify(str s, s.reserve(10), s.capacity() >= 10);
    scope_verify(str s, s.resize(10), s.size() == 10);
    scope_verify(str s, s.resize(10, 'a'), s.size() == 10 && s == "aaaaaaaaaa");
    verify(str().max_size() > 0);
    verify(str().get_allocator() == std::allocator<char>());
    verify(std::strcmp(str("c_str").c_str(), "c_str") == 0);

#if SZ_IS_CPP23_ && defined(__cpp_lib_string_resize_and_overwrite)
    // Test C++23 resize and overwrite functionality
    scope_verify(str s("hello"),
                 s.resize_and_overwrite(10,
                                        [](char *p, std::size_t count) noexcept {
                                            std::memset(p, 'X', count);
                                            return count;
                                        }),
                 s.size() == 10 && s == "XXXXXXXXXX");

    scope_verify(str s("test"),
                 s.resize_and_overwrite(8,
                                        [](char *p, std::size_t) noexcept {
                                            std::strcpy(p, "ABCDE");
                                            return 5;
                                        }),
                 s.size() == 5 && s == "ABCDE");

    scope_verify(str s("orig"),
                 s.try_resize_and_overwrite(6,
                                            [](char *p, std::size_t count) noexcept {
                                                std::strcpy(p, "works!");
                                                return count;
                                            }),
                 s.size() == 6 && s == "works!");
#endif

    // On 32-bit systems the base capacity can be larger than our `z::string::min_capacity`.
    // It's true for MSVC: https://github.com/ashvardanian/StringZilla/issues/168
    if (SZ_IS_64BIT_) scope_verify(str s = "hello", s.shrink_to_fit(), s.capacity() <= sz::string::min_capacity);

    // Concatenation.
    // Following are missing in strings, but are present in vectors.
    verify(str().append("test") == "test");
    verify(str("test") + "ing" == "testing");
    verify(str("test") + str("ing") == "testing");
    verify(str("test") + str("ing") + str("123") == "testing123");
    scope_verify(str s = "!?", s.push_back('a'), s == "!?a");
    scope_verify(str s = "!?", s.pop_back(), s == "!");

    // Incremental construction.
    verify(str("__").insert(1, "test") == "_test_");
    verify(str("__").insert(1, "test", 2) == "_te_");
    verify(str("__").insert(1, 5, 'a') == "_aaaaa_");
    verify(str("__").insert(1, str("test")) == "_test_");
    verify(str("__").insert(1, str("test"), 2) == "_st_");
    verify(str("__").insert(1, str("test"), 2, 1) == "_s_");

    // Inserting at a given iterator position yields back an iterator.
    scope_verify(str s = "__", s.insert(s.begin() + 1, 5, 'a'), s == "_aaaaa_");
    scope_verify(str s = "__", s.insert(s.begin() + 1, {'a', 'b', 'c'}), s == "_abc_");
    let_verify(str s = "__", s.insert(s.begin() + 1, 5, 'a') == (s.begin() + 1));
    let_verify(str s = "__", s.insert(s.begin() + 1, {'a', 'b', 'c'}) == (s.begin() + 1));

    // Handle exceptions.
    // The `length_error` might be difficult to catch due to a large `max_size()`.
    // throws_verify(large_string.insert(large_string.size() - 1, large_number_of_chars, 'a'), std::length_error);
    throws_verify(str("hello").insert(6, "world"), std::out_of_range);         // `index > size()` case from STL
    throws_verify(str("hello").insert(5, str("world"), 6), std::out_of_range); // `s_index > str.size()` case from STL

    // Erasure.
    verify(str("").erase(0, 3) == "");
    verify(str("test").erase(1, 2) == "tt");
    verify(str("test").erase(1) == "t");
    scope_verify(str s = "test", s.erase(s.begin() + 1), s == "tst");
    scope_verify(str s = "test", s.erase(s.begin() + 1, s.begin() + 2), s == "tst");
    scope_verify(str s = "test", s.erase(s.begin() + 1, s.begin() + 3), s == "tt");
    let_verify(str s = "test", s.erase(s.begin() + 1) == (s.begin() + 1));
    let_verify(str s = "test", s.erase(s.begin() + 1, s.begin() + 2) == (s.begin() + 1));
    let_verify(str s = "test", s.erase(s.begin() + 1, s.begin() + 3) == (s.begin() + 1));

    // Substitutions.
    verify(str("hello").replace(1, 2, "123") == "h123lo");
    verify(str("hello").replace(1, 2, str("123"), 1) == "h23lo");
    verify(str("hello").replace(1, 2, "123", 1) == "h1lo");
    verify(str("hello").replace(1, 2, "123", 1, 1) == "h2lo");
    verify(str("hello").replace(1, 2, str("123"), 1, 1) == "h2lo");
    verify(str("hello").replace(1, 2, 3, 'a') == "haaalo");

    // Substitutions with iterators.
    scope_verify(str s = "hello", s.replace(s.begin() + 1, s.begin() + 3, 3, 'a'), s == "haaalo");
    scope_verify(str s = "hello", s.replace(s.begin() + 1, s.begin() + 3, {'a', 'b'}), s == "hablo");

    // Some nice "tweetable" examples :)
    verify(str("Loose").replace(2, 2, str("vath"), 1) == "Loathe");
    verify(str("Loose").replace(2, 2, "vath", 1) == "Love");

    // Insertion is a special case of replacement.
    // Appending and assigning are special cases of insertion.
    // Still, we test them separately to make sure they are not broken.
    verify(str("hello").append("123") == "hello123");
    verify(str("hello").append(str("123")) == "hello123");
    verify(str("hello").append(str("123"), 1) == "hello23");
    verify(str("hello").append(str("123"), 1, 1) == "hello2");
    verify(str("hello").append({'1', '2'}) == "hello12");
    verify(str("hello").append(2, '!') == "hello!!");
    let_verify(str s = "123", str("hello").append(s.begin(), s.end()) == "hello123");
}

/** @brief Constructs StringZilla classes from STL and vice-versa to ensure that the conversions are working. */
void test_stl_conversions_unit() {
    // From a mutable STL string to StringZilla and vice-versa.
    {
        std::string stl {"hello"};
        sz::string sz = stl;
        sz::string_view szv = stl;
        sz::string_span szs = stl;
        verify(sz == "hello");
        verify(szv == "hello");
        verify(szs == "hello");
        szs[0] = 'H'; // A span aliases the source, so this writes through
        verify(stl == "Hello");
        verify(szv == "Hello"); // And a view sees the mutation
        verify(sz == "hello");  // While an owning copy predates it
        stl = sz;
        verify(stl == "hello");
    }
    // From StringZilla views back into a fresh STL string.
    {
        sz::string const sz {"hello"};
        std::string stl;
        stl = sz;
        verify(stl == "hello");
        stl = sz::string_view {"world"};
        verify(stl == "world");
    }
    // From an immutable STL string to StringZilla.
    {
        std::string const stl {"hello"};
        sz::string const sz = stl;
        sz::string_view const szv = stl;
        verify(sz == "hello");
        verify(szv == "hello");
        verify(szv.data() == stl.data()); // A view borrows, a string owns
    }
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
    // From STL `string_view` to StringZilla and vice-versa.
    {
        std::string_view stl {"hello"};
        sz::string sz = stl;
        sz::string_view szv = stl;
        verify(sz == "hello");
        verify(szv == "hello");
        stl = sz;
        verify(stl == "hello");
        stl = szv;
        verify(stl == "hello");
    }
#endif
}

/** @brief Tests STL containers keyed by StringZilla strings, and STL containers ordered & hashed by `sz` functors. */
void test_stl_containers_unit() {

    // Byte order and length disagree across these: "Zebra" wins on its first byte despite being longer,
    // and each prefix precedes its own extension.
    char const *const ascending_keys[] = {"Zebra", "app", "apple", "apples", "banana"};

    // The `sz::string` keys use the native ordering, the `std::string` keys go through `sz::less`.
    std::map<sz::string, int> sorted_words_sz;
    std::map<std::string, int, sz::less> sorted_words_stl;
    for (int insertion = 4; insertion >= 0; --insertion) { // Reverse order, so sorting has work to do
        sorted_words_sz.emplace(ascending_keys[insertion], insertion);
        sorted_words_stl.emplace(ascending_keys[insertion], insertion);
    }
    verify(sorted_words_sz.size() == 5);
    verify(sorted_words_stl.size() == 5);

    std::size_t rank_sz = 0;
    for (auto const &entry : sorted_words_sz) {
        verify(entry.first == ascending_keys[rank_sz] && "sz::string map produced the wrong key at sorted rank_sz");
        verify(entry.second == static_cast<int>(rank_sz) &&
               "sz::string map produced the wrong value at sorted rank_sz");
        ++rank_sz;
    }
    verify(rank_sz == 5);

    std::size_t rank_stl = 0;
    for (auto const &entry : sorted_words_stl) {
        verify(entry.first == ascending_keys[rank_stl] &&
               "std::string map via sz::less produced the wrong key at sorted rank_stl");
        verify(entry.second == static_cast<int>(rank_stl) &&
               "std::string map via sz::less produced the wrong value at sorted rank_stl");
        ++rank_stl;
    }
    verify(rank_stl == 5);

    verify(sorted_words_sz.at("apple") == 2);
    verify(sorted_words_stl.at("apple") == 2);
    verify(sorted_words_sz.count("Apple") == 0); // Comparisons are case-sensitive
    verify(sorted_words_stl.count("Apple") == 0);
    verify(sorted_words_sz.count("appl") == 0); // A prefix of a present key is still absent
    verify(sorted_words_sz.erase("app") == 1);
    verify(sorted_words_sz.erase("app") == 0);
    verify(sorted_words_sz.size() == 4);
    verify(sorted_words_sz.begin()->first == "Zebra");
    verify(sorted_words_sz.lower_bound("apple")->first == "apple");
    verify(sorted_words_sz.upper_bound("apple")->first == "apples");

    // Equal-valued keys assembled from different storage must hash alike and compare equal,
    // so the second insertion collapses onto the first instead of adding a bucket.
    std::unordered_map<sz::string, int> words_sz;
    words_sz.emplace("banana", 7);
    sz::string grown_sz = "bana";
    grown_sz.append("na");
    verify(std::hash<sz::string> {}(grown_sz) == std::hash<sz::string> {}(sz::string("banana")) &&
           "std::hash disagreed for equal-content strings built via different construction paths");
    verify(words_sz.find(grown_sz) != words_sz.end());
    verify(words_sz.emplace(grown_sz, 9).second == false);
    verify(words_sz.at(grown_sz) == 7);
    verify(words_sz.size() == 1);
    verify(words_sz.find("bananas") == words_sz.end());

    // The same, but with `std::string` keys routed through `sz::hash` and `sz::equal_to`.
    std::unordered_map<std::string, int, sz::hash, sz::equal_to> words_stl;
    std::string const heap_key(200, 'x'); // Long enough to escape any small-string buffer
    std::string grown_stl;
    for (int repetition = 0; repetition < 200; ++repetition) grown_stl.push_back('x');
    words_stl.emplace(heap_key, 7);
    verify(sz::hash {}(heap_key) == sz::hash {}(grown_stl) &&
           "sz::hash disagreed for equal-content strings built via different construction paths");
    verify(sz::equal_to {}(heap_key, grown_stl));
    verify(sz::equal_to {}(heap_key, "x") == false);
    verify(words_stl.find(grown_stl) != words_stl.end());
    verify(words_stl.emplace(grown_stl, 9).second == false);
    verify(words_stl.at(grown_stl) == 7);
    verify(words_stl.size() == 1);
    verify(words_stl.find("xyz") == words_stl.end());

    // Empty keys are valid, and order before everything else.
    verify(sz::less {}("", "a"));
    verify(sz::less {}("a", "") == false);
    verify(sz::equal_to {}("", ""));
    sorted_words_sz.emplace("", -1);
    verify(sorted_words_sz.begin()->first == "");
    words_stl.emplace("", -1);
    verify(words_stl.at("") == -1);
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
    verify(str("hello").sat(0) == 'h');
    verify(str("hello").sat(-1) == 'o');
    verify(str("rest").sat(1) == 'e');
    verify(str("rest").sat(-1) == 't');
    verify(str("rest").sat(-4) == 'r');

    verify(str("front").front() == 'f');
    verify(str("front").front(1) == "f");
    verify(str("front").front(2) == "fr");
    verify(str("front").front(2) == "fr");
    verify(str("front").front(-2) == "fro");
    verify(str("front").front(0) == "");
    verify(str("front").front(5) == "front");
    verify(str("front").front(-5) == "");

    verify(str("back").back() == 'k');
    verify(str("back").back(1) == "ack");
    verify(str("back").back(2) == "ck");
    verify(str("back").back(-1) == "k");
    verify(str("back").back(-2) == "ck");
    verify(str("back").back(-4) == "back");
    verify(str("back").back(4) == "");

    verify(str("hello").sub(1) == "ello");
    verify(str("hello").sub(-1) == "o");
    verify(str("hello").sub(1, 2) == "e");
    verify(str("hello").sub(1, 100) == "ello");
    verify(str("hello").sub(100, 100) == "");
    verify(str("hello").sub(-2, -1) == "l");
    verify(str("hello").sub(-2, -2) == "");
    verify(str("hello").sub(100, -100) == "");

    // Passing initializer lists to `operator[]`.
    // Put extra braces to correctly estimate the number of macro arguments :)
    verify((str("hello")[{1, 2}] == "e"));
    verify((str("hello")[{1, 100}] == "ello"));
    verify((str("hello")[{100, 100}] == ""));
    verify((str("hello")[{100, -100}] == ""));
    verify((str("hello")[{-100, -100}] == ""));

    // Checksums
    auto accumulate_bytes = [](str const &s) -> std::size_t {
        return std::accumulate(s.begin(), s.end(), (std::size_t)0,
                               [](std::size_t sum, char c) { return sum + static_cast<unsigned char>(c); });
    };
    verify(str("a").bytesum() == (std::size_t)'a');
    verify(str("0").bytesum() == (std::size_t)'0');
    verify(str("0123456789").bytesum() == arithmetic_sum('0', '9'));
    verify(str("abcdefghijklmnopqrstuvwxyz").bytesum() == arithmetic_sum('a', 'z'));
    verify(str("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz").bytesum() ==
           arithmetic_sum('a', 'z') * 3);
    let_verify(str s = "近来，加文出席微博之夜时对着镜头频繁摆出假笑表情、一度累" //
                       "瘫睡倒在沙发上的照片被广泛转发，引发对他失去童年、被过度" //
                       "消费的担忧。八岁的加文，已当网红近六年了，可以说，自懂事" //
                       "以来，他没有过过一天没有名气的日子。",
               s.bytesum() == accumulate_bytes(s));
}

/** @brief Exercises StringZilla's non-STL mutating string extensions on `sz::string`. */
void test_extensions_updates_unit() {
    using str = sz::string;

    // Try methods.
    verify(str("obsolete").try_assign("hello"));
    verify(str().try_reserve(10));
    verify(str().try_resize(10));
    verify(str("__").try_insert(1, "test"));
    verify(str("test").try_erase(1, 2));
    verify(str("test").try_clear());
    verify(str("test").try_replace(1, 2, "aaaa"));
    verify(str("test").try_push_back('a'));
    verify(str("test").try_shrink_to_fit());

    // Self-referencing methods.
    scope_verify(str s = "test", s.try_assign(s.view()), s == "test");
    scope_verify(str s = "test", s.try_assign(s.view().sub(1, 2)), s == "e");
    scope_verify(str s = "test", s.try_append(s.view().sub(1, 2)), s == "teste");

    // Try methods going beyond and beneath capacity threshold.
    scope_verify(str s = "0123456789012345678901234567890123456789012345678901234567890123", // 64 symbols at start
                 s.try_append(s) && s.try_append(s) && s.try_append(s) && s.try_append(s) && s.try_clear() &&
                     s.try_shrink_to_fit(),
                 s.capacity() < sz::string::min_capacity);

    // Same length replacements.
    scope_verify(str s = "hello", s.replace_all("xx", "xx"), s == "hello");
    scope_verify(str s = "hello", s.replace_all("l", "1"), s == "he11o");
    scope_verify(str s = "hello", s.replace_all("he", "al"), s == "alllo");
    scope_verify(str s = "hello", s.replace_all("x"_bs, "!"), s == "hello");
    scope_verify(str s = "hello", s.replace_all("o"_bs, "!"), s == "hell!");
    scope_verify(str s = "hello", s.replace_all("ho"_bs, "!"), s == "!ell!");

    // Shorter replacements.
    scope_verify(str s = "hello", s.replace_all("xx", "x"), s == "hello");
    scope_verify(str s = "hello", s.replace_all("l", ""), s == "heo");
    scope_verify(str s = "hello", s.replace_all("h", ""), s == "ello");
    scope_verify(str s = "hello", s.replace_all("o", ""), s == "hell");
    scope_verify(str s = "hello", s.replace_all("llo", "!"), s == "he!");
    scope_verify(str s = "hello", s.replace_all("x"_bs, ""), s == "hello");
    scope_verify(str s = "hello", s.replace_all("lo"_bs, ""), s == "he");

    // Longer replacements.
    scope_verify(str s = "hello", s.replace_all("xx", "xxx"), s == "hello");
    scope_verify(str s = "hello", s.replace_all("l", "ll"), s == "hellllo");
    scope_verify(str s = "hello", s.replace_all("h", "hh"), s == "hhello");
    scope_verify(str s = "hello", s.replace_all("o", "oo"), s == "helloo");
    scope_verify(str s = "hello", s.replace_all("llo", "llo!"), s == "hello!");
    scope_verify(str s = "hello", s.replace_all("x"_bs, "xx"), s == "hello");
    scope_verify(str s = "hello", s.replace_all("lo"_bs, "lo"), s == "helololo");

    // Directly mapping bytes using a Look-Up Table.
    sz::look_up_table invert_case = sz::look_up_table::identity();
    for (char c = 'a'; c <= 'z'; c++) invert_case[c] = c - 'a' + 'A';
    for (char c = 'A'; c <= 'Z'; c++) invert_case[c] = c - 'A' + 'a';
    scope_verify(str s = "hello", s.lookup(invert_case), s == "HELLO");
    scope_verify(str s = "HeLLo", s.lookup(invert_case), s == "hEllO");
    scope_verify(str s = "H-lL0", s.lookup(invert_case), s == "h-Ll0");

    // Concatenation.
    verify(str(str("a") | str("b")) == "ab");
    verify(str(str("a") | str("b") | str("ab")) == "abab");

    verify(str(sz::concatenate("a"_sv, "b"_sv)) == "ab");
    verify(str(sz::concatenate("a"_sv, "b"_sv, "c"_sv)) == "abc");

    // The cases above pass only rvalues carrying a `::value_type`. Named lvalues deduce to a
    // reference and raw literals to a character array, neither of which has member typedefs.
    {
        str name = "ash", domain = "mail", tld = "com";
        verify(str(sz::concatenate(name, "@", domain, ".", tld)) == "ash@mail.com");
        verify(str(name | "@" | domain | "." | tld) == "ash@mail.com");
        verify(str(sz::concatenate(name, domain)) == "ashmail");
        verify(str(sz::concatenate("@", name)) == "@ash");

        // Materializing uses an implicit conversion, so the concatenation constructor is not explicit.
        sz::string email = name | "@" | domain;
        verify(email == "ash@mail");
    }

    // Range members slice the string they were called on, so offsets stay inside its own buffer.
    {
        str text = "hello brave new world";
        for (auto segment : text.utf8_wordbreaks()) {
            std::ptrdiff_t const offset = segment.data() - text.data();
            verify(offset >= 0 && offset <= static_cast<std::ptrdiff_t>(text.size()) &&
                   "utf8_wordbreaks segment landed outside the caller's own buffer");
        }
        for (auto token : text.utf8_split_whitespaces()) {
            std::ptrdiff_t const offset = token.data() - text.data();
            verify(offset >= 0 && offset <= static_cast<std::ptrdiff_t>(text.size()) &&
                   "utf8_split_whitespaces token landed outside the caller's own buffer");
        }
        for (auto field : text.utf8_split_delimiters()) {
            std::ptrdiff_t const offset = field.data() - text.data();
            verify(offset >= 0 && offset <= static_cast<std::ptrdiff_t>(text.size()) &&
                   "utf8_split_delimiters field landed outside the caller's own buffer");
        }
        str sso = "a b c";
        for (auto token : sso.utf8_split_whitespaces()) {
            std::ptrdiff_t const offset = token.data() - sso.data();
            verify(offset >= 0 && offset <= static_cast<std::ptrdiff_t>(sso.size()) &&
                   "utf8_split_whitespaces token landed outside the small-string-optimized buffer");
        }
    }

    // Randomization.
    verify(str::random(0).empty());
    verify(str::random(4).size() == 4);
    verify(str::random(4, 42).size() == 4);
}

#pragma endregion // Extensions

/**
 *  @brief The lazy search ranges and their inverses - `find_all`, `rfind_all`, `split`, `rsplit`, `partition`.
 *
 *  Not a template over the string type, unlike its neighbours: these cases deliberately mix owning strings,
 *  borrowed views and literals in one expression, because what they pin is how the range holds its operands.
 *  A haystack passed as an lvalue is borrowed, so a match's `data()` must land inside the caller's own buffer
 *  and not inside a private copy - which under the small-string optimization would still produce plausible
 *  offsets. A needle, by contrast, is copied into the matcher, so a temporary one may outlive the expression.
 */
void test_extensions_ranges_unit() {
    std::printf("  - testing lazy search ranges and splitting...\n");

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
                       matches[1].data() - haystack.data() == 13 &&
                       "Match offsets did not land inside the borrowed lvalue haystack's own buffer");
        let_verify(auto in_sso = sz::find_all(sso, "a").template to<std::vector<sz::string_view>>(),
                   in_sso.size() == 2 &&                     //
                       in_sso[0].data() - sso.data() == 0 && //
                       in_sso[1].data() - sso.data() == 4 &&
                       "Match offsets did not land inside the small-string-optimized haystack's own buffer");
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

#pragma region String Class

/** @brief Tests copy constructor and copy-assignment constructor of `sz::string`. */
void test_string_constructors_unit() {
    std::string alphabet {sz::ascii_printables(), sizeof(sz::ascii_printables())};
    std::vector<sz::string> strings;
    for (std::size_t alphabet_slice = 0; alphabet_slice != alphabet.size(); ++alphabet_slice)
        strings.push_back(alphabet.substr(0, alphabet_slice));
    std::vector<sz::string> copies {strings};
    verify(copies.size() == strings.size());
    for (size_t i = 0; i < copies.size(); ++i) {
        verify(copies[i].size() == strings[i].size() && "Copy-constructed string has the wrong length at index i");
        verify(copies[i] == strings[i] && "Copy-constructed string diverged from its source at index i");
        for (size_t j = 0; j < strings[i].size(); j++)
            verify(copies[i][j] == strings[i][j] && "Copy-constructed string mismatched a byte at index i, j");
    }
    std::vector<sz::string> assignments = strings;
    for (size_t i = 0; i < assignments.size(); ++i) {
        verify(assignments[i].size() == strings[i].size() && "Copy-assigned string has the wrong length at index i");
        verify(assignments[i] == strings[i] && "Copy-assigned string diverged from its source at index i");
        for (size_t j = 0; j < strings[i].size(); j++)
            verify(assignments[i][j] == strings[i][j] && "Copy-assigned string mismatched a byte at index i, j");
    }
    verify(std::equal(strings.begin(), strings.end(), copies.begin()));
    verify(std::equal(strings.begin(), strings.end(), assignments.begin()));
}

/**
 *  @brief Validates that shrinking `reserve` calls are harmless no-ops, just like in the STL.
 *         Regression test: shrinking used to overflow the heap buffer in release builds.
 */
void test_string_reserve_unit() {
    // C API: grow, then shrink - the buffer, length, and contents must stay intact.
    {
        sz_memory_allocator_t alloc;
        sz_memory_allocator_init_default(&alloc);

        sz_string_t str;
        sz_ptr_t start = sz_string_init_length(&str, 100, &alloc);
        verify(start != nullptr);
        std::memset(start, 'a', 100);

        sz_ptr_t grown = sz_string_reserve(&str, 200, &alloc);
        verify(grown != nullptr);
        verify(sz_string_length(&str) == 100);

        // Shrinking must be a no-op: same buffer, same length, same contents.
        sz_ptr_t shrunk = sz_string_reserve(&str, 50, &alloc);
        verify(shrunk == grown);
        verify(sz_string_length(&str) == 100);
        for (sz_size_t i = 0; i != 100; ++i) verify(shrunk[i] == 'a');

        sz_string_free(&str, &alloc);
    }
    // C++ API: `sz::string::reserve` shrinking must match `std::string` behavior - keep the contents.
    {
        sz::string str(100, 'a');
        std::size_t const capacity_before = str.capacity();
        str.reserve(50);
        verify(str.size() == 100);
        verify(str.capacity() == capacity_before);
        verify(str == sz::string(100, 'a'));
    }
}

/** @brief Checks for memory leaks in the string class using the `accounting_allocator`. */
void test_memory_stability_equivalence(std::size_t length, std::size_t iterations) {

    verify(accounting_allocator::counter_ref() == 0 && "Allocator counter was not zero before the stability run");
    using string = sz::basic_string<char, accounting_allocator>;
    string base;

    for (std::size_t i = 0; i < length; ++i) base.push_back('c');
    verify(base.length() == length && "Base string has the wrong length after `push_back` construction");

    // Do copies leak?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string copy(base);
            verify(copy.length() == length && "Copy-constructed string has the wrong length at iteration i");
            verify(copy == base && "Copy-constructed string diverged from `base` at iteration i");
        }
    });

    // How about assignments?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string copy;
            copy = base;
            verify(copy.length() == length && "Copy-assigned string has the wrong length at iteration i");
            verify(copy == base && "Copy-assigned string diverged from `base` at iteration i");
        }
    });

    // How about the move constructor?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string unique_item(base);
            verify(unique_item.length() == length && "Pre-move string has the wrong length at iteration i");
            verify(unique_item == base && "Pre-move string diverged from `base` at iteration i");
            string copy(std::move(unique_item));
            verify(copy.length() == length && "Move-constructed string has the wrong length at iteration i");
            verify(copy == base && "Move-constructed string diverged from `base` at iteration i");
        }
    });

    // And the move assignment operator with an empty target payload?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string unique_item(base);
            string copy;
            copy = std::move(unique_item);
            verify(copy.length() == length &&
                   "Move-assigned (empty target) string has the wrong length at iteration i");
            verify(copy == base && "Move-assigned (empty target) string diverged from `base` at iteration i");
        }
    });

    // And move assignment where the target had a payload?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string unique_item(base);
            string copy;
            for (std::size_t j = 0; j < 317; j++) copy.push_back('q');
            copy = std::move(unique_item);
            verify(copy.length() == length &&
                   "Move-assigned (occupied target) string has the wrong length at iteration i");
            verify(copy == base && "Move-assigned (occupied target) string diverged from `base` at iteration i");
        }
    });

    // Now let's clear the base and check that we're back to zero
    base = string();
    verify(accounting_allocator::counter_ref() == 0 && "Allocator counter did not return to zero after clearing");
}

/** @brief Tests the correctness of the string class update methods, such as `push_back` and `erase`. */
void test_string_updates_equivalence(std::size_t repetitions) {
    // Compare STL and StringZilla strings append functionality.
    char const alphabet_chars[] = "abcdefghijklmnopqrstuvwxyz";
    auto &generator = global_random_generator();
    for (std::size_t repetition = 0; repetition != repetitions; ++repetition) {
        std::string stl_string;
        sz::string sz_string;
        for (std::size_t length = 1; length != 200; ++length) {
            char c = alphabet_chars[generator() % 26];
            stl_string.push_back(c);
            sz_string.push_back(c);
            verify(sz::string_view(stl_string) == sz::string_view(sz_string) &&
                   "sz::string diverged from std::string after `push_back`");
        }

        // Compare STL and StringZilla strings erase functionality.
        while (stl_string.length()) {
            std::size_t offset_to_erase = generator() % stl_string.length();
            std::size_t chars_to_erase = generator() % (stl_string.length() - offset_to_erase) + 1;
            stl_string.erase(offset_to_erase, chars_to_erase);
            sz_string.erase(offset_to_erase, chars_to_erase);
            verify(sz::string_view(stl_string) == sz::string_view(sz_string) &&
                   "sz::string diverged from std::string after `erase`");
        }
    }
}

#pragma endregion // String Class

#pragma region Equivalence

/**
 *  @brief One backend's memory-movement primitives (copy/move/fill), stored by pointer so the differential driver
 *         can iterate a table. Members are named for the call sites (`reference.copy(...)` etc.), invoked directly.
 */
struct memory_backend_t {
    char const *name;
    sz_copy_t copy;
    sz_move_t move;
    sz_fill_t fill;
};

/** @brief One backend's byte-lookup (transform) kernel, stored by pointer; `reference.lookup(...)` invokes it. */
struct lookup_backend_t {
    char const *name;
    sz_lookup_t lookup;
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
 *  `inputs` is the number of random source patterns fuzzed at each length.
 */
template <typename reference_, typename candidate_>
void check_memory_equivalence_(reference_ reference, candidate_ candidate, sz_size_t inputs) {

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
                if (length)
                    verify(std::memcmp(reference_output.data(), target, length) == 0 &&
                           "Candidate copy backend diverged from the serial reference");

                reference.fill(reference_output.data(), length, fill_value);
                candidate.fill(target, length, fill_value);
                if (length)
                    verify(std::memcmp(reference_output.data(), target, length) == 0 &&
                           "Candidate fill backend diverged from the serial reference");
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
                    verify(std::memcmp(buffer, reference_buffer.data(), length) == 0 &&
                           "Candidate move backend diverged from reference on forward overlap");

                    // Backward overlap: destination behind the source.
                    std::memcpy(buffer, source, length);
                    std::memcpy(reference_buffer.data(), source, length);
                    candidate.move(buffer, buffer + shift, moved);
                    reference.move(reference_buffer.data(), reference_buffer.data() + shift, moved);
                    verify(std::memcmp(buffer, reference_buffer.data(), length) == 0 &&
                           "Candidate move backend diverged from reference on backward overlap");
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
 *  `inputs` is the number of random source patterns fuzzed at each length.
 */
template <typename reference_, typename candidate_>
void check_lookup_equivalence_(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    char upper_table[256], lower_table[256], ascii_table[256];
    sz_lookup_init_upper(upper_table);
    sz_lookup_init_lower(lower_table);
    sz_lookup_init_ascii(ascii_table);
    struct named_table_t {
        char const *name;
        char const *table;
    };
    named_table_t const named_tables[] = {
        {"upper", upper_table},
        {"lower", lower_table},
        {"ascii", ascii_table},
    };

    std::vector<sz_size_t> const lengths = memory_equivalence_lengths();
    sz_size_t const max_length = lengths.back();

    for (named_table_t const &named_table : named_tables)
        for (sz_size_t length : lengths) {
            for (sz_size_t input = 0; input != inputs; ++input) {

                std::vector<char> source_storage(length + SZ_CACHE_LINE_WIDTH, '\0');
                sz_cptr_t const source = source_storage.data() + (input % SZ_CACHE_LINE_WIDTH);
                if (length) randomize_string(const_cast<char *>(source), length);

                for_each_cacheline_offset_(max_length, [&](sz_ptr_t target, std::size_t) {
                    std::vector<char> reference_output(length, '\0');
                    reference.lookup(reference_output.data(), length, source, named_table.table);
                    candidate.lookup(target, length, source, named_table.table);
                    if (length)
                        verify(std::memcmp(reference_output.data(), target, length) == 0 &&
                               "Candidate lookup output diverged from reference for this lookup table");
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

    char upper_table[256], lower_table[256], ascii_table[256];
    sz_lookup_init_upper(upper_table);
    sz_lookup_init_lower(lower_table);
    sz_lookup_init_ascii(ascii_table);
    char const *const lookup_tables[] = {upper_table, lower_table, ascii_table};

    for (char const *lookup_table : lookup_tables) {
        lookup(nullptr, 0, nullptr, lookup_table); // Zero-length must touch nothing

        for (std::size_t length : {(std::size_t)1, (std::size_t)8, (std::size_t)64, (std::size_t)257})
            with_guarded_buffer_(length, [&](sz_ptr_t destination, std::size_t usable_length) {
                std::vector<char> source(usable_length, '\0');
                for (std::size_t byte = 0; byte != usable_length; ++byte)
                    source[byte] = (char)((byte % 3) ? 'a' + (byte % 26) : 0);
                lookup(destination, usable_length, source.data(), lookup_table);
            });
    }
}

/**
 *  @brief Adversarial safety driver: feeds zero-length, tiny, overlapping, and embedded-NUL inputs through
 *         the dispatched, serial, and every natively-compiled movement/lookup kernel, asserting that canary
 *         bytes guarding both sides of the destination remain intact and that nothing crashes.
 */
void test_memory_safety() {

    // Dispatched (automatic kernel resolution).
    check_memory_safety_(sz_copy, sz_move, sz_fill);

    // Manual propagation to each natively-compiled backend kernel.
    check_memory_safety_(sz_copy_serial, sz_move_serial, sz_fill_serial);
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

    // Dispatched (automatic kernel resolution).
    check_lookup_safety_(sz_lookup);

    // Manual propagation to each natively-compiled backend kernel.
    check_lookup_safety_(sz_lookup_serial);
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
 *  @brief The memory-movement (copy/move/fill) backends compiled on this target. The always-present `dispatched`
 *         entry keeps the table non-empty on a baseline build. This tier set has Skylake but no Icelake.
 */
static memory_backend_t const memory_backends[] = {
    {"dispatched", sz_copy, sz_move, sz_fill},
#if SZ_USE_HASWELL
    {"haswell", sz_copy_haswell, sz_move_haswell, sz_fill_haswell},
#endif
#if SZ_USE_SKYLAKE
    {"skylake", sz_copy_skylake, sz_move_skylake, sz_fill_skylake},
#endif
#if SZ_USE_NEON
    {"neon", sz_copy_neon, sz_move_neon, sz_fill_neon},
#endif
#if SZ_USE_SVE
    {"sve", sz_copy_sve, sz_move_sve, sz_fill_sve},
#endif
#if SZ_USE_V128
    {"v128", sz_copy_v128, sz_move_v128, sz_fill_v128},
#endif
#if SZ_USE_V128RELAXED
    {"v128relaxed", sz_copy_v128relaxed, sz_move_v128relaxed, sz_fill_v128relaxed},
#endif
#if SZ_USE_RVV
    {"rvv", sz_copy_rvv, sz_move_rvv, sz_fill_rvv},
#endif
#if SZ_USE_LASX
    {"lasx", sz_copy_lasx, sz_move_lasx, sz_fill_lasx},
#endif
#if SZ_USE_POWERVSX
    {"powervsx", sz_copy_powervsx, sz_move_powervsx, sz_fill_powervsx},
#endif
};

/**
 *  @brief The byte-lookup (transform) backends compiled on this target. The always-present `dispatched` entry keeps
 *         the table non-empty on a baseline build. This tier set has Icelake but no Skylake.
 */
static lookup_backend_t const lookup_backends[] = {
    {"dispatched", sz_lookup},
#if SZ_USE_HASWELL
    {"haswell", sz_lookup_haswell},
#endif
#if SZ_USE_ICELAKE
    {"icelake", sz_lookup_icelake},
#endif
#if SZ_USE_NEON
    {"neon", sz_lookup_neon},
#endif
#if SZ_USE_SVE
    {"sve", sz_lookup_sve},
#endif
#if SZ_USE_V128
    {"v128", sz_lookup_v128},
#endif
#if SZ_USE_V128RELAXED
    {"v128relaxed", sz_lookup_v128relaxed},
#endif
#if SZ_USE_RVV
    {"rvv", sz_lookup_rvv},
#endif
#if SZ_USE_LASX
    {"lasx", sz_lookup_lasx},
#endif
#if SZ_USE_POWERVSX
    {"powervsx", sz_lookup_powervsx},
#endif
};

/**
 *  @brief Drives the serial-vs-SIMD movement and lookup differential tests across every backend compiled on this
 *         target (dispatched first). Copy/move/fill and lookup carry their own (differing) tier sets.
 */
void test_memory_all() {
    sz_size_t const inputs = (sz_size_t)scale_iterations(2);

    memory_backend_t const memory_serial {"serial", sz_copy_serial, sz_move_serial, sz_fill_serial};
    for (memory_backend_t const &backend : memory_backends) check_memory_equivalence_(memory_serial, backend, inputs);

    lookup_backend_t const lookup_serial {"serial", sz_lookup_serial};
    for (lookup_backend_t const &backend : lookup_backends) check_lookup_equivalence_(lookup_serial, backend, inputs);
}

#pragma endregion // Drivers

// Explicit template instantiations for the entry points invoked from `main()` (see `stringzilla.cpp`).
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
