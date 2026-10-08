/**
 *  @file test/string.cpp
 *  @author Ash Vardanian
 *  @date December 21, 2023
 *  @brief Tests for the string class and the utilities beneath it.
 *
 *  Covers arithmetic and struct plumbing, ASCII utilities, memory, STL compatibility, conversions
 *  and the extensions beyond the STL.
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
#include <cstdlib> // `std::malloc`, `std::free`
#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::transform`
#include <forward_list>  // `std::forward_list`
#include <iterator>      // `std::distance`
#include <map>           // `std::map`
#include <memory>        // `std::allocator`
#include <numeric>       // `std::accumulate`
#include <random>        // `std::random_device`
#include <set>           // `std::set`
#include <span>          // `std::span`
#include <sstream>       // `std::ostringstream`
#include <unordered_map> // `std::unordered_map`
#include <unordered_set> // `std::unordered_set`
#include <vector>        // `std::vector`

#include <string>      // Baseline
#include <string_view> // Baseline

#include <fmt/format.h>

#include "cross.hpp"   // `check_memory_unit_`, `memory_backend_t`, `lookup_backend_t`
#include "harness.hpp" // `randomize_string`, `test_context_t`

namespace ashvardanian::stringzilla::test {
using sz::literals::operator""_sv; // for `sz::string_view_t`
using sz::literals::operator""_bs; // for `sz::byteset_t`

using namespace std::literals; // for ""sv

#pragma region Helpers

/** Compares two byte ranges and fails with a localized diagnostic on the first mismatch. */
inline void expect_equality(char const *first, char const *second, std::size_t size) {
    if (std::memcmp(first, second, size) == 0) return;
    std::size_t mismatch_position = 0;
    for (; mismatch_position < size; ++mismatch_position)
        if (first[mismatch_position] != second[mismatch_position]) break;
    fmt::println(stderr, "Mismatch at position {}", mismatch_position);
    verify(first[mismatch_position] == second[mismatch_position]);
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

/** Stateful allocator charging each byte to its counter, so two over distinct counters differ. */
struct accounting_allocator_t {
    using value_type = char;
    std::size_t *live_bytes;

    explicit accounting_allocator_t(std::size_t *live_bytes) noexcept : live_bytes(live_bytes) {}

    char *allocate(std::size_t count) {
        *live_bytes += count;
        return std::allocator<char> {}.allocate(count);
    }

    void deallocate(char *block, std::size_t count) {
        verify(count <= *live_bytes && "Deallocated more bytes than were tracked as allocated");
        *live_bytes -= count;
        std::allocator<char> {}.deallocate(block, count);
    }

    bool operator==(accounting_allocator_t const &) const noexcept = default;
};

/** Allocator that throws on every request, as @c std::allocator does once memory runs out. */
struct throwing_allocator_t {
    using value_type = char;
    char *allocate(std::size_t) { throw std::bad_alloc(); }
    void deallocate(char *, std::size_t) noexcept {}
    bool operator==(throwing_allocator_t const &) const noexcept = default;
};

/** Runs @p callback and asserts that it leaves @p live_bytes unchanged. */
template <typename callback_type_>
void assert_balanced_memory(std::size_t const &live_bytes, callback_type_ callback) {
    std::size_t const before = live_bytes;
    callback();
    verify(live_bytes == before && "Callback leaked or double-freed tracked allocator bytes");
}

/** The copy, move and fill dispatch points, in the shape of their kernels. */
static memory_backend_t const memory_dispatched {"dispatched", cpu_best<sz_copy_best>, cpu_best<sz_move_best>,
                                                 cpu_best<sz_fill_best>};

/** The lookup dispatch point, in the shape of its kernels. */
static lookup_backend_t const lookup_dispatched {"dispatched", cpu_best<sz_lookup_best>};

#pragma endregion Helpers

#pragma region Arithmetic

/** Several string processing operations rely on computing integer logarithms. Failures in such
 *  operations will result in wrong @c resize outcomes and heap corruption. */
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

#if STRINGZILLA_ARCH_64BIT_
    verify(sz_size_bit_ceil(4000000000ull) == (1ull << 32));
    verify(sz_size_bit_ceil(8000000000ull) == (1ull << 33));
    verify(sz_size_bit_ceil(16000000000ull) == (1ull << 34));

    verify(sz_size_bit_ceil((1ull << 62)) == (1ull << 62));
    verify(sz_size_bit_ceil((1ull << 62) + 1) == (1ull << 63));
    verify(sz_size_bit_ceil((1ull << 63)) == (1ull << 63));
#endif
}

#pragma endregion Arithmetic

#pragma region Sequence

/** Validates @c sz_sequence_t and related construction utilities. */
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
        sequence.get_start = [](void const *handle, sz_size_t index) -> sz_cptr_t {
            auto const &strings = *static_cast<strings_vector_t const *>(handle);
            return strings[index].data();
        };
        sequence.get_length = [](void const *handle, sz_size_t index) -> sz_size_t {
            auto const &strings = *static_cast<strings_vector_t const *>(handle);
            return strings[index].size();
        };

        verify(sequence.count == 3);
        verify("banana"_sv == sequence.get_start(sequence.handle, 0));
        verify("apple"_sv == sequence.get_start(sequence.handle, 1));
        verify("cherry"_sv == sequence.get_start(sequence.handle, 2));
    }
}

/** Verifies assignment from forward iterators and appending an element from the same tape. */
void test_tape_assign_unit() {
    sz::tape<char, std::uint32_t> tape;

    // A forward list is walked forward only, but any number of times - all that `assign` needs.
    std::forward_list<std::string> strings {"alpha", "", "gamma"};
    verify(tape.assign(strings.begin(), strings.end()) == sz::status_t::success_k);
    verify(tape.size() == 3);
    verify(sz::string_view_t(tape[0].data(), tape[0].size()) == "alpha"_sv);
    verify(tape[1].size() == 0);
    verify(sz::string_view_t(tape[2].data(), tape[2].size()) == "gamma"_sv);
    verify(tape.append(tape[0]) == sz::status_t::success_k);
    verify(sz::string_view_t(tape[3].data(), tape[3].size()) == "alpha"_sv);
    verify(tape.assign(tape.begin(), tape.end()) == sz::status_t::success_k);
    verify(tape.size() == 4 && sz::string_view_t(tape[3].data(), tape[3].size()) == "alpha"_sv);
    auto const terminated_bytes = tape.buffer();
    verify(terminated_bytes[tape[0].size()] == '\0');
    auto moved = std::move(tape);
    verify(tape.size() == 0 && moved.size() == 4);
    verify(tape.assign(strings.begin(), strings.end()) == sz::status_t::success_k);
    tape = std::move(moved);
    auto &same = tape;
    tape = std::move(same);
    verify(tape.size() == 4);
    verify(tape.assign(strings.end(), strings.end()) == sz::status_t::success_k);
    verify(tape.view().size() == 0);

    std::uint32_t const offsets[] = {0, 5, 5};
    sz::packed_tape_view<char, std::uint32_t> const packed({"alpha", 5}, offsets);
    verify(packed[1].empty());

    auto const [allocator, status] = sz::unified_alloc<char, sz_cap_serial_k>::make();
    verify(status == sz::status_t::success_k);
    tape_t owned(allocator), independent(allocator);
    verify(owned.assign(strings.begin(), strings.end()) == sz::status_t::success_k);
    verify(owned.size() == 3 && owned[1].empty());
    std::array<sz_string_view_t, 3> views {{{"a\0b", 3}, {"", 0}, {"de", 2}}};
    verify(owned.assign(views) == sz::status_t::success_k);
    verify(owned.allocation_bytes() == 4 * sizeof(sz_u64_t) + 5);
    verify(owned.view().tape_total_bytes() == 5);
    verify(owned.assign(owned.sequence()) == sz::status_t::success_k);
    verify(independent.assign(owned.sequence()) == sz::status_t::success_k);
    owned.reset();
    verify(independent.size() == 3 && independent[1].empty());
    verify(sz::string_view_t(independent[0].data(), independent[0].size()) == "a\0b"_sv);
    verify(independent.append(independent[0]) == sz::status_t::success_k);
    verify(independent.size() == 4 && independent[3].size() == 3);
    verify(independent.sequence().get_start == sz_sequence_tape_start);
    auto const descriptor = independent.sequence();
    owned = std::move(independent);
    verify(independent.size() == 0 && owned.sequence().handle == descriptor.handle);
    sz_sequence_t oversized = owned.sequence();
    oversized.count = STRINGZILLA_SIZE_MAX;
    verify(owned.assign(oversized) == sz::status_t::bad_alloc_k);
    verify(owned.size() == 4);
    verify(owned.assign(std::span<sz_string_view_t const> {}) == sz::status_t::success_k);
    verify(owned.size() == 0);

    sz::tape<char16_t, std::uint32_t> wide;
    std::array<std::u16string_view, 2> const units {u"alpha", u""};
    verify(wide.assign(units.begin(), units.end()) == sz::status_t::success_k);
    verify(wide.append(wide[0]) == sz::status_t::success_k);
    verify(std::u16string_view(wide[2].data(), wide[2].size()) == units[0]);
    auto const terminated_units = wide.buffer();
    verify(terminated_units[units[0].size()] == u'\0');
}

/** Validates that @c tape refuses to grow past the range of its offset type. */
void test_tape_overflow_unit() {
    // 8-bit offsets hit the same code path as 32-bit offsets past 4 GB, but already at 256 bytes.
    using small_tape_t = sz::tape<char, std::uint8_t>;

    // Appending past the offset range must fail cleanly and leave the stored strings untouched.
    {
        small_tape_t tape;
        std::string const oversized_string(200, 'x');
        verify(tape.append(std::span<char const>(oversized_string)) == sz::status_t::success_k);
        // Two 200-byte strings need 402 bytes of buffer, past the 255 maximum of 8-bit offsets.
        verify(tape.append(std::span<char const>(oversized_string)) == sz::status_t::overflow_risk_k);
        verify(tape.size() == 1);
        // The first string must still sit at offset 0, ending at 201 with its NULL terminator.
        verify(tape.offsets()[0] == 0);
        verify(tape.offsets()[1] == 201);
        verify(std::memcmp(tape.buffer().data(), oversized_string.data(), oversized_string.size()) == 0);
    }

    // Same for bulk assignment: the combined size must fit the offset range.
    {
        small_tape_t tape;
        std::string const stored_string(10, 'z');
        verify(tape.append(std::span<char const>(stored_string)) == sz::status_t::success_k);
        std::vector<std::string> strings {std::string(200, 'x'), std::string(200, 'y')};
        verify(tape.assign(strings.begin(), strings.end()) == sz::status_t::overflow_risk_k);
        verify(tape.size() == 1 && tape[0].size() == stored_string.size());
    }
    {
        struct allocation_state_t {
            sz_allocator_t heap {};
            std::size_t calls = 0, fail_at = 0, live_bytes = 0;
        } state;
        verify(sz_allocator_init_heap(&state.heap) == sz_success_k);
        sz_allocator_t allocator {};
        allocator.handle = &state;
        allocator.allocate = [](sz_size_t bytes, void *handle, sz_stream_t stream) -> void * {
            auto &state = *static_cast<allocation_state_t *>(handle);
            if (++state.calls == state.fail_at) return nullptr;
            void *block = state.heap.allocate(bytes, state.heap.handle, stream);
            if (block) state.live_bytes += bytes;
            return block;
        };
        allocator.free = [](void *block, sz_size_t bytes, void *handle, sz_stream_t stream) {
            auto &state = *static_cast<allocation_state_t *>(handle);
            verify(bytes <= state.live_bytes);
            state.live_bytes -= bytes;
            state.heap.free(block, bytes, state.heap.handle, stream);
        };
        using allocated_tape_t = sz::tape<char, std::uint32_t, sz::unified_alloc<char>>;
        {
            allocated_tape_t tape {sz::unified_alloc<char>(allocator)};
            std::array<std::string_view, 1> const texts {"kept"};
            verify(tape.assign(texts.begin(), texts.end()) == sz::status_t::success_k);
            std::size_t const live_bytes = state.live_bytes;
            for (std::size_t fail_after : {1u, 2u}) {
                state.fail_at = state.calls + fail_after;
                verify(tape.assign(texts.begin(), texts.end()) == sz::status_t::bad_alloc_k);
                verify(tape.size() == 1 && sz::string_view_t(tape[0].data(), tape[0].size()) == "kept"_sv);
                verify(state.live_bytes == live_bytes);
            }
        }
        verify(state.live_bytes == 0);
    }
}

#pragma endregion Sequence

#pragma region Allocator

/** Validates @c sz_allocator_t and related construction utilities. */
void test_allocator_unit() {
    // Our behavior for `malloc(0)` is to return a NULL pointer,
    // while the standard is implementation-defined.
    {
        sz_allocator_t allocator;
        verify(sz_allocator_init_heap(&allocator) == sz_success_k);
        verify(allocator.allocate(0, allocator.handle, nullptr) == nullptr);
    }

    // Non-NULL allocation
    {
        sz_allocator_t allocator;
        verify(sz_allocator_init_heap(&allocator) == sz_success_k);
        void *byte = allocator.allocate(1, allocator.handle, nullptr);
        verify(byte != nullptr && "Default allocator returned NULL for a non-zero-length allocation");
        allocator.free(byte, 1, allocator.handle, nullptr);
    }

    {
        alignas(sz_size_t) char buffer[1025];
        sz_allocator_t allocator;
        verify(sz_allocator_init_arena(&allocator, buffer + 1, sizeof(buffer) - 1, 64) == sz_success_k);
        verify(allocator.allocate(STRINGZILLA_SIZE_MAX, allocator.handle, nullptr) == nullptr);
        void *byte = allocator.allocate(1, allocator.handle, nullptr);
        verify(byte != nullptr && reinterpret_cast<sz_size_t>(byte) % 64 == 0);
        void *next = allocator.allocate(1, allocator.handle, nullptr);
        verify(next != nullptr && reinterpret_cast<sz_size_t>(next) % 64 == 0 && next != byte);
        allocator.free(byte, 1, allocator.handle, nullptr);
        sz_allocator_t const original = allocator;
        sz_arena_t_ const *metadata = static_cast<sz_arena_t_ const *>(allocator.handle);
        sz_size_t const consumed = metadata->consumed;
        verify(sz_allocator_init_arena(&allocator, buffer + 1, sizeof(buffer) - 1, 0) == sz_unexpected_dimensions_k);
        verify(sz_allocator_init_arena(&allocator, buffer + 1, sizeof(buffer) - 1, 3) == sz_unexpected_dimensions_k);
        verify(sz_allocator_init_arena(&allocator, buffer + 1, sizeof(sz_size_t), 64) == sz_bad_alloc_k);
        verify(sz_allocator_init_arena(&allocator, nullptr, sizeof(buffer), 64) == sz_bad_alloc_k);
        verify(sz_allocator_init_arena(&allocator, buffer, STRINGZILLA_SIZE_MAX, 64) == sz_overflow_risk_k);
        verify(allocator.allocate == original.allocate && allocator.free == original.free &&
               allocator.handle == original.handle);
        verify(metadata->consumed == consumed);
        verify(sz_allocator_init_arena(&allocator, buffer + 1, sizeof(buffer) - 1, 1) == sz_success_k);
        verify(allocator.allocate(1, allocator.handle, nullptr) != nullptr);
    }
    {
        sz_allocator_t allocator;
        verify(sz_allocator_init_heap(&allocator) == sz_success_k);
        sz_allocator_t const original = allocator;
        verify(sz_allocator_init_device_best(&allocator, sz_cap_serial_k) == sz_missing_kernel_k);
        verify(sz_allocator_init_pinned_best(&allocator, sz_cap_serial_k) == sz_missing_kernel_k);
        verify(allocator.allocate == original.allocate && allocator.free == original.free &&
               allocator.handle == original.handle);
    }
    {
        auto const [allocator, status] = sz::unified_alloc<char, sz_cap_serial_k>::make();
        verify(status == sz::status_t::success_k);
        char *empty = allocator.allocate(0);
        verify(empty != nullptr);
        allocator.deallocate(empty, 0);
        sz::unified_alloc<sz_size_t, sz_cap_serial_k> words(allocator);
        verify(words.allocate(STRINGZILLA_SIZE_MAX) == nullptr);
    }
}

#pragma endregion Allocator

#pragma region Byteset

/** Validates @c sz_byteset_t and related construction utilities. */
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

/** Known-answer coverage for the ASCII classification methods of @c sz::string_t and
 *  @c sz::string_view_t, such as @c is_alpha, @c is_digit and @c contains_only. */
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

#pragma endregion Byteset

#pragma region Memory

/**
 *  @brief Known-answer tests for the memory dispatch points and the C++ string wrappers.
 *
 *  A regression that the serial-vs-SIMD agreement tests would miss - because both share a wrong
 *  constant - is still caught against an external ground truth; the kernels of each capability face
 *  the same vectors in `cross_<arch>.cpp`.
 */
void test_memory_unit() {
    check_memory_unit_(memory_dispatched);
    check_lookup_unit_(lookup_dispatched);

    // The case tables follow the Latin-1 letter rule, keeping × and ÷ and ÿ as is.
    {
        char upper_table[256], lower_table[256];
        sz_lookup_init_upper(upper_table);
        sz_lookup_init_lower(lower_table);
        for (int byte = 0; byte != 256; ++byte) {
            bool const is_lower_letter = (byte >= 'a' && byte <= 'z') || (byte >= 0xE0 && byte <= 0xFE && byte != 0xF7);
            bool const is_upper_letter = (byte >= 'A' && byte <= 'Z') || (byte >= 0xC0 && byte <= 0xDE && byte != 0xD7);
            verify((unsigned char)upper_table[byte] == (is_lower_letter ? byte - 32 : byte));
            verify((unsigned char)lower_table[byte] == (is_upper_letter ? byte + 32 : byte));
        }
    }

    // C++ wrapper sanity: a couple of `sz::string_t` / `sz::string_view_t` known-answer reads alongside the C API.
    {
        sz::string_view_t const view = "Hello, World!"_sv;
        verify(view.size() == 13u);
        verify(view.substr(7, 5) == "World"_sv);
        verify(view.front() == 'H' && view.back() == '!');

        sz::string_t const owned = "Hello, World!";
        verify(owned.size() == 13u);
        verify(owned == view);
        verify(sz::string_t("apple").compare("banana") < 0);
    }

    // Embedded NUL must be preserved verbatim by a stored `sz::string_t`: the size is the full byte length, and
    // indexing past the interior NUL reaches the trailing bytes rather than stopping at the C-string boundary.
    {
        char const with_nul[] = {'a', 'b', '\0', 'c', 'd'};
        sz::string_t const owned(with_nul, sizeof(with_nul));
        verify(owned.size() == sizeof(with_nul));   // Full length, NUL is a stored byte
        verify(owned[2] == '\0');                   // The interior NUL survives
        verify(owned[3] == 'c' && owned[4] == 'd'); // Indexing past the NUL works
        verify(owned == sz::string_view_t(with_nul, sizeof(with_nul)));
    }

    // Copies past 1 MB run in both directions on Haswell and Skylake. The length is from
    // GitHub issue #228, and the misaligned ends exercise the head and tail of that path.
    {
        std::size_t const length = 1024ull * 10ull * 103ull;
        std::vector<char> source(length + 7), target(length + 11);
        for (std::size_t i = 0; i != source.size(); ++i) source[i] = static_cast<char>('a' + i % 26);
        verify(memory_dispatched.copy(target.data() + 11, source.data() + 7, length, nullptr) == sz_success_k);
        expect_equality(source.data() + 7, target.data() + 11, length);
    }
}

#pragma endregion Memory

#pragma region STL Reads

/** Invokes different C++ member methods of immutable strings to cover all STL APIs. This test
 *  guarantees API @b compatibility with STL @c std::basic_string template. */
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

    // Spaceship operator instead of conventional comparions.
    verify((str("a") <=> str("b")) == std::strong_ordering::less);
    verify((str("b") <=> str("a")) == std::strong_ordering::greater);
    verify((str("b") <=> str("b")) == std::strong_ordering::equal);
    verify((str("a") <=> str("aa")) == std::strong_ordering::less);

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

#if defined(__cpp_lib_string_contains)
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

    verify(std::equal_to<str> {}("hello", "world") == false);
    verify(std::less<str> {}("hello", "world") == true);
}

#pragma endregion STL Reads

#pragma region STL Updates

/** Invokes different C++ member methods of the memory-owning string class to make sure they all
 *  pass compilation, guaranteeing API compatibility with the STL @c std::basic_string template. */
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

#if defined(__cpp_lib_string_resize_and_overwrite)
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

    let_verify(str s("orig"), sz::succeeded(s.try_resize_and_overwrite(6,
                                                                       [](char *p, std::size_t count) noexcept {
                                                                           std::strcpy(p, "works!");
                                                                           return count;
                                                                       })) &&
                                  s.size() == 6 && s == "works!");
#endif

    // On 32-bit systems the base capacity can be larger than our `z::string::min_capacity`.
    // It's true for MSVC: https://github.com/ashvardanian/StringZilla/issues/168
    if (STRINGZILLA_ARCH_64BIT_)
        scope_verify(str s = "hello", s.shrink_to_fit(), s.capacity() <= sz::string_t::min_capacity);

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

/** Constructs StringZilla classes from STL and vice-versa, ensuring the conversions work. */
void test_stl_conversions_unit() {
    // From a mutable STL string to StringZilla and vice-versa.
    {
        std::string stl {"hello"};
        sz::string_t sz = stl;
        sz::string_view_t szv = stl;
        sz::string_span_t szs = stl;
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
        sz::string_t const sz {"hello"};
        std::string stl;
        stl = sz;
        verify(stl == "hello");
        stl = sz::string_view_t {"world"};
        verify(stl == "world");
    }
    // From an immutable STL string to StringZilla.
    {
        std::string const stl {"hello"};
        sz::string_t const sz = stl;
        sz::string_view_t const szv = stl;
        verify(sz == "hello");
        verify(szv == "hello");
        verify(szv.data() == stl.data()); // A view borrows, a string owns
    }
    // From STL `string_view_t` to StringZilla and vice-versa.
    {
        std::string_view stl {"hello"};
        sz::string_t sz = stl;
        sz::string_view_t szv = stl;
        verify(sz == "hello");
        verify(szv == "hello");
        stl = sz;
        verify(stl == "hello");
        stl = szv;
        verify(stl == "hello");
    }
}

/** Tests STL containers keyed by StringZilla strings, and ordered and hashed by @c sz functors. */
void test_stl_containers_unit() {

    // Byte order and length disagree across these: "Zebra" wins on its first byte despite being longer,
    // and each prefix precedes its own extension.
    char const *const ascending_keys[] = {"Zebra", "app", "apple", "apples", "banana"};

    // The `sz::string_t` keys use the native ordering, the `std::string` keys go through `sz::less`.
    std::map<sz::string_t, int> sorted_words_sz;
    std::map<std::string, int, sz::less> sorted_words_stl;
    for (int insertion = 4; insertion >= 0; --insertion) { // Reverse order, so sorting has work to do
        sorted_words_sz.emplace(ascending_keys[insertion], insertion);
        sorted_words_stl.emplace(ascending_keys[insertion], insertion);
    }
    verify(sorted_words_sz.size() == 5);
    verify(sorted_words_stl.size() == 5);

    std::size_t rank_sz = 0;
    for (auto const &entry : sorted_words_sz) {
        verify(entry.first == ascending_keys[rank_sz] && "sz::string_t map produced the wrong key at sorted rank_sz");
        verify(entry.second == static_cast<int>(rank_sz) &&
               "sz::string_t map produced the wrong value at sorted rank_sz");
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
    std::unordered_map<sz::string_t, int> words_sz;
    words_sz.emplace("banana", 7);
    sz::string_t grown_sz = "bana";
    grown_sz.append("na");
    verify(std::hash<sz::string_t> {}(grown_sz) == std::hash<sz::string_t> {}(sz::string_t("banana")) &&
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

#pragma endregion STL Updates

#pragma region Extensions

/** Invokes C++ member methods of immutable strings, covering extensions beyond the STL API. */
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

/** Exercises StringZilla's non-STL mutating string extensions on @c sz::string_t. */
void test_extensions_updates_unit() {
    using str = sz::string_t;

    // Try methods.
    verify(sz::succeeded(str("obsolete").try_assign("hello")));
    verify(sz::succeeded(str().try_reserve(10)));
    verify(sz::succeeded(str().try_resize(10)));
    verify(sz::succeeded(str("__").try_insert(1, "test")));
    verify(str("test").try_erase(1, 2));
    verify(sz::succeeded(str("test").try_replace(1, 2, "aaaa")));
    verify(sz::succeeded(str("test").try_push_back('a')));
    verify(sz::succeeded(str("test").try_shrink_to_fit()));

    // Growing through a concatenation must succeed and land exactly on the concatenation's length.
    let_verify(str s = "ab", sz::succeeded(s.try_assign("hello"_sv | "world")) && s == "helloworld");

    // A throwing allocator must surface as `bad_alloc_k` from the `noexcept` twins.
    let_verify(sz::basic_string<throwing_allocator_t> s, s.try_reserve(1000) == sz::status_t::bad_alloc_k && s.empty());

    // Self-referencing methods.
    let_verify(str s = "test", sz::succeeded(s.try_assign(s.view())) && s == "test");
    let_verify(str s = "test", sz::succeeded(s.try_assign(s.view().sub(1, 2))) && s == "e");
    let_verify(str s = "test", sz::succeeded(s.try_append(s.view().sub(1, 2))) && s == "teste");

    // Try methods going beyond and beneath capacity threshold.
    {
        str s = "0123456789012345678901234567890123456789012345678901234567890123"; // 64 symbols at start
        for (int doubling = 0; doubling != 4; ++doubling) verify(sz::succeeded(s.try_append(s)));
        s.clear();
        verify(sz::succeeded(s.try_shrink_to_fit()));
        verify(s.capacity() < sz::string_t::min_capacity);
    }

    // Folding and normalizing allocate through the string's allocator, not a default-made one.
    {
        std::size_t live_bytes = 0;
        accounting_allocator_t const allocator {&live_bytes};
        {
            sz::basic_string<accounting_allocator_t> s(allocator);
            verify(sz::succeeded(s.try_assign("STRASSE AND STRASSE")));
            verify(sz::succeeded(s.try_utf8_uncased_fold()) && s == "strasse and strasse");
            verify(sz::succeeded(s.try_utf8_normalize(sz_normal_form_nfd_k)) && s == "strasse and strasse");
            verify(s.get_allocator() == allocator && live_bytes != 0);
        }
        verify(live_bytes == 0);
    }

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
    sz::look_up_table_t invert_case = sz::look_up_table_t::identity();
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
        sz::string_t email = name | "@" | domain;
        verify(email == "ash@mail");
    }

    // Range members slice the string they were called on, so offsets stay inside its own buffer.
    {
        str text = "hello brave new world";
        verify(offsets_within(text, text.utf8_wordbreaks()) ==
                   (std::vector<std::ptrdiff_t> {0, 5, 6, 11, 12, 15, 16}) &&
               "utf8_wordbreaks segment landed outside the caller's own buffer");
        verify(offsets_within(text, text.utf8_split_whitespaces()) == (std::vector<std::ptrdiff_t> {0, 6, 12, 16}) &&
               "utf8_split_whitespaces token landed outside the caller's own buffer");
        verify(offsets_within(text, text.utf8_split_delimiters()) == (std::vector<std::ptrdiff_t> {0, 6, 12, 16}) &&
               "utf8_split_delimiters field landed outside the caller's own buffer");
        str sso = "a b c";
        verify(offsets_within(sso, sso.utf8_split_whitespaces()) == (std::vector<std::ptrdiff_t> {0, 2, 4}) &&
               "utf8_split_whitespaces token landed outside the small-string-optimized buffer");
    }

    // Randomization.
    verify(str::random(0).empty());
    verify(str::random(4).size() == 4);
    verify(str::random(4, 42).size() == 4);
}

#pragma endregion Extensions

/**
 *  @brief The lazy search ranges and their inverses - @c find_all, @c rfind_all, @c split,
 *      @c rsplit, @c partition.
 *
 *  Not a template over the string type, unlike its neighbours: these cases deliberately mix owning
 *  strings, borrowed views and literals in one expression, because what they pin is how the range
 *  holds its operands. A haystack passed as an lvalue is borrowed, so a match's `data()` must land
 *  inside the caller's own buffer and not inside a private copy - which under the small-string
 *  optimization would still produce plausible offsets. A needle, by contrast, is copied into the
 *  matcher, so a temporary one may outlive the expression.
 */
void test_extensions_ranges_unit() {
    // Searching for a set of characters
    verify(sz::string_view_t("a").find_first_of("az") == 0);
    verify(sz::string_view_t("a").find_last_of("az") == 0);
    verify(sz::string_view_t("a").find_first_of("xz") == sz::string_view_t::npos);
    verify(sz::string_view_t("a").find_last_of("xz") == sz::string_view_t::npos);

    verify(sz::string_view_t("a").find_first_not_of("xz") == 0);
    verify(sz::string_view_t("a").find_last_not_of("xz") == 0);
    verify(sz::string_view_t("a").find_first_not_of("az") == sz::string_view_t::npos);
    verify(sz::string_view_t("a").find_last_not_of("az") == sz::string_view_t::npos);

    verify(sz::string_view_t("aXbYaXbY").find_first_of("XY") == 1);
    verify(sz::string_view_t("axbYaxbY").find_first_of("Y") == 3);
    verify(sz::string_view_t("YbXaYbXa").find_last_of("XY") == 6);
    verify(sz::string_view_t("YbxaYbxa").find_last_of("Y") == 4);
    verify(sz::string_view_t(sz::base64(), sizeof(sz::base64())).find_first_of("_") == sz::string_view_t::npos);
    verify(sz::string_view_t(sz::base64(), sizeof(sz::base64())).find_first_of("+") == 62);
    verify(sz::string_view_t(sz::ascii_printables(), sizeof(sz::ascii_printables())).find_first_of("~") !=
           sz::string_view_t::npos);

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

    verify(""_sv.find_all(".", sz::include_overlaps_t {}).size() == 0);
    verify(""_sv.find_all(".", sz::exclude_overlaps_t {}).size() == 0);
    verify("."_sv.find_all(".", sz::include_overlaps_t {}).size() == 1);
    verify("."_sv.find_all(".", sz::exclude_overlaps_t {}).size() == 1);
    verify(".."_sv.find_all(".", sz::include_overlaps_t {}).size() == 2);
    verify(".."_sv.find_all(".", sz::exclude_overlaps_t {}).size() == 2);
    verify(""_sv.rfind_all(".", sz::include_overlaps_t {}).size() == 0);
    verify(""_sv.rfind_all(".", sz::exclude_overlaps_t {}).size() == 0);
    verify("."_sv.rfind_all(".", sz::include_overlaps_t {}).size() == 1);
    verify("."_sv.rfind_all(".", sz::exclude_overlaps_t {}).size() == 1);
    verify(".."_sv.rfind_all(".", sz::include_overlaps_t {}).size() == 2);
    verify(".."_sv.rfind_all(".", sz::exclude_overlaps_t {}).size() == 2);

    // An empty needle matches at every offset from 0 to `size()` inclusive, in either direction.
    verify("abc"_sv.find_all("").size() == 4);
    verify("abc"_sv.find_all("", sz::exclude_overlaps_t {}).size() == 4);
    verify("abc"_sv.rfind_all("").size() == 4);
    verify(""_sv.find_all("").size() == 1);
    verify(sz::find_all(sz::string_t("abc"), "").size() == 4);

    verify("a.b.c.d"_sv.find_all(".").size() == 3);
    verify("a.,b.,c.,d"_sv.find_all(".,").size() == 3);
    verify("a.,b.,c.,d"_sv.rfind_all(".,").size() == 3);
    verify("a.b,c.d"_sv.find_all(".,"_bs).size() == 3);
    verify("a...b...c"_sv.rfind_all("..").size() == 4);
    verify("a...b...c"_sv.rfind_all("..", sz::include_overlaps_t {}).size() == 4);
    verify("a...b...c"_sv.rfind_all("..", sz::exclude_overlaps_t {}).size() == 2);

    let_verify(auto finds = "a.b.c"_sv.find_all("abcd"_bs).template to<std::vector<std::string>>(),
               finds.size() == 3 && finds[0] == "a");
    let_verify(auto rfinds = "a.b.c"_sv.rfind_all("abcd"_bs).template to<std::vector<std::string>>(),
               rfinds.size() == 3 && rfinds[0] == "c");

    // Test propagating strings and their non-owning views into temporary ranges and iterators
    verify(sz::find_all("abc"_sv, "b"_sv).size() == 1);
    verify(sz::find_all("hello"_sv, "l"_sv).size() == 2);
    verify(sz::rfind_all("abc"_sv, "b"_sv).size() == 1);

    {
        sz::string_t h("abc"), n("b");
        verify(sz::find_all(h, n).size() == 1);
    }
    {
        sz::string_t h("hello"), n("l");
        verify(sz::find_all(h, n).size() == 2);
    }
    {
        sz::string_t h("abc"), n("b");
        verify(sz::rfind_all(h, n).size() == 1);
    }

    verify(sz::find_all(sz::string_t("abc"), sz::string_t("b")).size() == 1);
    verify(sz::find_all(sz::string_t("hello"), sz::string_t("l")).size() == 2);
    verify(sz::rfind_all(sz::string_t("abc"), sz::string_t("b")).size() == 1);

    // Lvalue haystacks are borrowed, so slices land inside the caller's own buffer. A copied
    // haystack would offset into a private copy - and under SSO those offsets look plausible.
    {
        sz::string_t haystack("hello world, hello cpp");
        sz::string_t sso("a b a");
        verify(offsets_within(haystack, sz::find_all(haystack, "hello")) == (std::vector<std::ptrdiff_t> {0, 13}) &&
               "Match offsets did not land inside the borrowed lvalue haystack's own buffer");
        verify(offsets_within(sso, sz::find_all(sso, "a")) == (std::vector<std::ptrdiff_t> {0, 4}) &&
               "Match offsets did not land inside the small-string-optimized haystack's own buffer");
    }

    // Needles are copied into the matcher, so a temporary one outlives the expression that built it.
    verify(sz::find_all(sz::string_t("hello world, hello cpp"), sz::string_t("hello")).size() == 2);

    // Haystack and needle need not share a type - literals, views, and owning strings mix.
    {
        sz::string_t owning("a-b-c");
        sz::string_view_t view("a-b-c");
        sz::string_t needle("-");
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
    verify("a.b"_sv.split("").size() == 1 && *"a.b"_sv.split("").begin() == "a.b"); // Empty separators never split
    verify("a.b"_sv.rsplit("").size() == 1);
    verify(sz::string_view_t {}.split(",").size() == 1 && sz::string_view_t {}.rsplit(",").size() == 1);

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

/** Tests copy constructor and copy-assignment constructor of @c sz::string_t. */
void test_string_constructors_unit() {
    std::string alphabet {sz::ascii_printables(), sizeof(sz::ascii_printables())};
    std::vector<sz::string_t> strings;
    for (std::size_t alphabet_slice = 0; alphabet_slice != alphabet.size(); ++alphabet_slice)
        strings.push_back(alphabet.substr(0, alphabet_slice));
    std::vector<sz::string_t> copies {strings};
    verify(copies.size() == strings.size());
    for (size_t i = 0; i < copies.size(); ++i) {
        verify(copies[i].size() == strings[i].size() && "Copy-constructed string has the wrong length at index i");
        verify(copies[i] == strings[i] && "Copy-constructed string diverged from its source at index i");
        for (size_t j = 0; j < strings[i].size(); j++)
            verify(copies[i][j] == strings[i][j] && "Copy-constructed string mismatched a byte at index i, j");
    }
    std::vector<sz::string_t> assignments = strings;
    for (size_t i = 0; i < assignments.size(); ++i) {
        verify(assignments[i].size() == strings[i].size() && "Copy-assigned string has the wrong length at index i");
        verify(assignments[i] == strings[i] && "Copy-assigned string diverged from its source at index i");
        for (size_t j = 0; j < strings[i].size(); j++)
            verify(assignments[i][j] == strings[i][j] && "Copy-assigned string mismatched a byte at index i, j");
    }
    verify(std::equal(strings.begin(), strings.end(), copies.begin()));
    verify(std::equal(strings.begin(), strings.end(), assignments.begin()));
}

/** Validates that shrinking @c reserve calls are harmless no-ops, just like in the STL, and that
 *  every reallocation carries the terminator. Regression tests: shrinking used to overflow the heap
 *  buffer in release builds, and growing or fitting left the new buffer unterminated. */
void test_string_reserve_unit() {
    // C API: grow, shrink, then fit - the buffer, length, contents and terminator stay intact.
    {
        // Fresh blocks arrive full of noise, so a terminator never copied cannot read as one.
        sz_allocator_t allocator;
        allocator.allocate = +[](sz_size_t length, void *, sz_stream_t) -> void * {
            void *const block = std::malloc(length);
            if (block) std::memset(block, '#', length);
            return block;
        };
        allocator.free = +[](void *block, sz_size_t, void *, sz_stream_t) { std::free(block); };
        allocator.handle = nullptr;

        sz_string_t str;
        sz_ptr_t start = sz_string_init_length(&str, 100, &allocator);
        verify(start != nullptr);
        std::memset(start, 'a', 100);

        sz_ptr_t grown = sz_string_reserve(&str, 200, &allocator);
        verify(grown != nullptr);
        verify(sz_string_length(&str) == 100);
        verify(grown[100] == '\0' && "Growing left the new buffer unterminated");

        // Shrinking must be a no-op: same buffer, same length, same contents.
        sz_ptr_t shrunk = sz_string_reserve(&str, 50, &allocator);
        verify(shrunk == grown);
        verify(sz_string_length(&str) == 100);
        for (sz_size_t i = 0; i != 100; ++i) verify(shrunk[i] == 'a');

        sz_ptr_t fitted = sz_string_shrink_to_fit(&str, &allocator);
        verify(fitted != nullptr);
        verify(sz_string_length(&str) == 100);
        verify(fitted[100] == '\0' && "Fitting left the new buffer unterminated");

        sz_string_free(&str, &allocator);
    }
    // C++ API: `sz::string_t::reserve` shrinking must match `std::string` behavior - keep the contents.
    {
        sz::string_t str(100, 'a');
        std::size_t const capacity_before = str.capacity();
        str.reserve(50);
        verify(str.size() == 100);
        verify(str.capacity() == capacity_before);
        verify(str == sz::string_t(100, 'a'));
    }
}

/** Checks the string class for leaks, and that blocks return to the allocator that granted them. */
void test_memory_stability_equivalence(test_context_t &context, std::size_t length) {
    using accounting_string_t = sz::basic_string<accounting_allocator_t>;
    static_assert(sizeof(accounting_string_t) == sizeof(sz::string_t) + sizeof(std::size_t *),
                  "Only a stateful allocator may widen the string");

    std::size_t const iterations = context.iterations(100);
    std::size_t live_bytes = 0, foreign_bytes = 0;
    accounting_allocator_t const allocator {&live_bytes}, foreign_allocator {&foreign_bytes};
    accounting_string_t base(allocator);
    for (std::size_t i = 0; i < length; ++i) base.push_back('c');
    verify(base.length() == length && "Base string has the wrong length after `push_back` construction");

    // Do copies leak?
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            accounting_string_t copy(base);
            verify(copy.get_allocator() == allocator);
            verify(copy.length() == length && "Copy-constructed string has the wrong length at iteration i");
            verify(copy == base && "Copy-constructed string diverged from `base` at iteration i");
        }
    });

    // How about assignments?
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            accounting_string_t copy(allocator);
            copy = base;
            verify(copy.length() == length && "Copy-assigned string has the wrong length at iteration i");
            verify(copy == base && "Copy-assigned string diverged from `base` at iteration i");
        }
    });

    // How about the move constructor?
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            accounting_string_t unique_item(base);
            verify(unique_item.length() == length && "Pre-move string has the wrong length at iteration i");
            verify(unique_item == base && "Pre-move string diverged from `base` at iteration i");
            accounting_string_t copy(std::move(unique_item));
            verify(copy.length() == length && "Move-constructed string has the wrong length at iteration i");
            verify(copy == base && "Move-constructed string diverged from `base` at iteration i");
        }
    });

    // And the move assignment operator with an empty target payload?
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            accounting_string_t unique_item(base);
            accounting_string_t copy(allocator);
            copy = std::move(unique_item);
            verify(copy.length() == length &&
                   "Move-assigned (empty target) string has the wrong length at iteration i");
            verify(copy == base && "Move-assigned (empty target) string diverged from `base` at iteration i");
        }
    });

    // And move assignment where the target had a payload?
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            accounting_string_t unique_item(base);
            accounting_string_t copy(allocator);
            for (std::size_t j = 0; j < 317; j++) copy.push_back('q');
            copy = std::move(unique_item);
            verify(copy.length() == length &&
                   "Move-assigned (occupied target) string has the wrong length at iteration i");
            verify(copy == base && "Move-assigned (occupied target) string diverged from `base` at iteration i");
        }
    });

    // Across unequal allocators, both assignments copy bytes and the target keeps its own allocator.
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            accounting_string_t unique_item(base);
            accounting_string_t moved(foreign_allocator), copied(foreign_allocator);
            for (std::size_t j = 0; j < 317; j++) moved.push_back('q');
            moved = std::move(unique_item);
            copied = base;
            verify(moved.get_allocator() == foreign_allocator && copied.get_allocator() == foreign_allocator);
            verify(moved == base && "Move-assigned (foreign target) string diverged from `base` at iteration i");
            verify(copied == base && "Copy-assigned (foreign target) string diverged from `base` at iteration i");
        }
        verify(foreign_bytes == 0);
    });

    // Now let's clear the base and check that we're back to zero
    base = accounting_string_t(allocator);
    verify(live_bytes == 0 && "Allocator counter did not return to zero after clearing");
}

/** Tests the correctness of the string class update methods, such as @c push_back and @c erase. */
void test_string_updates_equivalence(test_context_t &context, std::size_t repetitions) {
    // Compare STL and StringZilla strings append functionality.
    char const alphabet_chars[] = "abcdefghijklmnopqrstuvwxyz";
    std::mt19937 &generator = context.generator;
    for (std::size_t repetition = 0; repetition != repetitions; ++repetition) {
        std::string stl_string;
        sz::string_t sz_string;
        for (std::size_t length = 1; length != 200; ++length) {
            char c = alphabet_chars[generator() % 26];
            stl_string.push_back(c);
            sz_string.push_back(c);
            verify(sz::string_view_t(stl_string) == sz::string_view_t(sz_string) &&
                   "sz::string_t diverged from std::string after `push_back`");
        }

        // Compare STL and StringZilla strings erase functionality.
        while (stl_string.length()) {
            std::size_t offset_to_erase = generator() % stl_string.length();
            std::size_t chars_to_erase = generator() % (stl_string.length() - offset_to_erase) + 1;
            stl_string.erase(offset_to_erase, chars_to_erase);
            sz_string.erase(offset_to_erase, chars_to_erase);
            verify(sz::string_view_t(stl_string) == sz::string_view_t(sz_string) &&
                   "sz::string_t diverged from std::string after `erase`");
        }
    }
}

#pragma endregion String Class

#pragma region Drivers

/** Adversarial safety driver: feeds zero-length, tiny, overlapping, and embedded-NUL inputs through
 *  the movement and lookup dispatch points, asserting that canary bytes guarding both sides of the
 *  destination remain intact and nothing crashes; the cross files do the same per capability. */
void test_memory_safety() {
    check_memory_safety_(memory_dispatched);
    check_lookup_safety_(lookup_dispatched);
}

/** Drives the movement and lookup differential tests of the dispatch points against serial. */
void test_memory_all(test_context_t &context) {
    check_memory_equivalence_(context, memory_dispatched);
    check_lookup_equivalence_(context, lookup_dispatched);
}

#pragma endregion Drivers

/** Explicit template instantiations for the entry points invoked from @c main(). */
template void test_ascii_unit<sz::string_t>();
template void test_ascii_unit<sz::string_view_t>();
template void test_stl_reads_unit<std::string_view>();
template void test_stl_reads_unit<std::string>();
template void test_stl_reads_unit<sz::string_view_t>();
template void test_stl_reads_unit<sz::string_t>();
template void test_stl_updates_unit<std::string>();
template void test_stl_updates_unit<sz::string_t>();
template void test_extensions_reads_unit<sz::string_view_t>();
template void test_extensions_reads_unit<sz::string_t>();

} // namespace ashvardanian::stringzilla::test
