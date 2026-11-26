/**
 *  @file test/string.cpp
 *  @author Ash Vardanian
 *  @date December 21, 2023
 *  @brief Tests for the string class and the utilities beneath it.
 *
 *  Covers arithmetic and struct plumbing, ASCII utilities, string collections, STL compatibility,
 *  conversions and the extensions beyond the STL.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#undef STRINGZILLA_DEBUG
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
#include <iterator>      // `std::distance`
#include <map>           // `std::map`
#include <memory>        // `std::allocator`
#include <numeric>       // `std::accumulate`
#include <random>        // `std::random_device`
#include <span>          // `std::span`
#include <sstream>       // `std::ostringstream`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include <string>      // Baseline
#include <string_view> // Baseline

#include <fmt/format.h>

#include "cross.hpp"   // `check_strings_unit_`, `strings_backend_t`
#include "harness.hpp" // `randomize_string`, `case_stream_key`

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
    expect_equal(first[mismatch_position], second[mismatch_position]);
}

/** The owning string @c sz::string_t::make builds from @p content, a view or a lazy concatenation,
 *  failing the test when the factory reports an error. */
template <typename content_type_>
sz::string_t owned_(content_type_ const &content) {
    auto [made, status] = sz::string_t::make(content);
    expect_true(sz::succeeded(status));
    return std::move(made);
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

/** Stateful allocator charging each byte to its counter, so two over distinct counters differ.
 *  A string adopts it on move assignment, the one way a stateful allocator moves without a copy. */
struct accounting_allocator_t {
    using value_type = char;
    using propagate_on_container_move_assignment = std::true_type;
    std::size_t *live_bytes;

    explicit accounting_allocator_t(std::size_t *live_bytes) noexcept : live_bytes(live_bytes) {}

    char *allocate(std::size_t count) {
        *live_bytes += count;
        return std::allocator<char> {}.allocate(count);
    }

    void deallocate(char *block, std::size_t count) {
        expect_true(count <= *live_bytes, "Deallocated more bytes than were tracked as allocated");
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
    expect_equal(live_bytes, before, "Callback leaked or double-freed tracked allocator bytes");
}

/** The strings dispatch points, in the shape of their kernels. */
static strings_backend_t const strings_dispatched {"dispatched",
                                                   cpu_best<sz_strings_to_tape_best>,
                                                   cpu_best<sz_strings_to_rows_best>,
                                                   cpu_best<sz_strings_escape_best>,
                                                   cpu_best<sz_strings_unescape_best>,
                                                   cpu_best<sz_strings_lookup_best>,
                                                   cpu_best<sz_strings_normalize_best>,
                                                   cpu_best<sz_strings_split_best>,
                                                   cpu_best<sz_strings_split_lines_best>,
                                                   cpu_best<sz_strings_join_best>};

#pragma endregion Helpers

#pragma region Arithmetic

/** Several string processing operations rely on computing integer logarithms. Failures in such
 *  operations will result in wrong @c resize outcomes and heap corruption. */
void test_arithmetic_unit() {

    expect_equal(sz_u64_clz(0x0000000000000001ull), 63);
    expect_equal(sz_u64_clz(0x0000000000000002ull), 62);
    expect_equal(sz_u64_clz(0x0000000000000003ull), 62);
    expect_equal(sz_u64_clz(0x0000000000000004ull), 61);
    expect_equal(sz_u64_clz(0x0000000000000007ull), 61);
    expect_equal(sz_u64_clz(0x8000000000000001ull), 0);
    expect_equal(sz_u64_clz(0xffffffffffffffffull), 0);
    expect_equal(sz_u64_clz(0x4000000000000000ull), 1);

    expect_equal(sz_size_log2i_nonzero(1), 0);
    expect_equal(sz_size_log2i_nonzero(2), 1);
    expect_equal(sz_size_log2i_nonzero(3), 1);

    expect_equal(sz_size_log2i_nonzero(4), 2);
    expect_equal(sz_size_log2i_nonzero(5), 2);
    expect_equal(sz_size_log2i_nonzero(7), 2);

    expect_equal(sz_size_log2i_nonzero(8), 3);
    expect_equal(sz_size_log2i_nonzero(9), 3);

    expect_equal(sz_size_bit_ceil(0), 0);
    expect_equal(sz_size_bit_ceil(1), 1);

    expect_equal(sz_size_bit_ceil(2), 2);
    expect_equal(sz_size_bit_ceil(3), 4);
    expect_equal(sz_size_bit_ceil(4), 4);

    expect_equal(sz_size_bit_ceil(77), 128);
    expect_equal(sz_size_bit_ceil(127), 128);
    expect_equal(sz_size_bit_ceil(128), 128);

    expect_equal(sz_size_bit_ceil(1000000ull), 1ull << 20);
    expect_equal(sz_size_bit_ceil(2000000ull), 1ull << 21);
    expect_equal(sz_size_bit_ceil(4000000ull), 1ull << 22);
    expect_equal(sz_size_bit_ceil(8000000ull), 1ull << 23);

    expect_equal(sz_size_bit_ceil(16000000ull), 1ull << 24);
    expect_equal(sz_size_bit_ceil(32000000ull), 1ull << 25);
    expect_equal(sz_size_bit_ceil(64000000ull), 1ull << 26);

    expect_equal(sz_size_bit_ceil(128000000ull), 1ull << 27);
    expect_equal(sz_size_bit_ceil(256000000ull), 1ull << 28);
    expect_equal(sz_size_bit_ceil(512000000ull), 1ull << 29);

    expect_equal(sz_size_bit_ceil(1000000000ull), 1ull << 30);
    expect_equal(sz_size_bit_ceil(2000000000ull), 1ull << 31);

#if STRINGZILLA_ARCH_64BIT_
    expect_equal(sz_size_bit_ceil(4000000000ull), 1ull << 32);
    expect_equal(sz_size_bit_ceil(8000000000ull), 1ull << 33);
    expect_equal(sz_size_bit_ceil(16000000000ull), 1ull << 34);

    expect_equal(sz_size_bit_ceil((1ull << 62)), 1ull << 62);
    expect_equal(sz_size_bit_ceil((1ull << 62) + 1), 1ull << 63);
    expect_equal(sz_size_bit_ceil((1ull << 63)), 1ull << 63);
#endif
}

#pragma endregion Arithmetic

#pragma region Strings Views

/** Validates @c sz_strings_view_t construction and the views @c sz_strings_get_ reads back. */
void test_strings_view_unit() {
    auto string_at = [](sz_strings_view_t const &strings, sz_size_t index) {
        sz_string_view_t view {};
        sz_strings_get_(&strings, &index, 1, &view);
        return sz::string_view_t(view.start, view.length);
    };
    {
        sz_strings_view_t strings;
        sz_cptr_t const fruits[] = {"banana", "apple", "cherry"};
        sz_strings_from_null_terminated_strings(fruits, 3, &strings);
        expect_true(strings.layout == sz_strings_accessor_k && strings.count == 3);
        expect_equal(string_at(strings, 0), "banana"_sv);
        expect_equal(string_at(strings, 1), "apple"_sv);
        expect_equal(string_at(strings, 2), "cherry"_sv);
    }
    // Empty collections, empty members, and duplicates are all legal.
    {
        sz_strings_view_t strings;
        sz_cptr_t const fruits[] = {"", "apple", "apple", ""};
        sz_strings_from_null_terminated_strings(fruits, 0, &strings);
        expect_equal(strings.count, 0);
        sz_strings_from_null_terminated_strings(fruits, 4, &strings);
        expect_equal(strings.count, 4);
        expect_true(string_at(strings, 0).empty() && string_at(strings, 3).empty());
        expect_true(string_at(strings, 1) == "apple"_sv && string_at(strings, 2) == "apple"_sv);
    }
    // An array of views and a tape over the same strings read back alike, block after block.
    {
        std::vector<std::string> numbers;
        for (std::size_t index = 0; index != 100; ++index) numbers.push_back(std::to_string(index * 7919));
        std::vector<sz_string_view_t> views;
        std::string payload;
        std::vector<sz_u32_t> offsets {0};
        for (std::string const &number : numbers) {
            views.push_back({number.data(), number.size()});
            payload += number;
            offsets.push_back(static_cast<sz_u32_t>(payload.size()));
        }
        sz_strings_view_t from_views, from_tape;
        sz_strings_from_string_views(views.data(), views.size(), &from_views);
        from_tape.layout = sz_strings_u32_tape_k, from_tape.count = numbers.size();
        from_tape.data.u32_tape.data = payload.data(), from_tape.data.u32_tape.offsets = offsets.data();
        sz_string_view_t views_block[64], tape_block[64];
        for (sz_size_t block_start = 0; block_start < numbers.size(); block_start += 64) {
            sz_size_t const block_count = sz_strings_get_block_(&from_views, block_start, views_block);
            expect_equal(block_count, sz_strings_get_block_(&from_tape, block_start, tape_block));
            for (sz_size_t position = 0; position != block_count; ++position) {
                sz::string_view_t const expected = numbers[block_start + position];
                expect_equal(sz::string_view_t(views_block[position].start, views_block[position].length), expected);
                expect_equal(sz::string_view_t(tape_block[position].start, tape_block[position].length), expected);
            }
        }
    }
}

/** Pins @c sz::strings_view_t in each layout and the @c sz::strings its conversions return,
 *  which own their blocks alone and leave nothing behind on failure. */
void test_strings_columns_unit() {
    char const payload[] = "alphabetagamma";
    sz_u32_t const narrow_offsets[] = {0, 5, 9, 14};

    // Arrow `String` and `LargeString` columns read alike, and a slice keeps its parent's offsets
    // until a gather packs it from zero.
    {
        sz_u64_t const wide_offsets[] = {0, 5, 9, 14};
        sz::strings_view_t const narrow(payload, narrow_offsets, 3), wide(payload, wide_offsets, 3);
        expect_true(narrow.size() == 3 && narrow[1] == "beta"_sv && wide[2] == "gamma"_sv);
        expect_true(!narrow.empty() && sz::strings_view_t(std::span<sz::string_view_t const> {}).empty() &&
                    sz::strings_t {}.empty());
        expect_true(std::equal(narrow.begin(), narrow.end(), wide.begin(), wide.end()));
        sz_u64_t hashes[3];
        expect_true(wide.hashes(hashes) == sz::status_t::success_k && hashes[2] == wide[2].hash());
        char upper_table[256];
        sz_strings_lookup_init_upper(upper_table);
        auto const [upper, lookup_status] = wide.lookup(sz::look_up_table_t(upper_table));
        expect_true(lookup_status == sz::status_t::success_k && upper[0] == "ALPHA"_sv);

        sz::strings_view_t const slice(payload, narrow_offsets + 1, 2);
        expect_true(slice[0] == "beta"_sv && slice[1] == "gamma"_sv);
        auto const [gathered, gather_status] = slice.gather();
        expect_true(gather_status == sz::status_t::success_k && gathered.size() == 2);
        expect_true(gathered.raw.data.u32_tape.offsets[0] == 0 && gathered[1] == "gamma"_sv);
    }

    // An accessor over views reads the same way, and JSON escaping round-trips through unescaping.
    {
        std::array<sz::string_view_t, 3> const views {"Alpha"_sv, ""_sv, "a\"b\n"_sv};
        sz::strings_view_t const strings(views);
        expect_true(strings.raw.layout == sz_strings_accessor_k && strings[2] == "a\"b\n"_sv);
        std::size_t positions[3];
        expect_equal(strings.find("a"_sv, positions), sz::status_t::success_k);
        expect_true(positions[0] == 4 && positions[1] == STRINGZILLA_SIZE_MAX && positions[2] == 0);
        expect_equal(strings.find("a"_sv, std::span(positions).first(2)), sz::status_t::unexpected_dimensions_k);
        auto const [folded, normalize_status] = strings.normalize(sz_normal_form_nfkc_casefold_k);
        expect_true(normalize_status == sz::status_t::success_k && folded[0] == "alpha"_sv);

        auto const [escaped, escape_status] = strings.escape(sz_escape_json_k);
        expect_true(escape_status == sz::status_t::success_k && escaped[2] == "a\\\"b\\n"_sv);
        std::size_t malformed_index = 0;
        auto const [unescaped, unescape_status] = escaped.view().unescape(sz_escape_json_k, malformed_index);
        expect_equal(unescape_status, sz::status_t::success_k);
        expect_true(std::equal(unescaped.begin(), unescaped.end(), views.begin(), views.end()));
    }

    // Splits cut a text like `bytes.split` and `str.splitlines`, and joins frame the columns back
    // into one string, alone or side by side as CSV, as long as all the columns are equally long.
    {
        auto const [each, each_status] = sz::strings_t::split(",a,,b"_sv, ","_bs, sz_split_each_k);
        expect_true(each_status == sz::status_t::success_k && each.size() == 4 && each[0].empty() && each[3] == "b"_sv);
        auto const [runs, runs_status] = sz::strings_t::split(" a \t b "_sv, " \t"_bs, sz_split_runs_k);
        expect_true(runs_status == sz::status_t::success_k && runs.size() == 2 && runs[1] == "b"_sv);
        auto const [lines, lines_status] = sz::strings_t::split_lines("one\r\ntwo\xC2\x85three\n"_sv);
        expect_true(lines_status == sz::status_t::success_k && lines.size() == 3 && lines[2] == "three"_sv);

        auto const [joined, join_status] = each.view().join("|"_sv);
        expect_true(join_status == sz::status_t::success_k && joined == "|a||b"_sv);
        sz_strings_view_t const columns[] {lines.view().raw, sz::strings_view_t(payload, narrow_offsets, 3).raw};
        sz_string_view_t const frames[] {{"\"", 1}, {"\",", 2}, {"", 0}};
        auto const [csv, csv_status] = sz::strings_view_t::join(columns, frames, "\n"_sv);
        expect_true(csv_status == sz::status_t::success_k && csv == "\"one\",alpha\n\"two\",beta\n\"three\",gamma"_sv);
        sz_strings_view_t const ragged[] {lines.view().raw, each.view().raw};
        auto const [rejected, rejected_status] = sz::strings_view_t::join(ragged, frames, "\n"_sv);
        expect_true(rejected_status == sz::status_t::unexpected_dimensions_k && rejected.empty());
    }

    // Counting runes validates every string, naming the first ill-formed one.
    {
        std::array<sz::string_view_t, 3> const texts {"héllo"_sv, "ok"_sv, "\xC3("_sv};
        std::size_t runes[3], malformed_index = 0;
        expect_equal(sz::strings_view_t(std::span(texts).first(2)).utf8_count(runes, malformed_index),
                     sz::status_t::success_k);
        expect_true(runes[0] == 5 && runes[1] == 2 && malformed_index == 2);
        expect_equal(sz::strings_view_t(texts).utf8_count({}, malformed_index), sz::status_t::invalid_utf8_k);
        expect_equal(malformed_index, 2);
    }

    // A malformed string fails the whole column with nothing allocated, and moves hand blocks over
    // without copies, so every block returns exactly once.
    {
        struct allocation_state_t {
            sz_allocator_t heap {};
            std::size_t live_bytes = 0;
        } state;
        expect_equal(sz_allocator_init_heap(&state.heap), sz_success_k);
        sz_allocator_t counting {};
        counting.handle = &state;
        counting.allocate = [](sz_size_t bytes, void *handle, sz_stream_t stream) -> void * {
            auto &state = *static_cast<allocation_state_t *>(handle);
            void *block = state.heap.allocate(bytes, state.heap.handle, stream);
            if (block) state.live_bytes += bytes;
            return block;
        };
        counting.free = [](void *block, sz_size_t bytes, void *handle, sz_stream_t stream) {
            auto &state = *static_cast<allocation_state_t *>(handle);
            expect_true(bytes <= state.live_bytes);
            state.live_bytes -= bytes;
            state.heap.free(block, bytes, state.heap.handle, stream);
        };
        sz::unified_alloc<char> const allocator(counting);
        {
            std::array<sz::string_view_t, 2> const hex {"6869"_sv, "6g"_sv};
            std::size_t malformed_index = 0;
            sz::strings_view_t const escaped(hex);
            auto const [decoded, decode_status] = escaped.unescape(sz_escape_hex_k, malformed_index, allocator);
            expect_true(decode_status == sz::status_t::invalid_utf8_k && malformed_index == 1);
            expect_true(decoded.size() == 0 && decoded.raw.data.u32_tape.offsets == nullptr && state.live_bytes == 0);

            sz::strings_view_t const column(payload, narrow_offsets, 3);
            auto [first, first_status] = column.gather(allocator);
            auto [second, second_status] = column.gather(allocator);
            expect_true(first_status == sz::status_t::success_k && second_status == sz::status_t::success_k);
            sz_cptr_t const first_payload = first.raw.data.u32_tape.data;
            sz::strings<sz::unified_alloc<char>> moved(std::move(first));
            expect_true(first.size() == 0 && first.raw.data.u32_tape.data == nullptr);
            expect_true(moved.size() == 3 && moved.raw.data.u32_tape.data == first_payload);
            second = std::move(moved);
            expect_true(moved.size() == 0 && second.raw.data.u32_tape.data == first_payload);
            auto &same = second;
            second = std::move(same);
            expect_true(second.size() == 3 && second[2] == "gamma"_sv);
        }
        expect_equal(state.live_bytes, 0);
    }
}

#pragma endregion Strings Views

#pragma region Allocator

/** Validates @c sz_allocator_t and related construction utilities. */
void test_allocator_unit() {
    // Our behavior for `malloc(0)` is to return a NULL pointer,
    // while the standard is implementation-defined.
    {
        sz_allocator_t allocator;
        expect_equal(sz_allocator_init_heap(&allocator), sz_success_k);
        expect_equal(allocator.allocate(0, allocator.handle, nullptr), nullptr);
    }

    // Non-NULL allocation
    {
        sz_allocator_t allocator;
        expect_equal(sz_allocator_init_heap(&allocator), sz_success_k);
        void *byte = allocator.allocate(1, allocator.handle, nullptr);
        expect_true(byte != nullptr, "Default allocator returned NULL for a non-zero-length allocation");
        allocator.free(byte, 1, allocator.handle, nullptr);
    }

    {
        alignas(sz_size_t) char buffer[1025];
        sz_allocator_t allocator;
        expect_equal(sz_allocator_init_arena(&allocator, buffer + 1, sizeof(buffer) - 1, 64), sz_success_k);
        expect_equal(allocator.allocate(STRINGZILLA_SIZE_MAX, allocator.handle, nullptr), nullptr);
        void *byte = allocator.allocate(1, allocator.handle, nullptr);
        expect_true(byte != nullptr && reinterpret_cast<sz_size_t>(byte) % 64 == 0);
        void *next = allocator.allocate(1, allocator.handle, nullptr);
        expect_true(next != nullptr && reinterpret_cast<sz_size_t>(next) % 64 == 0 && next != byte);
        allocator.free(byte, 1, allocator.handle, nullptr);
        sz_allocator_t const original = allocator;
        sz_arena_t_ const *metadata = static_cast<sz_arena_t_ const *>(allocator.handle);
        sz_size_t const consumed = metadata->consumed;
        expect_equal(sz_allocator_init_arena(&allocator, buffer + 1, sizeof(buffer) - 1, 0),
                     sz_unexpected_dimensions_k);
        expect_equal(sz_allocator_init_arena(&allocator, buffer + 1, sizeof(buffer) - 1, 3),
                     sz_unexpected_dimensions_k);
        expect_equal(sz_allocator_init_arena(&allocator, buffer + 1, sizeof(sz_size_t), 64), sz_bad_alloc_k);
        expect_equal(sz_allocator_init_arena(&allocator, nullptr, sizeof(buffer), 64), sz_bad_alloc_k);
        expect_equal(sz_allocator_init_arena(&allocator, buffer, STRINGZILLA_SIZE_MAX, 64), sz_overflow_risk_k);
        expect_true(allocator.allocate == original.allocate && allocator.free == original.free &&
                    allocator.handle == original.handle);
        expect_equal(metadata->consumed, consumed);
        expect_equal(sz_allocator_init_arena(&allocator, buffer + 1, sizeof(buffer) - 1, 1), sz_success_k);
        expect_true(allocator.allocate(1, allocator.handle, nullptr) != nullptr);
    }
    {
        sz_allocator_t allocator;
        expect_equal(sz_allocator_init_heap(&allocator), sz_success_k);
        sz_allocator_t const original = allocator;
        expect_equal(sz_allocator_init_device_best(&allocator, sz_cap_serial_k), sz_missing_kernel_k);
        expect_equal(sz_allocator_init_pinned_best(&allocator, sz_cap_serial_k), sz_missing_kernel_k);
        expect_true(allocator.allocate == original.allocate && allocator.free == original.free &&
                    allocator.handle == original.handle);
    }
    {
        auto const [allocator, status] = sz::unified_alloc<char, sz_cap_serial_k>::make();
        expect_equal(status, sz::status_t::success_k);
        char *empty = allocator.allocate(0);
        expect_true(empty != nullptr);
        allocator.deallocate(empty, 0);
        sz::unified_alloc<sz_size_t, sz_cap_serial_k> words(allocator);
        expect_equal(words.allocate(STRINGZILLA_SIZE_MAX), nullptr);
    }
}

#pragma endregion Allocator

#pragma region Byteset

/** Validates @c sz_byteset_t and related construction utilities. */
void test_byteset_unit() {
    sz_byteset_t s;
    sz_byteset_init(&s);
    expect_equal(sz_byteset_contains(&s, 'a'), sz_false_k);
    sz_byteset_add(&s, 'a');
    expect_equal(sz_byteset_contains(&s, 'a'), sz_true_k);
    sz_byteset_add(&s, 'z');
    expect_equal(sz_byteset_contains(&s, 'z'), sz_true_k);
    sz_byteset_invert(&s);
    expect_equal(sz_byteset_contains(&s, 'a'), sz_false_k);
    expect_equal(sz_byteset_contains(&s, 'z'), sz_false_k);
    expect_equal(sz_byteset_contains(&s, 'b'), sz_true_k);
    sz_byteset_init_ascii(&s);
    expect_equal(sz_byteset_contains(&s, 'A'), sz_true_k);
}

/** Known-answer coverage for the ASCII classification methods of @c sz::string_t and
 *  @c sz::string_view_t, such as @c is_alpha, @c is_digit and @c contains_only. */
template <typename string_type>
void test_ascii_unit() {

    using str = string_type;

    expect_equal("aaa"_bs.size(), 1ull);
    expect_equal("\0\0"_bs.size(), 1ull);
    expect_equal("abc"_bs.size(), 3ull);
    expect_equal("a\0bc"_bs.size(), 4ull);

    expect_true(!"abc"_bs.contains('\0'));
    expect_true(str("bca").contains_only("abc"_bs));

    expect_true(!str("").is_alpha());
    expect_true(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ").is_alpha());
    expect_true(!str("abc9").is_alpha());

    expect_true(!str("").is_alnum());
    expect_true(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789").is_alnum());
    expect_true(!str("abc!").is_alnum());

    expect_true(str("").is_ascii());
    expect_true(str("\x00x7F").is_ascii());
    expect_true(!str("abc123🔥").is_ascii());

    expect_true(!str("").is_digit());
    expect_true(str("0123456789").is_digit());
    expect_true(!str("012a").is_digit());

    expect_true(!str("").is_lower());
    expect_true(str("abcdefghijklmnopqrstuvwxyz").is_lower());
    expect_true(!str("abcA").is_lower());
    expect_true(!str("abc\n").is_lower());

    expect_true(!str("").is_space());
    expect_true(str(" \t\n\r\f\v").is_space());
    expect_true(!str(" \t\r\na").is_space());

    expect_true(!str("").is_upper());
    expect_true(str("ABCDEFGHIJKLMNOPQRSTUVWXYZ").is_upper());
    expect_true(!str("ABCa").is_upper());

    expect_true(str("").is_printable());
    expect_true(str("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&" // `&*` is no dereference
                    "*()_+")
                    .is_printable());
    expect_true(!str("012🔥").is_printable());

    expect_true(str("").contains_only("abc"_bs));
    expect_true(str("abc").contains_only("abc"_bs));
    expect_true(!str("abcd").contains_only("abc"_bs));
}

#pragma endregion Byteset

#pragma region Strings

/**
 *  @brief Known-answer tests for the strings dispatch points and the C++ string wrappers.
 *
 *  A regression that the serial-vs-SIMD agreement tests would miss - because both share a wrong
 *  constant - is still caught against an external ground truth; the kernels of each capability face
 *  the same vectors in `cross_<arch>.cpp`.
 */
void test_strings_unit() {
    check_strings_unit_(strings_dispatched);

    // The case tables follow the Latin-1 letter rule, keeping × and ÷ and ÿ as is.
    {
        char upper_table[256], lower_table[256];
        sz_strings_lookup_init_upper(upper_table);
        sz_strings_lookup_init_lower(lower_table);
        for (int byte = 0; byte != 256; ++byte) {
            bool const is_lower_letter = (byte >= 'a' && byte <= 'z') || (byte >= 0xE0 && byte <= 0xFE && byte != 0xF7);
            bool const is_upper_letter = (byte >= 'A' && byte <= 'Z') || (byte >= 0xC0 && byte <= 0xDE && byte != 0xD7);
            expect_equal((unsigned char)upper_table[byte], is_lower_letter ? byte - 32 : byte);
            expect_equal((unsigned char)lower_table[byte], is_upper_letter ? byte + 32 : byte);
        }
    }

    // C++ wrapper sanity: a couple of `sz::string_t` / `sz::string_view_t` known-answer reads alongside the C API.
    {
        sz::string_view_t const view = "Hello, World!"_sv;
        expect_equal(view.size(), 13u);
        expect_equal(view.substr(7, 5), "World"_sv);
        expect_true(view.front() == 'H' && view.back() == '!');

        sz::string_t const owned = owned_(view);
        expect_equal(owned.size(), 13u);
        expect_equal(owned, view);
        expect_true(owned_("apple").compare("banana") < 0);
    }

    // Embedded NUL must be preserved verbatim by a stored `sz::string_t`: the size is the full byte length, and
    // indexing past the interior NUL reaches the trailing bytes rather than stopping at the C-string boundary.
    {
        char const with_nul[] = {'a', 'b', '\0', 'c', 'd'};
        sz::string_t const owned = owned_(sz::string_view_t(with_nul, sizeof(with_nul)));
        expect_equal(owned.size(), sizeof(with_nul));    // Full length, NUL is a stored byte
        expect_equal(owned[2], '\0');                    // The interior NUL survives
        expect_true(owned[3] == 'c' && owned[4] == 'd'); // Indexing past the NUL works
        expect_equal(owned, sz::string_view_t(with_nul, sizeof(with_nul)));
    }
}

#pragma endregion Strings

#pragma region STL Reads

/** Invokes different C++ member methods of immutable strings to cover all STL APIs. This test
 *  guarantees API @b compatibility with STL @c std::basic_string template. */
template <typename string_type>
void test_stl_reads_unit() {

    using str = string_type;
    // StringZilla clamps positions past the end like Python's slices, where the STL throws.
    constexpr bool clamps = std::is_same_v<str, sz::string_view_t>;

    // Constructors.
    expect_true(str().empty());
    expect_equal(str().size(), 0);
    expect_true(str("").empty());
    expect_equal(str("").size(), 0);
    expect_equal(str("hello").size(), 5);
    expect_equal(str("hello", 4), "hell");

    // Element access.
    str const rest_text("rest");
    expect_equal(rest_text[0], 'r');
    expect_equal(str("rest").at(1), 'e');
    expect_equal(*str("rest").data(), 'r');
    expect_equal(str("front").front(), 'f');
    expect_equal(str("back").back(), 'k');

    // Iterators.
    expect_true(*str("begin").begin() == 'b' && *str("cbegin").cbegin() == 'c');
    expect_true(*str("rbegin").rbegin() == 'n' && *str("crbegin").crbegin() == 'n');
    expect_true(str("size").size() == 4 && str("length").length() == 6);

    // Slices... out-of-bounds exceptions are asymmetric!
    expect_equal(str("hello world").substr(0, 5), "hello");
    expect_equal(str("hello world").substr(6, 5), "world");
    expect_equal(str("hello world").substr(6), "world");
    expect_equal(str("hello world").substr(6, 100), "world"); // 106 is beyond the length of the string, but its OK
    if constexpr (clamps)
        expect_true(str("hello world").substr(100).empty() && str("hello world").substr(20, 5).empty());
    else {
        expect_throws<std::out_of_range>(
            [&] { return str("hello world").substr(100); }); // 100 is beyond the length of the string
        expect_throws<std::out_of_range>(
            [&] { return str("hello world").substr(20, 5); }); // 20 is beyond the length of the string
    }
#if defined(__GNUC__) && !defined(__NVCC__) // -1 casts to unsigned without warnings on GCC, but not NVCC
    if constexpr (clamps) expect_true(str("hello world").substr(-1, 5).empty());
    else expect_throws<std::out_of_range>([&] { return str("hello world").substr(-1, 5); });
    expect_equal(str("hello world").substr(0, -1), "hello world");
#endif

    // Character search in normal and reverse directions.
    expect_equal(str("hello").find('e'), 1);
    expect_equal(str("hello").find('e', 1), 1);
    expect_equal(str("hello").find('e', 2), str::npos);
    expect_equal(str("hello").rfind('l'), 3);
    expect_equal(str("hello").rfind('l', 2), 2);
    expect_equal(str("hello").rfind('l', 1), str::npos);

    // Substring search in normal and reverse directions.
    expect_equal(str().find(""), 0);
    expect_equal(str().rfind(""), 0);
    expect_equal(str("hello").find("ell"), 1);
    expect_equal(str("hello").find("ell", 1), 1);
    expect_equal(str("hello").find("ell", 2), str::npos);
    expect_equal(str("hello").find("el", 1), 1);
    expect_equal(str("hello").find("ell", 1, 2), 1);
    expect_equal(str("hello").rfind("l"), 3);
    expect_equal(str("hello").rfind("l", 2), 2);
    expect_equal(str("hello").rfind("l", 1), str::npos);

    // The second argument is the last possible value of the returned offset.
    expect_equal(str("hello").rfind("el", 1), 1);
    expect_equal(str("hello").rfind("ell", 1), 1);
    expect_equal(str("hello").rfind("ello", 1), 1);
    expect_equal(str("hello").rfind("ell", 1, 2), 1);

    // More complex queries.
    expect_equal(str("abbabbaaaaaa").find("aa"), 6);
    expect_equal(str("abbabbaaaaaa").find("ba"), 2);
    expect_equal(str("abbabbaaaaaa").find("bb"), 1);
    expect_equal(str("abbabbaaaaaa").find("bab"), 2);
    expect_equal(str("abbabbaaaaaa").find("babb"), 2);
    expect_equal(str("abbabbaaaaaa").find("babba"), 2);
    expect_equal(str("abcdabcd").substr(2, 4).find("abc"), str::npos);
    expect_equal(str("hello, world!").substr(0, 11).find("world"), str::npos);
    expect_equal(str("axabbcxcaaabbccc").find("aaabbccc"), 8);
    expect_equal(str("abcdabcdabc________").find("abcd"), 0);
    expect_equal(str("________abcdabcdabc").find("abcd"), 8);

    // Cover every SWAR case for unique string sequences.
    auto lowercase_alphabet = str("abcdefghijklmnopqrstuvwxyz");
    for (std::size_t one_byte_offset = 0; one_byte_offset + 1 <= lowercase_alphabet.size(); ++one_byte_offset)
        expect_equal(lowercase_alphabet.find(lowercase_alphabet.substr(one_byte_offset, 1)), one_byte_offset,
                     "1-byte SWAR needle matched at the wrong offset");
    for (std::size_t two_byte_offset = 0; two_byte_offset + 2 <= lowercase_alphabet.size(); ++two_byte_offset)
        expect_equal(lowercase_alphabet.find(lowercase_alphabet.substr(two_byte_offset, 2)), two_byte_offset,
                     "2-byte SWAR needle matched at the wrong offset");
    for (std::size_t four_byte_offset = 0; four_byte_offset + 4 <= lowercase_alphabet.size(); ++four_byte_offset)
        expect_equal(lowercase_alphabet.find(lowercase_alphabet.substr(four_byte_offset, 4)), four_byte_offset,
                     "4-byte SWAR needle matched at the wrong offset");
    for (std::size_t three_byte_offset = 0; three_byte_offset + 3 <= lowercase_alphabet.size(); ++three_byte_offset)
        expect_equal(lowercase_alphabet.find(lowercase_alphabet.substr(three_byte_offset, 3)), three_byte_offset,
                     "3-byte SWAR needle matched at the wrong offset");
    for (std::size_t five_byte_offset = 0; five_byte_offset + 5 <= lowercase_alphabet.size(); ++five_byte_offset)
        expect_equal(lowercase_alphabet.find(lowercase_alphabet.substr(five_byte_offset, 5)), five_byte_offset,
                     "5-byte SWAR needle matched at the wrong offset");

    // Simple repeating patterns - with one "almost match" before an actual match in each direction.
    expect_equal(str("_ab_abc_").find("abc"), 4);
    expect_equal(str("_abc_ab_").rfind("abc"), 1);
    expect_equal(str("_abc_abcd_").find("abcd"), 5);
    expect_equal(str("_abcd_abc_").rfind("abcd"), 1);
    expect_equal(str("_abcd_abcde_").find("abcde"), 6);
    expect_equal(str("_abcde_abcd_").rfind("abcde"), 1);
    expect_equal(str("_abcde_abcdef_").find("abcdef"), 7);
    expect_equal(str("_abcdef_abcde_").rfind("abcdef"), 1);
    expect_equal(str("_abcdef_abcdefg_").find("abcdefg"), 8);
    expect_equal(str("_abcdefg_abcdef_").rfind("abcdefg"), 1);

    // ! `rfind` and `find_last_of` are not consistent in meaning of their arguments.
    expect_equal(str("hello").find_first_of("le"), 1);
    expect_equal(str("hello").find_first_of("le", 1), 1);
    expect_equal(str("hello").find_last_of("le"), 3);
    expect_equal(str("hello").find_last_of("le", 2), 2);
    expect_equal(str("hello").find_first_not_of("hel"), 4);
    expect_equal(str("hello").find_first_not_of("hel", 1), 4);
    expect_equal(str("hello").find_last_not_of("hel"), 4);
    expect_equal(str("hello").find_last_not_of("hel", 4), 4);

    // Try longer strings to enforce SIMD.
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find('x'), 23);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find('X'), 49);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind('x'), 23);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind('X'), 49);

    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xy"), 23);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XY"), 49);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("yz"), 24);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("YZ"), 50);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xy"), 23);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XY"), 49);

    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xyz"), 23);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XYZ"), 49);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xyz"), 23);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XYZ"), 49);

    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xyzA"), 23);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XYZ0"), 49);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xyzA"), 23);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XYZ0"), 49);

    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_first_of("xyz"), 23);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_first_of("XYZ"), 49);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_last_of("xyz"), 25);
    expect_equal(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_last_of("XYZ"), 51);

    // Using single-byte non-ASCII values, e.g., À (0xC0), Æ (0xC6). The `\xFA`/`0` boundary is
    // load-bearing: a literal hex digit after `\xFA` would extend the escape, so keep it split.
    {
        char const *non_ascii_set = "abcdefgh\x01\xC6ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\xC0\xFA" //
                                    "0123456789+-";                                                        // 68 bytes
        expect_equal(str(non_ascii_set, 68).find_first_of("\xC6\xC7"), 9);
        expect_equal(str(non_ascii_set, 68).find_first_of("\xC0\xC1"), 54);
        expect_equal(str(non_ascii_set, 68).find_last_of("\xC6\xC7"), 9);
        expect_equal(str(non_ascii_set, 68).find_last_of("\xC0\xC1"), 54);
    }

    // Boundary conditions.
    expect_equal(str("hello").rfind("ell", str::npos), 1);
    expect_equal(str("hello").find_last_of("le", str::npos), 3);
    expect_equal(str("hello").find_first_of("ox", 4), 4);
    expect_equal(str("hello").find_first_of("ox", 5), str::npos);
    expect_equal(str("hello").find_last_of("ox", 4), 4);
    expect_equal(str("hello").find_last_of("ox", 5), 4);
    expect_equal(str("hello").find_first_of("hx", 0), 0);
    expect_equal(str("hello").find_last_of("hx", 0), 0);

    // More complex relative patterns
    expect_true(str("0123456789012345678901234567890123456789012345678901234567890123") <=
                str("0123456789012345678901234567890123456789012345678901234567890123"));
    expect_true(str("0123456789012345678901234567890123456789012345678901234567890123") <=
                str("0223456789012345678901234567890123456789012345678901234567890123"));
    expect_true(str("0123456789012345678901234567890123456789012345678901234567890123") <=
                str("0213456789012345678901234567890123456789012345678901234567890123"));
    expect_true(str("12341234") <= str("12341234"));
    expect_true(str("12341234") > str("12241224"));
    expect_true(str("12341234") < str("13241324"));
    expect_equal(str("0123456789012345678901234567890123456789012345678901234567890123"),
                 str("0123456789012345678901234567890123456789012345678901234567890123"));
    expect_true(str("0123456789012345678901234567890123456789012345678901234567890123") !=
                str("0223456789012345678901234567890123456789012345678901234567890123"));

    // Comparisons.
    expect_true(str("a") != str("b"));
    expect_true(str("a") < str("b"));
    expect_true(str("a") <= str("b"));
    expect_true(str("b") > str("a"));
    expect_true(str("b") >= str("a"));
    expect_true(str("a") < str("aa"));

    // Spaceship operator instead of conventional comparions.
    expect_equal(str("a") <=> str("b"), std::strong_ordering::less);
    expect_equal(str("b") <=> str("a"), std::strong_ordering::greater);
    expect_equal(str("b") <=> str("b"), std::strong_ordering::equal);
    expect_equal(str("a") <=> str("aa"), std::strong_ordering::less);

    // Compare with another `str`.
    expect_equal(str("test").compare(str("test")), 0);
    expect_true(str("apple").compare(str("banana")) < 0);
    expect_true(str("banana").compare(str("apple")) > 0);

    // Compare with a C-string.
    expect_equal(str("test").compare("test"), 0);
    expect_true(str("alpha").compare("beta") < 0);
    expect_true(str("beta").compare("alpha") > 0);

    // Compare substring with another `str`.
    expect_equal(str("hello world").compare(0, 5, str("hello")), 0);
    expect_true(str("hello world").compare(6, 5, str("earth")) > 0);
    expect_true(str("hello world").compare(6, 5, str("worlds")) < 0);
    if constexpr (clamps) expect_true(str("hello world").compare(20, 5, str("worlds")) < 0);
    else expect_throws<std::out_of_range>([&] { return str("hello world").compare(20, 5, str("worlds")); });

    // Compare substring with another `str`'s substring.
    expect_equal(str("hello world").compare(0, 5, str("say hello"), 4, 5), 0);
    expect_equal(str("hello world").compare(6, 5, str("world peace"), 0, 5), 0);
    expect_equal(str("hello world").compare(6, 5, str("a better world"), 9, 5), 0);

    // Out of bounds cases for both compared strings.
    if constexpr (clamps)
        expect_true(str("hello world").compare(20, 5, str("a better world"), 9, 5) < 0 &&
                    str("hello world").compare(6, 5, str("a better world"), 90, 5) > 0);
    else {
        expect_throws<std::out_of_range>(
            [&] { return str("hello world").compare(20, 5, str("a better world"), 9, 5); });
        expect_throws<std::out_of_range>(
            [&] { return str("hello world").compare(6, 5, str("a better world"), 90, 5); });
    }

    // Compare substring with a C-string.
    expect_equal(str("hello world").compare(0, 5, "hello"), 0);
    expect_true(str("hello world").compare(6, 5, "earth") > 0);
    expect_true(str("hello world").compare(6, 5, "worlds") < 0);

    // Compare substring with a C-string's prefix.
    expect_equal(str("hello world").compare(0, 5, "hello Ash", 5), 0);
    expect_equal(str("hello world").compare(6, 5, "worlds", 5), 0);
    expect_true(str("hello world").compare(6, 5, "worlds", 6) < 0);

    // Prefix and suffix checks against strings.
    expect_equal(str("https://cppreference.com").starts_with(str("http")), true);
    expect_equal(str("https://cppreference.com").starts_with(str("ftp")), false);
    expect_equal(str("https://cppreference.com").ends_with(str("com")), true);
    expect_equal(str("https://cppreference.com").ends_with(str("org")), false);

    // Prefix and suffix checks against characters.
    expect_equal(str("C++20").starts_with('C'), true);
    expect_equal(str("C++20").starts_with('J'), false);
    expect_equal(str("C++20").ends_with('0'), true);
    expect_equal(str("C++20").ends_with('3'), false);

    // Prefix and suffix checks against C-style strings.
    expect_equal(str("string_view").starts_with("string"), true);
    expect_equal(str("string_view").starts_with("String"), false);
    expect_equal(str("string_view").ends_with("view"), true);
    expect_equal(str("string_view").ends_with("View"), false);

#if defined(__cpp_lib_string_contains)
    // Checking basic substring presence.
    expect_equal(str("hello").contains(str("ell")), true);
    expect_equal(str("hello").contains(str("oll")), false);
    expect_equal(str("hello").contains('l'), true);
    expect_equal(str("hello").contains('x'), false);
    expect_equal(str("hello").contains("lo"), true);
    expect_equal(str("hello").contains("lx"), false);
#endif

    // Exporting the contents of the string using the `str::copy` method.
    {
        char buf[5 + 1] = {0};
        str("hello").copy(buf, 5);
        expect_equal(std::strcmp(buf, "hello"), 0);
    }
    {
        char buf[4 + 1] = {0};
        str("hello").copy(buf, 4, 1);
        expect_equal(std::strcmp(buf, "ello"), 0);
    }
    if constexpr (clamps) {
        char buf[1] = {0};
        expect_true(str("hello").copy(buf, 1, 100) == 0 && buf[0] == 0);
    }
    else expect_throws<std::out_of_range>([&] { return str("hello").copy((char *)"", 1, 100); });

    // Swaps.
    for (str const first : {"", "hello", "hellohellohellohellohellohellohellohellohellohellohellohello"}) {
        for (str const second : {"", "world", "worldworldworldworldworldworldworldworldworldworldworldworld"}) {
            str first_copy = first;
            str second_copy = second;
            first_copy.swap(second_copy);
            expect_true(first_copy == second && second_copy == first,
                        "swap(other) did not exchange contents for this first/second pair");
            first_copy.swap(first_copy);
            expect_equal(first_copy, second, "Self-swap mutated the string for this first/second pair");
        }
    }

    // Make sure the standard hash and function-objects instantiate just fine.
    expect_true(std::hash<str> {}("hello") != 0);
    {
        std::ostringstream os;
        os << str("hello");
        expect_equal(os.str(), "hello");
    }

    expect_equal(std::equal_to<str> {}("hello", "world"), false);
    expect_equal(std::less<str> {}("hello", "world"), true);
}

#pragma endregion STL Reads

#pragma region STL Updates

/** Invokes every mutating member @c sz::string_t shares with @c std::basic_string, with the same
 *  arguments: each reports a status where the STL throws, and clamps positions past the end where
 *  the STL raises @c std::out_of_range. */
void test_stl_updates_unit() {

    using str = sz::string_t;
    auto const ok = [](sz::status_t status) noexcept { return sz::succeeded(status); };
    static_assert(!std::is_copy_constructible_v<str> && !std::is_copy_assignable_v<str>);

    // Construction goes through the factories, which report a failed allocation.
    expect_true(str().empty() && str().size() == 0);
    {
        str s = owned_("");
        expect_true(s.empty());
    }
    {
        str s = owned_("hello");
        expect_equal(s.size(), 5);
    }
    {
        auto const [copied, status] = owned_("hello").copy();
        expect_true(ok(status) && copied == "hello");
    }
    {
        auto const [joined, status] = str::make("hel"_sv | "lo");
        expect_true(ok(status) && joined == "hello");
    }
    {
        str s;
        expect_true(ok(s.assign(258, '0')) && s.find(std::string(256, '1')) == str::npos);
    }

    // Assignments.
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign("hello")) && s == "hello");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign("hello", 4)) && s == "hell");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign(5, 'a')) && s == "aaaaa");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign(32, 'a')) && s == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign({'h', 'e', 'l', 'l', 'o'})) && s == "hello");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign("hello"_sv, 2)) && s == "llo");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign("hello"_sv, 2, 2)) && s == "ll");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign("hello"_sv, 9)) && s.empty());
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign(s.begin(), s.end())) && s == "obsolete");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign(s, 4)) && s == "lete");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign(s, 4, 3)) && s == "let");
    }

    // Self-assignment is a special case of assignment.
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign(s)) && s == "obsolete");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign(s.data(), 2)) && s == "ob");
    }
    {
        str s = owned_("obsolete");
        expect_true(ok(s.assign(s.data(), s.size())) && s == "obsolete");
    }

    // Allocations, capacity and memory management.
    expect_true(str().max_size() > 0);
    expect_equal(str().get_allocator(), std::allocator<char>());
    expect_equal(std::strcmp(owned_("c_str").c_str(), "c_str"), 0);

    // Concatenation stays lazy until a verb materializes it.
    {
        str s;
        expect_true(ok(s.append("test")) && s == "test");
    }
    {
        str s;
        expect_true(ok(s.assign("test"_sv | "ing")) && s == "testing");
    }
    {
        str s;
        expect_true(ok(s.assign("test"_sv | "ing" | "123")) && s == "testing123");
    }
    {
        str s = owned_("!?");
        s.pop_back();
        expect_equal(s, "!");
    }

    // Incremental construction.
    {
        str s = owned_("__");
        expect_true(ok(s.insert(1, "test")) && s == "_test_");
    }
    {
        str s = owned_("__");
        expect_true(ok(s.insert(1, "test", 2)) && s == "_te_");
    }
    {
        str s = owned_("__");
        expect_true(ok(s.insert(1, 5, 'a')) && s == "_aaaaa_");
    }
    {
        str s = owned_("__");
        expect_true(ok(s.insert(1, owned_("test"))) && s == "_test_");
    }
    {
        str s = owned_("__");
        expect_true(ok(s.insert(1, "test"_sv, 2)) && s == "_st_");
    }
    {
        str s = owned_("__");
        expect_true(ok(s.insert(1, "test"_sv, 2, 1)) && s == "_s_");
    }

    // Inserting at a given iterator position yields back an iterator beside the status.
    {
        str s = owned_("__");
        expect_true(ok(s.insert(s.begin() + 1, 5, 'a').status) && s == "_aaaaa_");
    }
    {
        str s = owned_("__");
        sz::expected<str::iterator> const inserted = s.insert(s.begin() + 1, {'a', 'b', 'c'});
        expect_true(ok(inserted.status) && inserted.value == s.begin() + 1 && s == "_abc_");
    }

    // Positions past the end clamp to it.
    {
        str s = owned_("hello");
        expect_true(ok(s.insert(6, "world")) && s == "helloworld");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.insert(5, "world"_sv, 6)) && s == "hello");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.replace(9, 2, "!")) && s == "hello!");
    }
    {
        str s = owned_("hello");
        expect_true(s.append(str::npos, 'a') == sz::status_t::overflow_risk_k && s == "hello");
    }

    // Erasure.
    {
        str s;
        expect_equal(s.erase(0, 3), "");
    }
    {
        str s = owned_("test");
        expect_equal(s.erase(1, 2), "tt");
    }
    {
        str s = owned_("test");
        expect_equal(s.erase(1), "t");
    }
    {
        str s = owned_("test");
        expect_equal(s.erase(9), "test");
    }
    {
        str s = owned_("test");
        s.erase(s.begin() + 1);
        expect_equal(s, "tst");
    }
    {
        str s = owned_("test");
        s.erase(s.begin() + 1, s.begin() + 2);
        expect_equal(s, "tst");
    }
    {
        str s = owned_("test");
        s.erase(s.begin() + 1, s.begin() + 3);
        expect_equal(s, "tt");
    }
    {
        str s = owned_("test");
        expect_equal(s.erase(s.begin() + 1), s.begin() + 1);
    }
    {
        str s = owned_("test");
        expect_equal(s.erase(s.begin() + 1, s.begin() + 2), s.begin() + 1);
    }
    {
        str s = owned_("test");
        expect_equal(s.erase(s.begin() + 1, s.begin() + 3), s.begin() + 1);
    }

    // Substitutions.
    {
        str s = owned_("hello");
        expect_true(ok(s.replace(1, 2, "123")) && s == "h123lo");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.replace(1, 2, "123"_sv, 1)) && s == "h23lo");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.replace(1, 2, "123", 1)) && s == "h1lo");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.replace(1, 2, "123", 1, 1)) && s == "h2lo");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.replace(1, 2, "123"_sv, 1, 1)) && s == "h2lo");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.replace(1, 2, 3, 'a')) && s == "haaalo");
    }

    // Substitutions with iterators.
    {
        str s = owned_("hello");
        expect_true(ok(s.replace(s.begin() + 1, s.begin() + 3, 3, 'a')) && s == "haaalo");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.replace(s.begin() + 1, s.begin() + 3, {'a', 'b'})) && s == "hablo");
    }

    // Some nice "tweetable" examples :)
    {
        str s = owned_("Loose");
        expect_true(ok(s.replace(2, 2, "vath"_sv, 1)) && s == "Loathe");
    }
    {
        str s = owned_("Loose");
        expect_true(ok(s.replace(2, 2, "vath", 1)) && s == "Love");
    }

    // Insertion is a special case of replacement.
    // Appending and assigning are special cases of insertion.
    // Still, we test them separately to make sure they are not broken.
    {
        str s = owned_("hello");
        expect_true(ok(s.append("123")) && s == "hello123");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.append("123"_sv, 1)) && s == "hello23");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.append("123"_sv, 1, 1)) && s == "hello2");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.append({'1', '2'})) && s == "hello12");
    }
    {
        str s = owned_("hello");
        expect_true(ok(s.append(2, '!')) && s == "hello!!");
    }
    {
        sz::string_view_t const tail = "123";
        {
            str s = owned_("hello");
            expect_true(ok(s.append(tail.begin(), tail.end())) && s == "hello123");
        }
    }
}

/** Constructs StringZilla classes from STL and vice-versa, ensuring the conversions work. */
void test_stl_conversions_unit() {
    // From a mutable STL string to StringZilla and vice-versa.
    {
        std::string stl {"hello"};
        sz::string_t sz = owned_(stl);
        sz::string_view_t szv = stl;
        sz::string_span_t szs = stl;
        expect_equal(sz, "hello");
        expect_equal(szv, "hello");
        expect_equal(szs, "hello");
        szs[0] = 'H'; // A span aliases the source, so this writes through
        expect_equal(stl, "Hello");
        expect_equal(szv, "Hello"); // And a view sees the mutation
        expect_equal(sz, "hello");  // While an owning copy predates it
        stl = sz;
        expect_equal(stl, "hello");
    }
    // From StringZilla views back into a fresh STL string.
    {
        sz::string_t const sz = owned_("hello");
        std::string stl;
        stl = sz;
        expect_equal(stl, "hello");
        stl = sz::string_view_t {"world"};
        expect_equal(stl, "world");
    }
    // From an immutable STL string to StringZilla.
    {
        std::string const stl {"hello"};
        sz::string_t const sz = owned_(stl);
        sz::string_view_t const szv = stl;
        expect_equal(sz, "hello");
        expect_equal(szv, "hello");
        expect_equal(szv.data(), stl.data()); // A view borrows, a string owns
    }
    // From STL `string_view_t` to StringZilla and vice-versa.
    {
        std::string_view stl {"hello"};
        sz::string_t sz = owned_(stl);
        sz::string_view_t szv = stl;
        expect_equal(sz, "hello");
        expect_equal(szv, "hello");
        stl = sz;
        expect_equal(stl, "hello");
        stl = szv;
        expect_equal(stl, "hello");
    }
}

/** Tests STL containers keyed by StringZilla strings, and ordered and hashed by @c sz functors. */
void test_stl_containers_unit() {

    // Byte order and length disagree across these: "Zebra" wins on its first byte despite being longer,
    // and each prefix precedes its own extension.
    char const *const ascending_keys[] = {"Zebra", "app", "apple", "apples", "banana"};

    // Both maps order through the transparent `sz::less`, which also finds owning keys by views.
    std::map<sz::string_t, int, sz::less> sorted_words_sz;
    std::map<std::string, int, sz::less> sorted_words_stl;
    for (int insertion = 4; insertion >= 0; --insertion) { // Reverse order, so sorting has work to do
        sorted_words_sz.emplace(owned_(ascending_keys[insertion]), insertion);
        sorted_words_stl.emplace(ascending_keys[insertion], insertion);
    }
    expect_equal(sorted_words_sz.size(), 5);
    expect_equal(sorted_words_stl.size(), 5);

    std::size_t rank_sz = 0;
    for (auto const &entry : sorted_words_sz) {
        expect_equal(entry.first, ascending_keys[rank_sz], "sz::string_t map produced the wrong key at sorted rank_sz");
        expect_equal(entry.second, static_cast<int>(rank_sz),
                     "sz::string_t map produced the wrong value at sorted rank_sz");
        ++rank_sz;
    }
    expect_equal(rank_sz, 5);

    std::size_t rank_stl = 0;
    for (auto const &entry : sorted_words_stl) {
        expect_equal(entry.first, ascending_keys[rank_stl],
                     "std::string map via sz::less produced the wrong key at sorted rank_stl");
        expect_equal(entry.second, static_cast<int>(rank_stl),
                     "std::string map via sz::less produced the wrong value at sorted rank_stl");
        ++rank_stl;
    }
    expect_equal(rank_stl, 5);

    expect_equal(sorted_words_sz.find("apple")->second, 2);
    expect_equal(sorted_words_stl.at("apple"), 2);
    expect_equal(sorted_words_sz.count("Apple"), 0); // Comparisons are case-sensitive
    expect_equal(sorted_words_stl.count("Apple"), 0);
    expect_equal(sorted_words_sz.count("appl"), 0); // A prefix of a present key is still absent
    sorted_words_sz.erase(sorted_words_sz.find("app"));
    expect_equal(sorted_words_sz.find("app"), sorted_words_sz.end());
    expect_equal(sorted_words_sz.size(), 4);
    expect_equal(sorted_words_sz.begin()->first, "Zebra");
    expect_equal(sorted_words_sz.lower_bound("apple")->first, "apple");
    expect_equal(sorted_words_sz.upper_bound("apple")->first, "apples");

    // Equal-valued keys assembled from different storage must hash alike and compare equal,
    // so the second insertion collapses onto the first instead of adding a bucket.
    std::unordered_map<sz::string_t, int, sz::hash_t, sz::equal_to_t> words_sz;
    words_sz.emplace(owned_("banana"), 7);
    sz::string_t grown_sz = owned_("bana");
    expect_true(sz::succeeded(grown_sz.append("na")));
    expect_equal(std::hash<sz::string_t> {}(grown_sz), std::hash<sz::string_t> {}(owned_("banana")),
                 "std::hash disagreed for equal-content strings built via different construction paths");
    expect_true(words_sz.find(grown_sz) != words_sz.end());
    expect_equal(words_sz.emplace(std::move(grown_sz), 9).second, false);
    expect_equal(words_sz.find("banana")->second, 7);
    expect_equal(words_sz.size(), 1);
    expect_equal(words_sz.find("bananas"), words_sz.end());

    // The same, but with `std::string` keys routed through `sz::hash_t` and `sz::equal_to_t`.
    std::unordered_map<std::string, int, sz::hash_t, sz::equal_to_t> words_stl;
    std::string const heap_key(200, 'x'); // Long enough to escape any small-string buffer
    std::string grown_stl;
    for (int repetition = 0; repetition < 200; ++repetition) grown_stl.push_back('x');
    words_stl.emplace(heap_key, 7);
    expect_equal(sz::hash_t {}(heap_key), sz::hash_t {}(grown_stl),
                 "sz::hash_t disagreed for equal-content strings built via different construction paths");
    expect_true(sz::equal_to_t {}(heap_key, grown_stl));
    expect_equal(sz::equal_to_t {}(heap_key, "x"), false);
    expect_true(words_stl.find(grown_stl) != words_stl.end());
    expect_equal(words_stl.emplace(grown_stl, 9).second, false);
    expect_equal(words_stl.at(grown_stl), 7);
    expect_equal(words_stl.size(), 1);
    expect_equal(words_stl.find("xyz"), words_stl.end());

    // Empty keys are valid, and order before everything else.
    expect_true(sz::less {}("", "a"));
    expect_equal(sz::less {}("a", ""), false);
    expect_true(sz::equal_to_t {}("", ""));
    sorted_words_sz.emplace(owned_(""), -1);
    expect_equal(sorted_words_sz.begin()->first, "");
    words_stl.emplace("", -1);
    expect_equal(words_stl.at(""), -1);
}

#pragma endregion STL Updates

#pragma region Extensions

/** Invokes C++ member methods of immutable strings, covering extensions beyond the STL API. */
template <typename string_type>
void test_extensions_reads_unit() {
    using str = string_type;

    // Signed offset lookups and slices.
    expect_equal(str("hello").sat(0), 'h');
    expect_equal(str("hello").sat(-1), 'o');
    expect_equal(str("rest").sat(1), 'e');
    expect_equal(str("rest").sat(-1), 't');
    expect_equal(str("rest").sat(-4), 'r');

    expect_equal(str("front").front(), 'f');
    expect_equal(str("front").front(1), "f");
    expect_equal(str("front").front(2), "fr");
    expect_equal(str("front").front(2), "fr");
    expect_equal(str("front").front(-2), "fro");
    expect_equal(str("front").front(0), "");
    expect_equal(str("front").front(5), "front");
    expect_equal(str("front").front(-5), "");

    expect_equal(str("back").back(), 'k');
    expect_equal(str("back").back(1), "ack");
    expect_equal(str("back").back(2), "ck");
    expect_equal(str("back").back(-1), "k");
    expect_equal(str("back").back(-2), "ck");
    expect_equal(str("back").back(-4), "back");
    expect_equal(str("back").back(4), "");

    expect_equal(str("hello").sub(1), "ello");
    expect_equal(str("hello").sub(-1), "o");
    expect_equal(str("hello").sub(1, 2), "e");
    expect_equal(str("hello").sub(1, 100), "ello");
    expect_equal(str("hello").sub(100, 100), "");
    expect_equal(str("hello").sub(-2, -1), "l");
    expect_equal(str("hello").sub(-2, -2), "");
    expect_equal(str("hello").sub(100, -100), "");

    // Passing initializer lists to `operator[]`.
    // Put extra braces to correctly estimate the number of macro arguments :)
    str const hello_text("hello");
    expect_true((hello_text[{1, 2}] == "e"));
    expect_true((hello_text[{1, 100}] == "ello"));
    expect_true((hello_text[{100, 100}] == ""));
    expect_true((hello_text[{100, -100}] == ""));
    expect_true((hello_text[{-100, -100}] == ""));

    // Checksums
    auto accumulate_bytes = [](str const &s) -> std::size_t {
        return std::accumulate(s.begin(), s.end(), (std::size_t)0,
                               [](std::size_t sum, char c) { return sum + static_cast<unsigned char>(c); });
    };
    expect_equal(str("a").bytesum(), (std::size_t)'a');
    expect_equal(str("0").bytesum(), (std::size_t)'0');
    expect_equal(str("0123456789").bytesum(), arithmetic_sum('0', '9'));
    expect_equal(str("abcdefghijklmnopqrstuvwxyz").bytesum(), arithmetic_sum('a', 'z'));
    expect_equal(str("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz").bytesum(),
                 arithmetic_sum('a', 'z') * 3);
    {
        str s = "近来，加文出席微博之夜时对着镜头频繁摆出假笑表情、一度累" //
                "瘫睡倒在沙发上的照片被广泛转发，引发对他失去童年、被过度" //
                "消费的担忧。八岁的加文，已当网红近六年了，可以说，自懂事" //
                "以来，他没有过过一天没有名气的日子。";
        expect_equal(s.bytesum(), accumulate_bytes(s));
    }
}

/** Exercises StringZilla's non-STL mutating string extensions on @c sz::string_t. */
void test_extensions_updates_unit() {
    using str = sz::string_t;

    // Allocating verbs return a status, and the signed-offset twins count negatives from the end.
    {
        str s = owned_("obsolete");
        expect_true(sz::succeeded(s.assign("hello")) && s == "hello");
    }
    {
        str s = owned_("obsolete");
        expect_true(sz::succeeded(s.assign(owned_("hello"))) && s == "hello");
    }
    {
        str s = owned_("obsolete");
        expect_true(sz::succeeded(s.assign(s)) && s == "obsolete");
    }
    {
        str s;
        expect_true(sz::succeeded(s.reserve(10)) && s.capacity() >= 10);
    }
    {
        str s;
        expect_true(sz::succeeded(s.resize(10)) && s.size() == 10);
    }
    {
        str s;
        expect_true(sz::succeeded(s.resize(10, 'a')) && s == "aaaaaaaaaa");
    }
    {
        str s = owned_("!?");
        expect_true(sz::succeeded(s.push_back('a')) && s == "!?a");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.append(owned_("123"))) && s == "hello123");
    }
    {
        str s = owned_("__");
        expect_true(sz::succeeded(s.insert_at(-1, "test")) && s == "_test_");
    }
    {
        str s = owned_("__");
        expect_true(sz::succeeded(s.insert_at(1, "test")) && s == "_test_");
    }
    {
        str s = owned_("__");
        expect_true(sz::succeeded(s.insert_at(9, "test")) && s == "__test");
    }
    {
        str s = owned_("test");
        expect_true(s.erase_slice(1, -1) == 2 && s == "tt");
    }
    {
        str s = owned_("test");
        expect_true(sz::succeeded(s.replace_slice(1, -1, "aaaa")) && s == "taaaat");
    }
    {
        str s;
        expect_true(s.reserve(s.max_size() + 1) == sz::status_t::overflow_risk_k && s.empty());
    }

    // On 32-bit systems the base capacity can be larger than our `z::string::min_capacity`.
    // It's true for MSVC: https://github.com/ashvardanian/StringZilla/issues/168
    if (STRINGZILLA_ARCH_64BIT_) {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.shrink_to_fit()) && s.capacity() <= sz::string_t::min_capacity);
    }

    // Resizing hands the callback the grown buffer and keeps only the length it reports.
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.resize_and_overwrite(10,
                                                         [](char *p, std::size_t count) noexcept {
                                                             std::memset(p, 'X', count);
                                                             return count;
                                                         })) &&
                    s == "XXXXXXXXXX");
    }
    {
        str s = owned_("test");
        expect_true(sz::succeeded(s.resize_and_overwrite(8,
                                                         [](char *p, std::size_t) noexcept {
                                                             std::strcpy(p, "ABCDE");
                                                             return 5;
                                                         })) &&
                    s.size() == 5 && s == "ABCDE");
    }
    {
        str s = owned_("orig");
        expect_true(sz::succeeded(s.resize_and_overwrite(6,
                                                         [](char *p, std::size_t count) noexcept {
                                                             std::strcpy(p, "works!");
                                                             return count;
                                                         })) &&
                    s.size() == 6 && s == "works!");
    }

    // Growing through a concatenation must succeed and land exactly on the concatenation's length.
    {
        str s = owned_("ab");
        expect_true(sz::succeeded(s.assign("hello"_sv | "world")) && s == "helloworld");
    }

    // A throwing allocator must surface as `bad_alloc_k` from the `noexcept` verbs.
    {
        sz::string<throwing_allocator_t> s;
        expect_true(s.reserve(1000) == sz::status_t::bad_alloc_k && s.empty());
    }

    // Self-referencing methods.
    {
        str s = owned_("test");
        expect_true(sz::succeeded(s.assign(s.view())) && s == "test");
    }
    {
        str s = owned_("test");
        expect_true(sz::succeeded(s.assign(s.view().sub(1, 2))) && s == "e");
    }
    {
        str s = owned_("test");
        expect_true(sz::succeeded(s.append(s.view().sub(1, 2))) && s == "teste");
    }

    // Growing beyond and shrinking beneath the capacity threshold.
    {
        str s = owned_("0123456789012345678901234567890123456789012345678901234567890123"); // 64 symbols at start
        for (int doubling = 0; doubling != 4; ++doubling) expect_true(sz::succeeded(s.append(s)));
        s.clear();
        expect_true(sz::succeeded(s.shrink_to_fit()));
        expect_true(s.capacity() < sz::string_t::min_capacity);
    }

    // Folding and normalizing allocate through the string's allocator, not a default-made one.
    {
        std::size_t live_bytes = 0;
        accounting_allocator_t const allocator {&live_bytes};
        {
            sz::string<accounting_allocator_t> s(allocator);
            expect_true(sz::succeeded(s.assign("STRASSE AND STRASSE")));
            expect_true(sz::succeeded(s.utf8_uncased_fold()) && s == "strasse and strasse");
            expect_true(sz::succeeded(s.utf8_normalize(sz_normal_form_nfd_k)) && s == "strasse and strasse");
            expect_true(s.get_allocator() == allocator && live_bytes != 0);
        }
        expect_equal(live_bytes, 0);
    }

    // Same length replacements.
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("xx", "xx")) && s == "hello");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("l", "1")) && s == "he11o");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("he", "al")) && s == "alllo");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("x"_bs, "!")) && s == "hello");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("o"_bs, "!")) && s == "hell!");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("ho"_bs, "!")) && s == "!ell!");
    }

    // Shorter replacements.
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("xx", "x")) && s == "hello");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("l", "")) && s == "heo");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("h", "")) && s == "ello");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("o", "")) && s == "hell");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("llo", "!")) && s == "he!");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("x"_bs, "")) && s == "hello");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("lo"_bs, "")) && s == "he");
    }

    // Longer replacements.
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("xx", "xxx")) && s == "hello");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("l", "ll")) && s == "hellllo");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("h", "hh")) && s == "hhello");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("o", "oo")) && s == "helloo");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("llo", "llo!")) && s == "hello!");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("x"_bs, "xx")) && s == "hello");
    }
    {
        str s = owned_("hello");
        expect_true(sz::succeeded(s.replace_all("lo"_bs, "lo")) && s == "helololo");
    }

    // Directly mapping bytes using a Look-Up Table.
    sz::look_up_table_t invert_case = sz::look_up_table_t::identity();
    for (char c = 'a'; c <= 'z'; c++) invert_case[c] = c - 'a' + 'A';
    for (char c = 'A'; c <= 'Z'; c++) invert_case[c] = c - 'A' + 'a';
    {
        str s = owned_("hello");
        s.lookup(invert_case);
        expect_equal(s, "HELLO");
    }
    {
        str s = owned_("HeLLo");
        s.lookup(invert_case);
        expect_equal(s, "hEllO");
    }
    {
        str s = owned_("H-lL0");
        s.lookup(invert_case);
        expect_equal(s, "h-Ll0");
    }

    // Concatenation stays lazy, and the factory reports a failed allocation.
    expect_equal(owned_(owned_("a") | owned_("b")), "ab");
    expect_equal(owned_(owned_("a") | owned_("b") | owned_("ab")), "abab");

    expect_equal(owned_(sz::concatenate("a"_sv, "b"_sv)), "ab");
    expect_equal(owned_(sz::concatenate("a"_sv, "b"_sv, "c"_sv)), "abc");

    // The cases above pass only rvalues carrying a `::value_type`. Named lvalues deduce to a
    // reference and raw literals to a character array, neither of which has member typedefs.
    {
        str name = owned_("ash"), domain = owned_("mail"), tld = owned_("com");
        expect_equal(owned_(sz::concatenate(name, "@", domain, ".", tld)), "ash@mail.com");
        expect_equal(owned_(name | "@" | domain | "." | tld), "ash@mail.com");
        expect_equal(owned_(sz::concatenate(name, domain)), "ashmail");
        expect_equal(owned_(sz::concatenate("@", name)), "@ash");
        {
            auto const [email, status] = str::make(name | "@" | domain);
            expect_true(sz::succeeded(status) && email == "ash@mail");
        }
    }

    // Range members slice the string they were called on, so offsets stay inside its own buffer.
    {
        str text = owned_("hello brave new world");
        expect_equal(offsets_within(text, text.utf8_wordbreaks()),
                     std::vector<std::ptrdiff_t> {0, 5, 6, 11, 12, 15, 16},
                     "utf8_wordbreaks segment landed outside the caller's own buffer");
        expect_equal(offsets_within(text, text.utf8_split_whitespaces()), std::vector<std::ptrdiff_t> {0, 6, 12, 16},
                     "utf8_split_whitespaces token landed outside the caller's own buffer");
        expect_equal(offsets_within(text, text.utf8_split_delimiters()), std::vector<std::ptrdiff_t> {0, 6, 12, 16},
                     "utf8_split_delimiters field landed outside the caller's own buffer");
        str sso = owned_("a b c");
        expect_equal(offsets_within(sso, sso.utf8_split_whitespaces()), std::vector<std::ptrdiff_t> {0, 2, 4},
                     "utf8_split_whitespaces token landed outside the small-string-optimized buffer");
    }

    // Randomization fills a string the caller sized.
    {
        str s;
        expect_true(sz::succeeded(s.resize(4)) && s.fill_random(42).size() == 4);
    }
    {
        str s;
        expect_true(s.fill_random().empty());
    }
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
    expect_equal(sz::string_view_t("a").find_first_of("az"), 0);
    expect_equal(sz::string_view_t("a").find_last_of("az"), 0);
    expect_equal(sz::string_view_t("a").find_first_of("xz"), sz::string_view_t::npos);
    expect_equal(sz::string_view_t("a").find_last_of("xz"), sz::string_view_t::npos);

    expect_equal(sz::string_view_t("a").find_first_not_of("xz"), 0);
    expect_equal(sz::string_view_t("a").find_last_not_of("xz"), 0);
    expect_equal(sz::string_view_t("a").find_first_not_of("az"), sz::string_view_t::npos);
    expect_equal(sz::string_view_t("a").find_last_not_of("az"), sz::string_view_t::npos);

    expect_equal(sz::string_view_t("aXbYaXbY").find_first_of("XY"), 1);
    expect_equal(sz::string_view_t("axbYaxbY").find_first_of("Y"), 3);
    expect_equal(sz::string_view_t("YbXaYbXa").find_last_of("XY"), 6);
    expect_equal(sz::string_view_t("YbxaYbxa").find_last_of("Y"), 4);
    expect_equal(sz::string_view_t(sz::base64(), sizeof(sz::base64())).find_first_of("_"), sz::string_view_t::npos);
    expect_equal(sz::string_view_t(sz::base64(), sizeof(sz::base64())).find_first_of("+"), 62);
    expect_true(sz::string_view_t(sz::ascii_printables(), sizeof(sz::ascii_printables())).find_first_of("~") !=
                sz::string_view_t::npos);

    // Forward searches starting past the end find nothing, even for an empty needle
    expect_equal("abc"_sv.find("", 3), 3);
    expect_equal("abc"_sv.find("", 4), sz::string_view_t::npos);
    expect_equal("abc"_sv.find("b", 4), sz::string_view_t::npos);
    expect_equal("abc"_sv.find('b', 4), sz::string_view_t::npos);
    expect_equal("abc"_sv.find_first_of("b", 4), sz::string_view_t::npos);
    expect_equal("abc"_sv.find_first_not_of("b", 4), sz::string_view_t::npos);
    expect_equal("abc"_sv.find('b', sz::string_view_t::npos), sz::string_view_t::npos); // Largest position
    expect_equal(sz::string_view_t().find("", 1), sz::string_view_t::npos);             // Default view has no data

    expect_equal("aabaa"_sv.remove_prefix("a"), "abaa");
    expect_equal("aabaa"_sv.remove_suffix("a"), "aaba");
    expect_equal("aabaa"_sv.lstrip("a"_bs), "baa");
    expect_equal("aabaa"_sv.rstrip("a"_bs), "aab");
    expect_equal("aabaa"_sv.strip("a"_bs), "b");

    // Check more advanced composite operations
    expect_equal("abbccc"_sv.partition('b').before.size(), 1);
    expect_equal("abbccc"_sv.partition("bb").before.size(), 1);
    expect_equal("abbccc"_sv.partition("bb").match.size(), 2);
    expect_equal("abbccc"_sv.partition("bb").after.size(), 3);
    expect_equal("abbccc"_sv.partition("bb").before, "a");
    expect_equal("abbccc"_sv.partition("bb").match, "bb");
    expect_equal("abbccc"_sv.partition("bb").after, "ccc");
    expect_equal("abb ccc"_sv.partition(sz::whitespaces_set()).after, "ccc");

    // Check ranges of search matches
    expect_equal("hello"_sv.find_all("l").size(), 2);
    expect_equal("hello"_sv.rfind_all("l").size(), 2);

    expect_equal(""_sv.find_all(".", sz::include_overlaps_t {}).size(), 0);
    expect_equal(""_sv.find_all(".", sz::exclude_overlaps_t {}).size(), 0);
    expect_equal("."_sv.find_all(".", sz::include_overlaps_t {}).size(), 1);
    expect_equal("."_sv.find_all(".", sz::exclude_overlaps_t {}).size(), 1);
    expect_equal(".."_sv.find_all(".", sz::include_overlaps_t {}).size(), 2);
    expect_equal(".."_sv.find_all(".", sz::exclude_overlaps_t {}).size(), 2);
    expect_equal(""_sv.rfind_all(".", sz::include_overlaps_t {}).size(), 0);
    expect_equal(""_sv.rfind_all(".", sz::exclude_overlaps_t {}).size(), 0);
    expect_equal("."_sv.rfind_all(".", sz::include_overlaps_t {}).size(), 1);
    expect_equal("."_sv.rfind_all(".", sz::exclude_overlaps_t {}).size(), 1);
    expect_equal(".."_sv.rfind_all(".", sz::include_overlaps_t {}).size(), 2);
    expect_equal(".."_sv.rfind_all(".", sz::exclude_overlaps_t {}).size(), 2);

    // An empty needle matches at every offset from 0 to `size()` inclusive, in either direction.
    expect_equal("abc"_sv.find_all("").size(), 4);
    expect_equal("abc"_sv.find_all("", sz::exclude_overlaps_t {}).size(), 4);
    expect_equal("abc"_sv.rfind_all("").size(), 4);
    expect_equal(""_sv.find_all("").size(), 1);
    expect_equal(sz::find_all(owned_("abc"), "").size(), 4);

    expect_equal("a.b.c.d"_sv.find_all(".").size(), 3);
    expect_equal("a.,b.,c.,d"_sv.find_all(".,").size(), 3);
    expect_equal("a.,b.,c.,d"_sv.rfind_all(".,").size(), 3);
    expect_equal("a.b,c.d"_sv.find_all(".,"_bs).size(), 3);
    expect_equal("a...b...c"_sv.rfind_all("..").size(), 4);
    expect_equal("a...b...c"_sv.rfind_all("..", sz::include_overlaps_t {}).size(), 4);
    expect_equal("a...b...c"_sv.rfind_all("..", sz::exclude_overlaps_t {}).size(), 2);

    {
        auto finds = "a.b.c"_sv.find_all("abcd"_bs).template to<std::vector<std::string>>();
        expect_true(finds.size() == 3 && finds[0] == "a");
    }
    {
        auto rfinds = "a.b.c"_sv.rfind_all("abcd"_bs).template to<std::vector<std::string>>();
        expect_true(rfinds.size() == 3 && rfinds[0] == "c");
    }

    // Test propagating strings and their non-owning views into temporary ranges and iterators
    expect_equal(sz::find_all("abc"_sv, "b"_sv).size(), 1);
    expect_equal(sz::find_all("hello"_sv, "l"_sv).size(), 2);
    expect_equal(sz::rfind_all("abc"_sv, "b"_sv).size(), 1);

    {
        sz::string_t h = owned_("abc"), n = owned_("b");
        expect_equal(sz::find_all(h, n).size(), 1);
    }
    {
        sz::string_t h = owned_("hello"), n = owned_("l");
        expect_equal(sz::find_all(h, n).size(), 2);
    }
    {
        sz::string_t h = owned_("abc"), n = owned_("b");
        expect_equal(sz::rfind_all(h, n).size(), 1);
    }

    expect_equal(sz::find_all(owned_("abc"), owned_("b")).size(), 1);
    expect_equal(sz::find_all(owned_("hello"), owned_("l")).size(), 2);
    expect_equal(sz::rfind_all(owned_("abc"), owned_("b")).size(), 1);

    // Lvalue haystacks are borrowed, so slices land inside the caller's own buffer. A copied
    // haystack would offset into a private copy - and under SSO those offsets look plausible.
    {
        sz::string_t haystack = owned_("hello world, hello cpp");
        sz::string_t sso = owned_("a b a");
        expect_equal(offsets_within(haystack, sz::find_all(haystack, "hello")), std::vector<std::ptrdiff_t> {0, 13},
                     "Match offsets did not land inside the borrowed lvalue haystack's own buffer");
        expect_equal(offsets_within(sso, sz::find_all(sso, "a")), std::vector<std::ptrdiff_t> {0, 4},
                     "Match offsets did not land inside the small-string-optimized haystack's own buffer");
    }

    // Needles are copied into the matcher, so a temporary one outlives the expression that built it.
    expect_equal(sz::find_all(owned_("hello world, hello cpp"), owned_("hello")).size(), 2);

    // Haystack and needle need not share a type - literals, views, and owning strings mix.
    {
        sz::string_t owning = owned_("a-b-c");
        sz::string_view_t view("a-b-c");
        sz::string_t needle = owned_("-");
        expect_equal(sz::find_all(view, "-").size(), 2);
        expect_equal(sz::find_all(owning, "-").size(), 2);
        expect_equal(sz::find_all(owning, view.substr(1, 1)).size(), 2);
        expect_equal(sz::find_all(view, needle).size(), 2);
        expect_equal(sz::split(owning, "-").size(), 3);
        expect_equal(sz::rsplit(view, needle).size(), 3);
        expect_equal(sz::split_characters(owning, "-").size(), 3);
    }

    // Check splitting - the inverse of `find_all` ranges
    {
        auto splits = ".a..c."_sv.split("."_bs).template to<std::vector<std::string>>();
        expect_true(splits.size() == 5 && splits[0] == "" && splits[1] == "a" && splits[4] == "");
    }
    {
        auto line_splits = "line1\nline2\nline3"_sv.split("line3").template to<std::vector<std::string>>();
        expect_true(line_splits.size() == 2 && line_splits[0] == "line1\nline2\n" && line_splits[1] == "");
    }

    expect_equal(""_sv.split(".").size(), 1);
    expect_equal(""_sv.rsplit(".").size(), 1);
    expect_true("a.b"_sv.split("").size() == 1 && *"a.b"_sv.split("").begin() == "a.b"); // Empty separators never split
    expect_equal("a.b"_sv.rsplit("").size(), 1);
    expect_true(sz::string_view_t {}.split(",").size() == 1 && sz::string_view_t {}.rsplit(",").size() == 1);

    expect_equal("hello"_sv.split("l").size(), 3);
    expect_equal("hello"_sv.rsplit("l").size(), 3);
    expect_equal(*advanced("hello"_sv.split("l").begin(), 0), "he");
    expect_equal(*advanced("hello"_sv.rsplit("l").begin(), 0), "o");
    expect_equal(*advanced("hello"_sv.split("l").begin(), 1), "");
    expect_equal(*advanced("hello"_sv.rsplit("l").begin(), 1), "");
    expect_equal(*advanced("hello"_sv.split("l").begin(), 2), "o");
    expect_equal(*advanced("hello"_sv.rsplit("l").begin(), 2), "he");

    expect_equal("a.b.c.d"_sv.split(".").size(), 4);
    expect_equal("a.b.c.d"_sv.rsplit(".").size(), 4);
    expect_equal(*("a.b.c.d"_sv.split(".").begin()), "a");
    expect_equal(*("a.b.c.d"_sv.rsplit(".").begin()), "d");
    expect_equal(*advanced("a.b.c.d"_sv.split(".").begin(), 1), "b");
    expect_equal(*advanced("a.b.c.d"_sv.rsplit(".").begin(), 1), "c");
    expect_equal(*advanced("a.b.c.d"_sv.split(".").begin(), 3), "d");
    expect_equal(*advanced("a.b.c.d"_sv.rsplit(".").begin(), 3), "a");
    expect_equal("a.b.,c,d"_sv.split(".,").size(), 2);
    expect_equal("a.b,c.d"_sv.split(".,"_bs).size(), 4);

    {
        auto rsplits = ".a..c."_sv.rsplit("."_bs).template to<std::vector<std::string>>();
        expect_true(rsplits.size() == 5 && rsplits[0] == "" && rsplits[1] == "c" && rsplits[4] == "");
    }
}

#pragma region String Class

/** Copies @c sz::string_t through @c copy and @c assign, the two verbs that report a failed
 *  allocation, as the copy constructor and copy assignment are deleted. */
void test_string_constructors_unit() {
    std::string alphabet {sz::ascii_printables(), sizeof(sz::ascii_printables())};
    for (std::size_t alphabet_slice = 0; alphabet_slice != alphabet.size(); ++alphabet_slice) {
        sz::string_t const source = owned_(alphabet.substr(0, alphabet_slice));
        sz::expected<sz::string_t> const copied = source.copy();
        expect_true(sz::succeeded(copied.status) && copied.value == source, "`copy` diverged from its source");
        sz::string_t assigned = owned_("obsolete");
        expect_true(sz::succeeded(assigned.assign(source)) && assigned == source, "`assign` diverged from its source");
    }
}

/** Validates that shrinking @c reserve calls are harmless no-ops, just like in the STL, and that
 *  every reallocation carries the terminator. Regression tests: shrinking used to overflow the heap
 *  buffer in release builds, and growing or fitting left the new buffer unterminated. */
void test_string_reserve_unit() {
    // C API: grow, shrink, then fit - the buffer, length, contents and terminator stay intact.
    {
        // Fresh blocks arrive full of noise, so a terminator never copied cannot read as one.
        sz_allocator_t allocator {};
        allocator.allocate = +[](sz_size_t length, void *, sz_stream_t) -> void * {
            void *const block = std::malloc(length);
            if (block) std::memset(block, '#', length);
            return block;
        };
        allocator.free = +[](void *block, sz_size_t, void *, sz_stream_t) { std::free(block); };
        allocator.handle = nullptr;

        sz_string_t str;
        sz_ptr_t start = sz_string_init_length(&str, 100, &allocator);
        if (!expect_true(start != nullptr)) return;
        std::memset(start, 'a', 100);

        sz_ptr_t grown = sz_string_reserve(&str, 200, &allocator);
        if (!expect_true(grown != nullptr)) return;
        expect_equal(sz_string_length(&str), 100);
        expect_equal(grown[100], '\0', "Growing left the new buffer unterminated");

        // Shrinking must be a no-op: same buffer, same length, same contents.
        sz_ptr_t shrunk = sz_string_reserve(&str, 50, &allocator);
        expect_equal(shrunk, grown);
        expect_equal(sz_string_length(&str), 100);
        for (sz_size_t i = 0; i != 100; ++i) expect_equal(shrunk[i], 'a');

        sz_ptr_t fitted = sz_string_shrink_to_fit(&str, &allocator);
        if (!expect_true(fitted != nullptr)) return;
        expect_equal(sz_string_length(&str), 100);
        expect_equal(fitted[100], '\0', "Fitting left the new buffer unterminated");

        sz_string_free(&str, &allocator);
    }
    // C++ API: `sz::string_t::reserve` shrinking must match `std::string` behavior - keep the contents.
    {
        sz::string_t str;
        expect_true(sz::succeeded(str.assign(100, 'a')));
        std::size_t const capacity_before = str.capacity();
        expect_true(sz::succeeded(str.reserve(50)));
        expect_equal(str.size(), 100);
        expect_equal(str.capacity(), capacity_before);
        expect_equal(str, std::string(100, 'a'));
    }
}

/** Checks the string class for leaks, and that blocks return to the allocator that granted them. */
void test_memory_stability_equivalence(std::size_t length) {
    using accounting_string_t = sz::string<accounting_allocator_t>;
    static_assert(sizeof(accounting_string_t) == sizeof(sz::string_t) + sizeof(std::size_t *),
                  "Only a stateful allocator may widen the string");

    // A stateful allocator that neither propagates nor compares equal could only move by copying,
    // which might fail, so such strings have no move assignment at all.
    struct pinned_allocator_t {
        using value_type = char;
        std::size_t *live_bytes;
        char *allocate(std::size_t count);
        void deallocate(char *block, std::size_t count);
        bool operator==(pinned_allocator_t const &) const noexcept = default;
    };
    static_assert(!std::is_move_assignable_v<sz::string<pinned_allocator_t>>);
    static_assert(std::is_nothrow_move_assignable_v<accounting_string_t>);

    std::size_t const iterations = scaled_iterations(100);
    std::size_t live_bytes = 0, foreign_bytes = 0;
    accounting_allocator_t const allocator {&live_bytes}, foreign_allocator {&foreign_bytes};
    accounting_string_t base(allocator);
    for (std::size_t i = 0; i < length; ++i) expect_true(sz::succeeded(base.push_back('c')));
    expect_equal(base.length(), length, "Base string has the wrong length after `push_back` construction");

    // Do copies leak?
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            sz::expected<accounting_string_t> const copy = base.copy();
            expect_true(sz::succeeded(copy.status) && copy.value.get_allocator() == allocator);
            expect_equal(copy.value.length(), length, "Copied string has the wrong length at iteration i");
            expect_equal(copy.value, base, "Copied string diverged from `base` at iteration i");
        }
    });

    // How about assignments?
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            accounting_string_t copy(allocator);
            expect_true(sz::succeeded(copy.assign(base)));
            expect_equal(copy.length(), length, "Copy-assigned string has the wrong length at iteration i");
            expect_equal(copy, base, "Copy-assigned string diverged from `base` at iteration i");
        }
    });

    // How about the move constructor?
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            auto [unique_item, copy_status] = base.copy();
            expect_true(sz::succeeded(copy_status));
            expect_equal(unique_item.length(), length, "Pre-move string has the wrong length at iteration i");
            expect_equal(unique_item, base, "Pre-move string diverged from `base` at iteration i");
            accounting_string_t copy(std::move(unique_item));
            expect_equal(copy.length(), length, "Move-constructed string has the wrong length at iteration i");
            expect_equal(copy, base, "Move-constructed string diverged from `base` at iteration i");
        }
    });

    // And the move assignment operator with an empty target payload?
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            auto [unique_item, copy_status] = base.copy();
            expect_true(sz::succeeded(copy_status));
            accounting_string_t copy(allocator);
            copy = std::move(unique_item);
            expect_equal(copy.length(), length,
                         "Move-assigned (empty target) string has the wrong length at iteration i");
            expect_equal(copy, base, "Move-assigned (empty target) string diverged from `base` at iteration i");
        }
    });

    // And move assignment where the target had a payload?
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            auto [unique_item, copy_status] = base.copy();
            expect_true(sz::succeeded(copy_status));
            accounting_string_t copy(allocator);
            for (std::size_t j = 0; j < 317; j++) expect_true(sz::succeeded(copy.push_back('q')));
            copy = std::move(unique_item);
            expect_equal(copy.length(), length,
                         "Move-assigned (occupied target) string has the wrong length at iteration i");
            expect_equal(copy, base, "Move-assigned (occupied target) string diverged from `base` at iteration i");
        }
    });

    // Across unequal allocators, a move adopts the source's allocator, and `assign` copies the
    // bytes into the target's own.
    assert_balanced_memory(live_bytes, [&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            auto [unique_item, copy_status] = base.copy();
            expect_true(sz::succeeded(copy_status));
            accounting_string_t moved(foreign_allocator), copied(foreign_allocator);
            for (std::size_t j = 0; j < 317; j++) expect_true(sz::succeeded(moved.push_back('q')));
            moved = std::move(unique_item);
            expect_true(sz::succeeded(copied.assign(base)));
            expect_true(moved.get_allocator() == allocator && copied.get_allocator() == foreign_allocator);
            expect_equal(moved, base, "Move-assigned (foreign target) string diverged from `base` at iteration i");
            expect_equal(copied, base, "Copy-assigned (foreign target) string diverged from `base` at iteration i");
        }
    });
    expect_equal(foreign_bytes, 0);

    // Now let's clear the base and check that we're back to zero
    base = accounting_string_t(allocator);
    expect_equal(live_bytes, 0, "Allocator counter did not return to zero after clearing");
}

/** Tests the correctness of the string class update methods, such as @c push_back and @c erase. */
void test_string_updates_equivalence(std::size_t repetitions) {
    split_mix_generator_t generator {case_stream_key()};
    // Compare STL and StringZilla strings append functionality.
    char const alphabet_chars[] = "abcdefghijklmnopqrstuvwxyz";
    for (std::size_t repetition = 0; repetition != repetitions; ++repetition) {
        std::string stl_string;
        sz::string_t sz_string;
        for (std::size_t length = 1; length != 200; ++length) {
            char c = alphabet_chars[generator() % 26];
            stl_string.push_back(c);
            expect_true(sz::succeeded(sz_string.push_back(c)));
            expect_equal(sz::string_view_t(stl_string), sz::string_view_t(sz_string),
                         "sz::string_t diverged from std::string after `push_back`");
        }

        // Compare STL and StringZilla strings erase functionality.
        while (stl_string.length()) {
            std::size_t offset_to_erase = generator() % stl_string.length();
            std::size_t chars_to_erase = generator() % (stl_string.length() - offset_to_erase) + 1;
            stl_string.erase(offset_to_erase, chars_to_erase);
            sz_string.erase(offset_to_erase, chars_to_erase);
            expect_equal(sz::string_view_t(stl_string), sz::string_view_t(sz_string),
                         "sz::string_t diverged from std::string after `erase`");
        }
    }
}

#pragma endregion String Class

#pragma region Drivers

/** Drives the strings safety checks through the dispatch points; the cross files do the same per
 *  capability. */
void test_strings_safety() { check_strings_safety_(strings_dispatched); }

/** Drives the strings differential tests of the dispatch points against serial. */
void test_strings_all() {
    split_mix_generator_t generator {case_stream_key()};
    check_strings_equivalence_(generator, strings_dispatched);
}

#pragma endregion Drivers

/** Explicit template instantiations for the entry points invoked from @c main(). */
template void test_ascii_unit<sz::string_view_t>();
template void test_stl_reads_unit<std::string_view>();
template void test_stl_reads_unit<std::string>();
template void test_stl_reads_unit<sz::string_view_t>();
template void test_extensions_reads_unit<sz::string_view_t>();

} // namespace ashvardanian::stringzilla::test

/*  The owning string forwards its reads to its view, so the view runs the read tests, and this
 *  instantiation compiles every member of the owning string. */
template class ashvardanian::stringzilla::string<std::allocator<char>>;
