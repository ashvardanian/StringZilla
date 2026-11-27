/**
 *  @file include/stringzilla/strings.hpp
 *  @author Ash Vardanian
 *  @date March 28, 2025
 *  @brief Columns of strings: the borrowed @c strings_view_t and the owned @c strings tape.
 */
#ifndef STRINGZILLA_STRINGS_HPP_
#define STRINGZILLA_STRINGS_HPP_

#include "stringzilla/string.hpp"

namespace ashvardanian {
namespace stringzilla {

#pragma region String Collections

template <typename allocator_type_>
struct strings;

/**
 *  @brief A borrowed column in any layout of @c sz_strings_view_t, the twin of @c string_view_t.
 *
 *  The verbs over a column, like @c sz::gather and @c sz::join, live in the C++ SDK.
 */
struct strings_view_t {
    using value_type = string_view_t;
    using iterator = indexed_container_iterator<strings_view_t>;

    /** The C view every member reads. */
    sz_strings_view_t raw;

    /** Wraps a C view, like one over records or NUL-padded rows. */
    constexpr explicit strings_view_t(sz_strings_view_t raw) noexcept : raw(raw) {}

    /** An Apache Arrow @c String column: string i is <tt>[offsets[i], offsets[i + 1])</tt> of @p data. */
    constexpr strings_view_t(char const *data, sz_u32_t const *offsets, std::size_t count) noexcept : raw {} {
        raw.layout = sz_strings_u32_tape_k, raw.count = count;
        raw.data.u32_tape.data = data, raw.data.u32_tape.offsets = offsets;
    }

    /** An Apache Arrow @c LargeString column, the same with 64-bit offsets. */
    constexpr strings_view_t(char const *data, sz_u64_t const *offsets, std::size_t count) noexcept : raw {} {
        raw.layout = sz_strings_u64_tape_k, raw.count = count;
        raw.data.u64_tape.data = data, raw.data.u64_tape.offsets = offsets;
    }

    /** Views @p strings through an accessor, copying no bytes, so they must outlive the view. */
    strings_view_t(std::span<string_view_t const> strings) noexcept;

    /** Views the tape of @p strings. */
    template <typename allocator_type_>
    strings_view_t(strings<allocator_type_> const &strings) noexcept : strings_view_t(strings.view()) {}

    constexpr std::size_t size() const noexcept { return raw.count; }
    constexpr bool empty() const noexcept { return raw.count == 0; }
    constexpr iterator begin() const noexcept { return {*this, 0}; }
    constexpr iterator end() const noexcept { return {*this, size()}; }

    /** The string at @p index, read from a tape in place; an accessor runs only on the host. */
    constexpr string_view_t operator[](std::size_t index) const noexcept {
        if (raw.layout == sz_strings_u32_tape_k) {
            sz_u32_t const *offsets = raw.data.u32_tape.offsets + index;
            return {raw.data.u32_tape.data + offsets[0], offsets[1] - offsets[0]};
        }
        if (raw.layout == sz_strings_u64_tape_k) {
            sz_u64_t const *offsets = raw.data.u64_tape.offsets + index;
            return {raw.data.u64_tape.data + offsets[0], static_cast<std::size_t>(offsets[1] - offsets[0])};
        }
        sz_size_t const position = index;
        sz_string_view_t string {};
        raw.data.accessor.get(raw.data.accessor.handle, &position, 1, &string);
        return {string.start, string.length};
    }

    /** Deleted: <tt>sz::find(strings, needle, positions)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    void find(arguments_types_ &&...) const = delete;

    /** Deleted: <tt>sz::hashes(strings, hashes)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    void hashes(arguments_types_ &&...) const = delete;

    /** Deleted: <tt>sz::utf8_count(strings, runes, malformed_index)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    void utf8_count(arguments_types_ &&...) const = delete;

    /** Deleted: <tt>sz::to_rows(strings, rows, width)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    void to_rows(arguments_types_ &&...) const = delete;

    /** Deleted: <tt>sz::gather(strings)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    void gather(arguments_types_ &&...) const = delete;

    /** Deleted: <tt>sz::escape(strings, kind)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    void escape(arguments_types_ &&...) const = delete;

    /** Deleted: <tt>sz::unescape(strings, kind, malformed_index)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    void unescape(arguments_types_ &&...) const = delete;

    /** Deleted: <tt>sz::lookup(strings, table)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    void lookup(arguments_types_ &&...) const = delete;

    /** Deleted: <tt>sz::normalize(strings, form)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    void normalize(arguments_types_ &&...) const = delete;

    /** Deleted: <tt>sz::join(strings, between)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    void join(arguments_types_ &&...) const = delete;
};

/**
 *  @brief An owned column, the twin of @c string: the C span and the allocator of its blocks.
 *
 *  Only the column verbs and the splits of the C++ SDK fill one, so it never grows and moves
 *  without copies.
 *  Its offsets are 32-bit while the payload fits in 4 GiB and 64-bit beyond.
 */
template <typename allocator_type_ = heap_alloc<char>>
struct strings {
    using allocator_type = allocator_type_;
    using value_type = string_view_t;
    using iterator = indexed_container_iterator<strings>;

    /** The tape, zeroed while empty. */
    sz_strings_span_t raw {};

    /** Where both blocks of @c raw come from and return to. */
    sz_no_unique_address_ allocator_type_ allocator {};

    strings() noexcept = default;
    explicit strings(allocator_type_ allocator) noexcept : allocator(allocator) {}
    strings(strings const &) = delete;
    strings &operator=(strings const &) = delete;
    strings(strings &&other) noexcept : raw(other.raw), allocator(std::move(other.allocator)) { other.raw = {}; }
    strings &operator=(strings &&other) noexcept {
        if (this == &other) return *this;
        release_();
        raw = other.raw, allocator = std::move(other.allocator);
        other.raw = {};
        return *this;
    }
    ~strings() noexcept { release_(); }

    std::size_t size() const noexcept { return raw.count; }
    bool empty() const noexcept { return raw.count == 0; }
    string_view_t operator[](std::size_t index) const noexcept;
    iterator begin() const noexcept { return {*this, 0}; }
    iterator end() const noexcept { return {*this, size()}; }

    strings_view_t view() const noexcept;

    /** Deleted: <tt>sz::split_into(text, delimiters, kind)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    static void split(arguments_types_ &&...) = delete;

    /** Deleted: <tt>sz::split_lines(text)</tt>, a kernel of ours. */
    template <typename... arguments_types_>
    static void split_lines(arguments_types_ &&...) = delete;

  private:
    void release_() noexcept {
        [[maybe_unused]] status_t const status = with_allocator_(allocator, [&](sz_allocator_t &c_allocator) {
            sz_strings_span_free(&raw, &c_allocator, nullptr);
            return sz_success_k;
        });
    }
};

/** The owned column on the heap, the twin of @c string_t. */
using strings_t = strings<>;

inline strings_view_t::strings_view_t(std::span<string_view_t const> strings) noexcept : raw {} {
    raw.layout = sz_strings_accessor_k, raw.count = strings.size();
    raw.data.accessor.handle = strings.data();
    raw.data.accessor.get = [](void const *handle, sz_size_t const *indices, sz_size_t count, sz_string_view_t *views) {
        string_view_t const *strings = static_cast<string_view_t const *>(handle);
        for (sz_size_t position = 0; position != count; ++position) {
            views[position].start = strings[indices[position]].data();
            views[position].length = strings[indices[position]].size();
        }
    };
}

/** Defined once @c string_slice is complete: inside the class, nvcc's front end, emulating Clang,
 *  asserts resolving the @c strings_view_t span constructor against the incomplete slice. */
template <typename allocator_type_>
strings_view_t strings<allocator_type_>::view() const noexcept {
    sz_strings_view_t strings;
    sz_strings_from_span(&raw, &strings);
    return strings_view_t(strings);
}

template <typename allocator_type_>
string_view_t strings<allocator_type_>::operator[](std::size_t index) const noexcept {
    strings_view_t const strings = view();
    return strings[index];
}

#pragma endregion String Collections

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_STRINGS_HPP_
