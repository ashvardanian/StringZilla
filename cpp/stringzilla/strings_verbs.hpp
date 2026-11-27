/**
 *  @file cpp/stringzilla/strings_verbs.hpp
 *  @author Ash Vardanian
 *  @date March 28, 2025
 *  @brief Verbs over columns of strings and byte look-ups, through the dispatch points of
 *      @c strings.h and the column twins of @c find.h, @c hash.h and @c utf8_runes.h.
 */
#ifndef STRINGZILLA_STRINGS_VERBS_HPP_
#define STRINGZILLA_STRINGS_VERBS_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/find.h"       // `sz_find_strings_best`
#include "stringzilla/hash.h"       // `sz_hash_strings_best`
#include "stringzilla/strings.h"    // `sz_strings_to_tape_best`, `sz_strings_split_best`, `sz_strings_join_best`
#include "stringzilla/utf8_runes.h" // `sz_utf8_count_strings_best`

#include "stringzilla/strings.hpp"

#include "stringzilla/devices.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief Finds @p needle in every string, like a @c LIKE '%needle%' filter over a column.
 *  @param[out] positions The offset of each string's first match, or @c STRINGZILLA_SIZE_MAX.
 *  @sa sz_find_strings_best
 *
 *  @par Statuses:
 *  - @c unexpected_dimensions_k @p positions is too short.
 */
inline status_t find(strings_view_t strings, string_view_t needle, std::span<std::size_t> positions,
                     sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {
    if (positions.size() < strings.size()) return status_t::unexpected_dimensions_k;
    return static_cast<status_t>(sz_find_strings_best(&strings.raw, nullptr, strings.size(), needle.data(),
                                                      needle.size(), positions.data(), capabilities, stream));
}

/**
 *  @brief Hashes every string under one @p seed, each as @c sz::hash would.
 *  @sa sz_hash_strings_best
 *
 *  @par Statuses:
 *  - @c unexpected_dimensions_k @p hashes is too short.
 */
inline status_t hashes(strings_view_t strings, std::span<u64_t> hashes, u64_t seed = 0,
                       sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {
    if (hashes.size() < strings.size()) return status_t::unexpected_dimensions_k;
    return static_cast<status_t>(
        sz_hash_strings_best(&strings.raw, nullptr, strings.size(), seed, hashes.data(), capabilities, stream));
}

/**
 *  @brief Validates every string as UTF-8 and, unless @p runes is empty, counts its runes.
 *  @param[out] runes Empty, or each string's runes, unspecified from the first ill-formed one.
 *  @param[out] malformed_index The first ill-formed string, or the column size if none is.
 *  @sa sz_utf8_count_strings_best
 *
 *  @par Statuses:
 *  - @c invalid_utf8_k A string is ill-formed.
 *  - @c unexpected_dimensions_k @p runes is too short.
 */
inline status_t utf8_count(strings_view_t strings, std::span<std::size_t> runes, std::size_t &malformed_index,
                           sz_capability_t capabilities = default_capabilities(),
                           sz_stream_t stream = nullptr) noexcept {
    if (!runes.empty() && runes.size() < strings.size()) return status_t::unexpected_dimensions_k;
    return static_cast<status_t>(sz_utf8_count_strings_best(&strings.raw, nullptr, strings.size(),
                                                            runes.empty() ? nullptr : runes.data(), &malformed_index,
                                                            capabilities, stream));
}

/**
 *  @brief Writes every string into a NUL-padded row of @p width bytes, truncating longer ones.
 *  @return The length of the longest string, to retry with wider rows, or
 *      @c unexpected_dimensions_k when @p rows holds fewer rows than the column has strings.
 *  @sa sz_strings_to_rows_best
 */
inline expected<std::size_t> to_rows(strings_view_t strings, std::span<char> rows, std::size_t width,
                                     sz_capability_t capabilities = default_capabilities(),
                                     sz_stream_t stream = nullptr) noexcept {
    if (strings.size() && rows.size() / strings.size() < width) return {0, status_t::unexpected_dimensions_k};
    sz_strings_rows_t const destination {.data = rows.data(), .stride = width, .width = width};
    sz_size_t longest = 0;
    sz_status_t const status = sz_strings_to_rows_best(&strings.raw, nullptr, strings.size(), &destination, &longest,
                                                       capabilities, stream);
    return {longest, static_cast<status_t>(status)};
}

/**
 *  @brief Copies every string into one tape, whatever the layout.
 *  @sa sz_strings_to_tape_best
 */
template <typename allocator_type_ = heap_alloc<char>>
expected<strings<allocator_type_>> gather(strings_view_t strings, allocator_type_ allocator = {},
                                          sz_capability_t capabilities = default_capabilities(),
                                          sz_stream_t stream = nullptr) noexcept {
    stringzilla::strings<allocator_type_> column(allocator);
    status_t const status = with_allocator_(column.allocator, [&](sz_allocator_t &c_allocator) {
        return sz_strings_to_tape_best(&strings.raw, nullptr, strings.size(), &c_allocator, &column.raw, capabilities,
                                       stream);
    });
    return {std::move(column), status};
}

/**
 *  @brief Escapes every string for @p kind, like JSON or CSV.
 *  @sa sz_strings_escape_best
 */
template <typename allocator_type_ = heap_alloc<char>>
expected<strings<allocator_type_>> escape(strings_view_t strings, sz_escape_kind_t kind, allocator_type_ allocator = {},
                                          sz_capability_t capabilities = default_capabilities(),
                                          sz_stream_t stream = nullptr) noexcept {
    stringzilla::strings<allocator_type_> column(allocator);
    status_t const status = with_allocator_(column.allocator, [&](sz_allocator_t &c_allocator) {
        return sz_strings_escape_best(&strings.raw, nullptr, strings.size(), kind, &c_allocator, &column.raw,
                                      capabilities, stream);
    });
    return {std::move(column), status};
}

/**
 *  @brief Unescapes every string from @p kind, checking all of them before writing any.
 *  @param[out] malformed_index The first malformed string, set on @c invalid_utf8_k.
 *  @sa sz_strings_unescape_best
 */
template <typename allocator_type_ = heap_alloc<char>>
expected<strings<allocator_type_>> unescape(strings_view_t strings, sz_escape_kind_t kind, std::size_t &malformed_index,
                                            allocator_type_ allocator = {},
                                            sz_capability_t capabilities = default_capabilities(),
                                            sz_stream_t stream = nullptr) noexcept {
    stringzilla::strings<allocator_type_> column(allocator);
    status_t const status = with_allocator_(column.allocator, [&](sz_allocator_t &c_allocator) {
        return sz_strings_unescape_best(&strings.raw, nullptr, strings.size(), kind, &c_allocator, &column.raw,
                                        &malformed_index, capabilities, stream);
    });
    return {std::move(column), status};
}

/**
 *  @brief Maps every byte of every string through @p table, like @c tolower does.
 *  @sa sz_strings_lookup_best
 */
template <typename allocator_type_ = heap_alloc<char>>
expected<strings<allocator_type_>> lookup(strings_view_t strings, look_up_table_t const &table,
                                          allocator_type_ allocator = {},
                                          sz_capability_t capabilities = default_capabilities(),
                                          sz_stream_t stream = nullptr) noexcept {
    stringzilla::strings<allocator_type_> column(allocator);
    status_t const status = with_allocator_(column.allocator, [&](sz_allocator_t &c_allocator) {
        return sz_strings_lookup_best(&strings.raw, nullptr, strings.size(), table.raw(), &c_allocator, &column.raw,
                                      capabilities, stream);
    });
    return {std::move(column), status};
}

/**
 *  @brief Brings every string into the Unicode @p form, like NFC.
 *  @sa sz_strings_normalize_best
 */
template <typename allocator_type_ = heap_alloc<char>>
expected<strings<allocator_type_>> normalize(strings_view_t strings, sz_normal_form_t form,
                                             allocator_type_ allocator = {},
                                             sz_capability_t capabilities = default_capabilities(),
                                             sz_stream_t stream = nullptr) noexcept {
    stringzilla::strings<allocator_type_> column(allocator);
    status_t const status = with_allocator_(column.allocator, [&](sz_allocator_t &c_allocator) {
        return sz_strings_normalize_best(&strings.raw, nullptr, strings.size(), form, &c_allocator, &column.raw,
                                         capabilities, stream);
    });
    return {std::move(column), status};
}

/**
 *  @brief Writes row r of @p columns as
 *      <tt>frames[0] + columns[0][r] + frames[1] + … + frames[n]</tt>, with @p between between rows,
 *      like CSV lines or JSON arrays.
 *  @sa sz_strings_join_best
 *
 *  @par Statuses:
 *  - @c unexpected_dimensions_k Uneven columns or @p frames is not one longer.
 */
template <typename allocator_type_ = std::allocator<char>>
expected<string<allocator_type_>> join(std::span<sz_strings_view_t const> columns,
                                       std::span<sz_string_view_t const> frames, string_view_t between,
                                       allocator_type_ allocator = {},
                                       sz_capability_t capabilities = default_capabilities(),
                                       sz_stream_t stream = nullptr) noexcept {
    string<allocator_type_> joined(allocator);
    sz_size_t const rows = columns.empty() ? 0 : columns.front().count;
    if (frames.size() != columns.size() + 1) return {std::move(joined), status_t::unexpected_dimensions_k};
    for (sz_strings_view_t const &column : columns)
        if (column.count != rows) return {std::move(joined), status_t::unexpected_dimensions_k};

    sz_string_view_t const separator {between.data(), between.size()};
    sz_size_t length = 0;
    status_t status = static_cast<status_t>(sz_strings_join_best(columns.data(), columns.size(), rows, frames.data(),
                                                                 separator, nullptr, &length, capabilities, stream));
    if (failed(status)) return {std::move(joined), status};
    status_t const resized = joined.resize_and_overwrite(length, [&](char *output, std::size_t) {
        status = static_cast<status_t>(sz_strings_join_best(columns.data(), columns.size(), rows, frames.data(),
                                                            separator, output, &length, capabilities, stream));
        return length;
    });
    if (failed(resized)) status = resized;
    return {std::move(joined), status};
}

/**
 *  @brief Joins every string with @p between into one, like Python's <tt>str.join</tt>.
 *  @sa sz_strings_join_best
 */
template <typename allocator_type_ = std::allocator<char>>
expected<string<allocator_type_>> join(strings_view_t strings, string_view_t between, allocator_type_ allocator = {},
                                       sz_capability_t capabilities = default_capabilities(),
                                       sz_stream_t stream = nullptr) noexcept {
    sz_string_view_t const frames[2] {};
    return join(std::span<sz_strings_view_t const>(&strings.raw, 1), frames, between, allocator, capabilities, stream);
}

/**
 *  @brief Splits @p text at the bytes of @p delimiters into an owned column, like
 *      <tt>bytes.split</tt>, while the lazy @c sz::split walks a string without allocating.
 *  @sa sz_strings_split_best
 */
template <typename allocator_type_ = heap_alloc<char>>
expected<strings<allocator_type_>> split_into(string_view_t text, byteset_t const &delimiters, sz_split_kind_t kind,
                                              allocator_type_ allocator = {},
                                              sz_capability_t capabilities = default_capabilities(),
                                              sz_stream_t stream = nullptr) noexcept {
    strings<allocator_type_> column(allocator);
    status_t const status = with_allocator_(column.allocator, [&](sz_allocator_t &c_allocator) {
        return sz_strings_split_best(text.data(), text.size(), &delimiters.raw(), kind, &c_allocator, &column.raw,
                                     capabilities, stream);
    });
    return {std::move(column), status};
}

/**
 *  @brief Splits @p text into an owned column of lines without their breaks, like
 *      <tt>str.splitlines</tt>.
 *  @sa sz_strings_split_lines_best
 */
template <typename allocator_type_ = heap_alloc<char>>
expected<strings<allocator_type_>> split_lines(string_view_t text, allocator_type_ allocator = {},
                                               sz_capability_t capabilities = default_capabilities(),
                                               sz_stream_t stream = nullptr) noexcept {
    strings<allocator_type_> column(allocator);
    status_t const status = with_allocator_(column.allocator, [&](sz_allocator_t &c_allocator) {
        return sz_strings_split_lines_best(text.data(), text.size(), &c_allocator, &column.raw, capabilities, stream);
    });
    return {std::move(column), status};
}

/**
 *  @brief Maps all characters in the @p source string into the @p target buffer using the provided
 *      lookup @p table.
 *  @param[out] target Where the mapped characters go, @p source's own bytes for an in-place map.
 *  @sa sz_strings_lookup_best
 */
inline status_t lookup(string_view_t source, look_up_table_t const &table, char *target,
                       sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {
    // A one-string tape over the caller's buffers, in the offsets width the kernel writes, so it
    // allocates nothing; in place it shares the offsets, making it the very tape the kernel reads.
    sz_size_t const length = source.size();
    sz_u32_t narrow_source[2] = {0, (sz_u32_t)length}, narrow_target[2] = {0, (sz_u32_t)length};
    sz_u64_t wide_source[2] = {0, length}, wide_target[2] = {0, length};
    bool const in_place = source.data() == target;
    sz_strings_view_t strings;
    sz_strings_span_t tape;
    strings.count = tape.count = 1;
    if (length <= 0xFFFFFFFFull) {
        strings.layout = tape.layout = sz_strings_u32_tape_k;
        strings.data.u32_tape.data = source.data(), strings.data.u32_tape.offsets = narrow_source;
        tape.data.u32_tape.data = target, tape.data.u32_tape.offsets = in_place ? narrow_source : narrow_target;
    }
    else {
        strings.layout = tape.layout = sz_strings_u64_tape_k;
        strings.data.u64_tape.data = source.data(), strings.data.u64_tape.offsets = wide_source;
        tape.data.u64_tape.data = target, tape.data.u64_tape.offsets = in_place ? wide_source : wide_target;
    }
    return static_cast<status_t>(
        sz_strings_lookup_best(&strings, nullptr, 1, (sz_cptr_t)table.raw(), nullptr, &tape, capabilities, stream));
}

/**
 *  @brief Replaces @b (in-place) all characters in the string using the provided lookup table.
 *  @sa sz_strings_lookup_best
 */
inline status_t lookup(string_span_t string, look_up_table_t const &table,
                       sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {
    return lookup(string_view_t(string.data(), string.size()), table, string.data(), capabilities, stream);
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_STRINGS_VERBS_HPP_
