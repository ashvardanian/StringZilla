/**
 *  @file cpp/stringzilla/utf8_tokens.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief Lazy ranges split at Unicode newlines, whitespace and delimiters, over @c utf8_tokens.h.
 */
#ifndef STRINGZILLA_UTF8_TOKENS_HPP_
#define STRINGZILLA_UTF8_TOKENS_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/utf8_tokens.h"

#include "stringzilla/utf8_runes.hpp" // `utf8_split_view`, `utf8_segments_view`

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief A range of string slices @b between UTF-8 newlines, i.e. the lines, like a
 *      Unicode-aware <tt>splitlines()</tt>.
 *
 *  Splits on all 7 Unicode newline characters plus the CRLF sequence (LF, VT, FF, CR, NEL, LS, PS,
 *  CRLF) and yields the segments between them, while @c utf8_newlines_view yields the runs.
 *
 *  @tparam string_type_ String type, like @c string_view_t, @c string_slice_t, or @c std::string.
 */
template <typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k,
          empty_segments_t empties_ = empty_segments_t::keep_k>
using utf8_split_newlines_view =
    utf8_split_view<sz_utf8_newlines_best, string_type_, steps_, split_parts_t::between_k, empties_>;

/** A range of the UTF-8 newline runs themselves (LF, CR, CRLF, NEL, LS, PS, ...); see
 *  @c utf8_split_newlines_view. */
template <typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k,
          empty_segments_t empties_ = empty_segments_t::keep_k>
using utf8_newlines_view =
    utf8_split_view<sz_utf8_newlines_best, string_type_, steps_, split_parts_t::separators_k, empties_>;

/**
 *  @brief A range of string slices @b between UTF-8 whitespace runs - i.e. the tokens.
 *
 *  Splits on all 25 Unicode White_Space characters and yields the segments between them (empties
 *  kept by default). See @c utf8_whitespaces_view for the whitespace runs themselves.
 *
 *  @tparam string_type_ String type, like @c string_view_t, @c string_slice_t, or @c std::string.
 */
template <typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k,
          empty_segments_t empties_ = empty_segments_t::keep_k>
using utf8_split_whitespaces_view =
    utf8_split_view<sz_utf8_whitespaces_best, string_type_, steps_, split_parts_t::between_k, empties_>;

/** A range of the UTF-8 whitespace runs themselves; see @c utf8_split_whitespaces_view for the
 *  tokens between them. */
template <typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k,
          empty_segments_t empties_ = empty_segments_t::keep_k>
using utf8_whitespaces_view =
    utf8_split_view<sz_utf8_whitespaces_best, string_type_, steps_, split_parts_t::separators_k, empties_>;

/**
 *  @brief A range of string slices @b between any UTF-8 delimiter codepoints - i.e. the fields.
 *
 *  The superset of @c utf8_split_whitespaces_view: splits on every punctuation (P*), symbol (S*),
 *  or separator/whitespace (Z*) codepoint and yields the segments between them. See
 *  @c utf8_delimiters_view for the delimiter runs.
 *
 *  @tparam string_type_ String type, like @c string_view_t, @c string_slice_t, or @c std::string.
 */
template <typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k,
          empty_segments_t empties_ = empty_segments_t::keep_k>
using utf8_split_delimiters_view =
    utf8_split_view<sz_utf8_delimiters_best, string_type_, steps_, split_parts_t::between_k, empties_>;

/** A range of the UTF-8 delimiter runs themselves (P/S/Z categories); see
 *  @c utf8_split_delimiters_view. */
template <typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k,
          empty_segments_t empties_ = empty_segments_t::keep_k>
using utf8_delimiters_view =
    utf8_split_view<sz_utf8_delimiters_best, string_type_, steps_, split_parts_t::separators_k, empties_>;

/** Lazily splits @p text into the lines @b between Unicode newlines. */
template <a_string text_type_>
    requires a_borrowed_string<text_type_ &&>
utf8_split_newlines_view<string_view_t> utf8_split_newlines(text_type_ &&text) noexcept {
    return {view_of(text)};
}

/** Lazily yields the Unicode newline runs of @p text themselves. */
template <a_string text_type_>
    requires a_borrowed_string<text_type_ &&>
utf8_newlines_view<string_view_t> utf8_newlines(text_type_ &&text) noexcept {
    return {view_of(text)};
}

/** Lazily splits @p text into the tokens @b between Unicode whitespace runs. */
template <a_string text_type_>
    requires a_borrowed_string<text_type_ &&>
utf8_split_whitespaces_view<string_view_t> utf8_split_whitespaces(text_type_ &&text) noexcept {
    return {view_of(text)};
}

/** Lazily yields the Unicode whitespace runs of @p text themselves. */
template <a_string text_type_>
    requires a_borrowed_string<text_type_ &&>
utf8_whitespaces_view<string_view_t> utf8_whitespaces(text_type_ &&text) noexcept {
    return {view_of(text)};
}

/** Lazily splits @p text into the fields @b between Unicode delimiters. */
template <a_string text_type_>
    requires a_borrowed_string<text_type_ &&>
utf8_split_delimiters_view<string_view_t> utf8_split_delimiters(text_type_ &&text) noexcept {
    return {view_of(text)};
}

/** Lazily yields the Unicode delimiter runs of @p text themselves. */
template <a_string text_type_>
    requires a_borrowed_string<text_type_ &&>
utf8_delimiters_view<string_view_t> utf8_delimiters(text_type_ &&text) noexcept {
    return {view_of(text)};
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_UTF8_TOKENS_HPP_
