/**
 *  @file cpp/stringzilla/utf8_graphemes.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief The lazy range of UAX-29 grapheme clusters, through @c utf8_graphemes.h.
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_HPP_
#define STRINGZILLA_UTF8_GRAPHEMES_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/utf8_graphemes.h"

#include "stringzilla/utf8_runes.hpp" // `utf8_split_view`, `utf8_segments_view`

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief A range of string slices split at UAX-29 grapheme cluster boundaries, in order.
 *
 *  Unlike whitespace splitting, the graphemes tile the input: every byte belongs to exactly one
 *  grapheme, so consecutive graphemes are contiguous and no empty segments are produced. Drives
 *  @c sz_utf8_graphemes_best.
 *
 *  @tparam string_type_ String type, like @c string_view_t, @c string_slice_t, or @c std::string.
 */
template <typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k>
using utf8_graphemes_view = utf8_segments_view<sz_utf8_graphemes_best, string_type_, steps_>;

/** Lazily splits @p text into UAX-29 grapheme clusters. */
template <a_string text_type_>
    requires a_borrowed_string<text_type_ &&>
utf8_graphemes_view<string_view_t> utf8_graphemes(text_type_ &&text) noexcept {
    return {view_of(text)};
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_UTF8_GRAPHEMES_HPP_
