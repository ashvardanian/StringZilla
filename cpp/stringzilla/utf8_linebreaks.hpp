/**
 *  @file cpp/stringzilla/utf8_linebreaks.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief The lazy range of UAX-14 line-breakable segments, through @c utf8_linebreaks.h.
 */
#ifndef STRINGZILLA_UTF8_LINEBREAKS_HPP_
#define STRINGZILLA_UTF8_LINEBREAKS_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/utf8_linebreaks.h"

#include "stringzilla/utf8_runes.hpp" // `utf8_split_view`, `utf8_segments_view`

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief A range of string slices split at UAX-14 line break opportunities, in order.
 *
 *  Unlike whitespace splitting, the segments tile the input: every byte belongs to exactly one
 *  segment, so consecutive segments are contiguous and no empty segments are produced. Each segment
 *  ends at an allowed line break opportunity (both mandatory hard breaks and soft wrap points).
 *  Drives @c sz_utf8_linebreaks_best. Split with @c utf8_split_newlines for hard line breaks only.
 *
 *  @tparam string_type_ String type, like @c string_view_t, @c string_slice_t, or @c std::string.
 */
template <typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k>
using utf8_linebreaks_view = utf8_segments_view<sz_utf8_linebreaks_best, string_type_, steps_>;

/** Lazily splits @p text into UAX-14 line-breakable segments. */
template <a_string text_type_>
    requires a_borrowed_string<text_type_ &&>
utf8_linebreaks_view<string_view_t> utf8_linebreaks(text_type_ &&text) noexcept {
    return {view_of(text)};
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_UTF8_LINEBREAKS_HPP_
