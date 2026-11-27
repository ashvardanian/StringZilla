/**
 *  @file cpp/stringzilla/utf8_wordbreaks.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief The lazy range of UAX-29 word segments, through @c utf8_wordbreaks.h.
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_HPP_
#define STRINGZILLA_UTF8_WORDBREAKS_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/utf8_wordbreaks.h"

#include "stringzilla/utf8_runes.hpp" // `utf8_split_view`, `utf8_segments_view`

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief A range of UAX-29 word segments, in order.
 *
 *  Word segmentation, driven by @c sz_utf8_wordbreaks_best, tiles the input: every byte belongs
 *  to exactly one segment - alternating "words" (letter / number / mark / CJK runs) and the
 *  whitespace / punctuation between them - so consecutive segments are contiguous, never empty.
 *
 *  @tparam string_type_ String type, like @c string_view_t, @c string_slice_t, or @c std::string.
 *  @tparam steps_ Segments fetched per kernel call.
 */
template <typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k>
using utf8_wordbreaks_view = utf8_segments_view<sz_utf8_wordbreaks_best, string_type_, steps_>;

/** Lazily yields the UAX-29 word segments of @p text, tiling it. */
template <a_string text_type_>
    requires a_borrowed_string<text_type_ &&>
utf8_wordbreaks_view<string_view_t> utf8_wordbreaks(text_type_ &&text) noexcept {
    return {view_of(text)};
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_UTF8_WORDBREAKS_HPP_
