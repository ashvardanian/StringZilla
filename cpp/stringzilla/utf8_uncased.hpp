/**
 *  @file cpp/stringzilla/utf8_uncased.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief Case-insensitive UTF-8 search, order and folding, through @c utf8_uncased.h and
 *      @c utf8_uncased_fold.h.
 */
#ifndef STRINGZILLA_UTF8_UNCASED_HPP_
#define STRINGZILLA_UTF8_UNCASED_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/utf8_uncased.h"      // `sz_utf8_uncased_search_best`, `sz_utf8_uncased_order_best`
#include "stringzilla/utf8_uncased_fold.h" // `sz_utf8_uncased_fold_best`

#include "stringzilla/find.hpp" // `find_matches_view`, `a_borrowed_string`

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief A UTF-8 needle prepared once by @c sz_utf8_uncased_needle_init_best, for any number
 *      of uncased searches.
 *
 *  Borrows the needle's bytes, which must outlive it.
 *
 *  @code{.cpp}
 *  sz::utf8_uncased_needle_t pattern("hello");
 *  for (auto const& haystack : haystacks) {
 *      auto match = sz::utf8_uncased_search(haystack, pattern);
 *      if (match) { ... }
 *  }
 *  @endcode
 */
class utf8_uncased_needle_t {
    sz_utf8_uncased_needle_t prepared_;

  public:
    explicit utf8_uncased_needle_t(char const *needle, std::size_t length) noexcept {
        [[maybe_unused]] sz_status_t const status = sz_utf8_uncased_needle_init_best(needle, length, &prepared_,
                                                                                     default_capabilities(), nullptr);
        sz_assert_(status == sz_success_k);
    }

    explicit utf8_uncased_needle_t(string_view_t needle) noexcept
        : utf8_uncased_needle_t(needle.data(), needle.size()) {}

    template <std::size_t array_length_>
    explicit utf8_uncased_needle_t(char const (&needle)[array_length_]) noexcept
        : utf8_uncased_needle_t(needle, array_length_ - 1) {}

    char const *data() const noexcept { return prepared_.start; }
    std::size_t size() const noexcept { return prepared_.length; }
    sz_utf8_uncased_needle_t const &raw() const noexcept { return prepared_; }
};

/** The bytes of the UTF-8 sequence @p lead_byte opens, a stray continuation byte counting as one,
 *  as the decoders step. */
constexpr std::size_t utf8_lead_length_(sz_u8_t lead_byte) noexcept {
    return 1u + (lead_byte >= 0xC0u) + (lead_byte >= 0xE0u) + (lead_byte >= 0xF0u);
}

/**
 *  @brief Stateful matcher driving @ref find_matches_view over @b case-insensitive UTF-8 matches.
 *
 *  The uncased twin of @c matcher_find: @c sz_utf8_uncased_search_best resolves each match of the
 *  prepared @ref utf8_uncased_needle_t. Because case folding can change a match's byte length,
 *  @c needle_length reports the byte span of the @b last match rather than the needle's own
 *  length. Matches are reported @b non-overlapping and on codepoint boundaries, so an empty match
 *  steps over one codepoint: an empty needle matches at every boundary, the end included.
 *
 *  @tparam string_type_ The haystack view type, like @c string_view_t, from @ref find_matches_view.
 */
template <typename string_type_>
struct matcher_utf8_uncased_search {
    using size_type = typename string_type_::size_type;
    utf8_uncased_needle_t needle_;

    /** Byte span of the last match, as folding may change it. */
    size_type matched_length_ = 0;

    /** Bytes to step past the last match: all of it, or one codepoint for an empty one. */
    size_type skipped_length_ = 0;

    matcher_utf8_uncased_search(utf8_uncased_needle_t needle) noexcept : needle_(needle) {}
    size_type needle_length() const noexcept { return matched_length_; }
    size_type skip_length() const noexcept { return skipped_length_; }
    size_type operator()(string_type_ haystack) noexcept {
        sz_size_t match_length = 0;
        sz_cptr_t match = nullptr;
        [[maybe_unused]] sz_status_t const status = sz_utf8_uncased_search_best(
            haystack.data(), haystack.size(), &needle_.raw(), &match, &match_length, default_capabilities(), nullptr);
        sz_assert_(status == sz_success_k);
        if (!match) return string_type_::npos;
        size_type const offset = static_cast<size_type>(match - haystack.data());
        size_type const rest = haystack.size() - offset;
        matched_length_ = static_cast<size_type>(match_length);
        skipped_length_ = match_length || !rest ? matched_length_
                                                : sz_min_of_two(utf8_lead_length_(sz_bitcast_(sz_u8_t, *match)), rest);
        return offset;
    }
};

/** A case-insensitive match: its byte offset and its byte length, which folding may change. */
struct sized_match_t {
    std::size_t offset {};
    std::size_t length {};

    sized_match_t() noexcept = default;
    sized_match_t(std::size_t o, std::size_t l) noexcept : offset(o), length(l) {}

    operator bool() const noexcept { return offset != string_view_t::npos; }
    bool operator==(sized_match_t const &other) const noexcept {
        return offset == other.offset && length == other.length;
    }
    bool operator!=(sized_match_t const &other) const noexcept { return !(*this == other); }
};

/**
 *  @brief Finds the first case-insensitive match of a prepared @p needle in @p haystack.
 *  @return The match, or an offset of @c string_view_t::npos when there is none.
 *  @sa sz_utf8_uncased_search_best
 */
template <a_string haystack_type_>
expected<sized_match_t> utf8_uncased_search(haystack_type_ const &haystack, utf8_uncased_needle_t const &needle,
                                            sz_capability_t capabilities, sz_stream_t stream) noexcept {
    string_view_t const text = view_of(haystack);
    sz_size_t match_length = 0;
    sz_cptr_t match = nullptr;
    sz_status_t const status = sz_utf8_uncased_search_best(text.data(), text.size(), &needle.raw(), &match,
                                                           &match_length, capabilities, stream);
    if (!match) return {{string_view_t::npos, 0}, static_cast<status_t>(status)};
    return {{static_cast<std::size_t>(match - text.data()), match_length}, static_cast<status_t>(status)};
}

/** @copydoc utf8_uncased_search */
template <a_string haystack_type_>
sized_match_t utf8_uncased_search(haystack_type_ const &haystack, utf8_uncased_needle_t const &needle) noexcept {
    return default_mask_value_(utf8_uncased_search(haystack, needle, default_capabilities(), nullptr));
}

/** Finds the first case-insensitive match of @p needle, preparing it for this one search. */
template <a_string haystack_type_, a_string needle_type_>
sized_match_t utf8_uncased_search(haystack_type_ const &haystack, needle_type_ const &needle) noexcept {
    return utf8_uncased_search(haystack, utf8_uncased_needle_t(view_of(needle)));
}

/**
 *  @brief Lazily yields all @b non-overlapping case-insensitive matches of a prepared @p needle.
 *  @note An @b empty needle matches at every codepoint boundary, the end included.
 */
template <a_string haystack_type_>
    requires a_borrowed_string<haystack_type_ &&>
find_matches_view<string_view_t, matcher_utf8_uncased_search<string_view_t>> utf8_uncased_matches(
    haystack_type_ &&haystack, utf8_uncased_needle_t const &needle) noexcept {
    return {view_of(haystack), matcher_utf8_uncased_search<string_view_t>(needle)};
}

/** Lazily yields all @b non-overlapping case-insensitive matches of @p needle, whose bytes must
 *  outlive the range, as the prepared needle borrows them. */
template <a_string haystack_type_, a_string needle_type_>
    requires a_borrowed_string<haystack_type_ &&> && a_borrowed_string<needle_type_ &&>
find_matches_view<string_view_t, matcher_utf8_uncased_search<string_view_t>> utf8_uncased_matches(
    haystack_type_ &&haystack, needle_type_ &&needle) noexcept {
    return {view_of(haystack), matcher_utf8_uncased_search<string_view_t>(utf8_uncased_needle_t(view_of(needle)))};
}

/**
 *  @brief Orders @p first and @p second ignoring case, with lengths breaking ties.
 *  @return -1, 0 or 1.
 *  @sa sz_utf8_uncased_order_best
 */
template <a_string first_type_, a_string second_type_>
expected<int> utf8_uncased_order(first_type_ const &first, second_type_ const &second, sz_capability_t capabilities,
                                 sz_stream_t stream) noexcept {
    string_view_t const a = view_of(first), b = view_of(second);
    sz_ordering_t ordering = sz_equal_k;
    sz_status_t const status = sz_utf8_uncased_order_best(a.data(), a.size(), b.data(), b.size(), &ordering,
                                                          capabilities, stream);
    return {static_cast<int>(ordering), static_cast<status_t>(status)};
}

/** @copydoc utf8_uncased_order */
template <a_string first_type_, a_string second_type_>
int utf8_uncased_order(first_type_ const &first, second_type_ const &second) noexcept {
    return default_mask_value_(utf8_uncased_order(first, second, default_capabilities(), nullptr));
}

/**
 *  @brief Applies Unicode case folding to @p text @b in-place, which may grow it up to 3×, like
 *      U+00DF becoming "ss".
 *  @sa sz_utf8_uncased_fold_best
 */
template <typename allocator_type_>
status_t utf8_uncased_fold(string<allocator_type_> &text, sz_capability_t capabilities = default_capabilities(),
                           sz_stream_t stream = nullptr) noexcept {
    string_view_t const source = text.view();
    string<allocator_type_> result(text.get_allocator());
    status_t status = status_t::success_k;
    status_t const resized = result.resize_and_overwrite(source.size() * 3, [&](char *output, std::size_t) {
        sz_size_t length = 0;
        status = static_cast<status_t>(
            sz_utf8_uncased_fold_best(source.data(), source.size(), output, &length, capabilities, stream));
        return length;
    });
    if (failed(resized)) return resized;
    if (failed(status)) return status;
    text.swap(result);
    return status_t::success_k;
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_UTF8_UNCASED_HPP_
