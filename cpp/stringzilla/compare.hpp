/**
 *  @file cpp/stringzilla/compare.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief Prefix and suffix matching over any string, through the dispatch points of @c compare.h.
 */
#ifndef STRINGZILLA_COMPARE_HPP_
#define STRINGZILLA_COMPARE_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/compare.h" // `sz_equal_best`

#include "stringzilla/string.hpp"

#include "stringzilla/devices.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief Whether @p text starts with @p prefix.
 *  @sa sz_equal_best
 */
template <a_string text_type_, a_string prefix_type_>
expected<bool> starts_with(text_type_ const &text, prefix_type_ const &prefix, sz_capability_t capabilities,
                           sz_stream_t stream) noexcept {
    string_view_t const view = view_of(text), pattern = view_of(prefix);
    if (view.size() < pattern.size()) return {false, status_t::success_k};
    sz_bool_t equal = sz_true_k;
    sz_status_t const status = sz_equal_best(view.data(), pattern.data(), pattern.size(), &equal, capabilities, stream);
    return {equal == sz_true_k, static_cast<status_t>(status)};
}

/** @copydoc starts_with */
template <a_string text_type_, a_string prefix_type_>
bool starts_with(text_type_ const &text, prefix_type_ const &prefix) noexcept {
    return default_mask_value_(starts_with(text, prefix, default_capabilities(), nullptr));
}

/** Whether @p text starts with @p character. */
template <a_string text_type_>
bool starts_with(text_type_ const &text, char character) noexcept {
    string_view_t const view = view_of(text);
    return !view.empty() && view.front() == character;
}

/**
 *  @brief Whether @p text ends with @p suffix.
 *  @sa sz_equal_best
 */
template <a_string text_type_, a_string suffix_type_>
expected<bool> ends_with(text_type_ const &text, suffix_type_ const &suffix, sz_capability_t capabilities,
                         sz_stream_t stream) noexcept {
    string_view_t const view = view_of(text), pattern = view_of(suffix);
    if (view.size() < pattern.size()) return {false, status_t::success_k};
    sz_bool_t equal = sz_true_k;
    sz_status_t const status = sz_equal_best(view.data() + view.size() - pattern.size(), pattern.data(), pattern.size(),
                                             &equal, capabilities, stream);
    return {equal == sz_true_k, static_cast<status_t>(status)};
}

/** @copydoc ends_with */
template <a_string text_type_, a_string suffix_type_>
bool ends_with(text_type_ const &text, suffix_type_ const &suffix) noexcept {
    return default_mask_value_(ends_with(text, suffix, default_capabilities(), nullptr));
}

/** Whether @p text ends with @p character. */
template <a_string text_type_>
bool ends_with(text_type_ const &text, char character) noexcept {
    string_view_t const view = view_of(text);
    return !view.empty() && view.back() == character;
}

/** @p text without @p prefix when it starts with it, like Python's <tt>str.removeprefix</tt>. */
template <a_string text_type_, a_string prefix_type_>
string_view_t remove_prefix(text_type_ const &text, prefix_type_ const &prefix) noexcept {
    string_view_t const view = view_of(text), pattern = view_of(prefix);
    return starts_with(view, pattern) ? view.substr(pattern.size()) : view;
}

/** @p text without @p suffix when it ends with it, like Python's <tt>str.removesuffix</tt>. */
template <a_string text_type_, a_string suffix_type_>
string_view_t remove_suffix(text_type_ const &text, suffix_type_ const &suffix) noexcept {
    string_view_t const view = view_of(text), pattern = view_of(suffix);
    return ends_with(view, pattern) ? view.substr(0, view.size() - pattern.size()) : view;
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_COMPARE_HPP_
