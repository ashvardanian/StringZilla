/**
 *  @file cpp/stringzilla/utf8_norm.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief Unicode normalization checks and in-place normalization, through @c utf8_norm.h.
 */
#ifndef STRINGZILLA_UTF8_NORM_HPP_
#define STRINGZILLA_UTF8_NORM_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/utf8_norm.h" // `sz_utf8_norm_best`, `sz_utf8_find_denormalized_best`

#include "stringzilla/string.hpp"

#include "stringzilla/devices.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief The offset of the first byte of @p text breaking the normalization @p form.
 *  @param[in] form One of @c sz_normal_form_nfd_k, @c sz_normal_form_nfc_k,
 *      @c sz_normal_form_nfkd_k, or @c sz_normal_form_nfkc_k.
 *  @return The offset, or @ref npos_of the text when it is already in @p form.
 *  @sa sz_utf8_find_denormalized_best
 */
template <a_string text_type_>
expected<std::size_t> utf8_find_denormalized(text_type_ const &text, sz_normal_form_t form,
                                             sz_capability_t capabilities, sz_stream_t stream) noexcept {
    string_view_t const view = view_of(text);
    sz_cptr_t match = nullptr;
    sz_status_t const status = sz_utf8_find_denormalized_best(view.data(), view.size(), form, &match, capabilities,
                                                              stream);
    return {match ? static_cast<std::size_t>(match - view.data()) : npos_of<text_type_>(),
            static_cast<status_t>(status)};
}

/** @copydoc utf8_find_denormalized */
template <a_string text_type_>
std::size_t utf8_find_denormalized(text_type_ const &text, sz_normal_form_t form) noexcept {
    return default_mask_value_(utf8_find_denormalized(text, form, default_capabilities(), nullptr));
}

/** Whether @p text is already in the normalization @p form. */
template <a_string text_type_>
bool is_normalized(text_type_ const &text, sz_normal_form_t form) noexcept {
    return utf8_find_denormalized(view_of(text), form) == string_view_t::npos;
}

/**
 *  @brief Brings @p text @b in-place into the normalization @p form, which may grow it up to 18×.
 *  @sa sz_utf8_norm_best
 */
template <typename allocator_type_>
status_t utf8_normalize(string<allocator_type_> &text, sz_normal_form_t form,
                        sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {
    string_view_t const source = text.view();
    string<allocator_type_> result(text.get_allocator());
    status_t status = status_t::success_k;
    status_t const resized = result.resize_and_overwrite(source.size() * 18, [&](char *output, std::size_t) {
        sz_size_t length = 0;
        status = static_cast<status_t>(
            sz_utf8_norm_best(source.data(), source.size(), form, output, &length, capabilities, stream));
        return length;
    });
    if (failed(resized)) return resized;
    if (failed(status)) return status;
    text.swap(result);
    return status_t::success_k;
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_UTF8_NORM_HPP_
