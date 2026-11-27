/**
 *  @file cpp/stringzilla/hash.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief Hashes, checksums and random bytes over any string, through the dispatch points of
 *      @c hash.h, with the @c hash_t functor and the @c std::hash specializations.
 */
#ifndef STRINGZILLA_HASH_HPP_
#define STRINGZILLA_HASH_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include <cstdint> // `std::uint64_t`
#include <cstdlib> // `std::rand`

#include "stringzilla/hash.h" // `sz_hash_best`, `sz_hash_multiseed_best`, `sz_bytesum_best`

#include "stringzilla/string.hpp"

#include "stringzilla/devices.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief Hashes @p text under @p seed.
 *  @sa sz_hash_best
 */
template <a_string text_type_>
expected<std::uint64_t> hash(text_type_ const &text, std::uint64_t seed, sz_capability_t capabilities,
                             sz_stream_t stream) noexcept {
    string_view_t const view = view_of(text);
    sz_u64_t hashed = 0;
    sz_status_t const status = sz_hash_best(view.data(), view.size(), seed, &hashed, capabilities, stream);
    return {hashed, static_cast<status_t>(status)};
}

/** @copydoc hash */
template <a_string text_type_>
std::uint64_t hash(text_type_ const &text, std::uint64_t seed = 0) noexcept {
    return default_mask_value_(hash(text, seed, default_capabilities(), nullptr));
}

/**
 *  @brief Hashes @p text under each of @p seeds at once, one hash per seed into @p hashes.
 *  @note Equivalent to <tt>hashes[i] = hash(text, seeds[i])</tt>, but amortizes the input loading.
 *  @sa sz_hash_multiseed_best
 *
 *  @par Statuses:
 *  - @c unexpected_dimensions_k @p hashes and @p seeds differ in length.
 */
template <a_string text_type_>
status_t hash_multiseed(text_type_ const &text, std::span<std::uint64_t const> seeds, std::span<std::uint64_t> hashes,
                        sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {
    if (seeds.size() != hashes.size()) return status_t::unexpected_dimensions_k;
    string_view_t const view = view_of(text);
    return static_cast<status_t>(
        sz_hash_multiseed_best(view.data(), view.size(), reinterpret_cast<sz_u64_t const *>(seeds.data()), seeds.size(),
                               reinterpret_cast<sz_u64_t *>(hashes.data()), capabilities, stream));
}

/**
 *  @brief Sums the values of the bytes of @p text.
 *  @sa sz_bytesum_best
 */
template <a_string text_type_>
expected<std::uint64_t> bytesum(text_type_ const &text, sz_capability_t capabilities, sz_stream_t stream) noexcept {
    string_view_t const view = view_of(text);
    sz_u64_t sum = 0;
    sz_status_t const status = sz_bytesum_best(view.data(), view.size(), &sum, capabilities, stream);
    return {sum, static_cast<status_t>(status)};
}

/** @copydoc bytesum */
template <a_string text_type_>
std::uint64_t bytesum(text_type_ const &text) noexcept {
    return default_mask_value_(bytesum(text, default_capabilities(), nullptr));
}

/**
 *  @brief Overwrites the @p string slice with random bytes.
 *  @param[out] string The string to overwrite.
 *  @param[in] nonce "Number used once" seeding the random number generator, @b never repeat it!
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *  @sa sz_fill_random_best
 */
inline status_t fill_random(string_span_t string, std::uint64_t nonce,
                            sz_capability_t capabilities = default_capabilities(),
                            sz_stream_t stream = nullptr) noexcept {
    return static_cast<status_t>(
        sz_fill_random_best(string.data(), string.size(), static_cast<sz_u64_t>(nonce), capabilities, stream));
}

/**
 *  @brief Overwrites the @p string slice with random bytes using @c std::rand for the nonce.
 *  @sa sz_fill_random_best
 */
inline status_t fill_random(string_span_t string) noexcept {
    return fill_random(string, static_cast<std::uint64_t>(std::rand()));
}

/**
 *  @brief Helper function-like object to hash string-view convertible objects with StringZilla.
 *
 *  Similar to <tt>std::hash&lt;std::string_view&gt;</tt>, but doesn't require including the heavy
 *  <tt>&lt;functional&gt;</tt> header. Can be used to combine STL classes with StringZilla logic, like:
 *  <tt>std::unordered_map&lt;std::string, int, sz::hash_t, sz::equal_to_t&gt;</tt>.
 *
 *  @see std::hash: https://en.cppreference.com/w/cpp/utility/functional/hash
 */
struct hash_t {

    /** Lets hashed containers find owning keys by views, beside a transparent @ref equal_to_t. */
    using is_transparent = void;

    /** Hashes @p text with seed zero through @c sz_hash_best, as @c sz::hash(text) does. */
    std::size_t operator()(string_view_t text) const noexcept { return static_cast<std::size_t>(hash(text)); }
};

} // namespace stringzilla
} // namespace ashvardanian

namespace std {

template <>
struct hash<ashvardanian::stringzilla::string_view_t> {
    size_t operator()(ashvardanian::stringzilla::string_view_t str) const noexcept {
        return static_cast<size_t>(ashvardanian::stringzilla::hash(str));
    }
};

template <>
struct hash<ashvardanian::stringzilla::string_t> {
    size_t operator()(ashvardanian::stringzilla::string_t const &str) const noexcept {
        return static_cast<size_t>(ashvardanian::stringzilla::hash(str));
    }
};

} // namespace std

#endif // STRINGZILLA_HASH_HPP_
