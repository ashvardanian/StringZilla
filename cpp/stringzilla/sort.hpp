/**
 *  @file cpp/stringzilla/sort.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief Permutations that sort any container of strings, into a span or a fresh vector.
 */
#ifndef STRINGZILLA_SORT_HPP_
#define STRINGZILLA_SORT_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include <concepts> // `std::convertible_to`
#include <vector>   // vendor-boundary `std::vector`

#include "stringzilla/sort.h" // `sz_sort_order_best`, `sz_sort_uncased_order_best`

#include "stringzilla/string.hpp"

#include "stringzilla/devices.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief Internal data-structure wrapping arbitrary sequential containers for random lookups.
 *  @sa argsort, intersect
 */
template <typename container_type_, typename string_extractor_>
struct strings_args_ {

    /** The elements, read by index. */
    container_type_ const &container;

    /** Turns one element into the string it holds. */
    string_extractor_ const &extractor;

    /** The @c sz_strings_get_t accessor, extracting every requested element. */
    static void get(void const *handle, sz_size_t const *indices, sz_size_t count, sz_string_view_t *views) {
        strings_args_ const *args = reinterpret_cast<strings_args_ const *>(handle);
        for (sz_size_t position = 0; position != count; ++position) {
            string_view_t member = args->extractor(args->container[indices[position]]);
            views[position].start = member.data();
            views[position].length = static_cast<sz_size_t>(member.size());
        }
    }

    /** The accessor-backed collection of every element, borrowing these arguments. */
    sz_strings_view_t view() const noexcept {
        sz_strings_view_t strings;
        strings.layout = sz_strings_accessor_k, strings.count = static_cast<sz_size_t>(container.size());
        strings.data.accessor.get = &get, strings.data.accessor.handle = this;
        return strings;
    }
};

/** The order @c argsort arranges strings in. */
enum class sort_direction_t : bool {

    /** Smallest first. */
    ascending_k,

    /** Largest first. */
    descending_k,
};

/**
 *  @brief Computes the permutation of an array, that would lead to sorted order. The elements of
 *      the array must be convertible to a @c string_view_t with the given extractor. Unlike the
 *      @c sz_sort_order_best C interface, overwrites the output span.
 *  @sa sz_sort_order_best
 *
 *  @param[in] container The array of string-like elements to sort.
 *  @param[in] extractor The function object that extracts the string from the object.
 *  @param[out] order The caller-owned output span of indices, populated with the permutation.
 *      Must hold at least <tt>container.size()</tt> entries; the whole span receives a permutation.
 *  @param[in] top_count If non-zero, only the first @p top_count entries are guaranteed fully
 *      sorted (partial sort).
 *  @param[in] direction Whether the smallest or the largest strings come first.
 *  @param[in] allocator Serves the scratch memory of the sort.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @par Statuses:
 *  - @c unexpected_dimensions_k @p order is too short.
 */
template <typename container_type_, typename string_extractor_, typename allocator_type_ = std::allocator<char>>
status_t argsort(container_type_ const &container, string_extractor_ const &extractor, std::span<sz_size_t> order,
                 std::size_t top_count = 0, sort_direction_t direction = sort_direction_t::ascending_k,
                 allocator_type_ allocator = {}, sz_capability_t capabilities = default_capabilities(),
                 sz_stream_t stream = nullptr) noexcept {
    if (order.size() < container.size()) return status_t::unexpected_dimensions_k;

    // Pack the arguments into a single structure to reference it from the callback.
    using args_t = strings_args_<container_type_, string_extractor_>;
    args_t args {container, extractor};
    sz_strings_view_t const strings = args.view();

    using sz_alloc_type = sz_allocator_t;
    typename std::allocator_traits<allocator_type_>::template rebind_alloc<char> bytes_allocator(allocator);
    return with_allocator_(bytes_allocator, [&](sz_alloc_type &c_allocator) {
        return sz_sort_order_best(&strings, static_cast<sz_size_t>(top_count), static_cast<sz_bool_t>(direction),
                                  &c_allocator, order.data(), capabilities, stream);
    });
}

/**
 *  @brief Uncased (Unicode case-folded) counterpart of @c argsort over an output span.
 *  @sa sz_sort_uncased_order_best
 */
template <typename container_type_, typename string_extractor_, typename allocator_type_ = std::allocator<char>>
status_t argsort_utf8_uncased(container_type_ const &container, string_extractor_ const &extractor,
                              std::span<sz_size_t> order, std::size_t top_count = 0,
                              sort_direction_t direction = sort_direction_t::ascending_k,
                              allocator_type_ allocator = {}, sz_capability_t capabilities = default_capabilities(),
                              sz_stream_t stream = nullptr) noexcept {
    if (order.size() < container.size()) return status_t::unexpected_dimensions_k;

    using args_t = strings_args_<container_type_, string_extractor_>;
    args_t args {container, extractor};
    sz_strings_view_t const strings = args.view();

    using sz_alloc_type = sz_allocator_t;
    typename std::allocator_traits<allocator_type_>::template rebind_alloc<char> bytes_allocator(allocator);
    return with_allocator_(bytes_allocator, [&](sz_alloc_type &c_allocator) {
        return sz_sort_uncased_order_best(&strings, static_cast<sz_size_t>(top_count),
                                          static_cast<sz_bool_t>(direction), &c_allocator, order.data(), capabilities,
                                          stream);
    });
}

/** A callable exposing each element of @p container_type_ as something a @c string_view_t is built
 *  from. */
template <typename extractor_type_, typename container_type_>
concept a_string_extractor = requires(extractor_type_ const &extract, container_type_ const &container,
                                      std::size_t index) {
    { extract(container[index]) } -> std::convertible_to<string_view_t>;
};

/** Indices into a sorted or intersected array, in memory from the caller's allocator. */
template <typename allocator_type_>
using sorted_indices = std::vector<sz_size_t, // vendor-boundary
                                   typename std::allocator_traits<allocator_type_>::template rebind_alloc<sz_size_t>>;

/** Resizes @p container, reading its throw as @c bad_alloc_k, for the @c noexcept functions that
 *  return STL containers. */
template <typename container_type_>
status_t resize_or_bad_alloc_(container_type_ &container, std::size_t count) noexcept {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
    try {
        container.resize(count);
    }
    catch (...) {
        return status_t::bad_alloc_k;
    }
#else
    container.resize(count);
#endif
    return status_t::success_k;
}

/**
 *  @brief Computes the permutation of an array, that would lead to sorted order.
 *  @param[in] allocator Serves the returned indices and the scratch memory of the sort.
 *  @return The permutation.
 */
template <typename container_type_, a_string_extractor<container_type_> string_extractor_,
          typename allocator_type_ = std::allocator<char>>
expected<sorted_indices<allocator_type_>> argsort( //
    container_type_ const &container, string_extractor_ const &extractor, std::size_t top_count = 0,
    sort_direction_t direction = sort_direction_t::ascending_k, allocator_type_ allocator = {},
    sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {
    sorted_indices<allocator_type_> order(allocator);
    status_t status = resize_or_bad_alloc_(order, container.size());
    if (succeeded(status))
        status = argsort(container, extractor, {order.data(), order.size()}, top_count, direction, allocator,
                         capabilities, stream);
    return {std::move(order), status};
}

/** @overload */
template <typename container_type_, typename allocator_type_ = std::allocator<char>>
    requires std::is_convertible_v<typename container_type_::value_type, string_view_t>
expected<sorted_indices<allocator_type_>> argsort(container_type_ const &container, std::size_t top_count = 0,
                                                  sort_direction_t direction = sort_direction_t::ascending_k,
                                                  allocator_type_ allocator = {},
                                                  sz_capability_t capabilities = default_capabilities(),
                                                  sz_stream_t stream = nullptr) noexcept {
    using string_like_type = typename container_type_::value_type;
    return argsort(
        container, [](string_like_type const &s) -> string_view_t { return s; }, top_count, direction, allocator,
        capabilities, stream);
}

/**
 *  @brief Uncased (Unicode case-folded) permutation that would lead to sorted order.
 *  @param[in] allocator Serves the returned indices and the scratch memory of the sort.
 *  @return The permutation.
 */
template <typename container_type_, a_string_extractor<container_type_> string_extractor_,
          typename allocator_type_ = std::allocator<char>>
expected<sorted_indices<allocator_type_>> argsort_utf8_uncased( //
    container_type_ const &container, string_extractor_ const &extractor, std::size_t top_count = 0,
    sort_direction_t direction = sort_direction_t::ascending_k, allocator_type_ allocator = {},
    sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {
    sorted_indices<allocator_type_> order(allocator);
    status_t status = resize_or_bad_alloc_(order, container.size());
    if (succeeded(status))
        status = argsort_utf8_uncased(container, extractor, {order.data(), order.size()}, top_count, direction,
                                      allocator, capabilities, stream);
    return {std::move(order), status};
}

/** @overload */
template <typename container_type_, typename allocator_type_ = std::allocator<char>>
    requires std::is_convertible_v<typename container_type_::value_type, string_view_t>
expected<sorted_indices<allocator_type_>> argsort_utf8_uncased( //
    container_type_ const &container, std::size_t top_count = 0,
    sort_direction_t direction = sort_direction_t::ascending_k, allocator_type_ allocator = {},
    sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {
    using string_like_type = typename container_type_::value_type;
    return argsort_utf8_uncased(
        container, [](string_like_type const &s) -> string_view_t { return s; }, top_count, direction, allocator,
        capabilities, stream);
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SORT_HPP_
